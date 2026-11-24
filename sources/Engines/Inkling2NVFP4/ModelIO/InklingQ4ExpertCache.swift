import Foundation
import Metal
import TinyTitanCore
import TinyTitanFormat

/// Where the routed experts live while the decoder runs.
package enum InklingQ4ExpertResidency: Sendable, Equatable {
    /// No-copy buffers over the install's layer files: every expert is reached
    /// through the page cache. The reference the cached modes are checked against.
    case mapped
    /// `slots` experts per layer in the process's own memory, read from the
    /// install with the page cache bypassed. At the layer's expert count every
    /// expert is read once at load and routing needs no CPU step; below it,
    /// each layer visit stops after its router, loads the experts it chose
    /// that are not resident (evicting the least recently used), and continues.
    case slots(Int)
}

/// Per-layer expert slots read from `packed_experts/layer_NN.bin` (TT-121
/// phase 2). One Metal buffer per layer holds `slots` expert blobs at the
/// install's stride, so the kernels index a slot exactly as they index an
/// expert in the mapped file; the per-expert F32 scales are kept per slot.
package final class InklingQ4ExpertCache {
    /// unchecked-invariant: every field is a `let` set at load; the buffers'
    /// contents are written only by reads into slots no GPU work is reading.
    struct Layer: @unchecked Sendable {
        /// Bytes from one slot to the next (the repacked expert, TT-125).
        let stride: Int
        let file: Int32
        let path: String
        let slots: MTLBuffer
        let gateUpScale: MTLBuffer?
        let downScale: MTLBuffer?
        let gateUpOffset: Int
        let downOffset: Int
        /// Where the checkpoint keeps the expert, and how it is repacked.
        let plan: InklingNVFP4SlotPlan
    }

    package struct Statistics: Equatable {
        package var hits: UInt64 = 0
        package var misses: UInt64 = 0
        package var bytesRead: UInt64 = 0
        package var rounds: UInt64 = 0
        package var syncs: UInt64 = 0
        /// Wall time spent reading experts from the install on the critical
        /// path (prefetch reads run beside the GPU and are not counted).
        package var readSeconds: Double = 0
        /// Blocking load calls (one per round that misses).
        package var loadCalls: UInt64 = 0
        /// Lookahead routing: experts predicted for a visit, how many of
        /// them the visit's router then chose, prefetch reads issued, the
        /// prefetched experts a router used, and the time a visit waited for
        /// a prefetch still in flight.
        package var predicted: UInt64 = 0
        package var predictedCorrect: UInt64 = 0
        package var prefetchLoads: UInt64 = 0
        package var prefetchUsed: UInt64 = 0
        package var prefetchWaitSeconds: Double = 0
    }

    package let slotCount: Int
    let expertCount: Int
    /// Every expert resident and slot `e` holding expert `e`: routing ids are
    /// slot ids and no visit needs a CPU step.
    let isIdentity: Bool
    private(set) var layers: [Layer] = []
    private var slotOf: [[Int]]
    private var expertIn: [[Int]]
    private var lastUse: [[UInt64]]
    private var tick: UInt64 = 0
    package private(set) var statistics = Statistics()
    /// When set, the trace prefetch policies are measured against: each
    /// visit's routed ids (depth 0), the lookahead prediction for the next
    /// visit (depth 1), and diagnostic predictions further ahead (depth d).
    package var routingLog: [(layer: Int, ids: [Int32], depth: Int)]?
    /// The prediction each layer's next visit is scored against.
    private var predictions: [Set<Int>]
    /// Experts loaded by prefetch and not yet chosen by a router, per layer.
    private var prefetched: [Set<Int>]
    private let prefetchQueue = DispatchQueue(
        label: "inkling-q4-prefetch", qos: .userInitiated, attributes: .concurrent)
    private let prefetchGroup = DispatchGroup()
    /// Per layer and slot, the background read still writing it (nil once
    /// it is known to have landed).
    private var landing: [[DispatchGroup?]]
    /// A visit waits only for the in-flight reads of the experts it routed
    /// (true), or for every background read issued so far (false).
    package var waitPerExpert = true
    private let prefetchFailure = PrefetchFailure()

    /// Bytes of expert slots (and their scales) this cache holds.
    package var residentBytes: Int {
        layers.reduce(0) { total, layer in
            total + layer.slots.length + (layer.gateUpScale?.length ?? 0)
                + (layer.downScale?.length ?? 0)
        }
    }

    init(install: InklingQ4Install, manifest: InklingQ4Manifest, slots: Int, device: MTLDevice)
        throws
    {
        expertCount = manifest.expertCount
        guard slots >= manifest.expertsPerToken else {
            throw InklingQ4Install.Error.layout(
                "\(slots) expert slots per layer is fewer than the \(manifest.expertsPerToken) "
                    + "experts a token routes to")
        }
        slotCount = min(slots, expertCount)
        isIdentity = slotCount == expertCount
        slotOf = Array(
            repeating: Array(repeating: -1, count: expertCount), count: manifest.storedLayers)
        expertIn = Array(
            repeating: Array(repeating: -1, count: slotCount), count: manifest.storedLayers)
        lastUse = Array(
            repeating: Array(repeating: 0, count: slotCount), count: manifest.storedLayers)
        predictions = Array(repeating: [], count: manifest.storedLayers)
        prefetched = Array(repeating: [], count: manifest.storedLayers)
        landing = Array(
            repeating: Array(repeating: nil, count: slots), count: manifest.storedLayers)
        for index in 0..<manifest.storedLayers {
            guard let plan = install.slotPlans.first(where: { $0.layer == index }),
                let fileLayer = install.layout.layers.first(where: { $0.layer == index }),
                fileLayer.experts.count == expertCount
            else { throw InklingQ4Install.Error.layout("layer \(index) has no full expert list") }
            for (rank, expert) in fileLayer.experts.enumerated()
            where expert.offset != UInt64(rank * plan.fileStride) {
                throw InklingQ4Install.Error.layout(
                    "layer \(index) expert \(rank) breaks the uniform stride")
            }
            let path = install.directory.appendingPathComponent("packed_experts/" + plan.file).path
            let fd = open(path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { throw ModelError.posixFailed(call: "open(\(path))", errno: errno) }
            // The slots are the cache; the page cache would hold a second copy.
            _ = fcntl(fd, F_NOCACHE, 1)
            var info = stat()
            guard fstat(fd, &info) == 0, Int(info.st_size) >= expertCount * plan.fileStride else {
                close(fd)
                throw InklingQ4Install.Error.layout("\(plan.file) is shorter than its experts")
            }
            guard
                let buffer = device.makeBuffer(
                    length: slotCount * plan.slotStride, options: .storageModeShared)
            else {
                close(fd)
                throw MetalError.bufferAllocationFailed(
                    "Inkling NVFP4 expert slots, layer \(index)")
            }
            // NVFP4 layers carry a per-expert scale (amax / denominator) that the
            // kernels index by slot.
            func scaleBuffer() throws -> MTLBuffer? {
                guard plan.kind == .nvfp4 else { return nil }
                guard
                    let made = device.makeBuffer(length: slotCount * 4, options: .storageModeShared)
                else { throw MetalError.bufferAllocationFailed("Inkling NVFP4 expert scales") }
                return made
            }
            layers.append(
                Layer(
                    stride: plan.slotStride, file: fd, path: path, slots: buffer,
                    gateUpScale: try scaleBuffer(), downScale: try scaleBuffer(),
                    gateUpOffset: plan.gateUp.slotOffset, downOffset: plan.down.slotOffset,
                    plan: plan))
        }
        if isIdentity {
            for layer in 0..<layers.count {
                try load(
                    layer: layer, experts: Array(0..<expertCount), into: Array(0..<expertCount))
            }
            statistics = Statistics()
        }
    }

    deinit {
        prefetchGroup.wait()
        for layer in layers { close(layer.file) }
    }

    /// The layer's slot buffer as the graph's expert tensors: expert `s` of
    /// these tensors is slot `s`.
    func tensors(layer manifestLayer: InklingQ4Manifest.Layer, hidden: Int)
        -> (
            gateUp: InklingQ4Tensor, down: InklingQ4Tensor, gateUpScale: InklingQ4Tensor?,
            downScale: InklingQ4Tensor?
        )
    {
        let layer = layers[manifestLayer.index]
        let feedForward = manifestLayer.expertFeedForward
        let span = slotCount * layer.stride
        func scale(_ buffer: MTLBuffer?) -> InklingQ4Tensor? {
            buffer.map {
                InklingQ4Tensor(
                    operand: .init($0), type: .f32, rows: slotCount, columns: 1,
                    byteCount: slotCount * 4)
            }
        }
        return (
            InklingQ4Tensor(
                operand: .init(layer.slots, offset: layer.gateUpOffset),
                type: manifestLayer.gateUpType, rows: 2 * feedForward, columns: hidden,
                byteCount: span),
            InklingQ4Tensor(
                operand: .init(layer.slots, offset: layer.downOffset),
                type: manifestLayer.downType, rows: hidden, columns: feedForward,
                byteCount: span),
            scale(layer.gateUpScale), scale(layer.downScale)
        )
    }

    /// The rounds a visit's routing needs: each round is the slot id of every
    /// pair (`slotCount`, one past the last slot, for a pair a later round
    /// computes), with every expert the round uses resident when it returns.
    /// Experts already resident run in the first round, so a round never
    /// evicts what it is about to use. `body` encodes one round and is told
    /// whether it is the last: every earlier round must have finished on the
    /// GPU when `body` returns, because the next round's loads may overwrite
    /// its slots.
    func resolve(
        layer: Int, routed ids: [Int32],
        round body: (_ slots: [Int32], _ last: Bool) throws -> Void
    ) throws {
        statistics.syncs += 1
        routingLog?.append((layer, ids, 0))
        // A prefetch may still be writing slots. The visit's resident experts
        // must have landed before the GPU reads them; with `waitPerExpert`
        // that is all it waits for, so a mispredicted read in flight and the
        // blocking reads below overlap instead of queueing.
        let waitStarted = Date()
        if waitPerExpert {
            for id in Set(ids.map(Int.init)) where (0..<expertCount).contains(id) {
                let slot = slotOf[layer][id]
                guard slot >= 0, let group = landing[layer][slot] else { continue }
                group.wait()
                landing[layer][slot] = nil
            }
        } else {
            prefetchGroup.wait()
        }
        statistics.prefetchWaitSeconds += Date().timeIntervalSince(waitStarted)
        if let failure = prefetchFailure.take() { throw failure }
        let chosen = Set(ids.map(Int.init))
        statistics.predictedCorrect += UInt64(predictions[layer].intersection(chosen).count)
        predictions[layer] = []
        statistics.prefetchUsed += UInt64(prefetched[layer].intersection(chosen).count)
        prefetched[layer].subtract(chosen)
        var distinct: [Int] = []
        var seen = Set<Int>()
        for id in ids where seen.insert(Int(id)).inserted {
            guard (0..<expertCount).contains(Int(id)) else {
                throw MetalError.invalidState("router chose expert \(id) of \(expertCount)")
            }
            distinct.append(Int(id))
        }
        let resident = distinct.filter { slotOf[layer][$0] >= 0 }
        let missing = distinct.filter { slotOf[layer][$0] < 0 }
        statistics.hits += UInt64(resident.count)
        statistics.misses += UInt64(missing.count)
        var pending = resident + missing
        while !pending.isEmpty {
            let round = Array(pending.prefix(slotCount))
            pending.removeFirst(round.count)
            tick += 1
            for expert in round where slotOf[layer][expert] >= 0 {
                lastUse[layer][slotOf[layer][expert]] = tick
            }
            let loads = round.filter { slotOf[layer][$0] < 0 }
            if !loads.isEmpty {
                // Free slots first, then the least recently used ones; slots
                // this round uses were just stamped with the current tick.
                let victims = evictionOrder(layer: layer).prefix(loads.count)
                guard victims.count == loads.count else {
                    throw MetalError.invalidState("no free expert slot in layer \(layer)")
                }
                let slots = Array(victims)
                for (expert, slot) in zip(loads, slots) {
                    let previous = expertIn[layer][slot]
                    if previous >= 0 { slotOf[layer][previous] = -1 }
                    expertIn[layer][slot] = expert
                    slotOf[layer][expert] = slot
                    lastUse[layer][slot] = tick
                }
                try load(layer: layer, experts: loads, into: slots)
            }
            statistics.rounds += 1
            let members = Set(round)
            try body(
                ids.map {
                    members.contains(Int($0)) ? Int32(slotOf[layer][Int($0)]) : Int32(slotCount)
                }, pending.isEmpty)
        }
    }

    /// Starts reading `experts` of `layer` into slots in the background: the
    /// next visit's predicted routing, loaded while the GPU runs. Slots go to
    /// the least recently used experts that are not themselves predicted.
    /// Every GPU read of the layer's slots has finished when this is called
    /// (the caller waited at the router), and `resolve` waits for the reads.
    package func prefetch(layer: Int, experts predicted: [Int32]) {
        var distinct: [Int] = []
        for id in predicted.map(Int.init)
        where (0..<expertCount).contains(id) && !distinct.contains(id) {
            distinct.append(id)
        }
        routingLog?.append((layer, predicted, 1))
        statistics.predicted += UInt64(distinct.count)
        predictions[layer] = Set(distinct)
        tick += 1
        for expert in distinct where slotOf[layer][expert] >= 0 {
            lastUse[layer][slotOf[layer][expert]] = tick
        }
        let loads = distinct.filter { slotOf[layer][$0] < 0 }
        let victims = Array(evictionOrder(layer: layer).prefix(loads.count))
        guard !victims.isEmpty else { return }
        let assigned = Array(loads.prefix(victims.count))
        for (expert, slot) in zip(assigned, victims) {
            let previous = expertIn[layer][slot]
            if previous >= 0 {
                slotOf[layer][previous] = -1
                prefetched[layer].remove(previous)
            }
            expertIn[layer][slot] = expert
            slotOf[layer][expert] = slot
            lastUse[layer][slot] = tick
            prefetched[layer].insert(expert)
        }
        statistics.prefetchLoads += UInt64(assigned.count)
        statistics.bytesRead += UInt64(assigned.count * layers[layer].stride)
        let target = layers[layer]
        let count = slotCount
        let failure = prefetchFailure
        let group = DispatchGroup()
        for slot in victims { landing[layer][slot] = group }
        group.enter()
        prefetchQueue.async(group: prefetchGroup) {
            defer { group.leave() }
            do {
                try Self.read(target, experts: assigned, into: victims, slotCount: count)
            } catch {
                failure.record(error)
            }
        }
    }

    /// Slots that may be evicted now (not stamped with the current tick),
    /// least recently used first. (Evicting the expert routed least often in
    /// the layer's last 250 routings instead was measured and rejected: more
    /// misses at 14 slots and no faster at 32 or 64; see the plan document.)
    /// A slot a background read is still writing is never a victim: a second
    /// write to it would race the first.
    private func evictionOrder(layer: Int) -> [Int] {
        (0..<slotCount)
            .filter { lastUse[layer][$0] < tick && landed(layer: layer, slot: $0) }
            .sorted { lastUse[layer][$0] < lastUse[layer][$1] }
    }

    private func landed(layer: Int, slot: Int) -> Bool {
        guard let group = landing[layer][slot] else { return true }
        guard group.wait(timeout: .now()) == .success else { return false }
        landing[layer][slot] = nil
        return true
    }

    /// Reads `experts` into `slots` on the critical path.
    private func load(layer index: Int, experts: [Int], into slots: [Int]) throws {
        let started = Date()
        defer { statistics.readSeconds += Date().timeIntervalSince(started) }
        statistics.loadCalls += 1
        try Self.read(layers[index], experts: experts, into: slots, slotCount: slotCount)
        statistics.bytesRead += UInt64(experts.count * layers[index].plan.fileStride)
    }

    /// Reads `experts` into `slots` (in parallel, page cache bypassed) and
    /// copies their F32 scales into the per-slot arrays. Touches only those
    /// slots, so it may run beside GPU work on the layer's other slots.
    private static func read(
        _ layer: Layer, experts: [Int], into slots: [Int], slotCount: Int
    ) throws {
        let failures = LoadFailures(count: experts.count)
        let base = UInt(bitPattern: layer.slots.contents())
        let fd = layer.file
        let plan = layer.plan
        let fileStride = plan.fileStride
        let gateUpScales = layer.gateUpScale.map { UInt(bitPattern: $0.contents()) }
        let downScales = layer.downScale.map { UInt(bitPattern: $0.contents()) }
        DispatchQueue.concurrentPerform(iterations: experts.count) { job in
            guard
                let target = UnsafeMutableRawPointer(
                    bitPattern: base + UInt(slots[job] * layer.stride))
            else { return failures.record(job, EINVAL) }
            // The checkpoint's planar expert, then repacked into the slot.
            let staging = UnsafeMutableRawPointer.allocate(byteCount: fileStride, alignment: 16_384)
            defer { staging.deallocate() }
            var done = 0
            while done < fileStride {
                let got = pread(
                    fd, staging + done, fileStride - done,
                    off_t(experts[job] * fileStride + done))
                if got < 0, errno == EINTR { continue }
                guard got > 0 else { return failures.record(job, got < 0 ? errno : EIO) }
                done += got
            }
            for (matrix, interleaved) in [(plan.gateUp, true), (plan.down, false)] {
                let values = UnsafeRawPointer(staging + matrix.valuesOffset)
                let scales = UnsafeRawPointer(staging + matrix.scalesOffset)
                let out = target + matrix.slotOffset
                switch plan.kind {
                case .nvfp4:
                    InklingNVFP4Repack.nvfp4(
                        values: values, scales: scales, rows: matrix.rows, columns: matrix.columns,
                        interleaved: interleaved, into: out)
                case .mxfp8:
                    InklingNVFP4Repack.mxfp8(
                        values: values, scales: scales, rows: matrix.rows, columns: matrix.columns,
                        interleaved: interleaved, into: out)
                }
            }
            // NVFP4: the expert's scale is its amax over the layer's denominator.
            for (address, offset, denominator) in [
                (gateUpScales, plan.gateUp.amaxOffset, plan.gateUpDenominator),
                (downScales, plan.down.amaxOffset, plan.downDenominator),
            ] {
                guard let address, let offset, let denominator,
                    let scales = UnsafeMutablePointer<Float>(bitPattern: address)
                else { continue }
                scales[slots[job]] =
                    staging.loadUnaligned(fromByteOffset: offset, as: Float.self) / denominator
            }
        }
        if let (job, code) = failures.first {
            throw ModelError.posixFailed(
                call: "pread(\(layer.path), expert \(experts[job]))", errno: code)
        }
    }
}

/// The first error a background prefetch hit, handed to the next `resolve`.
/// unchecked-invariant: every access holds `lock`.
private final class PrefetchFailure: @unchecked Sendable {
    private let lock = NSLock()
    private var error: Error?

    func record(_ failure: Error) {
        lock.lock()
        defer { lock.unlock() }
        if error == nil { error = failure }
    }

    func take() -> Error? {
        lock.lock()
        defer { lock.unlock() }
        defer { error = nil }
        return error
    }
}

/// The first failed read of a parallel load.
/// unchecked-invariant: each worker of one concurrentPerform writes only its
/// own slot, and `first` is read after concurrentPerform has returned.
private final class LoadFailures: @unchecked Sendable {
    private let codes: UnsafeMutableBufferPointer<Int32>

    init(count: Int) {
        codes = .allocate(capacity: max(count, 1))
        codes.initialize(repeating: 0)
    }

    deinit { codes.deallocate() }

    func record(_ job: Int, _ code: Int32) { codes[job] = code }

    var first: (Int, Int32)? {
        codes.indices.first { codes[$0] != 0 }.map { ($0, codes[$0]) }
    }
}
