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
    struct Layer {
        let stride: Int
        let file: Int32
        let path: String
        let slots: MTLBuffer
        let gateUpScale: MTLBuffer?
        let downScale: MTLBuffer?
        let gateUpScaleOffset: Int?
        let downScaleOffset: Int?
        let gateUpOffset: Int
        let downOffset: Int
    }

    package struct Statistics: Equatable {
        package var hits: UInt64 = 0
        package var misses: UInt64 = 0
        package var bytesRead: UInt64 = 0
        package var rounds: UInt64 = 0
        package var syncs: UInt64 = 0
        /// Wall time spent reading experts from the install.
        package var readSeconds: Double = 0
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
        for index in 0..<manifest.storedLayers {
            guard let plan = install.layout.layers.first(where: { $0.layer == index }),
                plan.experts.count == expertCount, let first = plan.experts.first,
                let gateUp = first.tensors["gate_up"], let down = first.tensors["down"]
            else { throw InklingQ4Install.Error.layout("layer \(index) has no full expert list") }
            let stride = Int(expertCount > 1 ? plan.experts[1].offset - first.offset : first.size)
            for (rank, expert) in plan.experts.enumerated()
            where expert.offset != UInt64(rank * stride) || expert.tensors != first.tensors {
                throw InklingQ4Install.Error.layout(
                    "layer \(index) expert \(rank) breaks the uniform stride")
            }
            let path = install.directory.appendingPathComponent("packed_experts/" + plan.file).path
            let fd = open(path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { throw ModelError.posixFailed(call: "open(\(path))", errno: errno) }
            // The slots are the cache; the page cache would hold a second copy.
            _ = fcntl(fd, F_NOCACHE, 1)
            var info = stat()
            guard fstat(fd, &info) == 0, Int(info.st_size) >= expertCount * stride else {
                close(fd)
                throw InklingQ4Install.Error.layout("\(plan.file) is shorter than its experts")
            }
            guard
                let buffer = device.makeBuffer(
                    length: slotCount * stride, options: .storageModeShared)
            else {
                close(fd)
                throw MetalError.bufferAllocationFailed("Inkling Q4 expert slots, layer \(index)")
            }
            func scaleBuffer(_ key: String) throws -> (MTLBuffer?, Int?) {
                guard let slice = first.tensors[key] else { return (nil, nil) }
                guard slice.size == 4, slice.dtype == "F32",
                    let made = device.makeBuffer(length: slotCount * 4, options: .storageModeShared)
                else { throw InklingQ4Install.Error.layout("layer \(index) \(key) is not one F32") }
                return (made, Int(slice.offset))
            }
            let (gateUpScale, gateUpScaleOffset) = try scaleBuffer("gate_up_scale")
            let (downScale, downScaleOffset) = try scaleBuffer("down_scale")
            layers.append(
                Layer(
                    stride: stride, file: fd, path: path, slots: buffer,
                    gateUpScale: gateUpScale, downScale: downScale,
                    gateUpScaleOffset: gateUpScaleOffset, downScaleOffset: downScaleOffset,
                    gateUpOffset: Int(gateUp.offset), downOffset: Int(down.offset)))
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
                let victims = (0..<slotCount)
                    .filter { lastUse[layer][$0] < tick }
                    .sorted { lastUse[layer][$0] < lastUse[layer][$1] }
                    .prefix(loads.count)
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

    /// Reads `experts` into `slots` (in parallel, page cache bypassed) and
    /// copies their F32 scales into the per-slot arrays.
    private func load(layer index: Int, experts: [Int], into slots: [Int]) throws {
        let layer = layers[index]
        let started = Date()
        defer { statistics.readSeconds += Date().timeIntervalSince(started) }
        let failures = LoadFailures(count: experts.count)
        let base = UInt(bitPattern: layer.slots.contents())
        let fd = layer.file
        let stride = layer.stride
        DispatchQueue.concurrentPerform(iterations: experts.count) { job in
            guard let target = UnsafeMutableRawPointer(bitPattern: base + UInt(slots[job] * stride))
            else { return failures.record(job, EINVAL) }
            var done = 0
            while done < stride {
                let got = pread(
                    fd, target + done, stride - done, off_t(experts[job] * stride + done))
                if got < 0, errno == EINTR { continue }
                guard got > 0 else { return failures.record(job, got < 0 ? errno : EIO) }
                done += got
            }
        }
        if let (job, code) = failures.first {
            throw ModelError.posixFailed(
                call: "pread(\(layer.path), expert \(experts[job]))", errno: code)
        }
        statistics.bytesRead += UInt64(experts.count * stride)
        for (buffer, offset) in [
            (layer.gateUpScale, layer.gateUpScaleOffset), (layer.downScale, layer.downScaleOffset),
        ] {
            guard let buffer, let offset else { continue }
            let values = buffer.contents().bindMemory(to: Float.self, capacity: slotCount)
            for slot in slots {
                values[slot] = layer.slots.contents().loadUnaligned(
                    fromByteOffset: slot * stride + offset, as: Float.self)
            }
        }
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
