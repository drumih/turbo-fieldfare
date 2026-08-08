import Darwin
import Foundation
import Metal

/// Expert-readahead policy for `K3ExpertStreaming`.
public enum K3ExpertPrefetchPolicy: String, Sendable {
    /// Pure on-demand: every selected expert is read inside `beginLayer`.
    case off
    /// Temporal prediction (default): last token's routing per MoE layer is
    /// prefetched into the idle bank during the current layer's I/O wait.
    case predict
}

/// Maximum number of expert-file `pread` calls allowed to run at once.
/// Adaptive mode measures equal-sized K3 reads during the first few layers
/// and then pins the best observed worker count for the engine lifetime.
public enum K3ExpertIOWorkers: Sendable, Equatable {
    case adaptive
    case fixed(Int)

    public static let allowedFixedRange = 1...32
}

/// Darwin expert-read cache policy. `uncached` applies `F_NOCACHE` to each
/// lazily opened layer descriptor; it does not require multiple disks.
public enum K3ExpertIOCachePolicy: String, Sendable, Equatable {
    /// Use uncached reads for production-sized K3 expert blobs, buffered I/O
    /// for small synthetic/test layouts, and fall back if the hint is absent.
    case automatic = "auto"
    case buffered
    case uncached
}

/// Small online tuner for the bounded expert-I/O worker pool. Throughput is
/// compared as bytes/nanosecond, so demand and prefill batches with different
/// byte counts remain comparable. Each candidate gets two observations before
/// a winner is pinned; no model semantics or file placement changes.
struct K3ExpertIOAutotuner: Sendable {
    struct Plan: Sendable, Equatable {
        let workers: Int
        let candidateIndex: Int?
    }

    private let candidates: [Int]
    private let samplesPerCandidate: Int
    private var sampleCount: [Int]
    private var throughputSum: [Double]
    private var nextCandidateIndex = 0
    private(set) var selectedWorkers: Int?

    init(maxWorkers: Int = min(4, max(1, ProcessInfo.processInfo.activeProcessorCount)),
         samplesPerCandidate: Int = 2) {
        precondition(maxWorkers > 0)
        precondition(samplesPerCandidate > 0)
        var values = [1]
        while let last = values.last, last < maxWorkers {
            values.append(min(last * 2, maxWorkers))
        }
        self.candidates = Array(Set(values)).sorted()
        self.samplesPerCandidate = samplesPerCandidate
        self.sampleCount = [Int](repeating: 0, count: self.candidates.count)
        self.throughputSum = [Double](repeating: 0, count: self.candidates.count)
    }

    init(candidates: [Int], samplesPerCandidate: Int = 2) {
        precondition(!candidates.isEmpty && candidates.allSatisfy { $0 > 0 })
        precondition(samplesPerCandidate > 0)
        self.candidates = Array(Set(candidates)).sorted()
        self.samplesPerCandidate = samplesPerCandidate
        self.sampleCount = [Int](repeating: 0, count: self.candidates.count)
        self.throughputSum = [Double](repeating: 0, count: self.candidates.count)
    }

    mutating func plan(chunkCount: Int) -> Plan {
        precondition(chunkCount > 0)
        if let selectedWorkers {
            return Plan(workers: min(selectedWorkers, chunkCount), candidateIndex: nil)
        }
        // Interleave candidates by round (1,2,4,...,1,2,4,...) so cache
        // warming and thermals do not systematically favor the last worker
        // count. Skip candidates this batch cannot actually exercise.
        for offset in 0..<candidates.count {
            let index = (nextCandidateIndex + offset) % candidates.count
            if chunkCount >= candidates[index] {
                nextCandidateIndex = (index + 1) % candidates.count
                return Plan(workers: candidates[index], candidateIndex: index)
            }
        }
        return Plan(workers: 1, candidateIndex: nil)
    }

    mutating func record(plan: Plan, bytes: UInt64, nanos: UInt64) {
        guard selectedWorkers == nil,
              let index = plan.candidateIndex,
              bytes > 0, nanos > 0 else { return }
        sampleCount[index] += 1
        throughputSum[index] += Double(bytes) / Double(nanos)
        guard sampleCount.indices.allSatisfy({ sampleCount[$0] >= samplesPerCandidate })
        else { return }
        var best = 0
        for i in 1..<candidates.count {
            let lhs = throughputSum[i] / Double(max(sampleCount[i], 1))
            let rhs = throughputSum[best] / Double(max(sampleCount[best], 1))
            if lhs > rhs { best = i }
        }
        selectedWorkers = candidates[best]
    }

    var currentWorkers: Int {
        selectedWorkers ?? candidates[nextCandidateIndex]
    }

    var observationCount: Int { sampleCount.reduce(0, +) }
    var isComplete: Bool { selectedWorkers != nil }
}

/// Counters for the streaming diagnostics (CLI/HUD). All counts are cumulative
/// since init or the last `resetStats()`.
public struct K3ExpertStreamingStats: Sendable, Equatable {
    /// Selected experts already resident in the serving bank (prediction hits
    /// or lucky reuse).
    public var demandHits: UInt64 = 0
    /// Selected experts read on demand inside `beginLayer`.
    public var demandMisses: UInt64 = 0
    /// Expert reads issued as prediction prefetches.
    public var prefetchesIssued: UInt64 = 0
    /// Prefetch reads skipped because the target bank was still in use
    /// (runner did not release the previous layer's batch in time).
    public var prefetchSkippedBankBusy: UInt64 = 0
    /// Prefetch reads skipped because no prediction existed yet (first token
    /// after init / `resetPrediction()`).
    public var prefetchSkippedCold: UInt64 = 0
    public var demandBytes: UInt64 = 0
    public var prefetchBytes: UInt64 = 0
    /// Chunked-prefill expert bytes read through the shared bounded pool.
    public var prefillBytes: UInt64 = 0
    /// Successful bounded I/O batches and their aggregate wall time.
    public var ioBatches: UInt64 = 0
    public var ioNanos: UInt64 = 0
    /// Current fixed/adaptive worker limit and observed peak active preads.
    public var ioWorkerLimit: Int = 0
    public var peakConcurrentReads: Int = 0
    public var ioTuningObservations: Int = 0
    public var ioTuningComplete: Bool = false
    /// Resolved descriptor policy (`buffered` or `uncached`).
    public var ioCacheMode: String = K3ExpertIOCachePolicy.buffered.rawValue

    public init() {}

    public var bytesRead: UInt64 { demandBytes + prefetchBytes + prefillBytes }
    public var demandTotal: UInt64 { demandHits + demandMisses }
    public var hitRate: Double {
        demandTotal == 0 ? 0 : Double(demandHits) / Double(demandTotal)
    }
}

/// One layer's worth of resident experts: the bank buffer plus, per selected
/// expert (in request order), the byte offset of its slot inside `buffer`.
/// Bind directly into the `K3MoE` phase kernels (`experts` + `slotOffsets`).
/// `@unchecked Sendable` matches `TensorView`: the `MTLBuffer` is thread-safe.
public struct K3ExpertBatch: @unchecked Sendable {
    public let layer: Int
    public let bankIndex: Int
    public let buffer: MTLBuffer
    public let slotOffsets: [UInt64]

    public init(layer: Int, bankIndex: Int, buffer: MTLBuffer, slotOffsets: [UInt64]) {
        self.layer = layer
        self.bankIndex = bankIndex
        self.buffer = buffer
        self.slotOffsets = slotOffsets
    }
}

/// SSD streaming engine for K3 routed experts (docs/KIMI_K3_EVALUATION.md §5):
///
/// - **Global slot pool**: 2 banks × `slotsPerBank` slots, each slot exactly
///   one expert blob (`expertStride` bytes, 16 KB-aligned). Slots are generic
///   across layers — a slot's content is keyed `(layer, expert)`, and a bank
///   serves whichever MoE layer is currently decoding. Each bank is ONE
///   `posix_memalign`'d region wrapped in ONE bytesNoCopy shared `MTLBuffer`
///   so a layer's 16 experts bind as buffer + UInt64 offsets (the `K3MoE`
///   production binding).
/// - **Bounded pread fanout**: by default every selected K3 expert is one
///   17.5 MB `pread`; optional `ioSplits` keep page-aligned subreads available
///   for explicit A/B tests. The jobs are drained by a fixed or online-tuned
///   worker bound and waited as a group. Decode and chunked prefill share this
///   scheduler.
/// - **Temporal prediction**: `recordRouting` stores each MoE layer's actual
///   top-16 for the token; at `beginLayer(L)` the serving bank is expected to
///   hold the predicted set already (prefetched during the previous layer),
///   misses are demand-read, and — during the same I/O window — the next MoE
///   layer's predicted set is prefetched into the OTHER bank. Banks ping-pong
///   per MoE layer visit.
///
/// GPU-safety contract: a bank's bytes may be consumed by in-flight GPU work
/// until the runner calls `endLayer(_:)` (e.g. from the consuming command
/// buffer's completion handler). Prefetch never writes a live bank; if the
/// bank is still in use the prefetch is skipped and counted, degrading to
/// on-demand reads — correctness never depends on the runner's pacing.
///
/// Threading: `beginLayer` / `endLayer` / `recordRouting` are called from the
/// single decode-loop thread; the async prefetch task joins caller state via
/// its DispatchGroup. Stats reads are lock-guarded.
/// `@unchecked Sendable` matches `PreadExpertStreamer`.
public final class K3ExpertStreaming: @unchecked Sendable {
    public static let defaultSlotsPerBank = 16
    /// K3 expert files are already laid out as contiguous 17.5 MB blobs. One
    /// syscall per expert avoids multiplying queue depth and matched the M5
    /// Max whole-expert I/O probe; callers can still request split A/Bs.
    public static let defaultIOSplits = 1
    public static let slotAlignment = 16 * 1024

    public let policy: K3ExpertPrefetchPolicy
    public let slotsPerBank: Int
    public let ioSplits: Int
    public let ioWorkers: K3ExpertIOWorkers
    public let ioCachePolicy: K3ExpertIOCachePolicy
    public let expertStride: UInt64
    public let expertsPerLayer: Int
    /// Sorted 0-based MoE layer ids (the visit order of `beginLayer`).
    public let moeLayers: [Int]

    private let layerFileProvider: (Int) throws -> K3ExpertLayerFile
    private let ioQueue: DispatchQueue

    private final class Bank {
        let pointer: UnsafeMutableRawPointer
        let buffer: MTLBuffer
        var slotLayer: [Int32]    // -1 = empty
        var slotExpert: [Int32]
        /// True between `beginLayer` returning this bank and `endLayer`.
        var liveBatch = false
        var prefetchGroup: DispatchGroup?
        var prefetchError: Error?

        init(pointer: UnsafeMutableRawPointer, buffer: MTLBuffer, slots: Int) {
            self.pointer = pointer
            self.buffer = buffer
            self.slotLayer = [Int32](repeating: -1, count: slots)
            self.slotExpert = [Int32](repeating: -1, count: slots)
        }
    }

    private var banks: [Bank]
    private var layerFiles: [Int: K3ExpertLayerFile] = [:]
    private var predictions: [Int: [Int]] = [:]
    private var visitCount = 0
    private var statsStorage = K3ExpertStreamingStats()
    private let statsLock = NSLock()
    private var ioTuner: K3ExpertIOAutotuner
    private let ioTunerLock = NSLock()
    private let filePolicyLock = NSLock()
    private var configuredFileDescriptors: Set<Int32> = []
    private var uncachedIOEnabled: Bool

    /// Production initializer: layer files come from the model (lazy open +
    /// verify on first touch).
    public convenience init(model: K3Model,
                            policy: K3ExpertPrefetchPolicy = .predict,
                            slotsPerBank: Int = K3ExpertStreaming.defaultSlotsPerBank,
                            ioSplits: Int = K3ExpertStreaming.defaultIOSplits,
                            ioWorkers: K3ExpertIOWorkers = .adaptive,
                            ioCachePolicy: K3ExpertIOCachePolicy = .automatic) throws {
        try self.init(
            device: model.device,
            expertStride: model.expertsLayout.expertStride,
            expertsPerLayer: model.expertsLayout.expertsPerLayer,
            moeLayers: model.config.moeLayers0.sorted(),
            policy: policy,
            slotsPerBank: slotsPerBank,
            ioSplits: ioSplits,
            ioWorkers: ioWorkers,
            ioCachePolicy: ioCachePolicy,
            layerFileProvider: { try model.expertLayerFile($0) })
    }

    /// Designated initializer (tests drive it with synthetic layer files).
    public init(device: MTLDevice,
                expertStride: UInt64,
                expertsPerLayer: Int,
                moeLayers: [Int],
                policy: K3ExpertPrefetchPolicy = .predict,
                slotsPerBank: Int = K3ExpertStreaming.defaultSlotsPerBank,
                ioSplits: Int = K3ExpertStreaming.defaultIOSplits,
                ioWorkers: K3ExpertIOWorkers = .adaptive,
                ioCachePolicy: K3ExpertIOCachePolicy = .automatic,
                layerFileProvider: @escaping (Int) throws -> K3ExpertLayerFile) throws {
        precondition(expertStride > 0
                        && expertStride % UInt64(K3ExpertStreaming.slotAlignment) == 0,
                     "expertStride must be a positive multiple of 16 KB")
        precondition(expertsPerLayer > 0)
        precondition(slotsPerBank > 0)
        precondition(ioSplits > 0)
        if case .fixed(let workers) = ioWorkers {
            precondition(K3ExpertIOWorkers.allowedFixedRange.contains(workers),
                         "fixed expert I/O workers must be in "
                            + "\(K3ExpertIOWorkers.allowedFixedRange)")
        }
        precondition(!moeLayers.isEmpty && moeLayers == moeLayers.sorted())
        self.policy = policy
        self.slotsPerBank = slotsPerBank
        self.ioSplits = ioSplits
        self.ioWorkers = ioWorkers
        self.ioCachePolicy = ioCachePolicy
        self.expertStride = expertStride
        self.expertsPerLayer = expertsPerLayer
        self.moeLayers = moeLayers
        self.layerFileProvider = layerFileProvider
        self.ioQueue = DispatchQueue(label: "turbo-fieldfare.k3-expert-io",
                                     qos: .userInitiated,
                                     attributes: .concurrent)
        self.ioTuner = K3ExpertIOAutotuner()
        // Tiny fixture files benefit from ordinary cache semantics. The real
        // K3 blob is 17,547,264 bytes, safely above this threshold.
        self.uncachedIOEnabled = ioCachePolicy == .uncached
            || (ioCachePolicy == .automatic && expertStride >= 16 * 1024 * 1024)

        let bankBytes = slotsPerBank * Int(expertStride)
        var banks: [Bank] = []
        for index in 0..<2 {
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, K3ExpertStreaming.slotAlignment, bankBytes)
            guard result == 0, let pointer = raw else {
                throw StreamerError.allocFailed(errno: result)
            }
            nonisolated(unsafe) let capturedPointer = pointer
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: bankBytes,
                options: .storageModeShared,
                deallocator: { _, _ in free(capturedPointer) })
            else {
                free(pointer)
                throw StreamerError.bufferWrapFailed
            }
            buffer.label = "k3.experts.bank\(index)"
            banks.append(Bank(pointer: pointer, buffer: buffer, slots: slotsPerBank))
        }
        self.banks = banks
        self.statsStorage.ioCacheMode = self.uncachedIOEnabled
            ? K3ExpertIOCachePolicy.uncached.rawValue
            : K3ExpertIOCachePolicy.buffered.rawValue
    }

    deinit {
        for file in layerFiles.values { close(file.fileDescriptor) }
    }

    /// Total slot-pool bytes (both banks).
    public var slotPoolBytes: Int { 2 * slotsPerBank * Int(expertStride) }

    // MARK: - Per-layer service

    /// Serve layer `layer0`'s actual top-k routing. Blocks on the bank's
    /// in-flight prefetch (if any), demand-reads misses into the serving bank,
    /// then schedules the next MoE layer's predicted set into the idle bank.
    /// Returns the bank buffer + per-expert slot offsets for the MoE kernels.
    ///
    /// The caller MUST call `endLayer(_:)` once the GPU work consuming the
    /// returned batch has completed, before the bank's next `beginLayer`.
    public func beginLayer(_ layer0: Int, actualExperts: [Int]) throws -> K3ExpertBatch {
        precondition(moeLayers.contains(layer0),
                     "layer \(layer0) is not an MoE layer")
        precondition(actualExperts.count <= slotsPerBank,
                     "top-k \(actualExperts.count) exceeds slots per bank \(slotsPerBank)")
        for expert in actualExperts {
            precondition(expert >= 0 && expert < expertsPerLayer,
                         "expert id \(expert) out of range")
        }
        let bankIndex = visitCount & 1
        let bank = banks[bankIndex]
        let other = banks[bankIndex ^ 1]
        precondition(!bank.liveBatch,
                     "bank \(bankIndex) still serves layer-visit \(visitCount - 2); "
                        + "call endLayer(_:) before re-entering it")

        // The prefetch for this layer was scheduled during the previous visit.
        if let group = bank.prefetchGroup {
            group.wait()
            bank.prefetchGroup = nil
            if let error = bank.prefetchError {
                bank.prefetchError = nil
                throw error
            }
        }

        let file = try layerFile(for: layer0)
        let assigned = assignSlots(bank: bank, layer: layer0, experts: actualExperts)
        var hits = 0
        var missJobs: [(expert: Int, slot: Int)] = []
        for (index, slot) in assigned.enumerated() where !slot.wasHit {
            missJobs.append((actualExperts[index], slot.slot))
        }
        for slot in assigned where slot.wasHit { hits += 1 }

        // Demand reads, waited. Update keys only after the bytes landed.
        if !missJobs.isEmpty {
            try readJobs(missJobs.map { (file: file, expert: $0.expert,
                                         destination: bank.pointer.advanced(
                                            by: $0.slot * Int(expertStride))) })
            for job in missJobs {
                bank.slotLayer[job.slot] = Int32(layer0)
                bank.slotExpert[job.slot] = Int32(job.expert)
            }
        }

        // Prediction prefetch of the next MoE layer into the idle bank,
        // scheduled inside the same I/O window (async; waited at its layer).
        var prefetchIssued = 0
        var skippedBusy = 0
        var skippedCold = 0
        if policy == .predict, let next = nextMoELayer(after: layer0) {
            if let predicted = predictions[next] {
                if other.liveBatch || other.prefetchGroup != nil {
                    skippedBusy = predicted.count
                } else {
                    let nextFile = try layerFile(for: next)
                    let jobs = planPrefetch(bank: other, file: nextFile,
                                            predicted: predicted)
                    prefetchIssued = jobs.count
                    if !jobs.isEmpty {
                        let group = DispatchGroup()
                        other.prefetchGroup = group
                        other.prefetchError = nil
                        nonisolated(unsafe) let jobs = jobs
                        nonisolated(unsafe) let other = other
                        ioQueue.async(group: group) { [self] in
                            do {
                                try readJobsSync(jobs, bank: other)
                            } catch {
                                other.prefetchError = error
                            }
                        }
                    }
                }
            } else {
                skippedCold = 1
            }
        }

        statsLock.lock()
        statsStorage.demandHits &+= UInt64(hits)
        statsStorage.demandMisses &+= UInt64(missJobs.count)
        statsStorage.demandBytes &+= UInt64(missJobs.count) &* expertStride
        statsStorage.prefetchesIssued &+= UInt64(prefetchIssued)
        statsStorage.prefetchBytes &+= UInt64(prefetchIssued) &* expertStride
        statsStorage.prefetchSkippedBankBusy &+= UInt64(skippedBusy)
        statsStorage.prefetchSkippedCold &+= UInt64(skippedCold)
        statsLock.unlock()

        bank.liveBatch = true
        visitCount += 1
        return K3ExpertBatch(
            layer: layer0,
            bankIndex: bankIndex,
            buffer: bank.buffer,
            slotOffsets: assigned.map { UInt64($0.slot) * expertStride })
    }

    /// Release the batch's bank after its consuming GPU work completed.
    public func endLayer(_ batch: K3ExpertBatch) {
        precondition(banks[batch.bankIndex].liveBatch,
                     "endLayer without a live batch on bank \(batch.bankIndex)")
        banks[batch.bankIndex].liveBatch = false
    }

    /// Record layer `layer0`'s actual routing for this token; it becomes the
    /// prediction prefetched on the NEXT token.
    public func recordRouting(layer0: Int, experts: [Int]) {
        predictions[layer0] = experts
    }

    /// Forget all predictions (fresh conversation, diagnostics).
    public func resetPrediction() {
        predictions.removeAll()
    }

    /// Snapshot only the semantic-free routing predictor. Expert bank bytes
    /// remain an opportunistic cache and are never required for correctness.
    func capturePredictions() -> [Int: [Int]] { predictions }

    /// Restore predictor history at a prefix boundary. Any read scheduled by
    /// a previous run is joined first; a failed speculative read is discarded
    /// and will degrade to a demand read on the next visit.
    func restorePredictions(_ restored: [Int: [Int]]) {
        for bank in banks {
            precondition(!bank.liveBatch,
                         "cannot restore predictions while a GPU batch is live")
            if let group = bank.prefetchGroup { group.wait() }
            bank.prefetchGroup = nil
            bank.prefetchError = nil
        }
        predictions = restored
    }

    public func stats() -> K3ExpertStreamingStats {
        statsLock.lock()
        defer { statsLock.unlock() }
        return statsStorage
    }

    public func resetStats() {
        let tuning = ioTuningSnapshot()
        filePolicyLock.lock()
        let ioCacheMode = uncachedIOEnabled
            ? K3ExpertIOCachePolicy.uncached.rawValue
            : K3ExpertIOCachePolicy.buffered.rawValue
        filePolicyLock.unlock()
        statsLock.lock()
        statsStorage = K3ExpertStreamingStats()
        statsStorage.ioWorkerLimit = tuning.workers
        statsStorage.ioTuningObservations = tuning.observations
        statsStorage.ioTuningComplete = tuning.complete
        statsStorage.ioCacheMode = ioCacheMode
        statsLock.unlock()
    }

    /// Chunked prefill uses the same bounded/adaptive I/O path as decode.
    /// Destinations are one-to-one with `experts` and must each have room for
    /// `expertStride` bytes.
    func readExpertsForPrefill(file: K3ExpertLayerFile,
                               experts: [Int],
                               destinations: [UnsafeMutableRawPointer]) throws {
        precondition(experts.count == destinations.count)
        try configureCachePolicy(for: file)
        let jobs = zip(experts, destinations).map {
            (file: file, expert: $0.0, destination: $0.1)
        }
        try readJobs(jobs)
        statsLock.lock()
        statsStorage.prefillBytes &+= UInt64(experts.count) &* expertStride
        statsLock.unlock()
    }

    /// Test hook: `(layer, expert)` currently keyed in a bank's slots.
    public func bankContents(_ bankIndex: Int) -> [(layer: Int32, expert: Int32)] {
        let bank = banks[bankIndex]
        return Array(zip(bank.slotLayer, bank.slotExpert))
    }

    // MARK: - Slot planning

    /// Hits keep their slots; misses take slots whose content is NOT in the
    /// actual set (stale or predicted-but-unused). `slotsPerBank >= topK`
    /// makes this total.
    private func assignSlots(bank: Bank, layer: Int, experts: [Int])
        -> [(slot: Int, wasHit: Bool)] {
        var assigned = [(slot: Int, wasHit: Bool)]()
        assigned.reserveCapacity(experts.count)
        var reserved = [Bool](repeating: false, count: slotsPerBank)
        for expert in experts {
            var found = -1
            for slot in 0..<slotsPerBank
                where !reserved[slot]
                    && bank.slotLayer[slot] == Int32(layer)
                    && bank.slotExpert[slot] == Int32(expert) {
                found = slot
                break
            }
            if found >= 0 {
                reserved[found] = true
                assigned.append((found, true))
            } else {
                assigned.append((-1, false))
            }
        }
        let wanted = Set(experts)
        for (index, _) in experts.enumerated() where assigned[index].slot == -1 {
            var target = -1
            for slot in 0..<slotsPerBank where !reserved[slot] {
                let holdsWanted = bank.slotLayer[slot] == Int32(layer)
                    && wanted.contains(Int(bank.slotExpert[slot]))
                if !holdsWanted {
                    target = slot
                    break
                }
            }
            if target == -1 {
                for slot in 0..<slotsPerBank where !reserved[slot] {
                    target = slot
                    break
                }
            }
            precondition(target != -1, "slot pool cannot place the requested set")
            reserved[target] = true
            assigned[index] = (target, false)
        }
        return assigned
    }

    /// Slots of `bank` not already holding `(layer, predicted)` content are
    /// refill candidates; predicted experts already resident are skipped.
    private func planPrefetch(bank: Bank, file: K3ExpertLayerFile, predicted: [Int])
        -> [(file: K3ExpertLayerFile, expert: Int, destination: UnsafeMutableRawPointer)] {
        var distinct: [Int] = []
        var seen = Set<Int>()
        for expert in predicted where seen.insert(expert).inserted {
            distinct.append(expert)
        }
        var jobs: [(file: K3ExpertLayerFile, expert: Int,
                    destination: UnsafeMutableRawPointer)] = []
        var reserved = [Bool](repeating: false, count: slotsPerBank)
        let layer = file.layer
        for expert in distinct {
            var resident = false
            for slot in 0..<slotsPerBank
                where bank.slotLayer[slot] == Int32(layer)
                    && bank.slotExpert[slot] == Int32(expert) {
                // Keep the resident copy; mark it reserved so it is not
                // overwritten by a later job in this same batch.
                if !reserved[slot] { reserved[slot] = true }
                resident = true
                break
            }
            if resident { continue }
            var target = -1
            for slot in 0..<slotsPerBank where !reserved[slot] {
                target = slot
                break
            }
            guard target != -1 else { break }
            reserved[target] = true
            jobs.append((file,
                         expert,
                         bank.pointer.advanced(by: target * Int(expertStride))))
        }
        return jobs
    }

    // MARK: - I/O

    private func layerFile(for layer0: Int) throws -> K3ExpertLayerFile {
        if let file = layerFiles[layer0] { return file }
        let file = try layerFileProvider(layer0)
        try configureCachePolicy(for: file)
        layerFiles[layer0] = file
        return file
    }

    /// Apply `F_NOCACHE` once per descriptor. Automatic mode falls back to
    /// buffered I/O if the platform refuses the hint; an explicit uncached
    /// request reports the failure instead of silently changing the A/B arm.
    private func configureCachePolicy(for file: K3ExpertLayerFile) throws {
        filePolicyLock.lock()
        defer { filePolicyLock.unlock() }
        guard uncachedIOEnabled,
              !configuredFileDescriptors.contains(file.fileDescriptor) else { return }
        if fcntl(file.fileDescriptor, F_NOCACHE, 1) == 0 {
            configuredFileDescriptors.insert(file.fileDescriptor)
            return
        }
        let failure = errno
        if ioCachePolicy == .automatic {
            uncachedIOEnabled = false
            statsLock.lock()
            statsStorage.ioCacheMode = K3ExpertIOCachePolicy.buffered.rawValue
            statsLock.unlock()
            return
        }
        throw ModelError.posixFailed(call: "fcntl(F_NOCACHE)", errno: failure)
    }

    /// Next MoE layer in visit order. Wraps around to the first MoE layer at
    /// the end of the schedule so the FIRST MoE layer of the next token is
    /// prefetched during the last layer's I/O window too.
    private func nextMoELayer(after layer0: Int) -> Int? {
        for layer in moeLayers where layer > layer0 { return layer }
        return moeLayers.first
    }

    /// Prefetch task body: read all jobs, then publish slot keys. Runs inside
    /// the bank's DispatchGroup, so `beginLayer`'s `group.wait()` orders the
    /// key publish before any hit lookup.
    private func readJobsSync(
        _ jobs: [(file: K3ExpertLayerFile, expert: Int,
                  destination: UnsafeMutableRawPointer)],
        bank: Bank
    ) throws {
        try readJobs(jobs)
        let strideBytes = Int(expertStride)
        for job in jobs {
            let slotIndex = (job.destination - bank.pointer) / strideBytes
            bank.slotLayer[slotIndex] = Int32(job.file.layer)
            bank.slotExpert[slotIndex] = Int32(job.expert)
        }
    }

    /// Split every job into page-aligned chunks, then drain that chunk list
    /// with at most the selected number of workers. This keeps synchronous
    /// `pread` queue depth bounded instead of asking the global pool to run
    /// all top-k × split operations simultaneously.
    private func readJobs(
        _ jobs: [(file: K3ExpertLayerFile, expert: Int,
                  destination: UnsafeMutableRawPointer)]
    ) throws {
        guard !jobs.isEmpty else { return }
        let strideBytes = Int(expertStride)
        let pageBytes = K3ExpertStreaming.slotAlignment
        let totalPages = strideBytes / pageBytes
        let splits = min(ioSplits, max(totalPages, 1))
        // Page counts per split, remainder distributed to the leading chunks.
        let basePages = totalPages / splits
        let extraPages = totalPages % splits

        struct Chunk {
            let fd: Int32
            let fileOffset: UInt64
            let destination: UnsafeMutableRawPointer
            let byteCount: Int
        }
        var chunkList: [Chunk] = []
        chunkList.reserveCapacity(jobs.count * splits)
        for job in jobs {
            guard job.expert >= 0 && job.expert < job.file.expertOffsets.count else {
                throw StreamerError.offsetOutOfRange(UInt64(job.expert))
            }
            let expertOffset = job.file.expertOffsets[job.expert]
            var pageCursor = 0
            for split in 0..<splits {
                let pages = basePages + (split < extraPages ? 1 : 0)
                if pages > 0 {
                    chunkList.append(Chunk(
                        fd: job.file.fileDescriptor,
                        fileOffset: expertOffset + UInt64(pageCursor * pageBytes),
                        destination: job.destination.advanced(by: pageCursor * pageBytes),
                        byteCount: pages * pageBytes))
                }
                pageCursor += pages
            }
        }
        let tuningPlan = ioPlan(chunkCount: chunkList.count)
        let errorLock = NSLock()
        let cursorLock = NSLock()
        let activityLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        nonisolated(unsafe) var nextChunk = 0
        nonisolated(unsafe) var activeReads = 0
        nonisolated(unsafe) var peakReads = 0
        nonisolated(unsafe) let chunks = chunkList
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        DispatchQueue.concurrentPerform(iterations: tuningPlan.workers) { _ in
            while true {
                cursorLock.lock()
                let index: Int?
                if nextChunk < chunks.count {
                    index = nextChunk
                    nextChunk += 1
                } else {
                    index = nil
                }
                cursorLock.unlock()
                guard let index else { return }

                let chunk = chunks[index]
                activityLock.lock()
                activeReads += 1
                peakReads = max(peakReads, activeReads)
                activityLock.unlock()
                do {
                    var filled = 0
                    while filled < chunk.byteCount {
                        let readCount = pread(
                            chunk.fd,
                            chunk.destination.advanced(by: filled),
                            chunk.byteCount - filled,
                            off_t(chunk.fileOffset) + off_t(filled))
                        if readCount < 0, errno == EINTR { continue }
                        if readCount < 0 { throw StreamerError.preadFailed(errno: errno) }
                        if readCount == 0 {
                            throw StreamerError.sizeMismatch(
                                expected: UInt64(chunk.byteCount), actual: UInt64(filled))
                        }
                        filled += readCount
                    }
                } catch {
                    errorLock.lock()
                    if firstError == nil { firstError = error }
                    errorLock.unlock()
                }
                activityLock.lock()
                activeReads -= 1
                activityLock.unlock()
            }
        }
        if let firstError { throw firstError }
        let nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
        let bytes = chunkList.reduce(UInt64(0)) { $0 &+ UInt64($1.byteCount) }
        recordIO(plan: tuningPlan, bytes: bytes, nanos: nanos, peakReads: peakReads)
    }

    private func ioPlan(chunkCount: Int) -> K3ExpertIOAutotuner.Plan {
        precondition(chunkCount > 0)
        switch ioWorkers {
        case .fixed(let workers):
            return K3ExpertIOAutotuner.Plan(
                workers: min(workers, chunkCount), candidateIndex: nil)
        case .adaptive:
            ioTunerLock.lock()
            defer { ioTunerLock.unlock() }
            return ioTuner.plan(chunkCount: chunkCount)
        }
    }

    private func recordIO(plan: K3ExpertIOAutotuner.Plan,
                          bytes: UInt64,
                          nanos: UInt64,
                          peakReads: Int) {
        if ioWorkers == .adaptive {
            ioTunerLock.lock()
            ioTuner.record(plan: plan, bytes: bytes, nanos: nanos)
            let tuning = (ioTuner.currentWorkers,
                          ioTuner.observationCount,
                          ioTuner.isComplete)
            ioTunerLock.unlock()
            statsLock.lock()
            statsStorage.ioWorkerLimit = tuning.0
            statsStorage.ioTuningObservations = tuning.1
            statsStorage.ioTuningComplete = tuning.2
        } else {
            statsLock.lock()
            statsStorage.ioWorkerLimit = plan.workers
            statsStorage.ioTuningComplete = true
        }
        statsStorage.ioBatches &+= 1
        statsStorage.ioNanos &+= nanos
        statsStorage.peakConcurrentReads = max(
            statsStorage.peakConcurrentReads, peakReads)
        statsLock.unlock()
    }

    private func ioTuningSnapshot() -> (workers: Int, observations: Int, complete: Bool) {
        switch ioWorkers {
        case .fixed(let workers):
            return (workers, 0, true)
        case .adaptive:
            ioTunerLock.lock()
            defer { ioTunerLock.unlock() }
            return (ioTuner.currentWorkers,
                    ioTuner.observationCount,
                    ioTuner.isComplete)
        }
    }
}
