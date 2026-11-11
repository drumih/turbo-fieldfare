import Foundation
import Metal

/// unchecked-invariant: the buffer is written once at fetch and then only read
/// by GPU commands; no mutating access escapes after `load` returns.
package struct GGUFExpertMatrix: @unchecked Sendable {
    package let buffer: MTLBuffer
    package let rows: Int
    package let columns: Int
    package let type: GGUFType
    package let byteCount: Int
}

/// unchecked-invariant: carries three immutable matrices; the cache owns their
/// lifetime and never rewrites them.
package struct GGUFExpertBuffers: @unchecked Sendable {
    package let layer: Int
    package let expert: Int
    package let gate: GGUFExpertMatrix
    package let up: GGUFExpertMatrix
    package let down: GGUFExpertMatrix
    /// The cache slot these matrices live in; the cache reuses slots, so a
    /// caller must not hold them across a slot's refill.
    package let slotIndex: Int

    package var fetchedBytes: Int {
        gate.byteCount + up.byteCount + down.byteCount
    }
}

package enum GGUFExpertBufferCacheError: Error, Equatable, CustomStringConvertible {
    case invalidCapacity(Int)
    case shortRead(name: String, expected: Int, got: Int)
    case missingTensor(String)
    case tooManyExperts(requested: Int, capacity: Int)

    package var description: String {
        switch self {
        case .invalidCapacity(let value):
            return "expert cache capacity \(value) must be at least 1"
        case .shortRead(let name, let expected, let got):
            return "short read of \(name): expected \(expected) bytes, got \(got)"
        case .missingTensor(let name):
            return "the model has no expert tensor \(name)"
        case .tooManyExperts(let requested, let capacity):
            return "\(requested) experts requested from a \(capacity)-slot cache"
        }
    }
}

/// Reads one expert's three matrices into Metal buffers on demand and keeps a
/// few of them, keyed by (layer, expert).
///
/// This is the streaming half of the as-is path. The engine's affine streamer
/// copies fixed-stride expert blobs into page-aligned slots; a GGUF expert is
/// three separate byte ranges in a shard with a block format per matrix, so the
/// fetch has to assemble them and the GEMV has to be told each matrix's type.
/// Reads go straight into the shared MTLBuffer, so the GPU sees the bytes the
/// fused GEMV will read without a host copy.
///
/// unchecked-invariant: every mutable field (`descriptors`, `order`, `entries`,
/// `slots`, `freeSlots`) is read and written only while `lock` is held - in
/// `experts(layer:experts:)` and the `load`/`descriptor` calls it makes under the
/// lock. A slot handed out is rewritten only when it is evicted, and with
/// `capacity >= top-k` that happens on a later layer's fetch, after the caller
/// has waited for the GPU work reading it.
package final class GGUFExpertBufferCache: @unchecked Sendable {
    private let model: GGUFModel
    private let device: MTLDevice
    private let capacity: Int
    private let lock = NSLock()
    private var descriptors: [Int: Int32] = [:]
    private var order: [CacheKey] = []
    private var entries: [CacheKey: GGUFExpertBuffers] = [:]
    /// Fixed slot storage: allocating a buffer per miss cost ~480 five-megabyte
    /// allocations a token and most of the routed phase. Each slot holds the
    /// three matrices at the largest size they can be, and a miss fills it.
    private var slots: [Slot] = []
    private var freeSlots: [Int] = []

    private struct CacheKey: Hashable {
        let layer: Int
        let expert: Int
    }

    private struct Slot {
        let gate: MTLBuffer
        let up: MTLBuffer
        let down: MTLBuffer
    }

    /// Largest routed matrices this checkpoint stores: gate/up are 640 x 2560
    /// at the widest format they use (IQ4_XS, 136 bytes per 256 values, layer
    /// 47), down is 2560 x 640 at IQ4_NL (18 bytes per 32 values).
    private static let gateBytes = 640 * 2_560 * 136 / 256
    private static let downBytes = 2_560 * 640 * 18 / 32

    package init(
        model: GGUFModel, device: MTLDevice, capacity: Int = 8
    ) throws {
        guard capacity >= 1 else {
            throw GGUFExpertBufferCacheError.invalidCapacity(capacity)
        }
        self.model = model
        self.device = device
        self.capacity = capacity
        for _ in 0..<capacity {
            guard
                let gate = device.makeBuffer(
                    length: Self.gateBytes, options: .storageModeShared),
                let up = device.makeBuffer(
                    length: Self.gateBytes, options: .storageModeShared),
                let down = device.makeBuffer(
                    length: Self.downBytes, options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed("expert cache slots")
            }
            slots.append(Slot(gate: gate, up: up, down: down))
            freeSlots.append(slots.count - 1)
        }
    }

    deinit {
        for descriptor in descriptors.values { close(descriptor) }
    }

    package func expert(layer: Int, expert: Int) throws -> GGUFExpertBuffers {
        try experts(layer: layer, experts: [expert])[0]
    }

    /// Fetches `experts` of `layer`, reading every missing matrix at once.
    ///
    /// A token's routed experts are known together, so their reads are issued
    /// together. Fetched one expert at a time, the drive saw at most three
    /// outstanding reads, and the decode trace measured the fetch at 583 ms of
    /// a 970 ms token (2026-11-11, 24 GB M3). Hits are refreshed first and
    /// misses evict from the least-recent end, so with `experts.count <=
    /// capacity` a miss never evicts an expert of the same request.
    package func experts(layer: Int, experts: [Int]) throws -> [GGUFExpertBuffers] {
        guard experts.count <= capacity else {
            throw GGUFExpertBufferCacheError.tooManyExperts(
                requested: experts.count, capacity: capacity)
        }
        lock.lock()
        defer { lock.unlock() }
        var misses: [CacheKey] = []
        for expert in experts {
            let key = CacheKey(layer: layer, expert: expert)
            if entries[key] != nil {
                if let position = order.firstIndex(of: key) {
                    order.remove(at: position)
                    order.append(key)
                }
            } else if !misses.contains(key) {
                misses.append(key)
            }
        }
        var planned: [(key: CacheKey, slot: Int)] = []
        for key in misses {
            if let free = freeSlots.popLast() {
                planned.append((key, free))
                continue
            }
            let evicted = order.removeFirst()
            guard let victim = entries.removeValue(forKey: evicted) else {
                throw GGUFExpertBufferCacheError.invalidCapacity(0)
            }
            planned.append((key, victim.slotIndex))
        }
        do {
            let loaded = try load(layer: layer, planned: planned)
            for (plan, buffers) in zip(planned, loaded) {
                entries[plan.key] = buffers
                order.append(plan.key)
            }
        } catch {
            // The planned slots hold partial reads; return them to the pool
            // rather than losing them.
            freeSlots.append(contentsOf: planned.map(\.slot))
            throw error
        }
        return try experts.map { expert in
            guard let buffers = entries[CacheKey(layer: layer, expert: expert)] else {
                throw GGUFExpertBufferCacheError.missingTensor("layer \(layer) expert \(expert)")
            }
            return buffers
        }
    }

    /// Reads one matrix into its buffer with `pread`, which carries its own
    /// offset, so concurrent reads can share one descriptor per shard.
    private static func read(
        _ range: GGUFTensorRange, into buffer: MTLBuffer, name: String, descriptor: Int32
    ) throws -> Int {
        guard Int(range.byteCount) <= buffer.length else {
            throw GGUFExpertBufferCacheError.shortRead(
                name: name, expected: Int(range.byteCount), got: buffer.length)
        }
        let destination = buffer.contents()
        var total = 0
        while total < Int(range.byteCount) {
            let count = min(Int(range.byteCount) - total, 1 << 20)
            let read = pread(
                descriptor, destination.advanced(by: total), count,
                off_t(range.fileOffset) + off_t(total))
            if read <= 0 {
                throw GGUFExpertBufferCacheError.shortRead(
                    name: name, expected: Int(range.byteCount), got: total)
            }
            total += read
        }
        return total
    }

    /// unchecked-invariant: each job index is run by exactly one task and the
    /// inputs are immutable; only the failure slot is shared, behind a lock.
    private final class ReadJobs: @unchecked Sendable {
        struct Job {
            let range: GGUFTensorRange
            let buffer: MTLBuffer
            let descriptor: Int32
            let name: String
        }

        let jobs: [Job]
        private let lock = NSLock()
        private var failure: Error?

        init(_ jobs: [Job]) { self.jobs = jobs }

        func run(_ index: Int) {
            let job = jobs[index]
            do {
                _ = try GGUFExpertBufferCache.read(
                    job.range, into: job.buffer, name: job.name, descriptor: job.descriptor)
            } catch {
                lock.lock()
                if failure == nil { failure = error }
                lock.unlock()
            }
        }

        var firstFailure: Error? {
            lock.lock()
            defer { lock.unlock() }
            return failure
        }
    }

    /// Reads the planned experts' matrices, all of them concurrently.
    private func load(layer: Int, planned: [(key: CacheKey, slot: Int)]) throws
        -> [GGUFExpertBuffers]
    {
        guard !planned.isEmpty else { return [] }
        /// GGUF dims are [columns, rows, experts]; the record's shape is
        /// [rows, columns], which the fused GEMV dispatch depends on.
        func shape(of role: String) throws -> (rows: Int, columns: Int) {
            let name = "blk.\(layer).\(role)_exps.weight"
            guard let location = model.tensor(named: name),
                location.tensor.dimensions.count == 3
            else {
                throw GGUFExpertBufferCacheError.missingTensor(name)
            }
            return (Int(location.tensor.dimensions[1]), Int(location.tensor.dimensions[0]))
        }
        let gateShape = try shape(of: "ffn_gate")
        let upShape = try shape(of: "ffn_up")
        let downShape = try shape(of: "ffn_down")
        let layouts = try planned.map {
            try GGUFExpertLayout.weights(layer: layer, expert: $0.key.expert, in: model)
        }
        var jobs: [ReadJobs.Job] = []
        jobs.reserveCapacity(planned.count * 3)
        for (plan, weights) in zip(planned, layouts) {
            let slot = slots[plan.slot]
            for (range, buffer, name) in [
                (weights.gate, slot.gate, "gate"), (weights.up, slot.up, "up"),
                (weights.down, slot.down, "down"),
            ] {
                jobs.append(
                    ReadJobs.Job(
                        range: range, buffer: buffer,
                        descriptor: try descriptor(for: range.shardIndex), name: name))
            }
        }
        let reads = ReadJobs(jobs)
        DispatchQueue.concurrentPerform(iterations: jobs.count) { reads.run($0) }
        if let error = reads.firstFailure { throw error }
        return zip(planned, layouts).map { plan, weights in
            let slot = slots[plan.slot]
            return GGUFExpertBuffers(
                layer: layer, expert: plan.key.expert,
                gate: GGUFExpertMatrix(
                    buffer: slot.gate, rows: gateShape.rows, columns: gateShape.columns,
                    type: weights.gate.type, byteCount: Int(weights.gate.byteCount)),
                up: GGUFExpertMatrix(
                    buffer: slot.up, rows: upShape.rows, columns: upShape.columns,
                    type: weights.up.type, byteCount: Int(weights.up.byteCount)),
                down: GGUFExpertMatrix(
                    buffer: slot.down, rows: downShape.rows, columns: downShape.columns,
                    type: weights.down.type, byteCount: Int(weights.down.byteCount)),
                slotIndex: plan.slot)
        }
    }

    private func descriptor(for shard: Int) throws -> Int32 {
        if let existing = descriptors[shard] { return existing }
        let opened = open(model.shards[shard].url.path, O_RDONLY | O_CLOEXEC)
        guard opened >= 0 else {
            throw MetalError.invalidState(
                "cannot open \(model.shards[shard].url.lastPathComponent): errno \(errno)")
        }
        descriptors[shard] = opened
        return opened
    }
}
