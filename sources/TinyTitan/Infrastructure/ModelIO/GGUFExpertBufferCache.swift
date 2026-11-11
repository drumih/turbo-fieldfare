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

/// Reads routed experts' three matrices into fixed Metal slots on demand and
/// keeps them keyed by (layer, expert).
///
/// This is the streaming half of the as-is path. A GGUF expert is three
/// separate byte ranges in a shard with a block format per matrix, so the fetch
/// assembles them and the GEMV is told each matrix's type. Reads go straight
/// into shared MTLBuffers, so the GPU reads the fetched bytes with no host copy.
///
/// **Pinning.** `experts(layer:experts:)` pins what it returns until the next
/// call: the caller encodes GPU work on those slots and waits for it before it
/// asks again, so a later miss never overwrites them meanwhile.
///
/// A speculative prefetch of the next layer's experts was built and measured
/// on this cache and rejected (2026-11-11, `docs/qwen38-gguf-backend-plan.md`):
/// the drive is already near its read limit, so early reads of guessed experts
/// only moved bandwidth around, and a top-6 guess was slower.
///
/// unchecked-invariant: every mutable field (`descriptors`, `order`, `entries`,
/// `pinned`, `counters`, `freeSlots`) is read and written only while `lock` is
/// held. The concurrent reads a request issues run while the request holds the
/// lock and touch only the slot buffers it assigned them.
package final class GGUFExpertBufferCache: @unchecked Sendable {
    private let model: GGUFModel
    private let device: MTLDevice
    private let capacity: Int
    private let lock = NSLock()
    private var descriptors: [Int: Int32] = [:]
    private var order: [CacheKey] = []
    private var entries: [CacheKey: GGUFExpertBuffers] = [:]
    private var pinned: Set<CacheKey> = []
    private var counters = Statistics()
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

    /// How requests were served, cumulative since the cache was built.
    package struct Statistics: Sendable, Equatable {
        /// Requested experts that were already resident.
        package var hits = 0
        /// Requested experts read on demand.
        package var misses = 0
    }

    package var statistics: Statistics {
        lock.lock()
        defer { lock.unlock() }
        return counters
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

    /// Fetches `experts` of `layer` and pins them until the next call.
    ///
    /// Resident experts are used as they are, and every remaining matrix is
    /// read at once (up to 30
    /// `pread`s): fetched one expert at a time, the drive saw at most three
    /// outstanding reads, which the decode trace measured at 583 ms of a 970 ms
    /// token (2026-11-11, 24 GB M3).
    package func experts(layer: Int, experts: [Int]) throws -> [GGUFExpertBuffers] {
        guard experts.count <= capacity else {
            throw GGUFExpertBufferCacheError.tooManyExperts(
                requested: experts.count, capacity: capacity)
        }
        lock.lock()
        defer { lock.unlock() }
        let keys = experts.map { CacheKey(layer: layer, expert: $0) }
        pinned = Set(keys)
        var misses: [CacheKey] = []
        for key in keys {
            if entries[key] != nil {
                touch(key)
                counters.hits += 1
            } else if !misses.contains(key) {
                misses.append(key)
            }
        }
        counters.misses += misses.count
        var planned: [(key: CacheKey, slot: Int)] = []
        for key in misses {
            guard let slot = claimSlot() else {
                freeSlots.append(contentsOf: planned.map(\.slot))
                throw GGUFExpertBufferCacheError.tooManyExperts(
                    requested: experts.count, capacity: capacity)
            }
            planned.append((key, slot))
        }
        do {
            let plan = try readPlan(layer: layer, planned: planned)
            try plan.reads.runAll()
            for (index, entry) in planned.enumerated() {
                entries[entry.key] = plan.buffers[index]
                order.append(entry.key)
            }
        } catch {
            // The planned slots hold partial reads; return them to the pool.
            freeSlots.append(contentsOf: planned.map(\.slot))
            throw error
        }
        return try keys.map { key in
            guard let buffers = entries[key] else {
                throw GGUFExpertBufferCacheError.missingTensor(
                    "layer \(key.layer) expert \(key.expert)")
            }
            return buffers
        }
    }

    private func touch(_ key: CacheKey) {
        if let position = order.firstIndex(of: key) {
            order.remove(at: position)
            order.append(key)
        }
    }

    /// A free slot, or the least recently used unpinned entry's. Nil when
    /// every slot is pinned. The caller holds `lock`.
    private func claimSlot() -> Int? {
        if let free = freeSlots.popLast() { return free }
        guard let position = order.firstIndex(where: { !pinned.contains($0) }) else {
            return nil
        }
        let victim = order.remove(at: position)
        return entries.removeValue(forKey: victim)?.slotIndex
    }

    private struct ReadPlan {
        let reads: ReadJobs
        let buffers: [GGUFExpertBuffers]
    }

    /// The reads that fill `planned` and the buffers they produce. The caller
    /// holds `lock` (descriptors are opened lazily).
    private func readPlan(layer: Int, planned: [(key: CacheKey, slot: Int)]) throws -> ReadPlan {
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
        guard !planned.isEmpty else { return ReadPlan(reads: ReadJobs([]), buffers: []) }
        let gate = try shape(of: "ffn_gate")
        let up = try shape(of: "ffn_up")
        let down = try shape(of: "ffn_down")
        var jobs: [ReadJobs.Job] = []
        var buffers: [GGUFExpertBuffers] = []
        for entry in planned {
            let weights = try GGUFExpertLayout.weights(
                layer: layer, expert: entry.key.expert, in: model)
            let slot = slots[entry.slot]
            for (range, buffer, name) in [
                (weights.gate, slot.gate, "gate"), (weights.up, slot.up, "up"),
                (weights.down, slot.down, "down"),
            ] {
                jobs.append(
                    ReadJobs.Job(
                        range: range, buffer: buffer,
                        descriptor: try descriptor(for: range.shardIndex), name: name))
            }
            buffers.append(
                GGUFExpertBuffers(
                    layer: layer, expert: entry.key.expert,
                    gate: GGUFExpertMatrix(
                        buffer: slot.gate, rows: gate.rows, columns: gate.columns,
                        type: weights.gate.type, byteCount: Int(weights.gate.byteCount)),
                    up: GGUFExpertMatrix(
                        buffer: slot.up, rows: up.rows, columns: up.columns,
                        type: weights.up.type, byteCount: Int(weights.up.byteCount)),
                    down: GGUFExpertMatrix(
                        buffer: slot.down, rows: down.rows, columns: down.columns,
                        type: weights.down.type, byteCount: Int(weights.down.byteCount)),
                    slotIndex: entry.slot))
        }
        return ReadPlan(reads: ReadJobs(jobs), buffers: buffers)
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
