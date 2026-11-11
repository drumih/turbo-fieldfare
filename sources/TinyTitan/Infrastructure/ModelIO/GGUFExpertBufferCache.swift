import Foundation
import Metal

/// unchecked-invariant: the buffer is written once at fetch and then only read
/// by GPU commands; no mutating access escapes after `load` returns.
package struct GGUFExpertMatrix: @unchecked Sendable {
    package let buffer: MTLBuffer
    /// Where the matrix starts in `buffer` (a slot of its layer's pool).
    package let offset: Int
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

/// Keeps routed experts resident in exact-size per-layer pools and reads the
/// missing ones in place, keyed by (layer, expert).
///
/// This is the streaming half of the as-is path. A GGUF expert is three byte
/// ranges in a shard with a block format per matrix; within one layer every
/// expert has the same three sizes (the format is per tensor), so each layer
/// owns one Metal buffer of `slotsPerLayer` equal slots and an expert's
/// matrices are addressed by offset. Reads go straight into that buffer, so the
/// GPU reads the fetched bytes with no host copy.
///
/// **Why a deep cache.** Routing is strongly reused across tokens: replaying
/// two 300-token generations (an essay and a code task, 2026-11-11), a per-layer
/// LRU of 21 / 42 / 64 / 84 / 125 experts hit 52-55% / 68-69% / 77% / 82% /
/// 88% of the requested experts in steady state. Every hit is a read the drive
/// no longer does, and the fetch was most of a token.
///
/// **Pinning.** `experts(layer:experts:)` pins what it returns until the next
/// call: the caller encodes GPU work on those slots and waits for it before it
/// asks again, so a later miss never overwrites them meanwhile.
///
/// A speculative prefetch of the next layer's experts was built and measured
/// and rejected (2026-11-11, `docs/qwen38-gguf-backend-plan.md`): the drive was
/// already near its read limit, so early reads of guessed experts only moved
/// bandwidth around.
///
/// unchecked-invariant: every mutable field (`descriptors`, `pools`, `pinned`,
/// `counters`) is read and written only while `lock` is held. The concurrent
/// reads a request issues run while the request holds the lock and touch only
/// the slots it assigned them.
package final class GGUFExpertBufferCache: @unchecked Sendable {
    private let model: GGUFModel
    /// Slots in every layer's pool; a request may ask for at most this many.
    package let capacity: Int
    private let lock = NSLock()
    private var descriptors: [Int: Int32] = [:]
    private var pools: [LayerPool] = []
    private var pinned: Set<CacheKey> = []
    private var counters = Statistics()

    private struct CacheKey: Hashable {
        let layer: Int
        let expert: Int
    }

    /// One layer's resident experts: an LRU over equal slots of one buffer.
    private struct LayerPool {
        let buffer: MTLBuffer
        let gateBytes: Int
        let upBytes: Int
        let downBytes: Int
        var entries: [Int: GGUFExpertBuffers] = [:]
        /// Resident experts, least recently used first.
        var order: [Int] = []
        var freeSlots: [Int]

        var slotBytes: Int { gateBytes + upBytes + downBytes }
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

    /// Resident bytes, all layers' pools together.
    package var residentBytes: Int {
        pools.reduce(0) { $0 + $1.buffer.length }
    }

    /// Bytes one slot of every layer takes together: what each unit of
    /// `capacity` costs.
    package static func bytesPerSlotAcrossLayers(model: GGUFModel) throws -> Int {
        try layerSizes(model: model).reduce(0) { $0 + $1.gate + $1.up + $1.down }
    }

    /// `capacity` slots in each layer's pool.
    package init(model: GGUFModel, device: MTLDevice, capacity: Int = 8) throws {
        guard capacity >= 1 else {
            throw GGUFExpertBufferCacheError.invalidCapacity(capacity)
        }
        self.model = model
        self.capacity = capacity
        for sizes in try Self.layerSizes(model: model) {
            let slotBytes = sizes.gate + sizes.up + sizes.down
            guard
                let buffer = device.makeBuffer(
                    length: capacity * slotBytes, options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed("expert pool")
            }
            pools.append(
                LayerPool(
                    buffer: buffer, gateBytes: sizes.gate, upBytes: sizes.up,
                    downBytes: sizes.down, freeSlots: Array((0..<capacity).reversed())))
        }
    }

    deinit {
        for descriptor in descriptors.values { close(descriptor) }
    }

    /// Each layer's expert matrix sizes, from expert 0 (every expert of a
    /// layer shares them).
    private static func layerSizes(model: GGUFModel) throws -> [(gate: Int, up: Int, down: Int)] {
        var sizes: [(gate: Int, up: Int, down: Int)] = []
        var layer = 0
        while model.tensor(named: "blk.\(layer).ffn_gate_exps.weight") != nil {
            let weights = try GGUFExpertLayout.weights(layer: layer, expert: 0, in: model)
            // Each matrix starts on a 256-byte boundary of its slot, so a
            // kernel's buffer offset is always aligned (every format here
            // already has sizes that are multiples of 256; this keeps it so).
            func aligned(_ bytes: UInt64) -> Int { (Int(bytes) + 255) / 256 * 256 }
            sizes.append(
                (
                    aligned(weights.gate.byteCount), aligned(weights.up.byteCount),
                    aligned(weights.down.byteCount)
                ))
            layer += 1
        }
        guard !sizes.isEmpty else {
            throw GGUFExpertBufferCacheError.missingTensor("blk.0.ffn_gate_exps.weight")
        }
        return sizes
    }

    package func expert(layer: Int, expert: Int) throws -> GGUFExpertBuffers {
        try experts(layer: layer, experts: [expert])[0]
    }

    /// Fetches `experts` of `layer` and pins them until the next call.
    ///
    /// Resident experts are used as they are, and every remaining matrix is
    /// read at once (up to 30 `pread`s): fetched one expert at a time, the
    /// drive saw at most three outstanding reads, which the decode trace
    /// measured at 583 ms of a 970 ms token (2026-11-11, 24 GB M3).
    package func experts(layer: Int, experts: [Int]) throws -> [GGUFExpertBuffers] {
        guard experts.count <= capacity else {
            throw GGUFExpertBufferCacheError.tooManyExperts(
                requested: experts.count, capacity: capacity)
        }
        guard pools.indices.contains(layer) else {
            throw GGUFExpertBufferCacheError.missingTensor("blk.\(layer).ffn_gate_exps.weight")
        }
        lock.lock()
        defer { lock.unlock() }
        pinned = Set(experts.map { CacheKey(layer: layer, expert: $0) })
        var misses: [Int] = []
        for expert in experts {
            if pools[layer].entries[expert] != nil {
                touch(layer: layer, expert: expert)
                counters.hits += 1
            } else if !misses.contains(expert) {
                misses.append(expert)
            }
        }
        counters.misses += misses.count
        var planned: [(expert: Int, slot: Int)] = []
        for expert in misses {
            guard let slot = claimSlot(layer: layer) else {
                pools[layer].freeSlots.append(contentsOf: planned.map(\.slot))
                throw GGUFExpertBufferCacheError.tooManyExperts(
                    requested: experts.count, capacity: capacity)
            }
            planned.append((expert, slot))
        }
        do {
            let plan = try readPlan(layer: layer, planned: planned)
            try plan.reads.runAll()
            for (index, entry) in planned.enumerated() {
                pools[layer].entries[entry.expert] = plan.buffers[index]
                pools[layer].order.append(entry.expert)
            }
        } catch {
            // The planned slots hold partial reads; return them to the pool.
            pools[layer].freeSlots.append(contentsOf: planned.map(\.slot))
            throw error
        }
        return try experts.map { expert in
            guard let buffers = pools[layer].entries[expert] else {
                throw GGUFExpertBufferCacheError.missingTensor(
                    "layer \(layer) expert \(expert)")
            }
            return buffers
        }
    }

    private func touch(layer: Int, expert: Int) {
        if let position = pools[layer].order.firstIndex(of: expert) {
            pools[layer].order.remove(at: position)
            pools[layer].order.append(expert)
        }
    }

    /// A free slot of `layer`, or its least recently used unpinned entry's.
    /// Nil when every slot is pinned. The caller holds `lock`.
    private func claimSlot(layer: Int) -> Int? {
        if let free = pools[layer].freeSlots.popLast() { return free }
        guard
            let position = pools[layer].order.firstIndex(where: {
                !pinned.contains(CacheKey(layer: layer, expert: $0))
            })
        else { return nil }
        let victim = pools[layer].order.remove(at: position)
        return pools[layer].entries.removeValue(forKey: victim)?.slotIndex
    }

    private struct ReadPlan {
        let reads: ReadJobs
        let buffers: [GGUFExpertBuffers]
    }

    /// The reads that fill `planned` and the buffers they produce. The caller
    /// holds `lock` (descriptors are opened lazily).
    private func readPlan(layer: Int, planned: [(expert: Int, slot: Int)]) throws -> ReadPlan {
        guard !planned.isEmpty else { return ReadPlan(reads: ReadJobs([]), buffers: []) }
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
        let gate = try shape(of: "ffn_gate")
        let up = try shape(of: "ffn_up")
        let down = try shape(of: "ffn_down")
        let pool = pools[layer]
        var jobs: [ReadJobs.Job] = []
        var buffers: [GGUFExpertBuffers] = []
        for entry in planned {
            let weights = try GGUFExpertLayout.weights(
                layer: layer, expert: entry.expert, in: model)
            let base = entry.slot * pool.slotBytes
            let offsets = (
                gate: base, up: base + pool.gateBytes, down: base + pool.gateBytes + pool.upBytes
            )
            for (range, offset, capacity, name) in [
                (weights.gate, offsets.gate, pool.gateBytes, "gate"),
                (weights.up, offsets.up, pool.upBytes, "up"),
                (weights.down, offsets.down, pool.downBytes, "down"),
            ] {
                jobs.append(
                    ReadJobs.Job(
                        range: range, buffer: pool.buffer, offset: offset, capacity: capacity,
                        descriptor: try descriptor(for: range.shardIndex), name: name))
            }
            buffers.append(
                GGUFExpertBuffers(
                    layer: layer, expert: entry.expert,
                    gate: GGUFExpertMatrix(
                        buffer: pool.buffer, offset: offsets.gate, rows: gate.rows,
                        columns: gate.columns, type: weights.gate.type,
                        byteCount: Int(weights.gate.byteCount)),
                    up: GGUFExpertMatrix(
                        buffer: pool.buffer, offset: offsets.up, rows: up.rows,
                        columns: up.columns, type: weights.up.type,
                        byteCount: Int(weights.up.byteCount)),
                    down: GGUFExpertMatrix(
                        buffer: pool.buffer, offset: offsets.down, rows: down.rows,
                        columns: down.columns, type: weights.down.type,
                        byteCount: Int(weights.down.byteCount)),
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
