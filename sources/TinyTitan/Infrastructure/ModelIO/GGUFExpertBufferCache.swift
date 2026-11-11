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

    package var description: String {
        switch self {
        case .invalidCapacity(let value):
            return "expert cache capacity \(value) must be at least 1"
        case .shortRead(let name, let expected, let got):
            return "short read of \(name): expected \(expected) bytes, got \(got)"
        case .missingTensor(let name):
            return "the model has no expert tensor \(name)"
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
/// `expert(layer:expert:)` and the `load`/`descriptor` calls it makes under the
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
        let key = CacheKey(layer: layer, expert: expert)
        lock.lock()
        defer { lock.unlock() }
        if let cached = entries[key] {
            if let position = order.firstIndex(of: key) {
                order.remove(at: position)
                order.append(key)
            }
            return cached
        }
        var slotIndex: Int
        if let free = freeSlots.popLast() {
            slotIndex = free
        } else {
            let evicted = order.removeFirst()
            guard let victim = entries.removeValue(forKey: evicted) else {
                throw GGUFExpertBufferCacheError.invalidCapacity(0)
            }
            slotIndex = victim.slotIndex
        }
        let loaded = try load(layer: layer, expert: expert, slotIndex: slotIndex)
        entries[key] = loaded
        order.append(key)
        return loaded
    }

    /// Reads one matrix into its buffer. Opens its own descriptor so the three
    /// reads of an expert can run concurrently without sharing file state.
    private static func read(
        _ range: GGUFTensorRange, into buffer: MTLBuffer, name: String, url: URL
    ) throws -> Int {
        guard Int(range.byteCount) <= buffer.length else {
            throw GGUFExpertBufferCacheError.shortRead(
                name: name, expected: Int(range.byteCount), got: buffer.length)
        }
        let fd = open(url.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else {
            throw MetalError.invalidState("open(\(url.path)) failed: \(errno)")
        }
        defer { close(fd) }
        let destination = buffer.contents()
        var total = 0
        while total < Int(range.byteCount) {
            let count = min(Int(range.byteCount) - total, 1 << 20)
            let read = pread(
                fd, destination.advanced(by: total), count,
                off_t(range.fileOffset) + off_t(total))
            if read <= 0 {
                throw GGUFExpertBufferCacheError.shortRead(
                    name: name, expected: Int(range.byteCount), got: total)
            }
            total += read
        }
        return total
    }

    /// unchecked-invariant: each job index is written by exactly one task and
    /// read only after the group completes; only the failure slot is shared, and
    /// it is behind a lock.
    private final class ReadJobs: @unchecked Sendable {
        let ranges: [GGUFTensorRange]
        let buffers: [MTLBuffer]
        let urls: [URL]
        let names: [String]
        private let lock = NSLock()
        private var failure: Error?

        init(ranges: [GGUFTensorRange], buffers: [MTLBuffer], urls: [URL], names: [String]) {
            self.ranges = ranges
            self.buffers = buffers
            self.urls = urls
            self.names = names
        }

        func run(_ index: Int) {
            do {
                _ = try GGUFExpertBufferCache.read(
                    ranges[index], into: buffers[index], name: names[index], url: urls[index])
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

    private func load(layer: Int, expert: Int, slotIndex: Int) throws -> GGUFExpertBuffers {
        let weights = try GGUFExpertLayout.weights(layer: layer, expert: expert, in: model)
        let slot = slots[slotIndex]
        /// GGUF dims are [columns, rows, experts]; the layout record's shape is
        /// [rows, columns], which the fused GEMV dispatch depends on.
        func shape(of role: String) throws -> (rows: Int, columns: Int) {
            let name = "blk.\(layer).\(role)_exps.weight"
            guard let location = model.tensor(named: name),
                location.tensor.dimensions.count == 3
            else {
                throw GGUFExpertBufferCacheError.missingTensor(name)
            }
            return (
                Int(location.tensor.dimensions[1]),
                Int(location.tensor.dimensions[0])
            )
        }
        let ranges = [weights.gate, weights.up, weights.down]
        let buffers = [slot.gate, slot.up, slot.down]
        let names = ["gate", "up", "down"]
        let jobs = ReadJobs(
            ranges: ranges, buffers: buffers,
            urls: ranges.map { model.shards[$0.shardIndex].url }, names: names)
        // The three reads are independent: run them together, which is where a
        // token's expert time is spent.
        DispatchQueue.concurrentPerform(iterations: 3) { jobs.run($0) }
        if let error = jobs.firstFailure { throw error }
        return GGUFExpertBuffers(
            layer: layer, expert: expert,
            gate: GGUFExpertMatrix(
                buffer: slot.gate, rows: try shape(of: "ffn_gate").rows,
                columns: try shape(of: "ffn_gate").columns, type: weights.gate.type,
                byteCount: Int(weights.gate.byteCount)),
            up: GGUFExpertMatrix(
                buffer: slot.up, rows: try shape(of: "ffn_up").rows,
                columns: try shape(of: "ffn_up").columns, type: weights.up.type,
                byteCount: Int(weights.up.byteCount)),
            down: GGUFExpertMatrix(
                buffer: slot.down, rows: try shape(of: "ffn_down").rows,
                columns: try shape(of: "ffn_down").columns, type: weights.down.type,
                byteCount: Int(weights.down.byteCount)),
            slotIndex: slotIndex)
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
