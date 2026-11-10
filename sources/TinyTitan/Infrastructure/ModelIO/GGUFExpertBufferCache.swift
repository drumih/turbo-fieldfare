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
        func fill(
            _ range: GGUFTensorRange, into buffer: MTLBuffer, name: String,
            shape: (rows: Int, columns: Int)
        ) throws -> GGUFExpertMatrix {
            guard Int(range.byteCount) <= buffer.length else {
                throw GGUFExpertBufferCacheError.shortRead(
                    name: name, expected: Int(range.byteCount), got: buffer.length)
            }
            let descriptor = try descriptor(for: range.shardIndex)
            let destination = buffer.contents()
            var total = 0
            while total < Int(range.byteCount) {
                let count = min(Int(range.byteCount) - total, 1 << 20)
                let read = pread(
                    descriptor,
                    destination.advanced(by: total),
                    count,
                    off_t(range.fileOffset) + off_t(total))
                if read <= 0 {
                    throw GGUFExpertBufferCacheError.shortRead(
                        name: name, expected: Int(range.byteCount), got: total)
                }
                total += read
            }
            return GGUFExpertMatrix(
                buffer: buffer, rows: shape.rows, columns: shape.columns, type: range.type,
                byteCount: Int(range.byteCount))
        }
        let gate = try fill(
            weights.gate, into: slot.gate, name: "gate", shape: try shape(of: "ffn_gate"))
        let up = try fill(
            weights.up, into: slot.up, name: "up", shape: try shape(of: "ffn_up"))
        let down = try fill(
            weights.down, into: slot.down, name: "down", shape: try shape(of: "ffn_down"))
        return GGUFExpertBuffers(
            layer: layer, expert: expert, gate: gate, up: up, down: down, slotIndex: slotIndex)
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
