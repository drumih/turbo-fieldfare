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

    package var fetchedBytes: Int {
        gate.byteCount + up.byteCount + down.byteCount
    }
}

package enum GGUFExpertBufferCacheError: Error, Equatable, CustomStringConvertible {
    case invalidCapacity(Int)
    case shortRead(name: String, expected: Int, got: Int)

    package var description: String {
        switch self {
        case .invalidCapacity(let value):
            return "expert cache capacity \(value) must be at least 1"
        case .shortRead(let name, let expected, let got):
            return "short read of \(name): expected \(expected) bytes, got \(got)"
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

    private struct CacheKey: Hashable {
        let layer: Int
        let expert: Int
    }

    package init(
        model: GGUFModel, device: MTLDevice, capacity: Int = 8
    ) throws {
        guard capacity >= 1 else {
            throw GGUFExpertBufferCacheError.invalidCapacity(capacity)
        }
        self.model = model
        self.device = device
        self.capacity = capacity
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
        let loaded = try load(layer: layer, expert: expert)
        entries[key] = loaded
        order.append(key)
        while order.count > capacity {
            let evicted = order.removeFirst()
            entries.removeValue(forKey: evicted)
        }
        return loaded
    }

    private func load(layer: Int, expert: Int) throws -> GGUFExpertBuffers {
        let weights = try GGUFExpertLayout.weights(layer: layer, expert: expert, in: model)
        func matrix(_ range: GGUFTensorRange, rows: Int, columns: Int, name: String) throws
            -> GGUFExpertMatrix
        {
            let descriptor = try descriptor(for: range.shardIndex)
            guard
                let buffer = device.makeBuffer(
                    length: Int(range.byteCount), options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed("expert \(name)")
            }
            // pread straight into the shared buffer: one syscall per 1 MiB
            // chunk, no FileHandle buffering, and the bytes the GPU reads are
            // the bytes that arrived.
            let destination = buffer.contents()
            var total = 0
            while total < Int(range.byteCount) {
                let count = min(Int(range.byteCount) - total, 1 << 20)
                let read = pread(
                    descriptor,
                    destination.advanced(by: total),
                    count,
                    off_t(range.fileOffset) + off_t(total))
                if read < 0 {
                    throw GGUFExpertBufferCacheError.shortRead(
                        name: name, expected: Int(range.byteCount), got: total)
                }
                if read == 0 { break }
                total += read
            }
            guard total == Int(range.byteCount) else {
                throw GGUFExpertBufferCacheError.shortRead(
                    name: name, expected: Int(range.byteCount), got: total)
            }
            return GGUFExpertMatrix(
                buffer: buffer, rows: rows, columns: columns, type: range.type,
                byteCount: Int(range.byteCount))
        }
        let gate = try matrix(weights.gate, rows: 640, columns: 2_560, name: "gate")
        let up = try matrix(weights.up, rows: 640, columns: 2_560, name: "up")
        let down = try matrix(weights.down, rows: 2_560, columns: 640, name: "down")
        return GGUFExpertBuffers(layer: layer, expert: expert, gate: gate, up: up, down: down)
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
