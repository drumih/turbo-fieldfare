import Foundation
import Metal
import TinyTitanFormat

package struct GGUFResidentArenaEntry: Sendable, Equatable {
    package let name: String
    package let offset: UInt64
    package let byteCount: UInt64
    package let rows: UInt32
    package let columns: UInt32
}

package enum GGUFResidentArenaError: Error, Equatable, CustomStringConvertible {
    case emptyTensorList
    case bufferAllocationFailed(UInt64)
    case unsupportedType(String, String)
    case commandFailed(String)

    package var description: String {
        switch self {
        case .emptyTensorList:
            return "the resident arena was given no tensors"
        case .bufferAllocationFailed(let bytes):
            return "could not allocate a \(bytes)-byte resident arena"
        case .unsupportedType(let name, let type):
            return "\(name) uses \(type), which the arena does not decode"
        case .commandFailed(let detail):
            return "resident arena command failed: \(detail)"
        }
    }
}

/// The resident half of the as-is GGUF model: every non-expert tensor decoded
/// to bf16 once, into a single Metal buffer, with an index the engine can
/// address by name.
///
/// bf16 is the format the existing kernels already read for resident weights
/// (`BF16GEMV`, the attention/GDN projections when promoted, the RMS norms), so
/// this turns GGUF blocks into the exact bytes those kernels expect without
/// touching the checkpoint on disk.
///
/// The PLE table is deliberately not part of this arena: at 51.2 G weights it
/// would be 102.4 GB as bf16. It stays in its IQ4_NL blocks and is gathered and
/// decoded per lookup.
package struct GGUFResidentArena {
    package let buffer: MTLBuffer
    package let entries: [String: GGUFResidentArenaEntry]
    package let byteCount: UInt64

    static let alignment: UInt64 = SSDAIFormatV1.alignmentBytes

    package static func build(
        names: [String],
        from model: GGUFModel,
        context: MetalContext
    ) throws -> GGUFResidentArena {
        guard !names.isEmpty else { throw GGUFResidentArenaError.emptyTensorList }

        // Lay the tensors out first: 16 KiB aligned, sizes from the element
        // count so the allocation is one buffer, not one per tensor.
        var entries: [String: GGUFResidentArenaEntry] = [:]
        var placements: [(name: String, offset: UInt64)] = []
        var cursor: UInt64 = 0
        for name in names {
            guard let location = model.tensor(named: name) else {
                throw GGUFResidentArenaError.unsupportedType(name, "missing")
            }
            let elements = location.tensor.elementCount
            let bytes = elements * 2
            let offset = (cursor + alignment - 1) / alignment * alignment
            entries[name] = GGUFResidentArenaEntry(
                name: name, offset: offset, byteCount: bytes,
                rows: rows(of: location.tensor.dimensions),
                columns: UInt32(location.tensor.dimensions[0]))
            placements.append((name, offset))
            cursor = offset + bytes
        }
        let total = (cursor + alignment - 1) / alignment * alignment
        guard
            let buffer = context.device.makeBuffer(
                length: Int(total), options: .storageModeShared)
        else {
            throw GGUFResidentArenaError.bufferAllocationFailed(total)
        }

        let decoders = try Decoders(context: context)
        for (name, offset) in placements {
            guard let location = model.tensor(named: name) else {
                throw GGUFResidentArenaError.unsupportedType(name, "missing")
            }
            let shard = model.shards[location.shardIndex]
            let handle = try FileHandle(forReadingFrom: shard.url)
            try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
            let raw = try handle.read(upToCount: Int(location.tensor.byteCount)) ?? Data()
            try handle.close()
            let tensor = location.tensor
            let count = Int(tensor.elementCount)
            switch tensor.type {
            case .f32, .f16, .bf16:
                let values = try GGUFDequant.decode(type: tensor.type, bytes: [UInt8](raw))
                let destination = buffer.contents()
                    .advanced(by: Int(offset))
                    .assumingMemoryBound(to: UInt16.self)
                for index in 0..<count {
                    destination[index] = Quantization.bf16Bits(values[index])
                }
            case .iq4NL, .iq2S, .iq3XXS, .iq4XS, .q2_0, .q6K, .q8_0:
                guard
                    let source = context.device.makeBuffer(
                        bytes: [UInt8](raw), length: raw.count,
                        options: .storageModeShared)
                else {
                    throw GGUFResidentArenaError.bufferAllocationFailed(UInt64(raw.count))
                }
                let commandBuffer = try context.queue.makeCommandBuffer().require()
                try decoders.encode(
                    type: tensor.type, commandBuffer: commandBuffer,
                    blocks: source, count: count, out: buffer, outOffset: Int(offset))
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                if let error = commandBuffer.error {
                    throw GGUFResidentArenaError.commandFailed("\(error)")
                }
            default:
                throw GGUFResidentArenaError.unsupportedType(
                    name, tensor.type.name)
            }
        }

        return GGUFResidentArena(buffer: buffer, entries: entries, byteCount: total)
    }

    private static func rows(of dimensions: [UInt64]) -> UInt32 {
        dimensions.count > 1 ? UInt32(dimensions[1]) : 1
    }

    /// The decoders, constructed once per arena rather than once per tensor:
    /// each compiles or resolves a pipeline.
    private struct Decoders {
        let iq4nl: DequantIQ4NL
        let iq2s: DequantIQ2S
        let iq3xxs: DequantIQ3XXS
        let iq4xs: DequantIQ4XS
        let q2: DequantQ20
        let q6k: DequantQ6K
        let q8: DequantQ80

        init(context: MetalContext) throws {
            iq4nl = try DequantIQ4NL(context: context)
            iq2s = try DequantIQ2S(context: context)
            iq3xxs = try DequantIQ3XXS(context: context)
            iq4xs = try DequantIQ4XS(context: context)
            q2 = try DequantQ20(context: context)
            q6k = try DequantQ6K(context: context)
            q8 = try DequantQ80(context: context)
        }

        func encode(
            type: GGUFType, commandBuffer: MTLCommandBuffer,
            blocks: MTLBuffer, count: Int, out: MTLBuffer, outOffset: Int
        ) throws {
            switch type {
            case .iq4NL:
                try iq4nl.encode(
                    commandBuffer: commandBuffer, blocks: blocks, count: count,
                    out: out, outOffset: outOffset)
            case .iq2S:
                try iq2s.encode(
                    commandBuffer: commandBuffer, blocks: blocks, count: count,
                    out: out, outOffset: outOffset)
            case .iq3XXS:
                try iq3xxs.encode(
                    commandBuffer: commandBuffer, blocks: blocks, count: count,
                    out: out, outOffset: outOffset)
            case .iq4XS:
                try iq4xs.encode(
                    commandBuffer: commandBuffer, blocks: blocks, count: count,
                    out: out, outOffset: outOffset)
            case .q2_0:
                try q2.encode(
                    commandBuffer: commandBuffer, blocks: blocks, count: count,
                    out: out, outOffset: outOffset)
            case .q6K:
                try q6k.encode(
                    commandBuffer: commandBuffer, blocks: blocks, count: count,
                    out: out, outOffset: outOffset)
            case .q8_0:
                try q8.encode(
                    commandBuffer: commandBuffer, blocks: blocks, count: count,
                    out: out, outOffset: outOffset)
            default:
                throw GGUFResidentArenaError.unsupportedType("tensor", type.name)
            }
        }
    }
}

extension Optional {
    fileprivate func require() throws -> Wrapped {
        guard let value = self else {
            throw GGUFResidentArenaError.commandFailed("no command buffer")
        }
        return value
    }
}
