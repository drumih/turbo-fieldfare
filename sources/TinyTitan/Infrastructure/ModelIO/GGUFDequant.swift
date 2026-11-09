import Foundation

/// Failures of GGUF block decoding.
package enum GGUFDequantError: Error, Equatable, CustomStringConvertible {
    case unsupportedType(String)
    case byteCountNotABlockMultiple(type: String, bytes: Int, blockBytes: Int)
    case truncatedBlock(type: String, offset: Int, needed: Int, available: Int)

    package var description: String {
        switch self {
        case .unsupportedType(let name):
            return "GGUF block decoding for \(name) is not implemented yet"
        case .byteCountNotABlockMultiple(let name, let bytes, let blockBytes):
            return
                "\(bytes) bytes of \(name) is not a multiple of its \(blockBytes)-byte block"
        case .truncatedBlock(let name, let offset, let needed, let available):
            return
                "\(name) block at byte \(offset) needs \(needed) bytes, \(available) are present"
        }
    }
}

/// Decodes GGUF tensor blocks to Float32.
///
/// This is deliberately the **host** decoder and deliberately first: it is the
/// component every later choice depends on, and its output can be compared
/// bit-for-bit against ggml's own `dequantize_row_*` functions without a model,
/// a GPU or a download. A wrong decode here would otherwise appear days later
/// as plausible-looking garbage.
///
/// Implemented for the block types this tree's checkpoint uses outside the IQ
/// grids: F32, F16, BF16, Q8_0, Q2_0, Q6_K, IQ4_NL and IQ4_XS. IQ2_S and
/// IQ3_XXS — the two lookup-table formats in the 48-layer expert tensors — are
/// milestone 2b and are refused by name rather than approximated.
package enum GGUFDequant {
    package static func decode(type: GGUFType, bytes: [UInt8]) throws -> [Float] {
        try bytes.withUnsafeBytes { try decode(type: type, bytes: $0) }
    }

    package static func decode(type: GGUFType, bytes: ArraySlice<UInt8>) throws -> [Float] {
        try decode(type: type, bytes: Array(bytes))
    }

    package static func decode(type: GGUFType, bytes: UnsafeRawBufferPointer) throws -> [Float] {
        let blockBytes = Int(type.blockBytes)
        guard bytes.count % blockBytes == 0 else {
            throw GGUFDequantError.byteCountNotABlockMultiple(
                type: type.name, bytes: bytes.count, blockBytes: blockBytes)
        }
        let blocks = bytes.count / blockBytes
        var values: [Float] = []
        values.reserveCapacity(blocks * Int(type.blockSize))
        for block in 0..<blocks {
            try decodeBlock(type: type, bytes: bytes, offset: block * blockBytes, into: &values)
        }
        return values
    }

    /// Decode exactly one block, appending `type.blockSize` values.
    package static func decodeBlock(
        type: GGUFType,
        bytes: UnsafeRawBufferPointer,
        offset: Int,
        into values: inout [Float]
    ) throws {
        let blockBytes = Int(type.blockBytes)
        guard offset >= 0, offset + blockBytes <= bytes.count else {
            throw GGUFDequantError.truncatedBlock(
                type: type.name, offset: offset, needed: blockBytes, available: bytes.count)
        }
        switch type {
        case .f32:
            values.append(Float(bitPattern: u32(bytes, offset)))
        case .f16:
            values.append(Float(Float16(bitPattern: u16(bytes, offset))))
        case .bf16:
            values.append(Float(bitPattern: UInt32(u16(bytes, offset)) << 16))
        case .q8_0:
            let scale = fp16(bytes, offset)
            for index in 0..<32 {
                let quant = Int(Int8(bitPattern: bytes[offset + 2 + index]))
                values.append(Float(quant) * scale)
            }
        case .q2_0:
            // 00 = -1, 01 = 0, 10 = +1, 11 = +2 (ggml-common.h, block_q2_0).
            let scale = fp16(bytes, offset)
            for index in 0..<64 {
                let packed = bytes[offset + 2 + index / 4]
                let quant = (packed >> UInt8((index % 4) * 2)) & 0x03
                values.append(Float(Int(quant) - 1) * scale)
            }
        case .q6K:
            decodeQ6K(bytes, offset: offset, into: &values)
        case .iq4NL:
            decodeIQ4NL(bytes, offset: offset, into: &values)
        case .iq4XS:
            decodeIQ4XS(bytes, offset: offset, into: &values)
        default:
            throw GGUFDequantError.unsupportedType(type.name)
        }
    }

    /// Q6_K: 128 low nibbles + 64 high 2-bit groups + 16 int8 scales + fp16 d.
    /// The four values a loop step produces are not adjacent: they are 32 apart.
    private static func decodeQ6K(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let scale = fp16(bytes, offset + 208)
        let ql = offset
        let qh = offset + 128
        let scales = offset + 192
        for half in 0..<2 {
            let qlBase = ql + half * 64
            let qhBase = qh + half * 32
            let scBase = scales + half * 8
            for l in 0..<32 {
                let iscale = l / 16
                let low =
                    Int(bytes[qlBase + l] & 0x0F)
                    | (Int(bytes[qhBase + l] & 0x03) << 4)
                let low2 =
                    Int(bytes[qlBase + 32 + l] & 0x0F)
                    | (Int((bytes[qhBase + l] >> 2) & 0x03) << 4)
                let high =
                    Int(bytes[qlBase + l] >> 4)
                    | (Int((bytes[qhBase + l] >> 4) & 0x03) << 4)
                let high2 =
                    Int(bytes[qlBase + 32 + l] >> 4)
                    | (Int((bytes[qhBase + l] >> 6) & 0x03) << 4)
                let s0 = Float(Int(Int8(bitPattern: bytes[scBase + iscale])))
                let s1 = Float(Int(Int8(bitPattern: bytes[scBase + iscale + 2])))
                let s2 = Float(Int(Int8(bitPattern: bytes[scBase + iscale + 4])))
                let s3 = Float(Int(Int8(bitPattern: bytes[scBase + iscale + 6])))
                values.append(scale * s0 * Float(low - 32))
                values.append(scale * s1 * Float(low2 - 32))
                values.append(scale * s2 * Float(high - 32))
                values.append(scale * s3 * Float(high2 - 32))
            }
            // The append order above is q1,q2,q3,q4 for one l; re-sort into the
            // block's element order l, l+32, l+64, l+96 at the end of the half.
            let start = values.count - 128
            var reordered = [Float](repeating: 0, count: 128)
            for l in 0..<32 {
                reordered[l] = values[start + l * 4]
                reordered[l + 32] = values[start + l * 4 + 1]
                reordered[l + 64] = values[start + l * 4 + 2]
                reordered[l + 96] = values[start + l * 4 + 3]
            }
            values.replaceSubrange(start..<values.count, with: reordered)
        }
    }

    private static func decodeIQ4NL(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let scale = fp16(bytes, offset)
        for index in 0..<16 {
            let packed = bytes[offset + 2 + index]
            values.append(scale * Float(iq4NL[Int(packed & 0x0F)]))
            values.append(scale * Float(iq4NL[Int(packed >> 4)]))
        }
        // ggml writes the low nibbles as elements 0...15 and the high nibbles as
        // 16...31; the append above interleaved them, so move them back.
        let start = values.count - 32
        var reordered = [Float](repeating: 0, count: 32)
        for index in 0..<16 {
            reordered[index] = values[start + index * 2]
            reordered[index + 16] = values[start + index * 2 + 1]
        }
        values.replaceSubrange(start..<values.count, with: reordered)
    }

    private static func decodeIQ4XS(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let scale = fp16(bytes, offset)
        let scalesHigh = u16(bytes, offset + 2)
        for sub in 0..<8 {
            let low = (bytes[offset + 4 + sub / 2] >> UInt8(4 * (sub % 2))) & 0x0F
            let high = (scalesHigh >> UInt16(2 * sub)) & 0x03
            let subScale = scale * Float((Int(low) | (Int(high) << 4)) - 32)
            for index in 0..<16 {
                let packed = bytes[offset + 8 + sub * 16 + index]
                values.append(subScale * Float(iq4NL[Int(packed & 0x0F)]))
                values.append(subScale * Float(iq4NL[Int(packed >> 4)]))
            }
        }
        // Low nibbles are elements 0...15 of each 32-element block, high nibbles
        // 16...31; de-interleave once at the end.
        let start = values.count - 256
        var reordered = [Float](repeating: 0, count: 256)
        for sub in 0..<8 {
            for index in 0..<16 {
                reordered[sub * 32 + index] = values[start + sub * 32 + index * 2]
                reordered[sub * 32 + index + 16] = values[start + sub * 32 + index * 2 + 1]
            }
        }
        values.replaceSubrange(start..<values.count, with: reordered)
    }

    /// `kvalues_iq4nl` from ggml-common.h, in the order ggml uses.
    private static let iq4NL: [Int8] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ]

    private static func u16(_ bytes: UnsafeRawBufferPointer, _ offset: Int) -> UInt16 {
        UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
    }

    private static func u32(_ bytes: UnsafeRawBufferPointer, _ offset: Int) -> UInt32 {
        UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }

    private static func fp16(_ bytes: UnsafeRawBufferPointer, _ offset: Int) -> Float {
        Float(Float16(bitPattern: u16(bytes, offset)))
    }
}
