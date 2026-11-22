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
        case .iq2S:
            decodeIQ2S(bytes, offset: offset, into: &values)
        case .iq3XXS:
            decodeIQ3XXS(bytes, offset: offset, into: &values)
        case .q2K:
            decodeQ2K(bytes, offset: offset, into: &values)
        case .q4K:
            decodeQ4K(bytes, offset: offset, into: &values)
        case .nvfp4:
            decodeNVFP4(bytes, offset: offset, into: &values)
        case .mxfp4:
            decodeMXFP4(bytes, offset: offset, into: &values)
        case .q5K:
            decodeQ5K(bytes, offset: offset, into: &values)
        case .iq2XXS:
            decodeIQ2XXS(bytes, offset: offset, into: &values)
        case .iq2XS:
            decodeIQ2XS(bytes, offset: offset, into: &values)
        case .iq1S:
            decodeIQ1S(bytes, offset: offset, into: &values)
        default:
            throw GGUFDequantError.unsupportedType(type.name)
        }
    }

    /// `kvalues_fp4` (ggml-common.h): E2M1 values doubled, so the block scales
    /// below carry the matching factor of one half.
    package static let fp4Values: [Float] = [
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    ]

    /// ggml's `ggml_ue4m3_to_fp32`: unsigned E4M3 (bias 7) times one half;
    /// 0x00 and 0x7F decode to zero and the top bit is ignored, as in ggml.
    package static func ue4m3Half(_ code: UInt8) -> Float {
        if code == 0 || code == 0x7F { return 0 }
        let exponent = Int((code >> 3) & 0x0F)
        let mantissa = Float(code & 0x07)
        let raw =
            exponent == 0
            ? Float(sign: .plus, exponent: -9, significand: mantissa)
            : Float(sign: .plus, exponent: exponent - 7, significand: 1 + mantissa / 8)
        return raw * 0.5
    }

    /// ggml's `ggml_e8m0_to_fp32_half`: 2^(e - 128), with the two subnormal codes.
    package static func e8m0Half(_ code: UInt8) -> Float {
        code < 2
            ? Float(bitPattern: UInt32(0x0020_0000) << UInt32(code))
            : Float(bitPattern: UInt32(code - 1) << 23)
    }

    /// NVFP4 (`block_nvfp4`): 64 values as four 16-value sub-blocks, each with
    /// a UE4M3 scale at bytes 0..<4; sub-block `s` packs value `j` in the low
    /// nibble and `j + 8` in the high nibble of byte `4 + 8s + j`.
    private static func decodeNVFP4(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        for sub in 0..<4 {
            let scale = ue4m3Half(bytes[offset + sub])
            let base = offset + 4 + sub * 8
            var high = [Float](repeating: 0, count: 8)
            for j in 0..<8 {
                let packed = bytes[base + j]
                values.append(fp4Values[Int(packed & 0x0F)] * scale)
                high[j] = fp4Values[Int(packed >> 4)] * scale
            }
            values.append(contentsOf: high)
        }
    }

    /// MXFP4 (`block_mxfp4`): 32 values, an E8M0 scale at byte 0, then value
    /// `j` in the low and `j + 16` in the high nibble of byte `1 + j`.
    private static func decodeMXFP4(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let scale = e8m0Half(bytes[offset])
        for j in 0..<16 {
            values.append(fp4Values[Int(bytes[offset + 1 + j] & 0x0F)] * scale)
        }
        for j in 0..<16 {
            values.append(fp4Values[Int(bytes[offset + 1 + j] >> 4)] * scale)
        }
    }

    /// Q2_K: 256 values, 84-byte block. Two 128-value halves, four 2-bit
    /// shifts per half; each 16-value group has its own 4-bit scale and
    /// 4-bit minimum. Ported from ggml's `dequantize_row_q2_K`.
    private static func decodeQ2K(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let d = fp16(bytes, offset + 80)
        let dmin = fp16(bytes, offset + 82)
        var scaleIndex = offset
        var q = offset + 16
        for _ in 0..<2 {
            var shift: UInt8 = 0
            for _ in 0..<4 {
                var sc = bytes[scaleIndex]
                var dl = d * Float(sc & 0x0F)
                var ml = dmin * Float(sc >> 4)
                for l in 0..<16 {
                    let quant = (bytes[q + l] >> shift) & 0x03
                    values.append(dl * Float(quant) - ml)
                }
                scaleIndex += 1
                sc = bytes[scaleIndex]
                dl = d * Float(sc & 0x0F)
                ml = dmin * Float(sc >> 4)
                for l in 0..<16 {
                    let quant = (bytes[q + 16 + l] >> shift) & 0x03
                    values.append(dl * Float(quant) - ml)
                }
                scaleIndex += 1
                shift += 2
            }
            q += 32
        }
    }

    /// Q4_K: 256 values, 144-byte block. Each 32-value group uses one 6-bit
    /// scale and one 6-bit minimum from the packed 12-byte scale table.
    private static func decodeQ4K(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let d = fp16(bytes, offset)
        let dmin = fp16(bytes, offset + 2)
        var scaleIndex = 0
        var q = offset + 16
        for _ in 0..<4 {
            var sc = 0
            var m = 0
            scaleMinK4(bytes, offset: offset, index: scaleIndex, sc: &sc, m: &m)
            let d1 = d * Float(sc)
            let m1 = dmin * Float(m)
            scaleMinK4(bytes, offset: offset, index: scaleIndex + 1, sc: &sc, m: &m)
            let d2 = d * Float(sc)
            let m2 = dmin * Float(m)
            for l in 0..<32 {
                values.append(d1 * Float(bytes[q + l] & 0x0F) - m1)
            }
            for l in 0..<32 {
                values.append(d2 * Float(bytes[q + l] >> 4) - m2)
            }
            q += 32
            scaleIndex += 2
        }
    }

    static func scaleMinK4(
        _ bytes: UnsafeRawBufferPointer, offset: Int, index: Int,
        sc: inout Int, m: inout Int
    ) {
        let scales = offset + 4
        if index < 4 {
            sc = Int(bytes[scales + index] & 0x3F)
            m = Int(bytes[scales + index + 4] & 0x3F)
        } else {
            sc =
                Int(bytes[scales + index + 4] & 0x0F)
                | (Int(bytes[scales + index - 4] >> 6) << 4)
            m =
                Int(bytes[scales + index + 4] >> 4)
                | (Int(bytes[scales + index] >> 6) << 4)
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

    /// IQ2_S: d, 32 quant indices, 32 sign bytes, 8 high-bit bytes, 8 scales.
    /// Each 32-element sub-block takes two scales (low and high nibble) and four
    /// 8-value grid entries; the 2 high bits per entry are in `qh`.
    private static func decodeIQ2S(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let scale = fp16(bytes, offset)
        let qs = offset + 2
        let qh = offset + 66
        let scales = offset + 74
        for sub in 0..<8 {
            let packed = bytes[scales + sub]
            let db0 = scale * (0.5 + Float(packed & 0x0F)) * 0.25
            let db1 = scale * (0.5 + Float(packed >> 4)) * 0.25
            for l in 0..<4 {
                let dl = l < 2 ? db0 : db1
                let index =
                    Int(bytes[qs + sub * 4 + l])
                    | ((Int(bytes[qh + sub]) << (8 - 2 * l)) & 0x300)
                let signs = bytes[qs + 32 + sub * 4 + l]
                let grid = GGUFIQTables.iq2sGrid[index]
                for j in 0..<8 {
                    let quant = Float(UInt8(truncatingIfNeeded: grid >> (8 * j)))
                    let factor: Float = signs & GGUFIQTables.kmask[j] != 0 ? -1 : 1
                    values.append(dl * quant * factor)
                }
            }
        }
    }

    /// IQ3_XXS: d, 64 grid-index bytes, then 32 scale/sign words. Each 32-element
    /// sub-block is one 32-bit word: top 4 bits are the scale, the other 28 are
    /// four 7-bit indices into `ksigns` (a byte per 8 elements).
    private static func decodeIQ3XXS(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let scale = fp16(bytes, offset)
        let qs = offset + 2
        for sub in 0..<8 {
            let aux = u32(bytes, qs + 64 + 4 * sub)
            let db = scale * (0.5 + Float(aux >> 28)) * 0.5
            for l in 0..<4 {
                let signs = GGUFIQTables.ksigns[Int((aux >> UInt32(7 * l)) & 127)]
                let first = GGUFIQTables.iq3xxsGrid[Int(bytes[qs + sub * 8 + 2 * l])]
                let second = GGUFIQTables.iq3xxsGrid[Int(bytes[qs + sub * 8 + 2 * l + 1])]
                for j in 0..<4 {
                    let quant = Float(UInt8(truncatingIfNeeded: first >> (8 * j)))
                    let factor: Float = signs & GGUFIQTables.kmask[j] != 0 ? -1 : 1
                    values.append(db * quant * factor)
                }
                for j in 0..<4 {
                    let quant = Float(UInt8(truncatingIfNeeded: second >> (8 * j)))
                    let factor: Float = signs & GGUFIQTables.kmask[j + 4] != 0 ? -1 : 1
                    values.append(db * quant * factor)
                }
            }
        }
    }

    /// `kvalues_iq4nl` from ggml-common.h, in the order ggml uses.
    private static let iq4NL: [Int8] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ]

    static func u16(_ bytes: UnsafeRawBufferPointer, _ offset: Int) -> UInt16 {
        UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
    }

    static func u32(_ bytes: UnsafeRawBufferPointer, _ offset: Int) -> UInt32 {
        UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }

    static func fp16(_ bytes: UnsafeRawBufferPointer, _ offset: Int) -> Float {
        Float(Float16(bitPattern: u16(bytes, offset)))
    }
}
