import Foundation

/// Host decoders for the low-bit types Unsloth's Inkling2 `UD-Q2_K_XL` uses:
/// IQ1_S, IQ2_XXS, IQ2_XS and Q5_K, each a port of ggml's
/// `dequantize_row_*` (ggml-quants.c), pinned bit-exact by
/// `GGUFDequantVectors.txt`.
extension GGUFDequant {
    /// IQ2_XXS: 66 bytes / 256. d, then per 32 values a 4-byte word of grid
    /// indices and a word of four 7-bit sign indices with a 4-bit scale on top.
    static func decodeIQ2XXS(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let d = fp16(bytes, offset)
        for ib32 in 0..<8 {
            let base = offset + 2 + ib32 * 8
            let aux = u32(bytes, base + 4)
            let db = d * (0.5 + Float(aux >> 28)) * 0.25
            for l in 0..<4 {
                let grid = GGUFIQTables.iq2xxsGrid[Int(bytes[base + l])]
                let signs = GGUFIQTables.ksigns[Int((aux >> UInt32(7 * l)) & 127)]
                for j in 0..<8 {
                    let quant = Float(UInt8(truncatingIfNeeded: grid >> (8 * j)))
                    values.append(db * quant * (signs & GGUFIQTables.kmask[j] != 0 ? -1 : 1))
                }
            }
        }
    }

    /// IQ2_XS: 74 bytes / 256. d, 32 uint16 (9-bit grid index, 7-bit sign
    /// index), then a byte of two 4-bit scales per 32 values.
    static func decodeIQ2XS(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let d = fp16(bytes, offset)
        for ib32 in 0..<8 {
            let packed = bytes[offset + 66 + ib32]
            let db = [
                d * (0.5 + Float(packed & 0x0F)) * 0.25, d * (0.5 + Float(packed >> 4)) * 0.25,
            ]
            for l in 0..<4 {
                let q = Int(u16(bytes, offset + 2 + (4 * ib32 + l) * 2))
                let grid = GGUFIQTables.iq2xsGrid[q & 511]
                let signs = GGUFIQTables.ksigns[q >> 9]
                for j in 0..<8 {
                    let quant = Float(UInt8(truncatingIfNeeded: grid >> (8 * j)))
                    values.append(
                        db[l / 2] * quant * (signs & GGUFIQTables.kmask[j] != 0 ? -1 : 1))
                }
            }
        }
    }

    /// IQ1_S: 50 bytes / 256. d, 32 grid-index low bytes, then per 32 values a
    /// uint16 with three high index bits per 8, a 3-bit scale and the delta sign.
    static func decodeIQ1S(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let d = fp16(bytes, offset)
        for ib in 0..<8 {
            let qh = Int(u16(bytes, offset + 34 + ib * 2))
            let dl = d * Float(2 * ((qh >> 12) & 7) + 1)
            let delta: Float = qh & 0x8000 != 0 ? -0.125 : 0.125
            for l in 0..<4 {
                let index = Int(bytes[offset + 2 + ib * 4 + l]) | (((qh >> (3 * l)) & 7) << 8)
                let grid = GGUFIQTables.iq1sGrid[index]
                for j in 0..<8 {
                    let quant = Float(Int8(truncatingIfNeeded: grid >> (8 * j)))
                    values.append(dl * (quant + delta))
                }
            }
        }
    }

    /// Q5_K: 176 bytes / 256. Q4_K's scales plus a 32-byte plane whose bit g of
    /// byte l is the fifth bit of value l of 32-value group g.
    static func decodeQ5K(
        _ bytes: UnsafeRawBufferPointer, offset: Int, into values: inout [Float]
    ) {
        let d = fp16(bytes, offset)
        let dmin = fp16(bytes, offset + 2)
        let qh = offset + 16
        var q = offset + 48
        var group = 0
        for _ in 0..<4 {
            var sc = 0
            var m = 0
            scaleMinK4(bytes, offset: offset, index: group, sc: &sc, m: &m)
            let d1 = d * Float(sc)
            let m1 = dmin * Float(m)
            scaleMinK4(bytes, offset: offset, index: group + 1, sc: &sc, m: &m)
            let d2 = d * Float(sc)
            let m2 = dmin * Float(m)
            for l in 0..<32 {
                let high = (bytes[qh + l] >> UInt8(group)) & 1 != 0 ? 16 : 0
                values.append(d1 * Float(Int(bytes[q + l] & 0x0F) + high) - m1)
            }
            for l in 0..<32 {
                let high = (bytes[qh + l] >> UInt8(group + 1)) & 1 != 0 ? 16 : 0
                values.append(d2 * Float(Int(bytes[q + l] >> 4) + high) - m2)
            }
            q += 32
            group += 2
        }
    }
}
