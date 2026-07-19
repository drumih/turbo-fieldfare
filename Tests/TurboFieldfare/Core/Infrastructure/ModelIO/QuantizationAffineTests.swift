import Testing
import Foundation
import TurboFieldfare

@Suite struct QuantizationAffineTests {

    @Test func quantizeDequantizeInt4Affine_roundtrip() {
        var row = [Float]()
        row.reserveCapacity(128)
        for k in 0..<64 { row.append(0.10 + Float(k) * 0.01) }   // 0.10 .. 0.73 (positive-only, bias > 0)
        for k in 0..<64 { row.append(-0.50 - Float(k) * 0.005) } // -0.50 .. -0.815 (negative-only, bias < 0)

        let q = Quantization.quantizeInt4Affine(row)
        #expect(q.packed.count == 64)
        #expect(q.scales.count == 2)
        #expect(q.biases.count == 2)

        let r = Quantization.dequantizeInt4Affine(q, n: 128)
        // 4-bit affine, group-0 scale ≈ (0.73 - 0.10) / 15 ≈ 0.042; group-1 scale
        // ≈ 0.315/15 ≈ 0.021. Worst-case rounding ≤ scale/2 plus BF16 rounding
        // of the scale (rel ≤ 2^-7) — bound under 0.025.
        for i in 0..<128 {
            #expect(abs(r[i] - row[i]) < 0.025,
                    "i=\(i) ref=\(row[i]) got=\(r[i])")
        }
    }

    @Test func quantizeDequantizeInt8Affine_roundtrip() {
        var row = [Float]()
        row.reserveCapacity(64)
        for k in 0..<64 { row.append(-0.3 + Float(k) * 0.02) }   // -0.30 .. +0.96

        let q = Quantization.quantizeInt8Affine(row)
        #expect(q.packed.count == 64)
        #expect(q.scales.count == 1)
        #expect(q.biases.count == 1)

        let r = Quantization.dequantizeInt8Affine(q, n: 64)
        // 8-bit affine, scale ≈ 1.26/255 ≈ 0.005 → rounding ≤ scale/2 ≈ 0.0025;
        // BF16 rounding of scale ≤ 2^-7 → extra ≤ 0.005 * 255/2 * 2^-7 ≈ 0.005.
        // Bound under 0.008 absorbs both contributions.
        for i in 0..<64 {
            #expect(abs(r[i] - row[i]) < 0.008,
                    "i=\(i) ref=\(row[i]) got=\(r[i])")
        }
    }

    @Test func bf16Roundtrip_isLossyButPredictable() {
        let values: [Float] = [0.0, 1.0, -1.0, 0.5, -0.5, 1.0 / 3.0, 12345.0]
        for v in values {
            let bits = Quantization.bf16Bits(v)
            let back = Quantization.bf16ToFloat(bits)
            // BF16 has 7 mantissa bits; rel error ≤ 2^-7 = 7.8e-3.
            let denom = max(abs(v), 1e-6)
            #expect(abs(back - v) / denom < 8e-3,
                    "v=\(v) back=\(back)")
        }
    }

    /// Constant-group sanity: a flat group reproduces exactly.
    @Test func quantizeInt4Affine_constantGroupRoundtripsExact() {
        let row = [Float](repeating: 0.42, count: 64)
        let q = Quantization.quantizeInt4Affine(row)
        let r = Quantization.dequantizeInt4Affine(q, n: 64)
        // Constant-group path stores bias = value, scale = 1; the only error is
        // the BF16 rounding of the bias.
        let bf16Rounded = Quantization.bf16ToFloat(Quantization.bf16Bits(0.42))
        for i in 0..<64 {
            #expect(abs(r[i] - bf16Rounded) < 1e-6,
                    "i=\(i) got=\(r[i]) ref=\(bf16Rounded)")
        }
    }

    /// Asymmetric range (positive-only with non-zero bias) — the failure mode
    /// the symmetric scheme cannot represent without wasted dynamic range.
    @Test func quantizeInt4Affine_positiveOnlyUsesFullCodebook() {
        let row = (0..<64).map { Float($0) / 63.0 + 1.0 } // 1.0 .. 2.0
        let q = Quantization.quantizeInt4Affine(row)
        var seenLow = false, seenHigh = false
        for b in q.packed {
            let lo = b & 0x0F
            let hi = b >> 4
            if lo == 0 || hi == 0   { seenLow = true }
            if lo == 15 || hi == 15 { seenHigh = true }
        }
        #expect(seenLow,  "affine 4-bit on positive-only range should hit q=0")
        #expect(seenHigh, "affine 4-bit on positive-only range should hit q=15")
    }
}
