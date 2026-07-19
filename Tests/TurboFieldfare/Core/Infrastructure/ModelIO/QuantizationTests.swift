import Testing
import Foundation
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct QuantizationTests {

    @Test func roundTripIsBoundedByAffineInt4Step() {
        var rng = SeedTree(0x51A7_1001).key("int4-affine-round-trip")
        var row = [Float](repeating: 0, count: 256)
        for i in 0..<row.count {
            row[i] = rng.uniform(-1.0, 1.0)
        }
        let q = Quantization.quantizeInt4Affine(row)
        let r = Quantization.dequantizeInt4Affine(q, n: row.count)

        // MLX affine 4-bit (16 levels covering [min, max]) gives a worst-case
        // per-sample error of half the quant step = (max - min) / 30 in the
        // worst group. We bound by amax * 2 / 14 + 1e-3 to absorb BF16
        // scale / bias rounding (rel ≤ 2^-7).
        for g in 0..<(row.count / Quantization.groupSize) {
            var amax: Float = 0
            for k in 0..<Quantization.groupSize {
                amax = max(amax, abs(row[g * Quantization.groupSize + k]))
            }
            let bound = amax / 7.0 + 1e-3
            for k in 0..<Quantization.groupSize {
                let i = g * Quantization.groupSize + k
                #expect(abs(r[i] - row[i]) <= bound,
                        "i=\(i) orig=\(row[i]) rec=\(r[i]) bound=\(bound)")
            }
        }
    }

    @Test func nibbleLayoutIsLowEvenHighOdd() {
        // Construct a row that exercises both nibble positions.
        let row: [Float] = (0..<64).map { Float($0) - 31.5 }   // 64 values

        let q = Quantization.quantizeInt4Affine(row)
        // Verify even-index weight comes from low nibble of byte 0.
        let b0 = q.packed[0]
        let lo = Int(b0 & 0x0F)
        let hi = Int(b0 >> 4)
        let scale = Quantization.bf16ToFloat(q.scales[0])
        let bias  = Quantization.bf16ToFloat(q.biases[0])
        let recon0 = Float(lo) * scale + bias
        let recon1 = Float(hi) * scale + bias
        #expect(abs(recon0 - row[0]) <= scale + 1e-3)
        #expect(abs(recon1 - row[1]) <= scale + 1e-3)
    }
}
