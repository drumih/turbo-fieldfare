import Testing
import Foundation
import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct DequantInt8GEMVReferenceTests {
    private static func scalarRef(
        weightRows: [Quantization.Int8AffineRow],
        x: [Float],
        n: Int
    ) -> [Float] {
        let m = weightRows.count
        var y = [Float](repeating: 0, count: m)
        for row in 0..<m {
            let w = Quantization.dequantizeInt8Affine(weightRows[row], n: n)
            var sum: Float = 0
            for i in 0..<n { sum += w[i] * x[i] }
            y[row] = sum
        }
        return y
    }

    @Test("vDSP ref matches scalar", arguments: [
        (4, 128, UInt64(0x141)),
        (8, 256, UInt64(0x142)),
        (16, 512, UInt64(0x143)),
        (128, 2816, UInt64(0x144)),
    ])
    func vDSPMatchesScalar(m: Int, n: Int, seed: UInt64) {
        var rng = SeedTree(seed).key("int8-gemv-ref-m\(m)-n\(n)")
        var rows: [Quantization.Int8AffineRow] = []
        rows.reserveCapacity(m)
        for _ in 0..<m {
            let raw = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
            rows.append(Quantization.quantizeInt8Affine(raw))
        }
        let x = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }

        let vdsp = DequantInt8GemvRef.apply(weightRows: rows, x: x, n: n)
        let scalar = Self.scalarRef(weightRows: rows, x: x, n: n)

        let rel = RelError.compute(actual: vdsp, reference: scalar)
        #expect(rel < Tolerance.identity, "M=\(m) N=\(n) rel=\(rel)")
    }

    /// INT8 affine round-trip bound. Codebook step `scale` covers the ±0.5 LSB
    /// rounding plus BF16 rounding of scale/bias.
    @Test("INT8 affine round-trip stays within derived bound", arguments: [128, 256, 2816])
    func quantizationDerivedBound(n: Int) {
        var rng = SeedTree(0x145).key("int8-affine-roundtrip-n\(n)")
        let raw = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let q = Quantization.quantizeInt8Affine(raw)
        let recovered = Quantization.dequantizeInt8Affine(q, n: n)

        let groups = n / Quantization.groupSize
        for g in 0..<groups {
            let scale = Quantization.bf16ToFloat(q.scales[g])
            let bound = scale + 1e-4
            for k in 0..<Quantization.groupSize {
                let i = g * Quantization.groupSize + k
                #expect(abs(recovered[i] - raw[i]) <= bound,
                        "group=\(g) k=\(k) diff=\(abs(recovered[i] - raw[i])) bound=\(bound)")
            }
        }
    }
}
