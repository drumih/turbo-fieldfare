import Testing
import Foundation
import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct MoEReferenceTests {
    @Test("gelu_pytorch_tanh matches scalar definition")
    func geluConstantsCorrect() {
        // Compare against the literal Gemma 4 / PyTorch reference points.
        let y = MoeRef.geluTanh([0.0, 1.0, -1.0, 2.0])
        #expect(abs(y[0]) < 1e-7, "y(0)=\(y[0])")
        let expectedAt1: Float = 0.5 * (1 + Foundation.tanh(0.7978845608028654 * (1 + 0.044715)))
        #expect(abs(y[1] - expectedAt1) < 1e-5, "y(1)=\(y[1]) expected=\(expectedAt1)")
        #expect(y[1] > 0)
        #expect(y[2] < 0)
        #expect(y[3] > y[1])
    }

    @Test("FFN runs without divergence on small shape")
    func ffnRunsOnSmallShape() {
        let d = 128, f = 64
        var rng = SeedTree(0x181).key("moe-ref-ffn")
        func randRow() -> Quantization.Int4AffineRow {
            let raw = (0..<d).map { _ in rng.uniform(-0.3, 0.3) }
            return Quantization.quantizeInt4Affine(raw)
        }
        func randDownRow() -> Quantization.Int4AffineRow {
            let raw = (0..<f).map { _ in rng.uniform(-0.3, 0.3) }
            return Quantization.quantizeInt4Affine(raw)
        }
        let gate = (0..<f).map { _ in randRow() }
        let up   = (0..<f).map { _ in randRow() }
        let down = (0..<d).map { _ in randDownRow() }
        let x    = (0..<d).map { _ in rng.uniform(-0.3, 0.3) }

        let y = MoeRef.runFFN(gateRows: gate, upRows: up, downRows: down,
                              x: x, d: d, f: f)
        #expect(y.count == d)
        for v in y { #expect(v.isFinite) }
    }
}
