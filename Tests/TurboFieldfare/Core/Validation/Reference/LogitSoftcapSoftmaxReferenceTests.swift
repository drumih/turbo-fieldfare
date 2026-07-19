import Testing
import Foundation
import TurboFieldfareValidationSupport

@Suite struct LogitSoftcapSoftmaxReferenceTests {
    /// Scalar reference: explicit per-element softcap, then explicit
    /// max/exp/sum/divide. No vDSP / vForce / cblas.
    private static func scalarRef(x: [Float], softcap: Float) -> [Float] {
        let v = x.count
        var capped = [Float](repeating: 0, count: v)
        for i in 0..<v {
            capped[i] = softcap * Foundation.tanh(x[i] / softcap)
        }
        var mx = -Float.infinity
        for v in capped { if v > mx { mx = v } }
        var es = [Float](repeating: 0, count: v)
        var sum: Float = 0
        for i in 0..<v {
            let e = Foundation.exp(capped[i] - mx)
            es[i] = e
            sum += e
        }
        return es.map { $0 / sum }
    }

    @Test("Accelerate ref matches scalar", arguments: [
        (128, Float(30.0), UInt64(0x10E)),
        (1024, Float(30.0), UInt64(0x10F)),
        (4096, Float(30.0), UInt64(0x110)),
    ])
    func acceleratedMatchesScalar(v: Int, softcap: Float, seed: UInt64) {
        var rng = SeedTree(seed).key("softcap-softmax-v\(v)")
        // Wide-range logits — exercises both the cap and the softmax tails.
        let x = (0..<v).map { _ in rng.uniform(-100.0, 100.0) }

        let accel = LogitSoftcapSoftmaxRef.apply(x: x, softcap: softcap)
        let scalar = Self.scalarRef(x: x, softcap: softcap)

        let rel = RelError.compute(actual: accel, reference: scalar)
        #expect(rel < Tolerance.identity, "V=\(v) rel=\(rel)")

        // Probability axioms.
        let sum = accel.reduce(0, +)
        #expect(abs(sum - 1.0) < 1e-4, "softmax sum=\(sum)")
        for p in accel { #expect(p >= 0 && p <= 1, "out-of-range prob \(p)") }
    }
}
