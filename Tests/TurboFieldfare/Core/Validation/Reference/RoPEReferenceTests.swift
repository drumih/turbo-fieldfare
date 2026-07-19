import Testing
import Foundation
import TurboFieldfareValidationSupport

/// Cross-validates `RopeRef.apply` against an independent reference
/// formulation that recomputes `cos`/`sin` per pair inline (no table
/// caching). Both must agree to `Tolerance.identity`.
@Suite struct RoPEReferenceTests {

    /// Inline-trig reference. No precomputed tables, no Accelerate.
    private static func scalarRef(input: [Float],
                                  numTokens: Int,
                                  numHeads: Int,
                                  headDim: Int,
                                  rotaryDim: Int,
                                  position: Int,
                                  theta: Float) -> [Float] {
        var out = input
        let pairs = rotaryDim / 2
        for t in 0..<numTokens {
            for h in 0..<numHeads {
                let base = (t * numHeads + h) * headDim
                for k in 0..<pairs {
                    let exponent = -Float(2 * k) / Float(rotaryDim)
                    let freq = powf(theta, exponent)
                    let angle = Float(position) * freq
                    let c = cosf(angle)
                    let s = sinf(angle)
                    let i0 = base + 2 * k
                    let i1 = i0 + 1
                    let x0 = input[i0]
                    let x1 = input[i1]
                    out[i0] = x0 * c - x1 * s
                    out[i1] = x0 * s + x1 * c
                }
            }
        }
        return out
    }

    @Test("vForce ref matches scalar ref",
          arguments: [
            (1, 16, 256, 256,   7, Float(10_000),    UInt64(0xD1)),
            (4,  8, 256, 256,  33, Float(10_000),    UInt64(0xD2)),
            (1, 16, 512, 128,  11, Float(1_000_000), UInt64(0xD3)),
            (2,  4,  64,  64,   5, Float(10_000),    UInt64(0xD4)),
          ])
    func vForceMatchesScalar(
        numTokens: Int,
        numHeads: Int,
        headDim: Int,
        rotaryDim: Int,
        position: Int,
        theta: Float,
        seed: UInt64
    ) {
        var rng = SeedTree(seed).key("rope-ref-h\(numHeads)-d\(headDim)")
        let count = numTokens * numHeads * headDim
        let input = (0..<count).map { _ in rng.uniform(-1.0, 1.0) }

        let vforce = RopeRef.apply(
            input: input,
            numTokens: numTokens, numHeads: numHeads,
            headDim: headDim, rotaryDim: rotaryDim,
            position: position, theta: theta
        )
        let scalar = Self.scalarRef(
            input: input,
            numTokens: numTokens, numHeads: numHeads,
            headDim: headDim, rotaryDim: rotaryDim,
            position: position, theta: theta
        )

        let rel = RelError.compute(actual: vforce, reference: scalar)
        #expect(rel < Tolerance.identity, "rel err = \(rel)")
    }

    @Test("Partial rotation leaves passthrough region byte-identical")
    func partialPassthroughUntouched() {
        var rng = SeedTree(0xD5).key("rope-passthrough")
        let numTokens = 2, numHeads = 4, headDim = 64, rotaryDim = 32
        let count = numTokens * numHeads * headDim
        let input = (0..<count).map { _ in rng.uniform(-1.0, 1.0) }

        let out = RopeRef.apply(
            input: input,
            numTokens: numTokens, numHeads: numHeads,
            headDim: headDim, rotaryDim: rotaryDim,
            position: 9, theta: 10_000
        )
        for t in 0..<numTokens {
            for h in 0..<numHeads {
                let base = (t * numHeads + h) * headDim
                for i in rotaryDim..<headDim {
                    #expect(out[base + i] == input[base + i],
                            "passthrough drifted at t=\(t) h=\(h) i=\(i)")
                }
            }
        }
    }
}
