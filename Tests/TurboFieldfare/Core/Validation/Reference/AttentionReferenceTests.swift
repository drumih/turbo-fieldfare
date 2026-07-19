import Testing
import Foundation
import TurboFieldfareValidationSupport

@Suite struct AttentionReferenceTests {
    /// Independent reference: scalar per-element accumulation, no vDSP.
    /// Same math, no Accelerate.
    private static func scalarRef(
        q: [Float], k: [Float], v: [Float],
        headDim: Int, numQHeads: Int, numKVHeads: Int,
        seqLen: Int, window: Int?
    ) -> [Float] {
        let groupSize = numQHeads / numKVHeads
        let scale = 1.0 / Float(headDim).squareRoot()
        let kvStart: Int = (window.map { w in max(0, seqLen - w) }) ?? 0
        var out = [Float](repeating: 0, count: numQHeads * headDim)
        for qh in 0..<numQHeads {
            let kvHead = qh / groupSize
            let qBase = qh * headDim
            var scores = [Float](repeating: 0, count: seqLen)
            var mx: Float = -.greatestFiniteMagnitude
            for p in kvStart..<seqLen {
                var s: Float = 0
                let kBase = (p * numKVHeads + kvHead) * headDim
                for d in 0..<headDim { s += q[qBase + d] * k[kBase + d] }
                s *= scale
                scores[p] = s
                if s > mx { mx = s }
            }
            var sumExp: Float = 0
            for p in kvStart..<seqLen {
                scores[p] = Foundation.exp(scores[p] - mx)
                sumExp += scores[p]
            }
            let invSum = 1.0 / sumExp
            for d in 0..<headDim {
                var acc: Float = 0
                for p in kvStart..<seqLen {
                    let vBase = (p * numKVHeads + kvHead) * headDim
                    acc += scores[p] * invSum * v[vBase + d]
                }
                out[qh * headDim + d] = acc
            }
        }
        return out
    }

    @Test("vDSP ref matches scalar", arguments: [
        (Int(64), 4, 2, 128, Int?(64),  UInt64(0x161)),
        (Int(64), 4, 2,  32, Int?(128), UInt64(0x162)),
        (Int(64), 8, 1, 128, Int?(nil), UInt64(0x163)),
        (Int(128), 8, 4, 64, Int?(nil), UInt64(0x164)),
    ])
    func vDSPMatchesScalar(
        headDim: Int, numQHeads: Int, numKVHeads: Int,
        seqLen: Int, window: Int?, seed: UInt64
    ) {
        var rng = SeedTree(seed).key("attention-ref-h\(numQHeads)-d\(headDim)-T\(seqLen)")
        let qCount = numQHeads * headDim
        let kvCount = seqLen * numKVHeads * headDim
        let q = (0..<qCount).map { _ in rng.uniform(-0.5, 0.5) }
        let k = (0..<kvCount).map { _ in rng.uniform(-0.5, 0.5) }
        let v = (0..<kvCount).map { _ in rng.uniform(-0.5, 0.5) }

        let vdsp = AttentionRef.apply(
            q: q, k: k, v: v,
            headDim: headDim, numQHeads: numQHeads,
            numKVHeads: numKVHeads, seqLen: seqLen, window: window
        )
        let scalar = Self.scalarRef(
            q: q, k: k, v: v,
            headDim: headDim, numQHeads: numQHeads,
            numKVHeads: numKVHeads, seqLen: seqLen, window: window
        )
        let rel = RelError.compute(actual: vdsp, reference: scalar)
        #expect(rel < Tolerance.identity, "rel=\(rel)")
    }
}
