import Foundation
import Accelerate

/// FP32 attention reference. Materializes the full attention matrix per Q
/// head: compute all `seqLen` scores via `vDSP_dotpr`, apply the optional
/// SWA window mask, softmax, then output via per-dim `vDSP_dotpr` against
/// the V columns. The kernel runs FlashAttention-style tiled online
/// softmax; this reference does no online merging at all.
public enum AttentionRef {
    /// Q layout: `[numQHeads, headDim]`.
    /// K, V layout: `[seqLen, numKVHeads, headDim]`. (V may alias K when
    /// the caller passes the same array — the math is unchanged.)
    /// Output: `[numQHeads, headDim]`.
    public static func apply(
        q: [Float],
        k: [Float],
        v: [Float],
        headDim: Int,
        numQHeads: Int,
        numKVHeads: Int,
        seqLen: Int,
        window: Int? = nil,
        scale: Float? = nil
    ) -> [Float] {
        precondition(numQHeads % numKVHeads == 0,
                     "numQHeads must be a multiple of numKVHeads")
        precondition(q.count == numQHeads * headDim)
        precondition(k.count == seqLen * numKVHeads * headDim)
        precondition(v.count == seqLen * numKVHeads * headDim)

        let groupSize = numQHeads / numKVHeads
        let scale = scale ?? (1.0 / Float(headDim).squareRoot())

        let kvStart: Int
        if let w = window, seqLen > w {
            kvStart = seqLen - w
        } else {
            kvStart = 0
        }

        var out = [Float](repeating: 0, count: numQHeads * headDim)

        // Per-Q-head: build score row via vDSP_dotpr (one dot per key
        // position), softmax via the per-element reference path, output
        // column-wise via vDSP_dotpr against V's per-dim stride.
        for qh in 0..<numQHeads {
            let kvHead = qh / groupSize
            let qBase = qh * headDim

            var scores = [Float](repeating: 0, count: seqLen - kvStart)
            q.withUnsafeBufferPointer { pq in
                for p in kvStart..<seqLen {
                    let kBase = (p * numKVHeads + kvHead) * headDim
                    var dot: Float = 0
                    k.withUnsafeBufferPointer { pk in
                        vDSP_dotpr(
                            pq.baseAddress! + qBase, 1,
                            pk.baseAddress! + kBase, 1,
                            &dot, vDSP_Length(headDim)
                        )
                    }
                    scores[p - kvStart] = dot * scale
                }
            }

            // Softmax (numerically stable).
            var mx = scores[0]
            for s in scores { if s > mx { mx = s } }
            var negMax = -mx
            scores.withUnsafeMutableBufferPointer { ps in
                vDSP_vsadd(ps.baseAddress!, 1, &negMax,
                           ps.baseAddress!, 1, vDSP_Length(ps.count))
            }
            scores = vForce.exp(scores)
            var sum: Float = 0
            scores.withUnsafeBufferPointer { ps in
                vDSP_sve(ps.baseAddress!, 1, &sum, vDSP_Length(ps.count))
            }
            var invSum = 1.0 / sum
            scores.withUnsafeMutableBufferPointer { ps in
                vDSP_vsmul(ps.baseAddress!, 1, &invSum,
                           ps.baseAddress!, 1, vDSP_Length(ps.count))
            }

            // Output[qh, d] = sum_p probs[p] * V[p, kvHead, d]
            // = dot of probs against V's column d (strided by numKVHeads*headDim).
            for d in 0..<headDim {
                var acc: Float = 0
                scores.withUnsafeBufferPointer { ps in
                    v.withUnsafeBufferPointer { pv in
                        let vColumnStart = (kvStart * numKVHeads + kvHead) * headDim + d
                        let stride = numKVHeads * headDim
                        vDSP_dotpr(
                            ps.baseAddress!, 1,
                            pv.baseAddress! + vColumnStart, vDSP_Stride(stride),
                            &acc, vDSP_Length(ps.count)
                        )
                    }
                }
                out[qh * headDim + d] = acc
            }
        }
        return out
    }
}
