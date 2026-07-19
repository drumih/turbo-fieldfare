import Foundation
import Accelerate

/// FP32 RoPE reference. Precomputes `(cos, sin)` tables for every rotation
/// pair via `vForce.cos`/`vForce.sin`, then applies the rotation per
/// `(token, head)` slice. The kernel walks pair-by-pair recomputing `cosf` /
/// `sinf` inside the inner loop; the reference computes them once in a
/// bulk vector call before any input is touched. Different precision chain
/// (vForce uses extended precision trig in some configurations) and
/// different operation order — kernel bugs that depend on per-pair
/// rounding won't be replicated here.
public enum RopeRef {
    /// Paired-convention RoPE. Pairs in `[0, rotaryDim/2)` rotate; pairs
    /// from `rotaryDim/2` to `headDim/2` pass through unchanged.
    /// Setting `rotaryDim == headDim` recovers full rotation.
    public static func apply(
        input: [Float],
        numTokens: Int,
        numHeads: Int,
        headDim: Int,
        rotaryDim: Int,
        position: Int,
        theta: Float
    ) -> [Float] {
        precondition(input.count == numTokens * numHeads * headDim,
                     "input must be numTokens * numHeads * headDim")
        precondition(rotaryDim % 2 == 0, "rotaryDim must be even")
        precondition(rotaryDim <= headDim, "rotaryDim cannot exceed headDim")

        let pairs = rotaryDim / 2

        // 1. Build angle table: angle[k] = position * theta^(-2k / rotaryDim)
        var angles = [Float](repeating: 0, count: pairs)
        let logTheta = Foundation.log(theta)
        let positionF = Float(position)
        for k in 0..<pairs {
            let exponent = -Float(2 * k) / Float(rotaryDim)
            angles[k] = positionF * Foundation.exp(exponent * logTheta)
        }

        // 2. Bulk-compute cos / sin once for all pairs.
        let cosTable = vForce.cos(angles)
        let sinTable = vForce.sin(angles)

        // 3. Apply rotation. Pass-through region copies input unchanged.
        var out = input
        for t in 0..<numTokens {
            for h in 0..<numHeads {
                let base = (t * numHeads + h) * headDim
                for k in 0..<pairs {
                    let i0 = base + 2 * k
                    let i1 = i0 + 1
                    let x0 = input[i0]
                    let x1 = input[i1]
                    let c = cosTable[k]
                    let s = sinTable[k]
                    out[i0] = x0 * c - x1 * s
                    out[i1] = x0 * s + x1 * c
                }
            }
        }
        return out
    }

    /// NeoX-convention RoPE. Pairs `(x[i], x[i+headDim/2])` for `i ∈ [0, rotatedPairs)`.
    /// Frequencies divide by `headDim` (not `2 * rotatedPairs`) — matching HF
    /// Gemma 4's proportional-RoPE init. Setting `rotatedPairs == headDim/2`
    /// recovers full NeoX rotation.
    public static func applyNeox(
        input: [Float],
        numTokens: Int,
        numHeads: Int,
        headDim: Int,
        rotatedPairs: Int,
        position: Int,
        theta: Float
    ) -> [Float] {
        precondition(input.count == numTokens * numHeads * headDim,
                     "input size mismatch")
        precondition(rotatedPairs * 2 <= headDim, "rotatedPairs * 2 must not exceed headDim")
        let halfDim = headDim / 2
        var angles = [Float](repeating: 0, count: rotatedPairs)
        let logTheta = Foundation.log(theta)
        let positionF = Float(position)
        for i in 0..<rotatedPairs {
            let exponent = -Float(2 * i) / Float(headDim)
            angles[i] = positionF * Foundation.exp(exponent * logTheta)
        }
        let cosTable = vForce.cos(angles)
        let sinTable = vForce.sin(angles)

        var out = input
        for t in 0..<numTokens {
            for h in 0..<numHeads {
                let base = (t * numHeads + h) * headDim
                for i in 0..<rotatedPairs {
                    let i0 = base + i
                    let i1 = base + halfDim + i
                    let x0 = input[i0]
                    let x1 = input[i1]
                    let c = cosTable[i]
                    let s = sinTable[i]
                    out[i0] = x0 * c - x1 * s
                    out[i1] = x0 * s + x1 * c
                }
            }
        }
        return out
    }
}
