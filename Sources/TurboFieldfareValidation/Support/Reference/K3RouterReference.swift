import Foundation
import Accelerate

/// FP32 reference for the K3 router (`KimiMoEGate` with sigmoid scoring,
/// `num_expert_group == 1`):
///
///   logits  = W_gate · x            (FP32; W_gate is [numExperts, d])
///   scores  = sigmoid(logits)
///   keys    = scores + e_score_correction_bias   (bias only for selection)
///   weights = gather(scores, topK(keys)) renormalized by (sum + 1e-20),
///             times routed_scaling_factor (1.0 on K3)
///
/// The GEMV rows dot with `vDSP_dotpr`; the kernel's lane-strided FP32
/// accumulation with a SIMD reduction sums in a different order. Selection
/// ties keep the lower expert index — deterministic, and identical to
/// `K3Router.selectTopK` by construction; distinct-key order matches any
/// correct top-k.
public enum K3RouterReference {
    public static func logits(x: [Float], gate: [Float],
                              numExperts: Int, d: Int) -> [Float] {
        precondition(x.count == d)
        precondition(gate.count == numExperts * d)
        var out = [Float](repeating: 0, count: numExperts)
        gate.withUnsafeBufferPointer { pg in
            x.withUnsafeBufferPointer { px in
                for e in 0..<numExperts {
                    var dot: Float = 0
                    vDSP_dotpr(pg.baseAddress! + e * d, 1,
                               px.baseAddress!, 1,
                               &dot, vDSP_Length(d))
                    out[e] = dot
                }
            }
        }
        return out
    }

    public static func sigmoidScores(_ logits: [Float]) -> [Float] {
        logits.map { 1 / (1 + exp(-$0)) }
    }

    /// Full gate: returns the raw sigmoid scores, the selection keys
    /// (scores + bias), the selected expert indices (descending key order),
    /// and the renormalized unbiased weights.
    public static func route(
        x: [Float],
        gate: [Float],
        bias: [Float],
        numExperts: Int,
        d: Int,
        topK: Int,
        scalingFactor: Float = 1.0
    ) -> (scores: [Float], keys: [Float], indices: [Int], weights: [Float]) {
        precondition(bias.count == numExperts)
        precondition(topK >= 1 && topK <= numExperts)
        let scores = sigmoidScores(logits(x: x, gate: gate,
                                          numExperts: numExperts, d: d))
        var keys = [Float](repeating: 0, count: numExperts)
        for e in 0..<numExperts {
            keys[e] = scores[e] + bias[e]
        }
        let order = (0..<numExperts).sorted { a, b in
            if keys[a] != keys[b] { return keys[a] > keys[b] }
            return a < b
        }
        var indices = [Int](repeating: 0, count: topK)
        var weights = [Float](repeating: 0, count: topK)
        var sum: Float = 0
        for slot in 0..<topK {
            let expert = order[slot]
            indices[slot] = expert
            weights[slot] = scores[expert]
            sum += scores[expert]
        }
        let denominator = sum + 1e-20
        for slot in 0..<topK {
            weights[slot] = weights[slot] / denominator * scalingFactor
        }
        return (scores, keys, indices, weights)
    }
}
