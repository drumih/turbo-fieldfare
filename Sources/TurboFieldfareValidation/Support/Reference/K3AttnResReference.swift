import Foundation

/// FP32 reference for K3 AttnRes (`_apply_attn_res` in
/// modeling_kimi_linear.py): vectors are the blocks followed by the prefix
/// (LAST); each is normalized without a learned weight, scored against the
/// fused `norm.weight ⊙ proj.weight` vector, softmaxed fp32
/// (max-subtracted), and the output is the weighted sum of the RAW vectors.
/// With zero blocks the result is exactly the prefix.
public enum K3AttnResReference {
    public static func apply(blocks: [[Float]],
                             prefix: [Float],
                             scoreVector: [Float],
                             eps: Float) -> [Float] {
        let hidden = prefix.count
        precondition(scoreVector.count == hidden)
        for block in blocks { precondition(block.count == hidden) }
        let vectors = blocks + [prefix]

        var scores = [Float](repeating: 0, count: vectors.count)
        for (i, v) in vectors.enumerated() {
            var ss: Float = 0
            for x in v { ss += x * x }
            let inv = 1 / (ss / Float(hidden) + eps).squareRoot()
            var score: Float = 0
            for j in 0..<hidden {
                score += (v[j] * inv) * scoreVector[j]
            }
            scores[i] = score
        }

        var m = scores[0]
        for i in 1..<scores.count { m = max(m, scores[i]) }
        var probs = [Float](repeating: 0, count: scores.count)
        var sum: Float = 0
        for i in scores.indices {
            let e = exp(scores[i] - m)
            probs[i] = e
            sum += e
        }
        for i in probs.indices { probs[i] /= sum }

        var out = [Float](repeating: 0, count: hidden)
        for (i, v) in vectors.enumerated() {
            for j in 0..<hidden {
                out[j] += probs[i] * v[j]
            }
        }
        return out
    }
}
