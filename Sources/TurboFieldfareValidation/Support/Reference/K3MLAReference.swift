import Foundation

/// FP32 reference for the K3 MLA decode path. This is deliberately the
/// NAIVE expanded form — per cached token it materializes k_nope/v through
/// kv_b and runs eager attention — which is the ground truth the absorbed
/// Metal kernels must match (kv_b is linear, so absorbing into q and
/// folding the value side is exact up to float rounding).
public enum K3MLAReference {
    /// KimiRMSNorm: fp32, `weight * (x * rsqrt(mean(x^2) + eps))`.
    public static func rmsNorm(_ x: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(x.count == weight.count)
        var ss: Float = 0
        for v in x { ss += v * v }
        let inv = 1 / (ss / Float(x.count) + eps).squareRoot()
        return zip(x, weight).map { $1 * $0 * inv }
    }

    /// The fp16 cache row written for a token: [ kv_a_layernorm(latent) |
    /// raw rope part ]. Input is the fp32 value of the fp16 kv_a projection.
    public static func cacheRow(kvA: [Float], normWeight: [Float], eps: Float,
                                latent: Int, rope: Int) -> [Float] {
        precondition(kvA.count == latent + rope)
        let normed = rmsNorm(Array(kvA[0..<latent]), weight: normWeight, eps: eps)
        return normed + kvA[latent..<(latent + rope)]
    }

    /// Naive expanded eager attention over the whole cache, then the
    /// sigmoid output gate. All inputs are the fp32 values of the
    /// corresponding fp16/fp32 buffers.
    ///
    /// - `kvB`: dequantized kv_b weight, row-major [H*(N+V)][L]; per head,
    ///   N k_nope rows then V value rows.
    /// - `q`: [H][N+R] (nope part, then rope part — never rotated).
    /// - `cache`: `seqLen` rows of [L+R] fp32-of-fp16.
    /// - returns the gated, flattened output [H*V].
    public static func attention(
        kvB: [Float],
        q: [Float],
        cache: [[Float]],
        gate: [Float],
        scale: Float,
        numHeads: Int, latent: Int, rope: Int, nope: Int, vHead: Int
    ) -> [Float] {
        let H = numHeads, L = latent, R = rope, N = nope, V = vHead
        let T = cache.count
        precondition(kvB.count == H * (N + V) * L)
        precondition(q.count == H * (N + R))
        precondition(gate.count == H * V)
        precondition(T >= 1)
        for row in cache { precondition(row.count == L + R) }

        var out = [Float](repeating: 0, count: H * V)
        for h in 0..<H {
            let wBase = h * (N + V) * L
            let qBase = h * (N + R)
            var scores = [Float](repeating: 0, count: T)
            // Per token: expand k_nope (N) and v (V) from the latent.
            var expandedV = [[Float]]()
            expandedV.reserveCapacity(T)
            for s in 0..<T {
                let lat = Array(cache[s][0..<L])
                var score: Float = 0
                for i in 0..<N {
                    var kn: Float = 0
                    for l in 0..<L {
                        kn += kvB[wBase + i * L + l] * lat[l]
                    }
                    score += q[qBase + i] * kn
                }
                for r in 0..<R {
                    score += q[qBase + N + r] * cache[s][L + r]
                }
                scores[s] = score * scale
                var vRow = [Float](repeating: 0, count: V)
                for j in 0..<V {
                    var acc: Float = 0
                    for l in 0..<L {
                        acc += kvB[wBase + (N + j) * L + l] * lat[l]
                    }
                    vRow[j] = acc
                }
                expandedV.append(vRow)
            }
            // Max-subtracted softmax in fp32.
            var m = scores[0]
            for s in 1..<T { m = max(m, scores[s]) }
            var probs = [Float](repeating: 0, count: T)
            var sum: Float = 0
            for s in 0..<T {
                let e = exp(scores[s] - m)
                probs[s] = e
                sum += e
            }
            for s in 0..<T { probs[s] /= sum }
            for j in 0..<V {
                var acc: Float = 0
                for s in 0..<T { acc += probs[s] * expandedV[s][j] }
                let g = 1 / (1 + exp(-gate[h * V + j]))
                out[h * V + j] = acc * g
            }
        }
        return out
    }
}
