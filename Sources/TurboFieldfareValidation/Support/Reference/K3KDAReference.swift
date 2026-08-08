import Foundation

/// FP32 reference for the K3 KDA single-token decode step — a straight port
/// of docs/K3_DATAFLOW.md "KDA layer" as pinned by the fla-verified
/// k3_ops.c (`k3_shortconv`, `k3_kda_decay`, `k3_kda_step`, `l2norm_`,
/// `k3_rmsnorm`). Sequential scalar loops in the same i-ascending order as
/// the C reference; the Metal kernel parallelizes along state columns, so
/// the summation orders match closely.
public enum K3KDAReference {
    /// l2norm_ from k3_ops.c: eps is added to the SUM of squares (not the
    /// mean), and the reciprocal uses a double accumulator there; fp32 here
    /// is inside the kernel tolerances.
    public static func l2Normalize(_ v: [Float], eps: Float = 1e-6) -> [Float] {
        var ss: Float = 0
        for x in v { ss += x * x }
        let inv = 1 / (ss + eps).squareRoot()
        return v.map { $0 * inv }
    }

    /// One depthwise causal-conv step for one tensor: history `state[c*3 +
    /// j]` holds the previous inputs, oldest first; taps are oldest..newest
    /// (tap 3 multiplies the current input); SiLU fused. State updated in
    /// place. Mirrors `k3_shortconv` with T == 1.
    public static func convStep(x: [Float], weights: [Float],
                                state: inout [Float]) -> [Float] {
        let channels = x.count
        precondition(weights.count == channels * 4,
                     "conv weights must be [channels][4]")
        precondition(state.count == channels * 3,
                     "conv state must be [channels][3]")
        var y = [Float](repeating: 0, count: channels)
        for c in 0..<channels {
            let w = c * 4
            let st = c * 3
            let cur = x[c]
            var acc = weights[w + 3] * cur
            acc += weights[w + 0] * state[st + 0]
            acc += weights[w + 1] * state[st + 1]
            acc += weights[w + 2] * state[st + 2]
            state[st + 0] = state[st + 1]
            state[st + 1] = state[st + 2]
            state[st + 2] = cur
            y[c] = acc * (1 / (1 + exp(-acc)))
        }
        return y
    }

    /// k3_kda_decay: per head a = exp(A_log[h]); per channel
    /// alpha = exp(-5 * sigmoid(a * z)), with dt_bias already folded into z.
    public static func decayAlpha(z: [Float], aLog: [Float],
                                  numHeads: Int, headDim: Int) -> [Float] {
        precondition(z.count == numHeads * headDim)
        precondition(aLog.count == numHeads)
        var alpha = [Float](repeating: 0, count: z.count)
        for h in 0..<numHeads {
            let a = exp(aLog[h])
            for d in 0..<headDim {
                let i = h * headDim + d
                let g = -5 * (1 / (1 + exp(-a * z[i])))
                alpha[i] = exp(g)
            }
        }
        return alpha
    }

    /// k3_kda_step for one head. `state` is [D][D] fp32 (row = key channel),
    /// updated in place: decay rows, u = Sᵀk (post-decay), rank-one delta
    /// write, then o = Sᵀq from the UPDATED state.
    public static func stepHead(state: inout [Float],
                                q: [Float], k: [Float], v: [Float],
                                alpha: [Float], beta: Float,
                                headDim: Int) -> [Float] {
        precondition(state.count == headDim * headDim)
        precondition(q.count == headDim && k.count == headDim
                     && v.count == headDim && alpha.count == headDim)
        let D = headDim
        for i in 0..<D {
            for j in 0..<D { state[i * D + j] *= alpha[i] }
        }
        var u = [Float](repeating: 0, count: D)
        for i in 0..<D {
            let ki = k[i]
            for j in 0..<D { u[j] += ki * state[i * D + j] }
        }
        for i in 0..<D {
            let c = beta * k[i]
            for j in 0..<D { state[i * D + j] += c * (v[j] - u[j]) }
        }
        var o = [Float](repeating: 0, count: D)
        for i in 0..<D {
            let qi = q[i]
            for j in 0..<D { o[j] += qi * state[i * D + j] }
        }
        return o
    }

    /// Full-token step across all heads: L2-normalizes q/k per head (eps
    /// 1e-6), scales q by headDim^-0.5, computes the decay gates and
    /// beta = sigmoid(logit), runs the recurrence, returns o [P]. `state`
    /// is [H][D][D] in place.
    public static func step(state: inout [Float],
                            q: [Float], k: [Float], v: [Float],
                            z: [Float], betaLogits: [Float], aLog: [Float],
                            numHeads: Int, headDim: Int) -> [Float] {
        let D = headDim
        let P = numHeads * D
        precondition(state.count == numHeads * D * D)
        precondition(q.count == P && k.count == P && v.count == P && z.count == P)
        precondition(betaLogits.count == numHeads && aLog.count == numHeads)
        let alpha = decayAlpha(z: z, aLog: aLog, numHeads: numHeads, headDim: D)
        let qScale = 1 / Float(D).squareRoot()
        var o = [Float](repeating: 0, count: P)
        for h in 0..<numHeads {
            let base = h * D
            let qNorm = l2Normalize(Array(q[base..<(base + D)]))
                .map { $0 * qScale }
            let kNorm = l2Normalize(Array(k[base..<(base + D)]))
            let beta = 1 / (1 + exp(-betaLogits[h]))
            var headState = Array(state[(h * D * D)..<((h + 1) * D * D)])
            let oHead = stepHead(
                state: &headState,
                q: qNorm, k: kNorm, v: Array(v[base..<(base + D)]),
                alpha: Array(alpha[base..<(base + D)]),
                beta: beta, headDim: D)
            state.replaceSubrange((h * D * D)..<((h + 1) * D * D),
                                  with: headState)
            o.replaceSubrange(base..<(base + D), with: oHead)
        }
        return o
    }

    /// Per-head gated output norm: RMSNorm with `weight` (eps = model
    /// rms_norm_eps), then elementwise * sigmoid(gate).
    public static func outputNorm(o: [Float], gate: [Float], weight: [Float],
                                  eps: Float,
                                  numHeads: Int, headDim: Int) -> [Float] {
        let D = headDim
        precondition(o.count == numHeads * D && gate.count == numHeads * D)
        precondition(weight.count == D)
        var out = [Float](repeating: 0, count: o.count)
        for h in 0..<numHeads {
            let base = h * D
            var ss: Float = 0
            for i in 0..<D { ss += o[base + i] * o[base + i] }
            let inv = 1 / (ss / Float(D) + eps).squareRoot()
            for i in 0..<D {
                let normed = weight[i] * o[base + i] * inv
                out[base + i] = normed * (1 / (1 + exp(-gate[base + i])))
            }
        }
        return out
    }
}
