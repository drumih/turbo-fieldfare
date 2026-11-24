import Foundation
import TinyTitanCore

/// Qwen3.5's dense models (2B, 4B and 9B) on the CPU, one token at a time.
///
/// Every dimension comes from the snapshot's config. The sizes differ in more
/// than width: the 2B has as many delta-rule value heads as key heads, the
/// 4B and 9B have twice as many; attention is 8 query heads over 2 KV heads
/// in the 2B and 16 over 4 in the other two. Both ratios are read, never
/// assumed, and the head-sharing tests pin the mapping for each. The 9B also
/// does not tie its output to the embedding, which `headWeightName` handles.
///
/// The side-engine's model: small enough to stay resident beside a 35B, and
/// run on cores the main engine leaves idle. Decode is one token at a time;
/// a prompt goes through `prefill(tokens:)`, which unpacks each quantized
/// matrix once and multiplies the whole chunk, then hands the KV cache and
/// the delta-rule state to the decode loop unchanged. The per-token path
/// remains reachable with `TINYTITAN_CPU_PREFILL=token`, which is how the two
/// are compared on a real model.
///
/// **Width is a scheduling decision.** Measured on this machine, a 35B
/// generation slows by 3% when this reads at one thread and by 31% at four.
/// So `threads` is settable, and the caller — which knows whether someone is
/// waiting on the GPU — chooses: one while a client generation is in flight,
/// four in the gaps.
///
/// Correctness is defined by `tools/qwen35_reference.py`, which is checked
/// in turn by continuations a 2B has no excuse for getting wrong.
public final class CPUQwen35 {

    public let snapshot: AffineSnapshot
    public let configuration: AffineSnapshot.Configuration
    /// Rows of the GEMVs to split across performance cores. See the note
    /// above: this is the knob that decides what the side-engine costs the
    /// model the person is waiting for.
    ///
    /// Set directly for a fixed width, or leave it to `contention` below,
    /// which is re-read before every token.
    public var threads: Int
    /// Whether someone is waiting on the main engine right now.
    ///
    /// When this is set the width becomes a decision rather than a constant,
    /// taken once per token: `busyThreads` while a client generation is in
    /// flight, `idleThreads` otherwise. Per token is the right granularity —
    /// changing width inside one would gain nothing, and a token is 50 ms.
    ///
    /// Nil leaves `threads` alone, which is what the benchmarks want.
    public var contention: (@Sendable () -> Bool)?
    /// Measured on this machine: one thread slows a 35B generation by 3%,
    /// which is inside its own run-to-run spread, and still gives this model
    /// about 6.7 tokens a second — enough to distil a session or check a
    /// claim. Two costs 13%, four costs 31%.
    public var busyThreads = 1
    /// With nothing waiting, take the performance cores. The four efficiency
    /// cores add about a gigabyte a second out of forty-five, so asking for
    /// eight buys nothing and can lose.
    public var idleThreads: Int

    let prefix = "language_model.model."
    var position = 0

    // Small tensors, read once and kept: 24 of each norm, the convolution
    // taps, and the delta rule's two per-head vectors.
    var inputNorm: [[Float]] = []
    var postNorm: [[Float]] = []
    var finalNorm: [Float] = []
    var gdnNorm: [Int: [Float]] = [:]
    var convTaps: [Int: [Float]] = [:]
    var aLog: [Int: [Float]] = [:]
    var dtBias: [Int: [Float]] = [:]
    var queryNorm: [Int: [Float]] = [:]
    var keyNorm: [Int: [Float]] = [:]
    /// The delta rule's two scalar-per-head projections, which the converter
    /// deliberately leaves at BF16: one row per value head (16 in the 2B, 32
    /// in the 4B), so quantizing them would save nothing and they feed an
    /// exponential, where a rounding error does not stay small.
    var deltaA: [Int: [Float]] = [:]
    var deltaB: [Int: [Float]] = [:]

    // Carried state. Every one of these is a place a wrong hand-off between
    // tokens can hide, which is why sequence parity is a separate gate from
    // position 0.
    var recurrent: [Int: [Float]] = [:]  // [Hv][Dv][Dk]
    var convolution: [Int: [Float]] = [:]  // [K-1][convDim]
    var keys: [Int: [Float]] = [:]  // [position][kvHeads * headDim]
    var values: [Int: [Float]] = [:]

    public init(snapshot: AffineSnapshot, threads: Int? = nil) throws {
        self.snapshot = snapshot
        configuration = snapshot.configuration
        // Both head maps below divide one count by the other. A ratio that is
        // not whole would floor into a mapping that runs and is wrong, and a
        // zero would trap mid-token; either is a config this engine does not
        // implement, so it is refused here, by name.
        for (wide, narrow, what) in [
            (configuration.heads, configuration.keyValueHeads, "attention heads over KV heads"),
            (
                configuration.linearValueHeads, configuration.linearKeyHeads,
                "delta-rule value heads over key heads"
            ),
        ] where narrow <= 0 || wide < narrow || wide % narrow != 0 {
            throw SafeTensorsFile.Failure.malformed("\(what) is \(wide)/\(narrow), not whole")
        }
        self.threads = threads ?? Int8AffineGEMV.preferredThreads
        idleThreads = threads ?? Int8AffineGEMV.preferredThreads
        for layer in 0..<configuration.layers {
            inputNorm.append(try snapshot.floats("\(prefix)layers.\(layer).input_layernorm.weight"))
            postNorm.append(
                try snapshot.floats(
                    "\(prefix)layers.\(layer).post_attention_layernorm.weight"))
            if configuration.isAttention(layer) {
                queryNorm[layer] = try snapshot.floats(
                    "\(prefix)layers.\(layer).self_attn.q_norm.weight")
                keyNorm[layer] = try snapshot.floats(
                    "\(prefix)layers.\(layer).self_attn.k_norm.weight")
            } else {
                let stem = "\(prefix)layers.\(layer).linear_attn."
                gdnNorm[layer] = try snapshot.floats(stem + "norm.weight")
                convTaps[layer] = try snapshot.floats(stem + "conv1d.weight")
                aLog[layer] = try snapshot.floats(stem + "A_log")
                dtBias[layer] = try snapshot.floats(stem + "dt_bias")
                deltaA[layer] = try snapshot.floats(stem + "in_proj_a.weight")
                deltaB[layer] = try snapshot.floats(stem + "in_proj_b.weight")
            }
        }
        finalNorm = try snapshot.floats("\(prefix)norm.weight")
    }

    /// Forget the conversation. Every carried state goes, and the next token
    /// is position zero again.
    public func reset() {
        position = 0
        recurrent.removeAll(keepingCapacity: true)
        convolution.removeAll(keepingCapacity: true)
        keys.removeAll(keepingCapacity: true)
        values.removeAll(keepingCapacity: true)
    }

    // MARK: - one token

    /// Returns the logits over the whole vocabulary.
    @discardableResult
    public func step(token: Int) throws -> [Float] {
        try step(token: token, needsLogits: true)
    }

    /// One token, optionally without the output head.
    ///
    /// For the 2B and 4B the head is the tied embedding: 248,320 rows over
    /// the hidden width, half a gigabyte of the 2B's 1.9 at 8 bits, read in
    /// full for every token. The 9B reads its own `lm_head` of the same
    /// shape. A prompt token's logits are thrown away — only the last one's
    /// are used — so computing them costs about a third of each prompt token
    /// for nothing.
    @discardableResult
    public func step(token: Int, needsLogits: Bool) throws -> [Float] {
        applyWidthPolicy()
        var h = try embedding(of: token)
        for layer in 0..<configuration.layers {
            var normalized = h
            CPUOps.rmsNorm(
                &normalized, gamma: inputNorm[layer],
                epsilon: configuration.normEpsilon)
            let mixed =
                configuration.isAttention(layer)
                ? try attention(layer: layer, x: normalized)
                : try gatedDeltaNet(layer: layer, x: normalized)
            for index in h.indices { h[index] += mixed[index] }

            normalized = h
            CPUOps.rmsNorm(
                &normalized, gamma: postNorm[layer],
                epsilon: configuration.normEpsilon)
            let feed = try mlp(layer: layer, x: normalized)
            for index in h.indices { h[index] += feed[index] }
        }
        CPUOps.rmsNorm(&h, gamma: finalNorm, epsilon: configuration.normEpsilon)
        position += 1
        return needsLogits ? try head(h) : []
    }

    /// Re-decide the width. Called before every token; separate so the
    /// decision can be tested without a forward pass.
    public func applyWidthPolicy() {
        guard let contention else { return }
        threads = contention() ? busyThreads : idleThreads
    }

    // MARK: - blocks

    func matrix(_ name: String) throws -> AffineSnapshot.Matrix {
        try snapshot.matrix(name)
    }

    func project(_ matrix: AffineSnapshot.Matrix, _ x: [Float]) throws -> [Float] {
        var out = [Float](repeating: 0, count: matrix.rows)
        try x.withUnsafeBufferPointer { input in
            try out.withUnsafeMutableBufferPointer { output in
                // An empty input or a zero-row matrix has no base address and
                // nothing to compute; the force unwrap made that a crash.
                guard let inputBase = input.baseAddress, let outputBase = output.baseAddress else {
                    return
                }
                try CPUOps.gemv(matrix, x: inputBase, out: outputBase, threads: threads)
            }
        }
        return out
    }

    /// `out = W · x` for a small dense row-major matrix. Only the delta
    /// rule's two per-head projections come through here.
    func dense(_ weights: [Float], rows: Int, x: [Float]) -> [Float] {
        let columns = weights.count / rows
        var out = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            var total: Float = 0
            let base = row * columns
            for column in 0..<columns { total += weights[base + column] * x[column] }
            out[row] = total
        }
        return out
    }

    func embedding(of token: Int) throws -> [Float] {
        let table = try matrix("\(prefix)embed_tokens.weight")
        precondition(token >= 0 && token < table.rows, "token \(token) out of range")
        return dequantize(row: token, of: table)
    }

    /// The output projection. The 2B and 4B tie it to the embedding table, so
    /// the logits are one GEMV over 248320 rows — the largest single read of
    /// every token, which is why it dominates the 1.9 GB per-token figure.
    ///
    ///
    /// The 9B is the vision-language build and does **not** tie: it ships its
    /// own `lm_head.weight`, which the converter carries through at
    /// `language_model.lm_head.weight` — note, without the `.model.` the
    /// embedding prefix carries. Reading the table there would produce
    /// plausible-looking but wrong logits, so the flag is honoured rather
    /// than assumed.
    func headWeightName() -> String {
        snapshot.configuration.tiedEmbedding
            ? "\(prefix)embed_tokens.weight"
            : "language_model.lm_head.weight"
    }

    func head(_ h: [Float]) throws -> [Float] {
        try project(try matrix(headWeightName()), h)
    }

    func mlp(layer: Int, x: [Float]) throws -> [Float] {
        let stem = "\(prefix)layers.\(layer).mlp."
        var gate = try project(try matrix(stem + "gate_proj.weight"), x)
        let up = try project(try matrix(stem + "up_proj.weight"), x)
        for index in gate.indices { gate[index] = CPUOps.silu(gate[index]) * up[index] }
        return try project(try matrix(stem + "down_proj.weight"), gate)
    }

    func attention(layer: Int, x: [Float]) throws -> [Float] {
        let stem = "\(prefix)layers.\(layer).self_attn."
        let heads = configuration.heads
        let kvHeads = configuration.keyValueHeads
        let dim = configuration.headDim

        // q_proj is twice as wide: the output gate is fused into it, packed
        // per head as [query ; gate]. Reading it as a plain query projection
        // gives a model that runs and is wrong.
        let packed = try project(try matrix(stem + "q_proj.weight"), x)
        var query = [Float](repeating: 0, count: heads * dim)
        var gate = [Float](repeating: 0, count: heads * dim)
        for head in 0..<heads {
            let source = head * 2 * dim
            for index in 0..<dim {
                query[head * dim + index] = packed[source + index]
                gate[head * dim + index] = packed[source + dim + index]
            }
        }
        var key = try project(try matrix(stem + "k_proj.weight"), x)
        let value = try project(try matrix(stem + "v_proj.weight"), x)

        // The per-layer norm weights must be in the snapshot; the force
        // unwraps below made a missing tensor a crash instead of an error.
        guard let queryNormWeights = queryNorm[layer],
            let keyNormWeights = keyNorm[layer]
        else {
            throw ModelError.internalInconsistency(
                detail: "attention layer \(layer) is missing its query/key norm weights")
        }
        CPUOps.rmsNormPerSlice(
            &query, width: dim, gamma: queryNormWeights,
            epsilon: configuration.normEpsilon)
        CPUOps.rmsNormPerSlice(
            &key, width: dim, gamma: keyNormWeights,
            epsilon: configuration.normEpsilon)
        CPUOps.applyRoPE(
            &query, headDim: dim, rotaryDim: configuration.rotaryDim,
            position: position, theta: configuration.ropeTheta)
        CPUOps.applyRoPE(
            &key, headDim: dim, rotaryDim: configuration.rotaryDim,
            position: position, theta: configuration.ropeTheta)

        // Taken out of the dictionary and put back after. Indexing
        // `keys[layer]!` inside the inner loop is a dictionary lookup and a
        // uniqueness check *per element*, which cost more than the attention
        // arithmetic it was wrapping -- measured, it more than halved the
        // engine's throughput.
        var cachedKeys = keys.removeValue(forKey: layer) ?? []
        var cachedValues = values.removeValue(forKey: layer) ?? []
        cachedKeys.append(contentsOf: key)
        cachedValues.append(contentsOf: value)
        let cached = cachedKeys.count / (kvHeads * dim)

        // Consecutive query heads share a KV head (transformers' `repeat_kv`
        // expands each KV head in place), so it is `head / group`, not
        // `head % kvHeads`. The 2B shares 2 KV heads among 8, the 4B 4 among 16.
        let group = heads / kvHeads
        let scale = 1 / Float(dim).squareRoot()
        var out = [Float](repeating: 0, count: heads * dim)
        var scores = [Float](repeating: 0, count: cached)
        cachedKeys.withUnsafeBufferPointer { keyStore in
            cachedValues.withUnsafeBufferPointer { valueStore in
                query.withUnsafeBufferPointer { queries in
                    out.withUnsafeMutableBufferPointer { output in
                        for head in 0..<heads {
                            let kvHead = head / group
                            let queryBase = head * dim
                            for step in 0..<cached {
                                let base = step * kvHeads * dim + kvHead * dim
                                var total: Float = 0
                                for index in 0..<dim {
                                    total += keyStore[base + index] * queries[queryBase + index]
                                }
                                scores[step] = total * scale
                            }
                            CPUOps.softmaxInPlace(&scores)
                            for step in 0..<cached {
                                let weight = scores[step]
                                if weight == 0 { continue }
                                let base = step * kvHeads * dim + kvHead * dim
                                for index in 0..<dim {
                                    output[queryBase + index] += weight * valueStore[base + index]
                                }
                            }
                        }
                    }
                }
            }
        }
        keys[layer] = cachedKeys
        values[layer] = cachedValues
        for index in out.indices { out[index] *= CPUOps.sigmoid(gate[index]) }
        return try project(try matrix(stem + "o_proj.weight"), out)
    }

    func gatedDeltaNet(layer: Int, x: [Float]) throws -> [Float] {
        let stem = "\(prefix)layers.\(layer).linear_attn."
        let hk = configuration.linearKeyHeads
        let hv = configuration.linearValueHeads
        let dk = configuration.linearKeyHeadDim
        let dv = configuration.linearValueHeadDim
        let kernel = configuration.convKernel

        let mixed = try project(try matrix(stem + "in_proj_qkv.weight"), x)
        let z = try project(try matrix(stem + "in_proj_z.weight"), x)
        // Every per-layer tensor this path needs, checked once: the force
        // unwraps below made a missing tensor a crash.
        guard let deltaWeightsA = deltaA[layer],
            let deltaWeightsB = deltaB[layer],
            let taps = convTaps[layer],
            let decayLog = aLog[layer],
            let decayBias = dtBias[layer],
            let normGamma = gdnNorm[layer]
        else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(layer) is missing delta-rule or norm weights")
        }
        let a = dense(deltaWeightsA, rows: hv, x: x)
        let b = dense(deltaWeightsB, rows: hv, x: x)
        let convDim = mixed.count

        // Causal depthwise convolution over the last `kernel` tokens. Tap
        // `k` reads `kernel - 1 - k` positions back, so the last tap is this
        // token.
        var tail = convolution[layer] ?? [Float](repeating: 0, count: (kernel - 1) * convDim)
        var convolved = [Float](repeating: 0, count: convDim)
        for channel in 0..<convDim {
            var total: Float = 0
            for step in 0..<(kernel - 1) {
                total += taps[channel * kernel + step] * tail[step * convDim + channel]
            }
            total += taps[channel * kernel + kernel - 1] * mixed[channel]
            convolved[channel] = CPUOps.silu(total)
        }
        // Slide the window: drop the oldest row, append this token's input.
        tail.removeFirst(convDim)
        tail.append(contentsOf: mixed)
        convolution[layer] = tail

        let keyWidth = hk * dk
        var query = Array(convolved[0..<keyWidth])
        var key = Array(convolved[keyWidth..<(2 * keyWidth)])
        let value = Array(convolved[(2 * keyWidth)...])
        CPUOps.l2NormalizePerSlice(&query, width: dk, epsilon: configuration.normEpsilon)
        CPUOps.l2NormalizePerSlice(&key, width: dk, epsilon: configuration.normEpsilon)

        var state = recurrent[layer] ?? [Float](repeating: 0, count: hv * dv * dk)
        var readout = [Float](repeating: 0, count: hv * dv)
        // Query and key are Hk heads wide, value and the state Hv. Where Hv
        // is a multiple -- the 4B's 32 over 16 -- each key head serves that
        // many *consecutive* value heads: transformers `repeat_interleave`s,
        // the oracle `np.repeat`s, gdn.metal divides. `head % hk` would pair
        // value head 1 with key head 1 and is the plausible wrong answer.
        let repeats = hv / hk
        for head in 0..<hv {
            let keyHead = head / repeats
            let beta = CPUOps.sigmoid(b[head])
            let decay = expf(
                -expf(decayLog[head])
                    * CPUOps.softplus(a[head] + decayBias[head]))
            let stateBase = head * dv * dk
            let keyBase = keyHead * dk
            for row in 0..<dv {
                let rowBase = stateBase + row * dk
                // Decay, read the stored value for this key, correct it
                // towards the new one, and write the correction back as an
                // outer product. This is the delta rule.
                var stored: Float = 0
                for column in 0..<dk {
                    let decayed = state[rowBase + column] * decay
                    state[rowBase + column] = decayed
                    stored += decayed * key[keyBase + column]
                }
                let correction = (value[head * dv + row] - stored) * beta
                var sum: Float = 0
                for column in 0..<dk {
                    let updated = state[rowBase + column] + key[keyBase + column] * correction
                    state[rowBase + column] = updated
                    sum += updated * query[keyBase + column]
                }
                readout[head * dv + row] = sum
            }
        }
        recurrent[layer] = state

        let inverse = 1 / Float(dv).squareRoot()
        for index in readout.indices { readout[index] *= inverse }
        CPUOps.rmsNormPerSlice(
            &readout, width: dv, gamma: normGamma,
            epsilon: configuration.normEpsilon)
        // SiLU, not sigmoid. The gate is `silu` in this lineage and
        // `sigmoid` in Qwen3.8-Flash-Next; getting it wrong produces a model
        // that runs, keeps healthy activations, and predicts a bare space
        // for "Once upon a".
        for index in readout.indices { readout[index] *= CPUOps.silu(z[index]) }
        return try project(try matrix(stem + "out_proj.weight"), readout)
    }

    // MARK: - generation

    /// Feed a prompt and continue it, greedily.
    ///
    /// Greedy because of what this engine is for. It distils a session into
    /// facts and checks a claim against a store; both want the model's best
    /// answer and neither wants variety, and a deterministic side-engine is
    /// one whose output can be compared between runs. Sampling can be added
    /// when something needs it.
    ///
    /// `onToken` sees each generated id as it is produced, so a caller can
    /// stream or stop early; returning false ends the generation.
    @discardableResult
    public func generate(
        prompt: [Int],
        maximumTokens: Int,
        stopping: Set<Int> = [],
        onToken: ((Int) -> Bool)? = nil
    ) throws -> [Int] {
        precondition(!prompt.isEmpty, "a generation needs a prompt")
        var logits = try prefill(tokens: prompt)
        var produced: [Int] = []
        for _ in 0..<maximumTokens {
            var best = 0
            for index in logits.indices where logits[index] > logits[best] { best = index }
            if stopping.contains(best) { break }
            produced.append(best)
            if let onToken, !onToken(best) { break }
            logits = try step(token: best)
        }
        return produced
    }

}

// TT-097: the Core CPU-engine contract. Every member already existed with
// this exact surface (the same file family in every module), so the
// conformance is a declaration only — no behavior changes.
