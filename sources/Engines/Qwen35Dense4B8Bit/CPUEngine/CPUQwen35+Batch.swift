import Foundation

/// Batched prompt prefill for the CPU engine.
///
/// The decode path is unchanged: `step` still runs one token through the
/// quantized GEMVs. A prompt is different — the same weight bytes would be
/// read once per token — so it goes through `CPUBatchedMatmul`, which unpacks
/// each matrix once and multiplies all T rows with the matrix engine.
///
/// Everything that carries state between tokens stays ordered: the chunk loop
/// hands one chunk's KV cache, convolution tail and delta-rule state to the
/// next, exactly as the per-token loop did. The gated-DeltaNet recurrence is
/// still sequential inside a chunk; its projections are batched.
///
/// `TINYTITAN_CPU_PREFILL=token` forces the per-token path and
/// `TINYTITAN_CPU_PREFILL=batch` forces the batched one, which is how the two
/// are compared on a real model at sizes below the automatic threshold.
extension CPUQwen35 {

    /// Tokens one batched pass carries. Bounds the T-by-intermediate buffers
    /// (at 2K tokens the 4B's 9,216-wide FFN is already 75 MB) and keeps each
    /// chunk's attention reading a cache it has just written. State handed
    /// between chunks is the state the per-token loop would carry.
    static let prefillChunkTokens = 2048

    /// Below this many tokens the per-token path wins. The batched path
    /// dequantizes every matrix in the model once; for a handful of prompt
    /// tokens that is more traffic than reading each weight a handful of
    /// times. Measured with twelve prompts per size on the 2B and 4B, ten
    /// tokens favour the per-token path and fifteen favour the batched one,
    /// so the crossover sits between them.
    static let prefillMinimumTokens = 12

    /// Run prompt tokens through the batched path, returning the logits of
    /// the last one. A one-token prompt is just `step`.
    public func prefill(tokens: [Int]) throws -> [Float] {
        precondition(!tokens.isEmpty, "a prefill needs a prompt")
        let mode = ProcessInfo.processInfo.environment["TINYTITAN_CPU_PREFILL"]
        if tokens.count < Self.prefillMinimumTokens && mode != "batch" {
            var logits: [Float] = []
            for (index, token) in tokens.enumerated() {
                logits = try step(token: token, needsLogits: index == tokens.count - 1)
            }
            return logits
        }
        if mode == "token" {
            var logits: [Float] = []
            for (index, token) in tokens.enumerated() {
                logits = try step(token: token, needsLogits: index == tokens.count - 1)
            }
            return logits
        }
        applyWidthPolicy()
        var logits: [Float] = []
        var start = 0
        while start < tokens.count {
            let end = min(start + Self.prefillChunkTokens, tokens.count)
            logits = try prefillChunk(Array(tokens[start..<end]), needsLogits: end == tokens.count)
            start = end
        }
        return logits
    }

    /// One chunk: every layer over every token in the chunk.
    func prefillChunk(_ tokens: [Int], needsLogits: Bool) throws -> [Float] {
        let count = tokens.count
        let hiddenWidth = configuration.hiddenSize
        var hidden = [Float](repeating: 0, count: count * hiddenWidth)
        for (row, token) in tokens.enumerated() {
            let embedded = try embedding(of: token)
            precondition(embedded.count == hiddenWidth, "embedding width is not the hidden size")
            let base = row * hiddenWidth
            for index in 0..<hiddenWidth { hidden[base + index] = embedded[index] }
        }

        // RoPE positions are the chunk's global positions. The position
        // counter advances only after the chunk, so a throw mid-chunk leaves
        // it where the next attempt's state expects it, as `step` does.
        let base = position
        for layer in 0..<configuration.layers {
            var normalized = hidden
            Self.rmsNormRows(
                &normalized, rows: count, gamma: inputNorm[layer],
                epsilon: configuration.normEpsilon)
            let mixed =
                configuration.isAttention(layer)
                ? try attentionBatch(
                    layer: layer, x: normalized, tokens: count, basePosition: base)
                : try gatedDeltaNetBatch(layer: layer, x: normalized, tokens: count)
            for index in hidden.indices { hidden[index] += mixed[index] }

            normalized = hidden
            Self.rmsNormRows(
                &normalized, rows: count, gamma: postNorm[layer],
                epsilon: configuration.normEpsilon)
            let feed = try mlpBatch(layer: layer, x: normalized, tokens: count)
            for index in hidden.indices { hidden[index] += feed[index] }
        }
        position += count
        guard needsLogits else { return [] }
        var last = Array(hidden[(count - 1) * hiddenWidth..<count * hiddenWidth])
        CPUOps.rmsNorm(&last, gamma: finalNorm, epsilon: configuration.normEpsilon)
        return try head(last)
    }

    /// The FFN for T rows: gate and up projected together, then down.
    func mlpBatch(layer: Int, x: [Float], tokens: Int) throws -> [Float] {
        let stem = "\(prefix)layers.\(layer).mlp."
        let gate = try CPUBatchedMatmul.multiply(
            try matrix(stem + "gate_proj.weight"), x: x, tokenCount: tokens, threads: threads)
        let up = try CPUBatchedMatmul.multiply(
            try matrix(stem + "up_proj.weight"), x: x, tokenCount: tokens, threads: threads)
        var gated = gate
        for index in gated.indices { gated[index] = CPUOps.silu(gate[index]) * up[index] }
        return try CPUBatchedMatmul.multiply(
            try matrix(stem + "down_proj.weight"), x: gated, tokenCount: tokens, threads: threads)
    }

    /// RMS norm each row independently with one gamma vector.
    static func rmsNormRows(
        _ values: inout [Float], rows: Int, gamma: [Float], epsilon: Float
    ) {
        let width = gamma.count
        precondition(values.count == rows * width, "rmsNormRows: shape does not match gamma")
        for row in 0..<rows {
            let base = row * width
            var sum: Float = 0
            for index in 0..<width {
                let value = values[base + index]
                sum += value * value
            }
            let scale = 1 / (sum / Float(width) + epsilon).squareRoot()
            for index in 0..<width {
                values[base + index] = values[base + index] * scale * gamma[index]
            }
        }
    }
}
