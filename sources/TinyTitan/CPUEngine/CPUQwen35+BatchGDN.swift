import Foundation
import TinyTitanKernelsC
import TinyTitanCore

/// Batched gated-DeltaNet layers.
///
/// The input projections and the two per-head scalar projections are batched
/// GEMMs over the whole prompt. The convolution is computed per token (it is
/// causal and cheap), and the delta rule runs over chunks of tokens with
/// `tinytitan_delta_rule_chunk`: the unrolled form of the recurrence reads and
/// writes each head's state once per chunk instead of once per token.
/// `benchmark/gdn_chunked_scan.py` is the specification for that form, and
/// `TINYTITAN_GDN_CHUNK` overrides the chunk length for the A/B.
extension CPUQwen35 {

    /// Tokens one delta-rule chunk carries. The state traffic falls with this
    /// number while the within-chunk Gram work grows with its square, so the
    /// default is a balance rather than a maximum; the environment variable
    /// exists to re-measure it on another machine.
    static var gdnChunkTokens: Int {
        let raw = ProcessInfo.processInfo.environment["TINYTITAN_GDN_CHUNK"]
        return max(1, min(128, Int(raw ?? "") ?? 8))
    }

    func gatedDeltaNetBatch(layer: Int, x: [Float], tokens: Int) throws -> [Float] {
        let stem = "\(prefix)layers.\(layer).linear_attn."
        let hk = configuration.linearKeyHeads
        let hv = configuration.linearValueHeads
        let dk = configuration.linearKeyHeadDim
        let dv = configuration.linearValueHeadDim
        let kernel = configuration.convKernel
        guard hk > 0, hv > 0, dk > 0, dv > 0, kernel > 0, hv % hk == 0 else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(layer): head geometry is not usable")
        }
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

        let mixed = try CPUBatchedMatmul.multiply(
            try matrix(stem + "in_proj_qkv.weight"), x: x, tokenCount: tokens, threads: threads)
        let z = try CPUBatchedMatmul.multiply(
            try matrix(stem + "in_proj_z.weight"), x: x, tokenCount: tokens, threads: threads)
        let aRows = try denseBatch(deltaWeightsA, rows: hv, x: x, tokens: tokens)
        let bRows = try denseBatch(deltaWeightsB, rows: hv, x: x, tokens: tokens)
        let convDim = mixed.count / tokens
        let keyWidth = hk * dk
        let valueWidth = hv * dv
        guard convDim == 2 * keyWidth + valueWidth, z.count == tokens * valueWidth else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(layer): projection widths do not match the config"
            )
        }

        var readout = [Float](repeating: 0, count: tokens * valueWidth)
        var state = recurrent[layer] ?? [Float](repeating: 0, count: hv * dv * dk)
        var tail = convolution[layer] ?? [Float](repeating: 0, count: (kernel - 1) * convDim)
        let inverse = 1 / Float(dv).squareRoot()
        let repeats = hv / hk

        let chunkLength = Self.gdnChunkTokens
        var start = 0
        while start < tokens {
            let count = min(chunkLength, tokens - start)
            var convolved = gdnConvolutionChunk(
                mixed: mixed, taps: taps, tail: &tail, start: start, count: count,
                convDim: convDim, kernel: kernel)
            var siluScratch = [Float](repeating: 0, count: count * convDim)
            convolved.withUnsafeMutableBufferPointer { values in
                siluScratch.withUnsafeMutableBufferPointer { scratch in
                    CPUOps.siluInPlace(values, scratch: scratch)
                }
            }
            gdnNormalizeQueries(
                &convolved, count: count, convDim: convDim, keyWidth: keyWidth, dk: dk)
            let (alphas, betas) = gdnAlphaBeta(
                aRows: aRows, bRows: bRows, decayLog: decayLog, decayBias: decayBias,
                start: start, count: count, heads: hv)
            gdnApplyChunk(
                convolved: convolved, alphas: alphas, betas: betas,
                count: count, convDim: convDim, keyWidth: keyWidth,
                valueWidth: valueWidth, dk: dk, dv: dv, repeats: repeats,
                state: &state, readout: &readout, readoutBase: start * valueWidth,
                threads: threads)
            start += count
        }
        recurrent[layer] = state
        convolution[layer] = tail

        gdnNormalizeReadout(
            readout: &readout, z: z, tokens: tokens, valueWidth: valueWidth, dv: dv,
            normGamma: normGamma, epsilon: configuration.normEpsilon, inverse: inverse)
        return try CPUBatchedMatmul.multiply(
            try matrix(stem + "out_proj.weight"), x: readout, tokenCount: tokens, threads: threads)
    }

    /// A dense (unquantized) matrix against T rows: the delta rule's two
    /// per-head projections are BF16 in the snapshot and tiny, so they are
    /// multiplied through the same BLAS wrapper without a dequantization step.
    func denseBatch(_ weights: [Float], rows: Int, x: [Float], tokens: Int) throws -> [Float] {
        let columns = weights.count / rows
        guard columns > 0, x.count == tokens * columns else {
            throw ModelError.internalInconsistency(
                detail: "batched dense projection: \(weights.count) weights for "
                    + "\(rows) rows and \(tokens)x\(x.count / max(tokens, 1)) activations")
        }
        var out = [Float](repeating: 0, count: tokens * rows)
        x.withUnsafeBufferPointer { xBuffer in
            weights.withUnsafeBufferPointer { wBuffer in
                out.withUnsafeMutableBufferPointer { outBuffer in
                    guard let xBase = xBuffer.baseAddress,
                        let wBase = wBuffer.baseAddress,
                        let outBase = outBuffer.baseAddress
                    else { return }
                    tinytitan_sgemm_rowmajor(
                        Int32(1), Int32(tokens), Int32(rows), Int32(columns),
                        xBase, wBase, outBase)
                }
            }
        }
        return out
    }

    /// Builds one chunk's convolved rows and advances the carried window.
    func gdnConvolutionChunk(
        mixed: [Float], taps: [Float], tail: inout [Float],
        start: Int, count: Int, convDim: Int, kernel: Int
    ) -> [Float] {
        var convolved = [Float](repeating: 0, count: count * convDim)
        for local in 0..<count {
            let mixedBase = (start + local) * convDim
            let outBase = local * convDim
            for channel in 0..<convDim {
                var total: Float = 0
                for step in 0..<(kernel - 1) {
                    total += taps[channel * kernel + step] * tail[step * convDim + channel]
                }
                total += taps[channel * kernel + kernel - 1] * mixed[mixedBase + channel]
                convolved[outBase + channel] = total
            }
            tail.removeFirst(convDim)
            tail.append(contentsOf: mixed[mixedBase..<(mixedBase + convDim)])
        }
        return convolved
    }

    /// L2-normalizes the key and query slices of every token in the chunk.
    func gdnNormalizeQueries(
        _ convolved: inout [Float], count: Int, convDim: Int, keyWidth: Int, dk: Int
    ) {
        for local in 0..<count {
            let base = local * convDim
            var query = Array(convolved[base..<(base + keyWidth)])
            CPUOps.l2NormalizePerSlice(
                &query, width: dk, epsilon: configuration.normEpsilon)
            for index in 0..<keyWidth { convolved[base + index] = query[index] }
            var key = Array(convolved[(base + keyWidth)..<(base + 2 * keyWidth)])
            CPUOps.l2NormalizePerSlice(
                &key, width: dk, epsilon: configuration.normEpsilon)
            for index in 0..<keyWidth { convolved[base + keyWidth + index] = key[index] }
        }
    }

    /// Per-token decay and update rates for the chunk, one column per head.
    func gdnAlphaBeta(
        aRows: [Float], bRows: [Float], decayLog: [Float], decayBias: [Float],
        start: Int, count: Int, heads: Int
    ) -> ([Float], [Float]) {
        var alphas = [Float](repeating: 0, count: count * heads)
        var betas = [Float](repeating: 0, count: count * heads)
        for local in 0..<count {
            let row = start + local
            for head in 0..<heads {
                let decayInput = aRows[row * heads + head] + decayBias[head]
                alphas[local * heads + head] =
                    expf(-expf(decayLog[head]) * CPUOps.softplus(decayInput))
                betas[local * heads + head] = CPUOps.sigmoid(bRows[row * heads + head])
            }
        }
        return (alphas, betas)
    }

    /// Runs the chunked delta rule for every head, splitting heads across the
    /// performance cores when there are enough of them.
    func gdnApplyChunk(
        convolved: [Float], alphas: [Float], betas: [Float],
        count: Int, convDim: Int, keyWidth: Int, valueWidth: Int,
        dk: Int, dv: Int, repeats: Int,
        state: inout [Float], readout: inout [Float], readoutBase: Int, threads: Int
    ) {
        let usable = max(1, min(threads, (alphas.count / count) / 8))
        let heads = alphas.count / count
        convolved.withUnsafeBufferPointer { convolvedBuffer in
            alphas.withUnsafeBufferPointer { alphaBuffer in
                betas.withUnsafeBufferPointer { betaBuffer in
                    state.withUnsafeMutableBufferPointer { stateBuffer in
                        readout.withUnsafeMutableBufferPointer { outputBuffer in
                            guard let convolvedBase = convolvedBuffer.baseAddress,
                                let alphaBase = alphaBuffer.baseAddress,
                                let betaBase = betaBuffer.baseAddress,
                                let stateBase = stateBuffer.baseAddress,
                                let outputBase = outputBuffer.baseAddress
                            else { return }
                            if usable <= 1 {
                                for head in 0..<heads {
                                    _ = tinytitan_delta_rule_chunk(
                                        convolvedBase + (head / repeats) * dk, convDim,
                                        convolvedBase + keyWidth + (head / repeats) * dk, convDim,
                                        convolvedBase + 2 * keyWidth + head * dv, convDim,
                                        alphaBase + head, heads,
                                        betaBase + head, heads,
                                        count, dk, dv,
                                        stateBase + head * dv * dk,
                                        outputBase + readoutBase + head * dv, valueWidth)
                                }
                            } else {
                                let headChunk = (heads + usable - 1) / usable
                                nonisolated(unsafe) let cBase = convolvedBase
                                nonisolated(unsafe) let aBase = alphaBase
                                nonisolated(unsafe) let bBase = betaBase
                                nonisolated(unsafe) let sBase = stateBase
                                nonisolated(unsafe) let oBase = outputBase
                                let base = readoutBase
                                DispatchQueue.concurrentPerform(iterations: usable) { slice in
                                    let first = slice * headChunk
                                    guard first < heads else { return }
                                    let last = min(first + headChunk, heads)
                                    for head in first..<last {
                                        _ = tinytitan_delta_rule_chunk(
                                            cBase + (head / repeats) * dk, convDim,
                                            cBase + keyWidth + (head / repeats) * dk, convDim,
                                            cBase + 2 * keyWidth + head * dv, convDim,
                                            aBase + head, heads,
                                            bBase + head, heads,
                                            count, dk, dv,
                                            sBase + head * dv * dk,
                                            oBase + base + head * dv, valueWidth)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The per-row norm, gate and inverse scaling after the recurrence.
    func gdnNormalizeReadout(
        readout: inout [Float], z: [Float], tokens: Int, valueWidth: Int, dv: Int,
        normGamma: [Float], epsilon: Float, inverse: Float
    ) {
        for row in 0..<tokens {
            let base = row * valueWidth
            var normalized = Array(readout[base..<(base + valueWidth)])
            for index in 0..<valueWidth { normalized[index] *= inverse }
            CPUOps.rmsNormPerSlice(&normalized, width: dv, gamma: normGamma, epsilon: epsilon)
            for index in 0..<valueWidth {
                normalized[index] *= CPUOps.silu(z[base + index])
            }
            for index in 0..<valueWidth { readout[base + index] = normalized[index] }
        }
    }
}
