import Foundation
import TinyTitanKernelsC
import TinyTitanCore

/// Batched full-attention layers.
///
/// The projections run as GEMMs; the attention itself keeps the per-head
/// shape the decode path uses, but the score and value products are matrix
/// multiplications instead of a loop over cached positions. The causal mask
/// is applied as a column limit per query row, which is the same set of keys
/// the per-token path sees when it runs one step at a time.
extension CPUQwen35 {

    func attentionBatch(
        layer: Int, x: [Float], tokens: Int, basePosition: Int
    ) throws -> [Float] {
        let stem = "\(prefix)layers.\(layer).self_attn."
        let heads = configuration.heads
        let kvHeads = configuration.keyValueHeads
        let dim = configuration.headDim
        guard heads > 0, kvHeads > 0, dim > 0, heads % kvHeads == 0 else {
            throw ModelError.internalInconsistency(
                detail: "attention layer \(layer): \(heads)/\(kvHeads) heads is not whole")
        }
        guard let queryNormWeights = queryNorm[layer],
            let keyNormWeights = keyNorm[layer]
        else {
            throw ModelError.internalInconsistency(
                detail: "attention layer \(layer) is missing its query/key norm weights")
        }

        let packed = try CPUBatchedMatmul.multiply(
            try matrix(stem + "q_proj.weight"), x: x, tokenCount: tokens, threads: threads)
        var keyRows = try CPUBatchedMatmul.multiply(
            try matrix(stem + "k_proj.weight"), x: x, tokenCount: tokens, threads: threads)
        let valueRows = try CPUBatchedMatmul.multiply(
            try matrix(stem + "v_proj.weight"), x: x, tokenCount: tokens, threads: threads)

        let queryWidth = heads * dim
        let keyWidth = kvHeads * dim
        guard packed.count == tokens * heads * 2 * dim,
            keyRows.count == tokens * keyWidth,
            valueRows.count == tokens * keyWidth
        else {
            throw ModelError.internalInconsistency(
                detail: "attention layer \(layer): projection widths do not match the config")
        }

        // q_proj is twice as wide per head: [query ; gate]. The gate is kept
        // aside and applied to the attention output, after the value product.
        var query = [Float](repeating: 0, count: tokens * queryWidth)
        var gate = [Float](repeating: 0, count: tokens * queryWidth)
        for row in 0..<tokens {
            let source = row * heads * 2 * dim
            for head in 0..<heads {
                let from = source + head * 2 * dim
                let to = row * queryWidth + head * dim
                for index in 0..<dim {
                    query[to + index] = packed[from + index]
                    gate[to + index] = packed[from + dim + index]
                }
            }
            var queryRow = Array(query[row * queryWidth..<(row + 1) * queryWidth])
            CPUOps.rmsNormPerSlice(
                &queryRow, width: dim, gamma: queryNormWeights,
                epsilon: configuration.normEpsilon)
            CPUOps.applyRoPE(
                &queryRow, headDim: dim, rotaryDim: configuration.rotaryDim,
                position: basePosition + row, theta: configuration.ropeTheta)
            let queryBase = row * queryWidth
            for index in 0..<queryWidth {
                query[queryBase + index] = queryRow[index]
            }

            var keyRow = Array(keyRows[row * keyWidth..<(row + 1) * keyWidth])
            CPUOps.rmsNormPerSlice(
                &keyRow, width: dim, gamma: keyNormWeights,
                epsilon: configuration.normEpsilon)
            CPUOps.applyRoPE(
                &keyRow, headDim: dim, rotaryDim: configuration.rotaryDim,
                position: basePosition + row, theta: configuration.ropeTheta)
            let keyBase = row * keyWidth
            for index in 0..<keyWidth {
                keyRows[keyBase + index] = keyRow[index]
            }
        }

        var cachedKeys = keys.removeValue(forKey: layer) ?? []
        var cachedValues = values.removeValue(forKey: layer) ?? []
        cachedKeys.append(contentsOf: keyRows)
        cachedValues.append(contentsOf: valueRows)
        let positions = cachedKeys.count / keyWidth

        let group = heads / kvHeads
        let scale = 1 / Float(dim).squareRoot()
        var out = [Float](repeating: 0, count: tokens * queryWidth)
        for head in 0..<heads {
            let kvHead = head / group
            let headOut = try attentionHead(
                query: query, tokens: tokens, heads: heads, head: head, dim: dim,
                cachedKeys: cachedKeys, cachedValues: cachedValues,
                positions: positions, kvHeads: kvHeads, kvHead: kvHead,
                basePosition: basePosition, scale: scale)
            for row in 0..<tokens {
                let to = row * queryWidth + head * dim
                for index in 0..<dim { out[to + index] = headOut[row * dim + index] }
            }
        }
        keys[layer] = cachedKeys
        values[layer] = cachedValues

        for index in out.indices { out[index] *= CPUOps.sigmoid(gate[index]) }
        return try CPUBatchedMatmul.multiply(
            try matrix(stem + "o_proj.weight"), x: out, tokenCount: tokens, threads: threads)
    }

    /// One head over the whole chunk: gather the head's rows out of the
    /// interleaved cache, then two GEMMs — scores and the weighted values.
    func attentionHead(
        query: [Float], tokens: Int, heads: Int, head: Int, dim: Int,
        cachedKeys: [Float], cachedValues: [Float],
        positions: Int, kvHeads: Int, kvHead: Int,
        basePosition: Int, scale: Float
    ) throws -> [Float] {
        let keyWidth = kvHeads * dim
        var queryHead = [Float](repeating: 0, count: tokens * dim)
        for row in 0..<tokens {
            let from = row * heads * dim + head * dim
            for index in 0..<dim { queryHead[row * dim + index] = query[from + index] }
        }
        var keyHead = [Float](repeating: 0, count: positions * dim)
        var valueHead = [Float](repeating: 0, count: positions * dim)
        for position in 0..<positions {
            let from = position * keyWidth + kvHead * dim
            for index in 0..<dim {
                keyHead[position * dim + index] = cachedKeys[from + index]
                valueHead[position * dim + index] = cachedValues[from + index]
            }
        }

        let scoreCount = tokens * positions
        var scores = [Float](repeating: 0, count: scoreCount)
        multiply(
            queryHead, keyHead, rows: tokens, columns: positions, inner: dim, transposed: true,
            into: &scores)
        let inverse = scale
        for row in 0..<tokens {
            let limit = basePosition + row
            guard limit < positions else {
                throw ModelError.internalInconsistency(
                    detail: "attention mask limit \(limit) is past the cache (\(positions))")
            }
            let base = row * positions
            // Column limit: the keys after this row's position do not exist
            // for it. Everything past the limit is dropped before the softmax
            // rather than exponentiated and multiplied by zero.
            var peak = -Float.infinity
            for column in 0...limit {
                scores[base + column] *= inverse
                if scores[base + column] > peak { peak = scores[base + column] }
            }
            var total: Float = 0
            for column in 0...limit {
                let value = expf(scores[base + column] - peak)
                scores[base + column] = value
                total += value
            }
            let normalization = 1 / total
            for column in 0...limit { scores[base + column] *= normalization }
            for column in (limit + 1)..<positions { scores[base + column] = 0 }
        }

        var result = [Float](repeating: 0, count: tokens * dim)
        multiply(
            scores, valueHead, rows: tokens, columns: dim, inner: positions, transposed: false,
            into: &result)
        return result
    }

    /// Row-major `out[rows, columns] = a[rows, inner] * b^T` or `b` per the
    /// flag, with `b` stored row-major as `[columns or inner, inner]`.
    func multiply(
        _ a: [Float], _ b: [Float], rows: Int, columns: Int, inner: Int,
        transposed: Bool, into out: inout [Float]
    ) {
        let trans = transposed ? 1 : 0
        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                out.withUnsafeMutableBufferPointer { outBuffer in
                    guard let aBase = aBuffer.baseAddress,
                        let bBase = bBuffer.baseAddress,
                        let outBase = outBuffer.baseAddress
                    else { return }
                    tinytitan_sgemm_rowmajor(
                        Int32(trans), Int32(rows), Int32(columns), Int32(inner),
                        aBase, bBase, outBase)
                }
            }
        }
    }
}
