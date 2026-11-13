import Foundation
import Metal

/// What a bf16-rest install needs beyond the per-projection dtype dispatch:
/// the embedding and the vocabulary head kept at the checkpoint's bf16.
///
/// Every other dense tensor already dispatches on its own dtype (`encodeRoleGEMV`,
/// `SlotGEMV`, the router's effective width). The embedding and the head did
/// not: they took their reading from the manifest's embedding slot, so a bf16
/// table inside a slot that still says 4 bits was read as nibbles. Both are
/// decided here from the tensor (`Model.embeddingWeightBits` /
/// `lmHeadWeightBits` return 16 for a bf16 tensor), once, when the runner is
/// built. An install whose embedding and head are quantized gets `embedding ==
/// nil` and `headIsBF16 == false`, and every call site below takes the branch it
/// took before this existed, byte for byte.
struct BF16RestKernels {
    /// The bf16 lookup, or nil when the embedding is quantized.
    let embedding: EmbedLookupBF16?
    /// The lm_head is bf16: `encodeHeadGEMV` and its multi-row forms read it
    /// through the bf16 GEMV with no scales or biases.
    let headIsBF16: Bool

    init(model: Model, context: MetalContext) throws {
        self.embedding =
            model.embeddingWeightBits == 16 ? try EmbedLookupBF16(context: context) : nil
        self.headIsBF16 = model.lmHeadWeightBits == 16
    }
}

extension RealForwardRunner {
    var headIsBF16: Bool { bf16Rest.headIsBF16 }

    /// One token's embedding row into `out`, scaled by `outScale`: the decode
    /// lookup for whatever the table is stored as.
    func encodeEmbeddingLookup(
        commandBuffer cb: MTLCommandBuffer,
        tokenId: UInt32,
        out: MTLBuffer, outOffset: Int = 0,
        outScale: Float
    ) throws {
        let D = UInt32(cfg.hiddenSize)
        let vocab = UInt32(cfg.vocabSize)
        let emb = try model.embedding()
        if let lookup = bf16Rest.embedding {
            try lookup.encode(
                commandBuffer: cb,
                table: emb.buffer, tableOffset: Int(emb.offset),
                out: out, outOffset: outOffset,
                tokenId: tokenId, d: D, outScale: outScale, vocab: vocab)
        } else if let affineEmbed {
            try affineEmbed.encode(
                commandBuffer: cb,
                table: emb.buffer, tableOffset: Int(emb.offset),
                scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                out: out, outOffset: outOffset,
                tokenId: tokenId, d: D, outScale: outScale, vocab: vocab)
        } else {
            try embedInt4.encode(
                commandBuffer: cb,
                table: emb.buffer, tableOffset: Int(emb.offset),
                scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                out: out, outOffset: outOffset,
                tokenId: tokenId, d: D, outScale: outScale, vocab: vocab)
        }
    }

    /// `t` tokens' embedding rows into `out`, token-major: the prefill lookup
    /// (and the MTP drafts' token branch) for whatever the table is stored as.
    func encodeEmbeddingBlock(
        commandBuffer cb: MTLCommandBuffer,
        tokens: MTLBuffer,
        out: MTLBuffer,
        t: Int,
        outScale: Float
    ) throws {
        let emb = try model.embedding()
        if let lookup = bf16Rest.embedding {
            try lookup.encodeBlock(
                commandBuffer: cb,
                table: emb.buffer, tableOffset: Int(emb.offset),
                tokens: tokens, out: out,
                t: UInt32(t), d: UInt32(cfg.hiddenSize),
                outScale: outScale, vocab: UInt32(cfg.vocabSize))
            return
        }
        try prefillEmbed.encode(
            commandBuffer: cb,
            table: emb.buffer,
            tableOffset: Int(emb.offset),
            scales: emb.buffer,
            scalesOffset: Int(emb.scaleOffset),
            biases: emb.buffer,
            biasesOffset: Int(emb.biasOffset),
            tokens: tokens,
            out: out,
            t: UInt32(t),
            d: UInt32(cfg.hiddenSize),
            outScale: outScale,
            vocab: UInt32(cfg.vocabSize))
    }

    /// One of the Qwen3.8 draft's `[D, D]` fusion projections over `rows`
    /// token-major rows: the int4 QMM it always used, or the bf16 GEMV when the
    /// draft keeps the projection at the checkpoint's bf16.
    func encodeDraftFusionProjection(
        commandBuffer cb: MTLCommandBuffer,
        projection: TensorView,
        x: MTLBuffer, y: MTLBuffer,
        rows: Int
    ) throws {
        let D = cfg.hiddenSize
        if Int(projection.dtype) == FusedKernelPolicy.bf16DType {
            try bf16Projection.encodeMultiRow(
                commandBuffer: cb,
                weights: projection.buffer,
                weightsOffset: Int(projection.offset),
                x: x, y: y,
                m: UInt32(D), n: UInt32(D),
                tokens: rows,
                xStrideElements: D,
                yStrideElements: D)
            return
        }
        try prefillQMM.encode(
            commandBuffer: cb,
            weights: projection.buffer,
            weightsOffset: Int(projection.offset),
            scales: projection.buffer,
            scalesOffset: Int(projection.scaleOffset),
            biases: projection.buffer,
            biasesOffset: Int(projection.biasOffset),
            x: x, y: y,
            t: rows, n: D, k: D)
    }
}
