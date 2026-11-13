import Foundation
import Metal

/// The Qwen3.8-Flash-Next draft head's fusion step.
///
/// Every other part of this port was checked against a reference. This one
/// could not be: upstream `transformers` drops the `mtp.*` namespace outright
/// (`_keys_to_ignore_on_load_unexpected`), and the MLX port infers the fusion
/// from tensor shapes, says so in its own comments, and applies the gamma
/// twice while doing it. So what follows is the reading the shapes support,
/// and the only evidence for it is the acceptance rate.
///
/// That is safe to be uncertain about in a way the rest of the port was not:
/// speculative decoding is exact. The emitted tokens are always the target's
/// own argmax, so a wrong fusion costs throughput and never output. A draft
/// that is never accepted is a slow correct model, not a wrong one.
///
/// The shapes: `pre_fc_norm_hidden` is `[hc_dim]` and `fc_hidden` is
/// `[D, D]`. A norm over the wide residual followed by a projection that only
/// accepts one stream's width means something collapses in between, and the
/// plain mean is the collapse that inverts `hc_init`. `fc_embedding` is
/// `[D, D]` and `pre_fc_norm_embedding` is `[D]`, so that branch is ordinary.
/// Two `[D, D]` matrices rather than one `[D, 2D]` is what says the branches
/// are summed rather than concatenated.
extension RealForwardRunner {
    /// Routes a draft step to the fusion its family uses. The two drafts share
    /// everything after the fusion, so this is the whole of the difference.
    func advanceMTPForFamily(
        tokens: ArraySlice<Int32>,
        targetHiddenRows: Data,
        startPosition: Int,
        predictNext: Bool
    ) async throws -> Int32? {
        switch cfg.family {
        case .qwen38flashMTP, .qwen4flashMTP:
            return try await advanceMTP38(
                tokens: tokens,
                targetHiddenRows: targetHiddenRows,
                startPosition: startPosition,
                predictNext: predictNext)
        default:
            return try await advanceMTP(
                tokens: tokens,
                targetHiddenRows: targetHiddenRows,
                startPosition: startPosition,
                predictNext: predictNext)
        }
    }

    /// Fuses the target's wide residual with the next token's embedding and
    /// runs the draft layer over the result.
    ///
    /// Mirrors `advanceMTP`, which serves the Qwen 3.6 draft; the two differ
    /// only in the fusion, so the chunk execution below is shared.
    func advanceMTP38(
        tokens: ArraySlice<Int32>,
        targetHiddenRows: Data,
        startPosition: Int,
        predictNext: Bool
    ) async throws -> Int32? {
        guard cfg.family == .qwen38flashMTP || cfg.family == .qwen4flashMTP else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        guard !tokens.isEmpty, tokens.count <= Self.mtpChunkCapacity else {
            throw PrefillError.chunkedUnsupported(
                "MTP adapter accepts 1...\(Self.mtpChunkCapacity) aligned rows")
        }
        let wide = residualWidth
        let expectedBytes = tokens.count * wide * MemoryLayout<Float16>.stride
        guard targetHiddenRows.count == expectedBytes else {
            throw PrefillError.chunkedUnsupported(
                "MTP target hidden payload has \(targetHiddenRows.count) bytes; "
                    + "expected \(expectedBytes)")
        }
        guard let tokenBuffer = mtpTokenBlock,
            let embeddingBlock = mtpEmbeddingBlock,
            let normalizedEmbedding = mtpNormalizedEmbeddingBlock,
            let collapsed = mtpNormalizedHiddenBlock,
            let normalizedWide = mtpConcatBlock,
            let projected = mtpProjectedBlock,
            let targetHidden = mtpTargetHiddenBlock,
            let elementwise
        else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        targetHiddenRows.copyBytes(
            to: targetHidden.contents()
                .assumingMemoryBound(to: UInt8.self), count: expectedBytes)
        let ids = tokens.map { UInt32(bitPattern: $0) }
        ids.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            tokenBuffer.contents().copyMemory(from: base, byteCount: bytes.count)
        }
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try encodeFusion(
            commandBuffer: cb, rows: tokens.count,
            targetHidden: targetHidden, collapsed: collapsed,
            normalizedWide: normalizedWide,
            tokenBuffer: tokenBuffer,
            embeddingBlock: embeddingBlock,
            normalizedEmbedding: normalizedEmbedding,
            projected: projected, elementwise: elementwise)
        cb.commit()
        try waitForCompletion(cb)
        return try await runDraftLayer(
            tokens: tokens,
            startPosition: startPosition,
            predictNext: predictNext,
            fused: projected)
    }

    /// The two projection branches and their sum. Split out so the fusion --
    /// the only inferred part of this family's draft -- reads as one thing.
    private func encodeFusion(
        commandBuffer cb: MTLCommandBuffer,
        rows: Int,
        targetHidden: MTLBuffer,
        collapsed: MTLBuffer,
        normalizedWide: MTLBuffer,
        tokenBuffer: MTLBuffer,
        embeddingBlock: MTLBuffer,
        normalizedEmbedding: MTLBuffer,
        projected: MTLBuffer,
        elementwise: Elementwise
    ) throws {
        let D = cfg.hiddenSize
        // Hidden branch: grouped norm across the streams, mean-collapse, project.
        let wideNorm = try model.mtpWideNorm()
        try rms.encodeBF16WGrouped(
            commandBuffer: cb,
            x: targetHidden,
            weight: wideNorm.buffer,
            weightOffset: Int(wideNorm.offset),
            out: normalizedWide,
            groupDim: UInt32(D),
            numGroups: residualStreamCount,
            eps: 1e-6, tokens: rows)
        try elementwise.encodeHCMean(
            commandBuffer: cb,
            streams: normalizedWide,
            out: collapsed,
            dim: D, streamCount: residualStreamCount,
            tokens: rows)
        try encodeDraftFusionProjection(
            commandBuffer: cb, projection: try model.mtpHiddenProjection(),
            x: collapsed, y: projected, rows: rows)

        // Token branch: embed, plain norm, project.
        try encodeEmbeddingBlock(
            commandBuffer: cb, tokens: tokenBuffer,
            out: embeddingBlock, t: rows, outScale: 1)
        let tokenNorm = try model.mtpTokenNorm()
        try prefillRMS.encodeBF16W(
            commandBuffer: cb,
            x: embeddingBlock,
            weight: tokenNorm.buffer,
            weightOffset: Int(tokenNorm.offset),
            out: normalizedEmbedding,
            t: UInt32(rows), d: UInt32(D), eps: 1e-6)
        try encodeDraftFusionProjection(
            commandBuffer: cb, projection: try model.mtpEmbeddingProjection(),
            x: normalizedEmbedding, y: embeddingBlock, rows: rows)
        // Sum the branches in place: `projected` becomes the fused input the
        // draft layer starts from.
        try elementwise.encodeResidualAdd(
            commandBuffer: cb,
            hidden: projected,
            delta: embeddingBlock,
            count: rows * D)
    }

    /// Runs the draft's single decoder layer over the fused input and reads
    /// back its prediction. Identical in shape to the Qwen 3.6 draft's tail.
    private func runDraftLayer(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        predictNext: Bool,
        fused: MTLBuffer
    ) async throws -> Int32? {
        let projected = fused
        let runtime = PrefillRuntimeConfig.production(chunkTokens: 32)
        let scratch = try ensurePrefillScratch(config: runtime)
        let mode: PrefillOutputMode = useFusedGreedyHead ? .greedyIfAvailable : .logits
        try await executePrefillChunk(
            tokens: tokens,
            startPosition: startPosition,
            outputMode: mode,
            logits: verificationLogits,
            scratch: scratch,
            config: runtime,
            writeFinalHead: predictNext,
            preparedHidden: projected)
        guard predictNext else { return nil }
        if useFusedGreedyHead { return Int32(bitPattern: lastGreedyToken) }
        let values = verificationLogits.contents()
            .assumingMemoryBound(to: Float16.self)
        var best = 0
        var bestValue = Float(values[0])
        for index in 1..<cfg.vocabSize where Float(values[index]) > bestValue {
            best = index
            bestValue = Float(values[index])
        }
        return Int32(best)
    }
}
