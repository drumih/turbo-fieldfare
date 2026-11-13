import Foundation
import Metal

/// Prefill of one chunk, layer by layer: the chunk orchestrator and the
/// per-layer pass it dispatches.
///
/// Split out of `RealForwardRunner+Prefill.swift` (2026-09-28) under the
/// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
/// `runPrefillLayer` stays `private`: it moved together with its only caller.
extension RealForwardRunner {
    /// lint:allow-long the orchestrator for one prefill chunk: scratch setup,
    /// the per-layer dispatch, and the head. Each stage it calls is its own
    /// method; what remains is the sequence, and inlining less of it would
    /// only hide the order the stages must run in.
    func executePrefillChunk(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int = 0,
        outputMode: PrefillOutputMode,
        logits: MTLBuffer,
        scratch: PrefillChunkScratchBuffers,
        config: PrefillRuntimeConfig,
        writeFinalHead: Bool,
        preparedHidden: MTLBuffer? = nil,
        snapshotGDNAfterFirstToken: Bool = false,
        useTwoRowProjection: Bool = false,
        verifyRoutedMoE: Bool = false
    ) async throws {
        // Layer-major prefill (one band of chunks walked layer by layer, with
        // a residual per chunk) is gone: every call runs the whole stack in
        // order, so the prologue and epilogue are unconditional.
        let runPrologue = true
        let runEpilogue = true
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires a KV cache")
        }
        let kvPosition = kv?.position(slot: slot) ?? 0
        // Chunk-major advances the cursor per chunk, so it always equals this
        // chunk's start. Layer-major writes a whole band ahead of the cursor
        // and advances once at the end, so the cursor is at or behind the
        // start. Writes themselves are position-addressed and validateRange
        // only bounds against maxContext, so running ahead is safe; this guard
        // is an invariant, not a mechanism.
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        // KV grows on demand rather than reserving maxContext, so make room for
        // this chunk before anything writes into it.
        try kv?.reserve(tokens: startPosition + tokens.count, slot: slot)
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)"
            )
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)"
            )
        }
        guard !snapshotGDNAfterFirstToken || tokens.count == 2 else {
            throw PrefillError.chunkedUnsupported(
                "Gated-DeltaNet speculative checkpoint requires two rows")
        }
        if let kv, kv.fp16RingEnabled,
            let ringLayer = (0..<cfg.numLayers).first(where: {
                kv.ringCapacity(layer: $0) > 0
            })
        {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)"
                )
            }
        }

        let layerViews = try makeLayerPrefillViews()

        // Reused UInt32 token-ID buffer, sized to the largest chunk seen so
        // far and grown on demand (R23); the prefill hot path never allocates
        // a Metal buffer per chunk.
        let tokenBytes = tokens.count * MemoryLayout<UInt32>.stride
        let tokenBuffer: MTLBuffer
        if let existing = prefillTokenBuffer, existing.length >= tokenBytes {
            tokenBuffer = existing
        } else {
            guard
                let made = ctx.device.makeBuffer(
                    length: tokenBytes,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "prefill.tokenIDs"
            prefillTokenBuffer = made
            tokenBuffer = made
        }
        let tokenPtr = tokenBuffer.contents().assumingMemoryBound(to: UInt32.self)
        for (i, token) in tokens.enumerated() {
            tokenPtr[i] = UInt32(bitPattern: token)
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let embedOutScale =
            cfg.embeddingScaledBySqrtHidden
            ? Float(D).squareRoot()
            : 1.0
        let t = tokens.count

        if runPrologue {
            prefillChunkState.markDirty(
                startPosition: startPosition,
                tokenCount: tokens.count)
        }
        // The n-gram rows depend only on token ids, so the whole chunk's
        // gather runs before any layer needs it.
        try gatherPLERowsPrefill(tokens: tokens)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        // The prologue seeds the residual once per chunk.
        if runPrologue {
            if let preparedHidden {
                // The caller hands over `[t, D]` rows -- an MTP draft's fused
                // hidden. A hyper-connection stack starts every stream from that
                // same vector, so it lands in the narrow staging buffer and is
                // widened exactly the way an embedding would be.
                let target = hyperConnection == nil ? scratch.hidden : scratch.normed
                guard let blit = cb.makeBlitCommandEncoder() else {
                    throw ModelError.residentBufferWrapFailed
                }
                blit.copy(
                    from: preparedHidden,
                    sourceOffset: 0,
                    to: target,
                    destinationOffset: 0,
                    size: t * D * MemoryLayout<Float16>.stride)
                blit.endEncoding()
                if hyperConnection != nil {
                    try requireElementwise().encodeHCExpand(
                        commandBuffer: cb,
                        source: scratch.normed, destination: scratch.hidden,
                        dim: D, streamCount: residualStreamCount, tokens: t)
                }
            } else {
                // A hyper-connection stack starts every stream from the token
                // embedding, so the lookup lands in a one-stream staging buffer
                // and is widened from there. `normed` is free until the first
                // layer's read gate writes it.
                let embedTarget = hyperConnection == nil ? scratch.hidden : scratch.normed
                try encodeEmbeddingBlock(
                    commandBuffer: cb, tokens: tokenBuffer,
                    out: embedTarget, t: t, outScale: embedOutScale)
                if hyperConnection != nil {
                    try requireElementwise().encodeHCExpand(
                        commandBuffer: cb,
                        source: scratch.normed, destination: scratch.hidden,
                        dim: D, streamCount: residualStreamCount, tokens: t)
                }
            }
        }

        // Track A: whether this chunk's full-attention layers run on the ANE.
        // The MTP verify (two-row projection / GDN snapshot), MTP adapter
        // chunks (preparedHidden), non-4096 chunk configs, and prompts beyond
        // the sidecar's history variants all stay on the GPU; continuity is
        // enforced inside eligibleChunk so a fallback mid-prompt sticks for
        // the rest of the request.
        let aneChunk: ANEPrefillAttention? = {
            guard let ane = anePrefill,
                kv?.streamsKVFromHost != true,
                slot == 0,
                !snapshotGDNAfterFirstToken,
                !useTwoRowProjection,
                !verifyRoutedMoE,
                preparedHidden == nil,
                ane.eligibleChunk(
                    startPosition: startPosition,
                    tokenCount: tokens.count,
                    configChunkTokens: config.chunkTokens)
            else { return nil }
            return ane
        }()

        let prefillProfile = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_PHASES"] != nil
        var prefillRouteNanos: UInt64 = 0
        var prefillAttentionNanos: UInt64 = 0
        var prefillTileNanos: UInt64 = 0
        var prefillTailNanos: UInt64 = 0
        var prefillActiveExperts: UInt64 = 0

        for L in 0..<cfg.numLayers {
            try await runPrefillLayer(
                L, cb: &cb, scratch: scratch, layerViews: layerViews,
                tokens: tokens, startPosition: startPosition, t: t, D: D,
                eps: eps, useTwoRowProjection: useTwoRowProjection,
                snapshotGDNAfterFirstToken: snapshotGDNAfterFirstToken,
                aneChunk: aneChunk, verifyRoutedMoE: verifyRoutedMoE,
                prefillRouteNanos: &prefillRouteNanos,
                prefillAttentionNanos: &prefillAttentionNanos,
                prefillTileNanos: &prefillTileNanos,
                prefillTailNanos: &prefillTailNanos,
                prefillActiveExperts: &prefillActiveExperts,
                slot: slot)
        }

        if prefillProfile {
            let prefillTotal = prefillRouteNanos + prefillTileNanos + prefillTailNanos
            let perLayer = Double(prefillActiveExperts) / Double(max(1, cfg.numLayers))
            // stderr, not stdout: this target is inside `TinyTitanLib`'s closure,
            // and stdout belongs to the program that embeds the library.
            let trace: String =
                "[prefill phases over \(t) tokens, \(prefillTotal / 1_000_000) ms total]\n"
                + "  route readback + GPU: \(String(format: "%.1f", Double(prefillRouteNanos) / 1e6)) ms\n"
                + "    of which attention + norms: \(String(format: "%.1f", Double(prefillAttentionNanos) / 1e6)) ms\n"
                + "    of which router + readback: \(String(format: "%.1f", Double(prefillRouteNanos - prefillAttentionNanos) / 1e6)) ms\n"
                + "  expert fetch + tiles: \(String(format: "%.1f", Double(prefillTileNanos) / 1e6)) ms\n"
                + "  tail + residual:      \(String(format: "%.1f", Double(prefillTailNanos) / 1e6)) ms\n"
                + "  active experts/layer: \(String(format: "%.2f", perLayer))"
                + " (topK=\(cfg.topKExperts), max possible \(t * cfg.topKExperts))\n"
            FileHandle.standardError.write(Data(trace.utf8))
        }

        if writeFinalHead, runEpilogue {
            try encodeFinalHead(
                logits: logits, scratch: scratch,
                tokenCount: t, hiddenSize: D, rmsEps: eps,
                outputMode: outputMode)
        }

        if runEpilogue {
            aneChunk?.finishChunk(
                startPosition: startPosition,
                tokenCount: tokens.count)
            kv?.advance(slot: slot, by: tokens.count)
            prefillChunkState.markCommitted()
        }
    }

    /// One layer's prefill pass over one chunk.
    ///
    /// Extracted verbatim from the layer loop so the loops can be inverted:
    /// layer-major prefill runs this for every chunk of a band before moving
    /// to the next layer, which is what makes each layer's experts stream once
    /// instead of once per chunk. Everything here is per (layer, chunk) except
    /// `scratch.hidden`, which is the residual and therefore the one buffer the
    /// caller must supply per chunk rather than per pass.
    private func runPrefillLayer(
        _ L: Int,
        cb: inout MTLCommandBuffer,
        scratch: PrefillChunkScratchBuffers,
        layerViews: [LayerPrefillQKVViews],
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        t: Int,
        D: Int,
        eps: Float,
        useTwoRowProjection: Bool,
        snapshotGDNAfterFirstToken: Bool,
        aneChunk: ANEPrefillAttention?,
        verifyRoutedMoE: Bool,
        prefillRouteNanos: inout UInt64,
        prefillAttentionNanos: inout UInt64,
        prefillTileNanos: inout UInt64,
        prefillTailNanos: inout UInt64,
        prefillActiveExperts: inout UInt64,
        slot: Int = 0
    ) async throws {
        try Task.checkCancellation()
        let prefillLayerStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        model.beginOpeningRoutedExpertStreamer(layer: L)
        let views = layerViews[L]

        if cfg.ple.layerIndices.contains(L) {
            try encodePLEPrefill(
                commandBuffer: cb,
                hidden: scratch.hidden,
                layer: L, tokens: t, eps: eps)
        }
        try encodeResidualEntryPrefill(
            commandBuffer: cb,
            hidden: scratch.hidden,
            norm: views.inputNorm,
            out: scratch.normed,
            sublayer: .attention, layer: L,
            tokens: t, eps: eps)
        // The indexer caches a key for every prefilled token, in or out
        // of the dense-exact window: decode crossing the boundary later
        // must not find holes behind it.
        let qsaSelection = try encodeQSAPrefill(
            cb: &cb, blockInput: scratch.normed,
            layer: L, startPosition: startPosition,
            tokens: t, eps: eps)
        try await encodePrefillAttentionBranch(
            cb: &cb, layer: L, views: views, scratch: scratch,
            tokenCount: t, hiddenSize: D, startPosition: startPosition, eps: eps,
            useTwoRowProjection: useTwoRowProjection,
            snapshotGDNAfterFirstToken: snapshotGDNAfterFirstToken,
            aneChunk: aneChunk, qsaSelection: qsaSelection, slot: slot)
        // Plain pre-norm residual block: hidden += attention branch,
        // then one post-attention norm feeds router, shared expert,
        // and routed phase 1 (routedX doubles as moeX).
        try encodeResidualExitPrefill(
            commandBuffer: cb,
            hidden: scratch.hidden,
            delta: scratch.h1,
            sublayer: .attention, layer: L,
            tokens: t)
        try encodeResidualEntryPrefill(
            commandBuffer: cb,
            hidden: scratch.hidden,
            norm: views.postAttention,
            out: scratch.routedX,
            sublayer: .mlp, layer: L,
            tokens: t, eps: eps)
        // The phase timer started before the attention block, so the split
        // between attention-plus-norms and the router is taken here.
        prefillAttentionNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - prefillLayerStart
        if verifyRoutedMoE {
            try await encodeRoutedMoEVerifyRows(
                cb: &cb, layer: L, rows: t, views: views, scratch: scratch,
                hiddenSize: D)
        } else {
            try await encodeRoutedMoEPrefill(
                cb: &cb, layer: L, views: views, scratch: scratch,
                tokenCount: t, startPosition: startPosition, hiddenSize: D,
                layerStart: prefillLayerStart,
                routeNanos: &prefillRouteNanos,
                tileNanos: &prefillTileNanos,
                tailNanos: &prefillTailNanos,
                activeExperts: &prefillActiveExperts)
        }
        dumpPrefillLayerOutputs(
            layer: L, scratch: scratch, tokenCount: t, hiddenSize: D,
            startPosition: startPosition)
    }

    /// One layer's attention branch for a prefill chunk: linear attention,
    /// ANE-covered full attention, or the GPU full-attention kernel. Extract of
    /// `runPrefillLayer`; the branch order and arguments are unchanged.
    private func encodePrefillAttentionBranch(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        views: LayerPrefillQKVViews,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        startPosition: Int,
        eps: Float,
        useTwoRowProjection: Bool,
        snapshotGDNAfterFirstToken: Bool,
        aneChunk: ANEPrefillAttention?,
        qsaSelection: QSASelection?,
        slot: Int
    ) async throws {
        let isLinear = cfg.layerIsLinear(L)
        let isFull = cfg.fullAttentionLayerMask[L] == 1
        let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
        let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
        let qDim = cfg.numHeads * headDim
        let kvDim = numKVHeads * headDim
        if isLinear {
            try encodeLinearAttentionPrefill(
                cb: cb, layer: L, views: views, scratch: scratch,
                tokenCount: t, hiddenSize: D,
                snapshotGDNAfterFirstToken: snapshotGDNAfterFirstToken,
                useTwoRowProjection: useTwoRowProjection,
                slot: slot)
        } else if let ane = aneChunk, ane.coveredLayers.contains(L) {
            // The indexer's selection is computed above for this layer whether
            // it runs here or on the GPU; the ANE has to be fed the same
            // choice, or it attends to keys the model drops.
            try await runANEFullAttentionPrefill(
                ane: ane, cb: &cb, layer: L, scratch: scratch,
                tokenCount: t, hiddenSize: D,
                startPosition: startPosition, kvDim: kvDim,
                selection: qsaSelection)
        } else {
            try encodeFullAttentionPrefill(
                cb: cb, layer: L, views: views, scratch: scratch,
                tokenCount: t, hiddenSize: D, startPosition: startPosition,
                isFull: isFull, headDim: headDim, numKVHeads: numKVHeads,
                qDim: qDim, kvDim: kvDim, rmsEps: eps,
                useTwoRowProjection: useTwoRowProjection,
                keepMask: qsaSelection,
                slot: slot)
        }
    }

    /// The activation-dump half of `runPrefillLayer`: every layer's post-MoE
    /// residual when the all-layers switch is on, then this layer's final
    /// hidden. Off by default; extract of the same function.
    private func dumpPrefillLayerOutputs(
        layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        startPosition: Int
    ) {
        // Debug: dump every layer's residual when asked, so a bisect can name
        // the layer where a run leaves the finite range. Off by default.
        if activationDumpActive(position: startPosition),
            ProcessInfo.processInfo
                .environment["TINYTITAN_ACT_DUMP_ALL_LAYERS"] == "1"
        {
            dumpActivationPrivate(
                "L\(L)_post_moe", scratch.hidden,
                count: t * D, position: startPosition)
        }
        // The stage above awaits its own command buffers, so the layer's output
        // hidden is complete here. This is the number a reference dump compares
        // against (`layerN`), which is what localizes a wrong stage to a layer.
        // Every row's residual after every layer, each in its own position's
        // directory, so row i of a multi-row pass sits beside what scalar
        // decode dumps for position startPosition + i (`L{L+1}_entry`). This is
        // the wide-verify identity bisect (docs/plan-wide-verify.md, task 3).
        for row in 0..<t where activationDumpActive(position: startPosition + row) {
            dumpActivationPrivate(
                "L\(L)_rows_after", scratch.hidden,
                count: residualWidth, position: startPosition + row,
                offset: row * residualWidth * MemoryLayout<Float16>.stride)
        }
        if activationDumpActive(position: startPosition), L <= dumpLayerLimit {
            dumpActivationPrivate(
                "L\(L)_after", scratch.hidden,
                count: t * D, position: startPosition)
            flushDeferredDumps()
        }
    }
}
