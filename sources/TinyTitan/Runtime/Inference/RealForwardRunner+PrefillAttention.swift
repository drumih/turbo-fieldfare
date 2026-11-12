import Foundation
import Metal

/// Chunked-prefill attention encoding, split out of `RealForwardRunner+Prefill.swift`.
///
/// The Gated-DeltaNet branch and the full-attention branch of one chunk, each a
/// single ordered pipeline over shared scratch. Pure code motion: the encoder
/// bodies are unchanged, moved so the chunk executor and the KV/head stages each
/// have their own file.

extension RealForwardRunner {
    /// Gated-DeltaNet (linear attention) branch of one chunked-prefill layer.
    ///
    /// lint:allow-long one layer's linear-attention pipeline is a single
    /// ordered sequence -- in-projection, causal conv, QK norm, delta step,
    /// gated norm, out-projection -- sharing scratch buffers at every step.
    /// Splitting it would thread a dozen buffers through sub-functions to make
    /// a line count smaller while making the data flow harder to follow.
    func encodeLinearAttentionPrefill(
        cb: MTLCommandBuffer, layer L: Int,
        views: LayerPrefillQKVViews, scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int,
        snapshotGDNAfterFirstToken: Bool, useTwoRowProjection: Bool,
        slot: Int = 0
    ) throws {
        // Gated-DeltaNet linear attention over the chunk: batched
        // projections, causal conv (+ tail carry), delta-rule
        // recurrence, gated norm, out_proj. No KV writes, no
        // attention, no blit.
        if usesDecodeOrderLinearAttention(snapshot: snapshotGDNAfterFirstToken) {
            try encodeLinearAttentionDecodeOrderRows(
                cb: cb, layer: L, scratch: scratch, tokenCount: t,
                hiddenSize: D, slot: slot)
            return
        }
        guard let gdn, let gdnState else {
            throw ModelError.internalInconsistency(
                detail:
                    "linear-attention layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        // `LayerPrefillQKVViews` fills the linear_attn slots exactly
        // when `layerIsLinear(L)`, so these are provably non-nil; the
        // guard turns a future arch/view regression into a thrown
        // error instead of a force-unwrap trap.
        guard let linQKV = views.linQKV,
            let linZ = views.linZ,
            let linA = views.linA,
            let linB = views.linB,
            let linConv = views.linConv,
            let linALog = views.linALog,
            let linDtBias = views.linDtBias,
            let linNorm = views.linNorm,
            let linOut = views.linOut
        else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(L) is missing a required linear_attn tensor view")
        }
        let la = cfg.linearAttention
        try encodeAffineProjection(
            commandBuffer: cb,
            family: .q,
            weightBits: model.gdnProjectionWeightBits,
            weights: linQKV,
            x: scratch.normed,
            y: scratch.q,
            rows: la.qkvDim,
            columns: D,
            tokenCount: t,
            xStrideElements: D,
            yStrideElements: la.qkvDim,
            useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(
            commandBuffer: cb,
            family: .kv,
            weightBits: model.gdnProjectionWeightBits,
            weights: linZ,
            x: scratch.normed,
            y: scratch.gdnZ,
            rows: la.valueDim,
            columns: D,
            tokenCount: t,
            xStrideElements: D,
            yStrideElements: la.valueDim,
            useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(
            commandBuffer: cb,
            family: .kv,
            weightBits: model.gdnProjectionWeightBits,
            weights: linA,
            x: scratch.normed,
            y: scratch.gdnA,
            rows: la.numVHeads,
            columns: D,
            tokenCount: t,
            xStrideElements: D,
            yStrideElements: la.numVHeads,
            useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(
            commandBuffer: cb,
            family: .kv,
            weightBits: model.gdnProjectionWeightBits,
            weights: linB,
            x: scratch.normed,
            y: scratch.gdnB,
            rows: la.numVHeads,
            columns: D,
            tokenCount: t,
            xStrideElements: D,
            yStrideElements: la.numVHeads,
            useTwoRowProjection: useTwoRowProjection)
        let convW = linConv
        let tail = gdnState.convTailSlot(layer: L, slot: slot)
        try gdn.encodeConvPrefill(
            commandBuffer: cb,
            tail: tail.buffer, tailOffset: tail.offset,
            qkvRows: scratch.q,
            convWeight: convW.buffer,
            convWeightOffset: Int(convW.offset),
            out: scratch.gdnConvOut,
            rows: t)
        if snapshotGDNAfterFirstToken {
            try gdn.encodeConvTailCheckpoint(
                commandBuffer: cb,
                tail: tail.buffer, tailOffset: tail.offset,
                qkvRows: scratch.q,
                checkpoint: gdnState.speculativeConvTailBuffer(layer: L))
        }
        try gdn.encodeConvTailUpdate(
            commandBuffer: cb,
            tail: tail.buffer, tailOffset: tail.offset,
            qkvRows: scratch.q,
            rows: t)
        try gdn.encodeQKNorm(
            commandBuffer: cb,
            convOut: scratch.gdnConvOut,
            rows: t)
        let aLog = linALog
        let dtBias = linDtBias
        let state = gdnState.stateSlot(layer: L, slot: slot)
        try gdn.encodeDeltaStepPrefill(
            commandBuffer: cb,
            convOut: scratch.gdnConvOut,
            aProj: scratch.gdnA,
            bProj: scratch.gdnB,
            aLog: aLog.buffer,
            aLogOffset: Int(aLog.offset),
            dtBias: dtBias.buffer,
            dtBiasOffset: Int(dtBias.offset),
            state: state.buffer, stateOffset: state.offset,
            checkpointState: snapshotGDNAfterFirstToken
                ? gdnState.speculativeStateBuffer(layer: L) : nil,
            y: scratch.gdnY,
            rows: t)
        if let window = wideVerifyRetain(layer: L) {
            try window.encodeRetain(
                commandBuffer: cb, layer: L,
                qkv: .init(scratch.q), a: .init(scratch.gdnA), b: .init(scratch.gdnB),
                rows: t)
        }
        let gatedNormW = linNorm
        try gdn.encodeGatedNorm(
            commandBuffer: cb,
            y: scratch.gdnY,
            z: scratch.gdnZ,
            weight: gatedNormW.buffer,
            weightOffset: Int(gatedNormW.offset),
            out: scratch.attentionOutput,
            rows: t)
        try encodeAffineProjection(
            commandBuffer: cb,
            family: .o,
            weightBits: model.gdnProjectionWeightBits,
            weights: linOut,
            x: scratch.attentionOutput,
            y: scratch.h1,
            rows: D,
            columns: la.valueDim,
            tokenCount: t,
            xStrideElements: la.valueDim,
            yStrideElements: D,
            useTwoRowProjection: useTwoRowProjection)
    }

    /// Softmax-attention branch of one chunked-prefill layer.
    ///
    /// lint:allow-long same shape as the linear branch: QKV projection, RoPE,
    /// KV-cache write and attention are one ordered pipeline over shared
    /// scratch, and the intermediate buffers have no meaning outside it.
    func encodeFullAttentionPrefill(
        cb: MTLCommandBuffer, layer L: Int,
        views: LayerPrefillQKVViews, scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int, startPosition: Int,
        isFull: Bool, headDim: Int, numKVHeads: Int,
        qDim: Int, kvDim: Int, rmsEps eps: Float,
        useTwoRowProjection: Bool,
        keepMask: QSASelection? = nil,
        slot: Int = 0
    ) throws {
        if usesDecodeOrderFullAttention(layer: L, keepMask: keepMask) {
            try encodeFullAttentionDecodeOrderRows(
                cb: cb, layer: L, scratch: scratch, tokenCount: t,
                hiddenSize: D, startPosition: startPosition, slot: slot)
            return
        }
        let qProjRows = cfg.attnOutputGate ? 2 * qDim : qDim
        try encodeAffineProjection(
            commandBuffer: cb,
            family: .q,
            weightBits: model.qoProjectionWeightBits,
            weights: try views.require(views.q, "q"),
            x: scratch.normed,
            y: scratch.q,
            rows: qProjRows,
            columns: D,
            tokenCount: t,
            xStrideElements: D,
            yStrideElements: qProjRows,
            useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(
            commandBuffer: cb,
            family: .kv,
            weightBits: model.kvProjectionWeightBits,
            weights: try views.require(views.k, "k"),
            x: scratch.normed,
            y: scratch.kStage,
            rows: kvDim,
            columns: D,
            tokenCount: t,
            xStrideElements: D,
            yStrideElements: kvDim,
            useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(
            commandBuffer: cb,
            family: .kv,
            weightBits: model.kvProjectionWeightBits,
            weights: try views.require(views.v, "v"),
            x: scratch.normed,
            y: scratch.vStage,
            rows: kvDim,
            columns: D,
            tokenCount: t,
            xStrideElements: D,
            yStrideElements: kvDim,
            useTwoRowProjection: useTwoRowProjection)
        if activationDumpActive(position: startPosition) {
            dumpActivationDeferred(
                "L\(L)_in", scratch.normed,
                count: t * D, position: startPosition)
            dumpActivationDeferred(
                "L\(L)_qpacked", scratch.q,
                count: t * qProjRows, position: startPosition)
            dumpActivationDeferred(
                "L\(L)_kraw", scratch.kStage,
                count: t * kvDim, position: startPosition)
            dumpActivationDeferred(
                "L\(L)_vraw", scratch.vStage,
                count: t * kvDim, position: startPosition)
        }

        // The attention input Q: the packed q_proj output is split
        // into per-head query/gate halves for gated architectures.
        let attnQ: MTLBuffer
        if cfg.attnOutputGate {
            try requireElementwise().encodeSplitQGate(
                commandBuffer: cb,
                packed: scratch.q,
                q: scratch.attnQ,
                gate: scratch.attnGate,
                heads: cfg.numHeads,
                dim: headDim,
                rows: t)
            attnQ = scratch.attnQ
        } else {
            attnQ = scratch.q
        }

        if cfg.ropeNeoxSubdim {
            let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
            try prefillQKVEpilogue.encodeNeoxSubdimNoVNorm(
                commandBuffer: cb,
                q: attnQ,
                k: scratch.kStage,
                qWeight: try views.require(views.qNorm, "qNorm").buffer,
                qWeightOffset: Int(try views.require(views.qNorm, "qNorm").offset),
                kWeight: try views.require(views.kNorm, "kNorm").buffer,
                kWeightOffset: Int(try views.require(views.kNorm, "kNorm").offset),
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVHeads),
                qTokenStrideElements: UInt32(qDim),
                kvTokenStrideElements: UInt32(kvDim),
                theta: Float(cfg.fullRopeTheta),
                rotaryDim: rotaryDim,
                eps: eps)
        } else {
            let rotatedPairs =
                isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDim / 2)
            try prefillQKVEpilogue.encode(
                commandBuffer: cb,
                q: attnQ,
                k: scratch.kStage,
                v: scratch.vStage,
                qWeight: try views.require(views.qNorm, "qNorm").buffer,
                qWeightOffset: Int(try views.require(views.qNorm, "qNorm").offset),
                kWeight: try views.require(views.kNorm, "kNorm").buffer,
                kWeightOffset: Int(try views.require(views.kNorm, "kNorm").offset),
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVHeads),
                qTokenStrideElements: UInt32(qDim),
                kvTokenStrideElements: UInt32(kvDim),
                theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                rotatedPairs: rotatedPairs,
                eps: eps)
        }

        if let kv {
            let bytes = t * kvDim * MemoryLayout<Float16>.stride
            try copyPrefillKVToCache(
                commandBuffer: cb,
                kv: kv,
                layer: L,
                startPosition: startPosition,
                tokenCount: t,
                slot: slot,
                keySource: scratch.kStage,
                valueSource: scratch.vStage,
                bytesPerToken: bytes / t)
        }
        let kvView = kv?.keyView(
            layer: L, slot: slot,
            validTokenCount: startPosition + t)
        let params = PrefillAttentionParams(
            startPosition: UInt32(startPosition),
            queryCount: UInt32(t),
            headDim: UInt32(headDim),
            numQHeads: UInt32(cfg.numHeads),
            numKVHeads: UInt32(numKVHeads),
            kvValidCount: UInt32(startPosition + t),
            slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
            kvTokenStrideElements: UInt32(kvDim),
            qTokenStrideElements: UInt32(qDim),
            oTokenStrideElements: UInt32(qDim),
            scale: Float(cfg.attentionScale),
            kvBits: UInt32(kvView?.precision.rawValue ?? 16),
            kvTokenStrideBytes: UInt32(kvView?.stride ?? (kvDim * 2)),
            kvValueBytes: UInt32(kvView?.valueBytes ?? (kvDim * 2)),
            kvGroupSize: UInt32(
                kvView?.groupSize
                    ?? KVCacheManager.quantizationGroupSize))
        if let kv {
            let keyView = kv.keyView(
                layer: L, slot: slot,
                validTokenCount: startPosition + t)
            let valueView = kv.valueView(
                layer: L, slot: slot,
                validTokenCount: startPosition + t)
            let ringCapacity = kv.ringCapacity(layer: L)
            let activeRingCapacity =
                ringCapacity > 0 && startPosition + t > ringCapacity
                ? UInt32(ringCapacity)
                : 0
            if kv.streamsLayer(L) {
                // The chunk's KV went to staging; gather the union of every
                // query row's selection into the window and let the kernel
                // read through the row indirection.
                let prepared = try kv.preparePrefillWindow(
                    layer: L, slot: slot,
                    startPosition: startPosition, tokenCount: t,
                    selection: keepMask,
                    commandBuffer: cb)
                try prefillAttention.encodeCausal(
                    commandBuffer: cb,
                    q: attnQ,
                    k: keyView.buffer, kOffset: keyView.offset,
                    v: valueView.buffer, vOffset: valueView.offset,
                    out: scratch.attentionOutput,
                    params: params,
                    kvRingCapacity: activeRingCapacity,
                    keepMask: prepared.mask,
                    keepStride: 0,
                    keepIndices: prepared.indices,
                    keepIndexStride: prepared.indexStride,
                    keepCounts: prepared.counts,
                    keepRows: prepared.rows,
                    path: .causalTiled)
            } else {
                try prefillAttention.encodeCausal(
                    commandBuffer: cb,
                    q: attnQ,
                    k: keyView.buffer, kOffset: keyView.offset,
                    v: valueView.buffer, vOffset: valueView.offset,
                    out: scratch.attentionOutput,
                    params: params,
                    kvRingCapacity: activeRingCapacity,
                    keepMask: keepMask?.mask,
                    keepStride: keepMask?.maskStride ?? 0,
                    keepIndices: keepMask?.indices,
                    keepIndexStride: keepMask?.indexStride ?? 0,
                    keepCounts: keepMask?.counts,
                    path: keepMask == nil
                        ? prefillAttentionPath
                        // Only the tiled kernel honours a selection.
                        : .causalTiled)
            }
        } else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill attention requires a KV cache")
        }
        if cfg.attnOutputGate {
            try requireElementwise().encodeSigmoidGateMul(
                commandBuffer: cb,
                out: scratch.attentionOutput,
                gate: scratch.attnGate,
                count: t * qDim)
        }
        try encodeAffineProjection(
            commandBuffer: cb,
            family: .o,
            weightBits: model.qoProjectionWeightBits,
            weights: try views.require(views.o, "o"),
            x: scratch.attentionOutput,
            y: scratch.h1,
            rows: D,
            columns: qDim,
            tokenCount: t,
            xStrideElements: qDim,
            yStrideElements: D,
            useTwoRowProjection: useTwoRowProjection)
    }
}
