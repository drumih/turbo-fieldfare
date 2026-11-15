import Foundation
import Metal
import TinyTitanCore

// The per-layer decode loop of `produceToken`. Split out of
// RealForwardRunner+Decode.swift (2026-09-28) under the 500-line-per-file rule as
// pure code motion: the loop body is unchanged, and the caller passes in the two
// locals and the deferred routed command the loop reads and returns.

extension RealForwardRunner {
    /// lint:allow-long the per-layer decode loop is one straight-line pass over the
    /// layers; every statement in it belongs to the layer it describes and splitting it
    /// means threading the layer's own locals through a new signature.
    func decodeLayerLoop(
        position: Int,
        slot: Int,
        into logits: MTLBuffer,
        emitHead: Bool,
        outputMode: PrefillOutputMode,
        D: UInt32,
        eps: Float,
        pendingRoutedCommand initialPending: PendingRoutedCommand?
    ) async throws -> PendingRoutedCommand? {
        var pendingRoutedCommand = initialPending
        for L in 0..<cfg.numLayers {
            // Dumping drains the previous layer's routed command first. The
            // residual is only settled once that has landed, and a dump taken
            // at encode time would read whatever the buffer held before the
            // GPU ran -- which reads exactly like a wrong answer.
            if activationDumpActive(position: position) {
                if let pending = pendingRoutedCommand {
                    try finishPendingRoutedCommand(pending, waitIfNeeded: true)
                    pendingRoutedCommand = nil
                    if L <= dumpLayerLimit {
                        dumpActivation(
                            "L\(L - 1)_mlp_out", h2Buf,
                            count: cfg.hiddenSize, position: position)
                        dumpActivation(
                            "L\(L - 1)_moe_acts", moeActs,
                            count: cfg.topKExperts
                                * cfg.moeIntermediateSize,
                            position: position)
                    }
                }
                dumpActivation("L\(L)_entry", hidden, count: residualWidth, position: position)
            }
            let tBodyStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let isLinear = cfg.layerIsLinear(L)
            // A dense model has no routed mixture at all: its FFN is the
            // shared-expert stage (whose projections the dense schema points at
            // `mlp.*`), so there is no router to read, no expert to fetch and
            // nothing to classify. Every MoE-only binding and stage below is
            // skipped rather than fed a zero-expert placeholder -- the router
            // role resolves to a name that cannot exist for this family, and
            // reading it would fail the layer.
            let denseFFN = cfg.numExperts == 0

            let inNorm = try model.inputNorm(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let nextRouterW: TensorView?
            if !denseFFN, nextLayerPredictionEnabled, L + 1 < cfg.numLayers {
                nextRouterW = try model.router(layer: L + 1)
            } else {
                nextRouterW = nil
            }
            let next2RouterW: TensorView? =
                (!denseFFN && Self.probe2TraceEnabled && L + 2 < cfg.numLayers)
                ? try model.router(layer: L + 2) : nil
            let residencyResources =
                (!denseFFN && decodeExpertExecution == .gpuResidency)
                ? try model.routedExpertResidency(layer: L) : nil
            let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
                (try requireOnesPerExpertScale(), 0)

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Attention+router split into measured sub-command-buffers
            // (TINYTITAN_KERNEL_STATS): attnCB = input norm + QKV + epilogue
            // (or the linear/gated attention), softmaxCB = the softmax
            // attention pass on full layers, tailCB = O-proj + residual +
            // post-norm + router. Same queue, same order, one wait on the
            // last CB; only the router readback forces the barrier.
            guard var attnCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            // The PLE block rewrites the wide residual before the attention
            // read gate sees it, so it is encoded ahead of the entry.
            try encodePLEDecode(
                commandBuffer: attnCB, layer: L,
                position: position, eps: eps)
            // Full-attention layers of a sparse-attention family run their
            // entry ahead of the rest, because the indexer's key cache and
            // its selection both hang off the block input.
            //
            // That entry can run on its own command buffer, committed before
            // this one, so a PLE block encoded here would rewrite the
            // residual *after* the entry had already read it. The two never
            // coincide in this architecture -- the n-gram layer is a linear
            // one -- and the guard is here so that stays a fact rather than
            // an assumption.
            precondition(
                !(qsaIndexer != nil
                    && cfg.fullAttentionLayerMask[L] == 1
                    && cfg.ple.layerIndices.contains(L)),
                "layer \(L) is both a PLE layer and a sparse-attention "
                    + "layer; the residual entry would be reordered "
                    + "around the n-gram block")
            var selection: DecodeKeySelection?
            if qsaIndexer != nil, cfg.fullAttentionLayerMask[L] == 1 {
                selection = try encodeQSAEntryAndSelect(
                    passthrough: &attnCB,
                    hidden: hidden, norm: inNorm, out: normed,
                    layer: L, position: position, eps: eps)
            } else {
                try encodeResidualEntryDecode(
                    commandBuffer: attnCB,
                    hidden: hidden, norm: inNorm,
                    out: normed, sublayer: .attention,
                    layer: L, eps: eps)
                try rotate(&attnCB, role: "glue.entry_attn")
            }
            var softmaxCB: MTLCommandBuffer?
            guard var tailCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }

            try encodeDecodeAttention(
                attnCB: &attnCB, tailCB: tailCB,
                softmaxCB: &softmaxCB,
                layer: L, position: position,
                slot: slot,
                isLinear: isLinear, rmsEps: eps,
                selection: selection)
            try encodeResidualExitDecode(
                commandBuffer: tailCB,
                hidden: hidden, delta: oOut,
                sublayer: .attention, layer: L)
            try rotate(&tailCB, role: "glue.exit_attn")
            try encodeResidualEntryDecode(
                commandBuffer: tailCB,
                hidden: hidden, norm: postAttn,
                out: routedX, sublayer: .mlp,
                layer: L, eps: eps)
            try rotate(&tailCB, role: "glue.entry_mlp")

            if !denseFFN {
                let routerW = try model.router(layer: L)
                try moe.encodeRouter(
                    commandBuffer: tailCB,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                    biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                    hidden: routedX,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: perExpertScale.buffer,
                    perExpertScaleOffset: perExpertScale.offset,
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
                try rotate(&tailCB, role: "glue.router")
                if let nextRouterW {
                    // Probe only: score the next router against the current
                    // post-attention normalized residual. The exact router above
                    // remains authoritative; this result never selects experts for
                    // this layer. It drives TINYTITAN_PREFETCH_TRACE and, when
                    // TINYTITAN_PREDICTIVE_PREFETCH is set, the speculative ring.
                    try moe.encodeRouter(
                        commandBuffer: tailCB,
                        weights: nextRouterW.buffer, weightsOffset: Int(nextRouterW.offset),
                        scales: nextRouterW.buffer, scalesOffset: Int(nextRouterW.scaleOffset),
                        biases: nextRouterW.buffer, biasesOffset: Int(nextRouterW.biasOffset),
                        hidden: routedX,
                        effectiveScale: effectiveScaleBuffers[L + 1],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: perExpertScale.offset,
                        outIndices: prefetchPredictionIndices,
                        outWeights: prefetchPredictionWeights,
                        numExperts: UInt32(cfg.numExperts), d: D,
                        topK: UInt32(cfg.topKExperts))
                }
                if let next2RouterW {
                    // Trace-only: layer L+2's router on layer L's residual.
                    try moe.encodeRouter(
                        commandBuffer: tailCB,
                        weights: next2RouterW.buffer, weightsOffset: Int(next2RouterW.offset),
                        scales: next2RouterW.buffer, scalesOffset: Int(next2RouterW.scaleOffset),
                        biases: next2RouterW.buffer, biasesOffset: Int(next2RouterW.biasOffset),
                        hidden: routedX,
                        effectiveScale: effectiveScaleBuffers[L + 2],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: perExpertScale.offset,
                        outIndices: prefetchPrediction2Indices,
                        outWeights: prefetchPrediction2Weights,
                        numExperts: UInt32(cfg.numExperts), d: D,
                        topK: UInt32(cfg.topKExperts))
                }
                if let residencyResources {
                    try moe.encodeResidencyClassification(
                        commandBuffer: tailCB,
                        topKIndices: outIndices,
                        residencyTable: residencyResources.table,
                        hitCount: residencyHitCount,
                        hitPositions: residencyHitPositions,
                        missCount: residencyMissCount,
                        missPositions: residencyMissPositions,
                        missExperts: residencyMissExperts,
                        resolvedSlots: residencyResolvedSlots,
                        resolvedGenerations: residencyResolvedGenerations,
                        topK: UInt32(cfg.topKExperts),
                        numExperts: UInt32(cfg.numExperts))
                }
            }
            attnCB.commit()
            if let attentionCB = softmaxCB {
                attentionCB.commit()
            }
            tailCB.commit()
            // Queued before the wait below, not after: the GPU runs the shared
            // MLP while the CPU blocks on tailCB for the routing.
            let overlapCompletionClock = runnerStatsEnabled ? CommandCompletionClock() : nil
            let sharedCB = try encodeAndCommitSharedExpert(
                layer: L,
                completionClock: overlapCompletionClock)
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try waitForCompletion(tailCB)
            // After the tail wait every attention-side buffer is settled and
            // nothing overwrites them until the next layer, so the dumps cost
            // no extra synchronization.
            if activationDumpActive(position: position) && L <= dumpLayerLimit {
                if let ple = pleBlock, cfg.ple.layerIndices.contains(L) {
                    let wide = residualWidth
                    dumpActivation("ple_key", ple.keyBuf, count: wide, position: position)
                    dumpActivation(
                        "ple_value", ple.valueBuf, count: cfg.hiddenSize, position: position)
                    dumpActivation("ple_key_normed", ple.keyNormed, count: wide, position: position)
                    dumpActivation("ple_query", ple.queryNormed, count: wide, position: position)
                    dumpActivation(
                        "ple_score", ple.scoreBuf,
                        count: cfg.hyperConnections.count, position: position)
                    dumpActivation(
                        "ple_gate", ple.gateBuf,
                        count: cfg.hyperConnections.count, position: position)
                    dumpActivation("ple_gated", ple.gatedBuf, count: wide, position: position)
                    dumpActivation("ple_conv", ple.convOut, count: wide, position: position)
                }
                dumpActivationPrivate(
                    "L\(L)_attn_in", normed, count: cfg.hiddenSize, position: position)
                dumpActivation("L\(L)_attn_out", oOut, count: cfg.hiddenSize, position: position)
                dumpActivation("L\(L)_mlp_in", routedX, count: cfg.hiddenSize, position: position)
                dumpActivation(
                    "L\(L)_hidden_post_attn", hidden,
                    count: residualWidth, position: position)
            }
            recordKernelGPU(role: "attn_norm_qkv", attnCB)
            // Split-mode kernels were committed ahead of attnCB on the same
            // queue, so they completed before the tail wait above.
            for (role, cb) in splitTimedBuffers { recordKernelGPU(role: role, cb) }
            splitTimedBuffers.removeAll(keepingCapacity: true)
            if let attentionCB = softmaxCB {
                recordKernelGPU(role: "attn_softmax", attentionCB)
            }
            recordKernelGPU(role: "attn_tail_router", tailCB)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            totalWaitNanos &+= waitNanos
            var prevRoutedUs: Double = 0
            if let pending = pendingRoutedCommand {
                prevRoutedUs = (pending.cb.gpuEndTime - pending.cb.gpuStartTime) * 1_000_000
                try finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos
            let predictedNextLayer: [Int]
            var predictedNextLayerWeights: [Float] = []
            if nextLayerPredictionEnabled, L + 1 < cfg.numLayers {
                let ptr = prefetchPredictionIndices.contents().bindMemory(
                    to: UInt32.self, capacity: cfg.topKExperts)
                predictedNextLayer = (0..<cfg.topKExperts).map {
                    min(Int(ptr[$0]), cfg.numExperts - 1)
                }
                if prefetchTraceFD >= 0 {
                    // The probe's routing weights, for the trace only: the
                    // prefetch gate that used them (TINYTITAN_PREFETCH_MIN_MARGIN)
                    // measured -4.2% and is gone. Width follows the buffer:
                    // fp32 at 4 bytes per entry, fp16 otherwise.
                    let k = cfg.topKExperts
                    if prefetchPredictionWeights.length >= k * MemoryLayout<Float>.stride {
                        let w = prefetchPredictionWeights.contents().bindMemory(
                            to: Float.self, capacity: k)
                        predictedNextLayerWeights = (0..<k).map { w[$0] }
                    } else {
                        let w = prefetchPredictionWeights.contents().bindMemory(
                            to: Float16.self, capacity: k)
                        predictedNextLayerWeights = (0..<k).map { Float(w[$0]) }
                    }
                }
            } else {
                predictedNextLayer = []
            }
            let predictedNext2Layer: [Int]
            if Self.probe2TraceEnabled, L + 2 < cfg.numLayers {
                let ptr = prefetchPrediction2Indices.contents().bindMemory(
                    to: UInt32.self, capacity: cfg.topKExperts)
                predictedNext2Layer = (0..<cfg.topKExperts).map {
                    min(Int(ptr[$0]), cfg.numExperts - 1)
                }
            } else {
                predictedNext2Layer = []
            }
            self.lastPredictedNext2Layer = predictedNext2Layer

            // CPU readback to fetch routed-expert blobs from disk. The expert
            // id list is reused host scratch (R16); the runner is single-flight
            // per generation, so it never aliases concurrent decode work.
            if denseFFN {
                // The shared-expert stage *is* this family's FFN: its output is
                // the layer's MLP contribution, added to the residual exactly
                // as the routed stage's phase-2 reduce adds a mixture's. The
                // buffer is submitted after `sharedCB` on the same queue, so the
                // read of `h1Buf` is ordered behind its write.
                guard let denseCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try encodeResidualExitDecode(
                    commandBuffer: denseCB,
                    hidden: hidden, delta: h1Buf,
                    sublayer: .mlp, layer: L)
                recordKernelGPU(role: "dense_ffn_exit", denseCB)
                denseCB.commit()
            } else {
                try await encodeDecodeRoutedMoE(
                    layer: L, position: position, sharedProj: sharedProj,
                    attnCB: attnCB, tailCB: tailCB,
                    sharedCB: sharedCB,
                    overlapCompletionClock: overlapCompletionClock,
                    pending: &pendingRoutedCommand,
                    bodyStart: tBodyStart, cb1Start: tCb1Start,
                    waitMark: tWait, waitNanos: waitNanos,
                    previousRoutedMicros: prevRoutedUs,
                    predictedNextLayer: predictedNextLayer,
                    predictedNextLayerWeights: predictedNextLayerWeights)
            }
        }
        return pendingRoutedCommand
    }
}
