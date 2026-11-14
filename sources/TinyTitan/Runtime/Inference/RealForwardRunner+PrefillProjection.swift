import Foundation
import Metal
import TinyTitanCore

/// Prefill projections and head: the per-layer tensor views, the affine GEMV
/// wrapper, and the final norm + lm_head.
///
/// Split out of `RealForwardRunner+Prefill.swift` (2026-09-28) under the
/// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
extension RealForwardRunner {
    /// Per-layer tensor views resolved once before the chunk loop.
    struct LayerPrefillQKVViews {
        let inputNorm: TensorView
        let postAttention: TensorView
        /// The router, or nil for a family with no routed mixture (dense).
        let router: TensorView?
        // Softmax-attention layers only (nil on linear-attention layers).
        let q: TensorView?
        let k: TensorView?
        let v: TensorView?
        let o: TensorView?
        let qNorm: TensorView?
        let kNorm: TensorView?
        // Gated-DeltaNet linear-attention layers only.
        let linQKV: TensorView?
        let linZ: TensorView?
        let linA: TensorView?
        let linB: TensorView?
        let linOut: TensorView?
        let linConv: TensorView?
        let linALog: TensorView?
        let linDtBias: TensorView?
        let linNorm: TensorView?

        /// The view for a weight this path requires, or a thrown error naming it.
        ///
        /// A softmax-attention layer must carry q/k/v/o and the two norms; the
        /// call sites used to force-unwrap them, so a profile or family
        /// mismatch crashed the process instead of naming the missing weight.
        func require(_ view: TensorView?, _ name: String) throws -> TensorView {
            guard let view else {
                throw ModelError.internalInconsistency(
                    detail: "full-attention prefill requires the \(name) weights, "
                        + "which this layer does not carry")
            }
            return view
        }
    }

    /// `weightBits` is the *role's* width, not the attention slot's: the dense
    /// Qwen 3.5 installs keep k/v at 8 bits with q/o at 4, and the int4-only
    /// batched paths below would read an 8-bit tensor as packed nibbles.
    func encodeAffineProjection(
        commandBuffer: MTLCommandBuffer,
        family: PrefillProjectionFamily,
        weightBits: Int,
        weights: TensorView,
        x: MTLBuffer,
        y: MTLBuffer,
        rows: Int,
        columns: Int,
        tokenCount: Int,
        xStrideElements: Int,
        yStrideElements: Int,
        useTwoRowProjection: Bool
    ) throws {
        // A promoted tensor carries no scales or biases, so none of the
        // batched paths below can read it -- they would take the companions
        // from offset zero, which is the file header. Fall straight to the
        // per-row GEMV, which dispatches on dtype.
        //
        // The batching lost here is cheap: every promoted family is among the
        // smallest tensors in the model, which is why they were chosen.
        if weights.dtype == 1 {
            // One dispatch for the whole chunk when there is more than a row:
            // the per-row loop costs a dispatch per token per projection, and a
            // 1500-token prompt on this family issues ~180k of them. The bytes
            // are unchanged, so this is launch amortisation, not a bandwidth
            // win -- see the shader's note.
            //
            // `TINYTITAN_BF16_PREFILL_SINGLE_ROW=1` forces the per-row path so
            // the two can be compared on one prompt: the arithmetic order is
            // identical, so the text has to be identical too.
            let singleRow =
                ProcessInfo.processInfo.environment["TINYTITAN_BF16_PREFILL_SINGLE_ROW"] == "1"
            if BF16PrefillDispatch.usesMultiRow(
                tokenCount: tokenCount, singleRowForced: singleRow)
            {
                try bf16Projection.encodeMultiRow(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    x: x, y: y,
                    m: UInt32(rows), n: UInt32(columns),
                    tokens: tokenCount,
                    xStrideElements: xStrideElements,
                    yStrideElements: yStrideElements)
                return
            }
            for row in 0..<tokenCount {
                try encodeRoleGEMV(
                    commandBuffer: commandBuffer,
                    projection: weights,
                    weightBits: weightBits,
                    x: x,
                    xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                    y: y,
                    yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                    m: UInt32(rows),
                    n: UInt32(columns))
            }
            return
        }
        if family == .q || family == .kv || family == .o,
            let candidate = prefillMPPAffineInt4
        {
            // K5: the same TensorOpsPolicy seam that gates attention gates
            // these projections. `select` also enforces the MPP kernel's
            // contiguity rules, so a promoted or padded layout keeps the
            // scalar path instead of reading the wrong row.
            let shape = MPPPrefillInt4QMM.Shape(
                tokenCount: tokenCount,
                rows: rows,
                columns: columns,
                weightBits: weightBits,
                xStrideElements: xStrideElements,
                yStrideElements: yStrideElements)
            if candidate.select(shape).selected {
                let path = try candidate.encode(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: tokenCount,
                    n: rows,
                    k: columns,
                    xStrideElements: xStrideElements,
                    yStrideElements: yStrideElements)
                if path == .affineThreadgroupF16 {
                    return
                }
            }
        }
        if useTwoRowProjection && tokenCount == 2
            && xStrideElements == columns && yStrideElements == rows
        {
            if weightBits == 4 {
                try int4.encodeTwoRows(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: UInt32(rows),
                    n: UInt32(columns))
            } else {
                try requireAffine().encodeTwoRows(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: UInt32(rows),
                    n: UInt32(columns))
            }
            return
        }
        if weightBits == model.attentionWeightBits,
            PrefillProjectionDispatchPolicy.selectedDispatch(
                for: family,
                chunkTokens: tokenCount) == .qmm
        {
            try prefillQMM.encode(
                commandBuffer: commandBuffer,
                weights: weights.buffer,
                weightsOffset: Int(weights.offset),
                scales: weights.buffer,
                scalesOffset: Int(weights.scaleOffset),
                biases: weights.buffer,
                biasesOffset: Int(weights.biasOffset),
                x: x,
                y: y,
                t: tokenCount,
                n: rows,
                k: columns)
            return
        }
        for row in 0..<tokenCount {
            try encodeRoleGEMV(
                commandBuffer: commandBuffer,
                projection: weights,
                weightBits: weightBits,
                x: x,
                xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                y: y,
                yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                m: UInt32(rows),
                n: UInt32(columns))
        }
    }

    /// Resolve every layer's tensor views once, before the chunk loop.
    func makeLayerPrefillViews() throws -> [LayerPrefillQKVViews] {
        try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isLinear = cfg.layerIsLinear(L)
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                router: cfg.numExperts == 0 ? nil : try model.router(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: isLinear ? nil : try model.kProj(layer: L),
                v: isLinear
                    ? nil
                    : ((isFull && cfg.attentionKEqV)
                        ? (try model.kProj(layer: L))
                        : (try model.vProj(layer: L))),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: isLinear ? nil : try model.qNorm(layer: L),
                kNorm: isLinear ? nil : try model.kNorm(layer: L),
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: isLinear ? try model.linearInProjZ(layer: L) : nil,
                linA: isLinear ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear ? try model.linearOutProj(layer: L) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear ? try model.linearALog(layer: L) : nil,
                linDtBias: isLinear ? try model.linearDtBias(layer: L) : nil,
                linNorm: isLinear ? try model.linearNorm(layer: L) : nil)
        }
    }

    /// Final norm and lm_head for the last chunk, writing logits or a fused
    /// greedy token depending on the output mode.
    func encodeFinalHead(
        logits: MTLBuffer,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        rmsEps eps: Float,
        outputMode: PrefillOutputMode
    ) throws {
        let finalNorm = try model.finalNorm()
        let lm = try model.lmHead()
        guard let finalCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        if let hc = hyperConnection {
            // The stack ends by collapsing the streams through the
            // model-level mixer, and only the last row feeds the head, so the
            // one-row decode gate serves here. `normed` is free once the last
            // layer has run.
            let rowBytes = D * residualStreamCount * MemoryLayout<Float16>.stride
            try hc.encodeRead(
                commandBuffer: finalCB,
                streamsBuffer: scratch.hidden,
                streamsOffset: (t - 1) * rowBytes,
                hcNorm: finalNorm.buffer,
                hcNormOffset: Int(finalNorm.offset),
                down: gateWeightsPublic(try model.hcMixerDown()),
                up: gateWeightsPublic(try model.hcMixerUp()),
                blockInput: scratch.normed, eps: eps)
            try encodeHeadGEMV(
                commandBuffer: finalCB,
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                x: scratch.normed, y: logits,
                m: UInt32(cfg.vocabSize), n: UInt32(D))
            finalCB.commit()
            try waitForCompletion(finalCB)
            recordKernelGPU(role: "prefill_head", finalCB)
            if activationDumpDirectory != nil {
                dumpActivation(
                    "prefill_logits", logits, count: cfg.vocabSize,
                    position: 0)
            }
            return
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            try fusionHead.encodeGreedyDecode(
                commandBuffer: finalCB,
                hidden: scratch.hidden,
                hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                normWeight: finalNorm.buffer,
                normOffset: Int(finalNorm.offset),
                weights: lm.buffer,
                weightsOffset: Int(lm.offset),
                scales: lm.buffer,
                scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer,
                biasesOffset: Int(lm.biasOffset),
                outToken: greedyTokenBuf,
                d: UInt32(D),
                vocab: UInt32(cfg.vocabSize),
                rmsEps: eps)
        } else {
            try prefillFinalRowHead.encodeLogits(
                commandBuffer: finalCB,
                hiddenBlock: scratch.hidden,
                row: t - 1,
                rowStrideElements: D,
                normWeight: finalNorm.buffer,
                normWeightOffset: Int(finalNorm.offset),
                weights: lm.buffer,
                weightsOffset: Int(lm.offset),
                scales: lm.buffer,
                scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer,
                biasesOffset: Int(lm.biasOffset),
                logits: logits,
                d: UInt32(D),
                vocab: UInt32(cfg.vocabSize),
                rmsEps: eps)
        }
        finalCB.commit()
        try waitForCompletion(finalCB)
        if activationDumpDirectory != nil {
            // The head has run and is complete, so these are the numbers the
            // reference's top-k printout compares against.
            dumpActivation(
                "prefill_logits", logits, count: cfg.vocabSize,
                position: 0)
            dumpActivationPrivate(
                "final_normed", scratch.normed,
                count: D, position: 0)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
        }
    }

}
