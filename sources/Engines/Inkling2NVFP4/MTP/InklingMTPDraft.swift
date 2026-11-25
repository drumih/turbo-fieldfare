import Foundation
import Metal
import TinyTitanCore

/// Inkling 2 DSpark MTP draft engine (TT-129): stores target-feature context
/// K/V from `[5, 23, 41, 48]`, runs the 3-layer bidirectional block transformer
/// over `[anchor, mask, ..., mask]` (block size 8), and runs the 7-step
/// `rnn_hidden` Markov head to propose 7 draft tokens per verify pass.
final class InklingMTPDraft {
    typealias Operand = GGUFGemvIds.Operand

    struct MixRMSArgs {
        var taps: UInt32
        var hidden: UInt32
        var epsilon: Float
    }
    struct RMSBF16Args {
        var width: UInt32
        var hasResidual: UInt32
        var epsilon: Float
    }
    struct QKNormRoPEArgs {
        var heads: UInt32
        var headDim: UInt32
        var basePosition: UInt32
        var hasNorm: UInt32
        var epsilon: Float
        var ropeTheta: Float
    }
    struct StoreKVArgs {
        var kvWidth: UInt32
        var basePosition: UInt32
        var capacity: UInt32
        var ring: UInt32
    }
    struct BlockAttnArgs {
        var numHeads: UInt32
        var numKVHeads: UInt32
        var headDim: UInt32
        var ctxLength: UInt32
        var blockTokens: UInt32
        var capacity: UInt32
        var ring: UInt32
        var window: UInt32
        var scale: Float
    }
    struct ConvSnapshotArgs {
        var channels: UInt32
        var slots: UInt32
        var tokens: UInt32
        var snapStrideFloats: UInt32
        var channelOffset: UInt32
    }

    let weights: InklingMTPWeights
    let state: InklingMTPState
    let targetWeights: InklingQ4Weights
    let manifest: InklingQ4Manifest
    let ops: InklingQ4Ops
    private let context: MetalContext
    private let pipelines: [String: MTLComputePipelineState]

    init(
        weights: InklingMTPWeights, state: InklingMTPState,
        targetWeights: InklingQ4Weights, manifest: InklingQ4Manifest,
        ops: InklingQ4Ops, context: MetalContext
    ) throws {
        self.weights = weights
        self.state = state
        self.targetWeights = targetWeights
        self.manifest = manifest
        self.ops = ops
        self.context = context
        var built: [String: MTLComputePipelineState] = [:]
        for name in [
            "draft_mix_rmsnorm", "draft_rms_norm_bf16", "draft_qk_norm_rope",
            "draft_store_kv_fp16", "draft_block_attention", "draft_silu_mul",
            "draft_copy_bf16_broadcast", "draft_markov_gate", "draft_add_softcap",
            "draft_conv_snapshots", "draft_capture_tap", "draft_add_bf16_bias",
            "draft_copy_f32",
        ] {
            built[name] = try context.pipeline(name)
        }
        self.pipelines = built
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let p = pipelines[name] else { throw MetalError.missingFunction(name) }
        return p
    }

    func reset() { state.reset() }

    /// Mixes captured target features for `count` committed tokens starting at
    /// `startPosition` and stores their projected K/V into each draft layer's
    /// context cache.
    func storeContextKV(startPosition: Int, count: Int) throws {
        guard count > 0 else { return }
        guard let cmd = context.queue.makeCommandBuffer(),
            let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { throw MetalError.commandEncoderFailed }
        let cfg = weights.config
        let taps = cfg.targetLayerIDs.count
        for layer in weights.layers {
            var mixArgs = MixRMSArgs(
                taps: UInt32(taps), hidden: UInt32(cfg.hiddenSize), epsilon: cfg.rmsEpsilon)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try pipeline("draft_mix_rmsnorm"))
            enc.setBuffer(state.targetTaps, offset: 0, index: 0)
            enc.setBuffer(weights.fuseAlpha, offset: layer.index * taps * 4, index: 1)
            enc.setBuffer(
                layer.dflareNorm.operand.buffer, offset: layer.dflareNorm.operand.offset, index: 2)
            enc.setBuffer(state.ctxMixed, offset: 0, index: 3)
            enc.setBytes(&mixArgs, length: MemoryLayout<MixRMSArgs>.stride, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            try ops.gemv(
                layer.kProjCtx, x: .init(state.ctxMixed), y: .init(state.ctxK), tokens: count,
                encoder: enc)
            try ops.gemv(
                layer.vProjCtx, x: .init(state.ctxMixed), y: .init(state.ctxV), tokens: count,
                encoder: enc)

            var ropeArgs = QKNormRoPEArgs(
                heads: UInt32(cfg.numKVHeads), headDim: UInt32(cfg.headDim),
                basePosition: UInt32(startPosition), hasNorm: 1,
                epsilon: cfg.rmsEpsilon, ropeTheta: cfg.ropeTheta)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try pipeline("draft_qk_norm_rope"))
            enc.setBuffer(state.ctxK, offset: 0, index: 0)
            enc.setBuffer(
                layer.kNormCtx.operand.buffer, offset: layer.kNormCtx.operand.offset, index: 1)
            enc.setBytes(&ropeArgs, length: MemoryLayout<QKNormRoPEArgs>.stride, index: 2)
            enc.dispatchThreadgroups(
                MTLSize(width: count * cfg.numKVHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: cfg.headDim / 2, height: 1, depth: 1))

            var storeArgs = StoreKVArgs(
                kvWidth: UInt32(cfg.kvWidth), basePosition: UInt32(startPosition),
                capacity: UInt32(state.capacities[layer.index]),
                ring: layer.isLocal ? 1 : 0)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try pipeline("draft_store_kv_fp16"))
            enc.setBuffer(state.ctxK, offset: 0, index: 0)
            enc.setBuffer(state.ctxV, offset: 0, index: 1)
            enc.setBuffer(state.kCaches[layer.index], offset: 0, index: 2)
            enc.setBuffer(state.vCaches[layer.index], offset: 0, index: 3)
            enc.setBytes(&storeArgs, length: MemoryLayout<StoreKVArgs>.stride, index: 4)
            enc.dispatchThreads(
                MTLSize(width: cfg.kvWidth, height: count, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, cfg.kvWidth), height: 1, depth: 1))
        }
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error { throw err }
    }

    /// Runs the 3-layer DSpark draft backbone on `[anchorToken, mask, ...]` at
    /// `position` (with `position` context tokens already in the draft KV cache)
    /// and the 7-step `rnn_hidden` Markov head, returning `blockSize - 1` (7)
    /// greedy draft token IDs.
    func draftBlock(
        anchorToken: Int, position: Int, lastTapRow: Int
    ) throws -> [Int] {
        try runBackbone(anchorToken: anchorToken, position: position, lastTapRow: lastTapRow)
        return try runMarkovChain(anchorToken: anchorToken)
    }

    private func runBackbone(anchorToken: Int, position: Int, lastTapRow: Int) throws {
        state.anchorToken.contents().storeBytes(of: Int32(anchorToken), as: Int32.self)
        guard let cmd = context.queue.makeCommandBuffer(),
            let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { throw MetalError.commandEncoderFailed }
        let cfg = weights.config
        let b = cfg.blockSize
        let h = cfg.hiddenSize
        let taps = cfg.targetLayerIDs.count

        // 1. Compute f_rnn from the last committed token's captured target taps.
        var mixArgs = MixRMSArgs(taps: UInt32(taps), hidden: UInt32(h), epsilon: cfg.rmsEpsilon)
        enc.setComputePipelineState(try pipeline("draft_mix_rmsnorm"))
        enc.setBuffer(state.targetTaps, offset: lastTapRow * taps * h * 4, index: 0)
        enc.setBuffer(weights.fuseAlpha, offset: cfg.numLayers * taps * 4, index: 1)
        enc.setBuffer(
            weights.markovDflareNorm.operand.buffer,
            offset: weights.markovDflareNorm.operand.offset, index: 2)
        enc.setBuffer(state.fRnn, offset: 0, index: 3)
        enc.setBytes(&mixArgs, length: MemoryLayout<MixRMSArgs>.stride, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        // 2. Embed block: row 0 = embed_norm(tokenEmbedding[anchor]), rows 1..<B = mask_embed.
        try ops.embed(
            targetWeights.tokenEmbedding, token: .init(state.anchorToken),
            out: .init(state.blockX), tokens: 1, encoder: enc)
        try ops.rmsNorm(
            .init(state.blockX), weight: targetWeights.tokenNorm.operand,
            out: .init(state.blockResidual), width: h, count: 1,
            epsilon: manifest.rmsEpsilon, encoder: enc)
        var widthU32 = UInt32(h)
        var rowOffsetU32: UInt32 = 1
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(try pipeline("draft_copy_bf16_broadcast"))
        enc.setBuffer(
            weights.maskEmbed.operand.buffer, offset: weights.maskEmbed.operand.offset, index: 0)
        enc.setBuffer(state.blockResidual, offset: 0, index: 1)
        enc.setBytes(&widthU32, length: 4, index: 2)
        enc.setBytes(&rowOffsetU32, length: 4, index: 3)
        enc.dispatchThreads(
            MTLSize(width: h, height: b - 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        // 3. Run the 3 dense BF16 transformer layers.
        for layer in weights.layers {
            try encodeDraftLayer(layer, position: position, enc: enc)
        }

        // 4. Final norm: residual += blockMLPOut; headHidden = rmsnorm(residual) * finalNorm.
        try encodeRMSBF16(
            x: state.blockMLPOut, residual: state.blockResidual, weight: weights.finalNorm,
            out: state.headHidden, rows: b, hasResidual: true, enc: enc)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error { throw err }
    }

    private func encodeDraftLayer(
        _ layer: InklingMTPLayerWeights, position: Int, enc: MTLComputeCommandEncoder
    ) throws {
        let cfg = weights.config
        let b = cfg.blockSize
        // Pre-attention norm (layer 0 reads blockResidual directly; layers 1+ fuse residual += prev MLP)
        try encodeRMSBF16(
            x: layer.index == 0 ? state.blockResidual : state.blockMLPOut,
            residual: state.blockResidual, weight: layer.inputNorm, out: state.blockNormed,
            rows: b, hasResidual: layer.index > 0, enc: enc)
        try ops.gemv(
            layer.qProj, x: .init(state.blockNormed), y: .init(state.blockQ), tokens: b,
            encoder: enc)
        try ops.gemv(
            layer.kProj, x: .init(state.blockNormed), y: .init(state.blockK), tokens: b,
            encoder: enc)
        try ops.gemv(
            layer.vProj, x: .init(state.blockNormed), y: .init(state.blockV), tokens: b,
            encoder: enc)

        // Per-head Q/K norm + NeoX RoPE
        try encodeQKNormRoPE(
            vec: state.blockQ, weight: layer.qNorm, heads: cfg.numHeads,
            basePosition: position, tokens: b, enc: enc)
        try encodeQKNormRoPE(
            vec: state.blockK, weight: layer.kNorm, heads: cfg.numKVHeads,
            basePosition: position, tokens: b, enc: enc)

        // Bidirectional block attention over [context KV, block KV]
        var attnArgs = BlockAttnArgs(
            numHeads: UInt32(cfg.numHeads), numKVHeads: UInt32(cfg.numKVHeads),
            headDim: UInt32(cfg.headDim), ctxLength: UInt32(position),
            blockTokens: UInt32(b), capacity: UInt32(state.capacities[layer.index]),
            ring: layer.isLocal ? 1 : 0,
            window: layer.isLocal ? UInt32(cfg.slidingWindow) : 0,
            scale: 1.0 / Float(cfg.headDim).squareRoot())
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(try pipeline("draft_block_attention"))
        enc.setBuffer(state.blockQ, offset: 0, index: 0)
        enc.setBuffer(state.kCaches[layer.index], offset: 0, index: 1)
        enc.setBuffer(state.vCaches[layer.index], offset: 0, index: 2)
        enc.setBuffer(state.blockK, offset: 0, index: 3)
        enc.setBuffer(state.blockV, offset: 0, index: 4)
        enc.setBuffer(state.blockAttn, offset: 0, index: 5)
        enc.setBytes(&attnArgs, length: MemoryLayout<BlockAttnArgs>.stride, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: b * cfg.numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: cfg.headDim, height: 1, depth: 1))

        try ops.gemv(
            layer.oProj, x: .init(state.blockAttn), y: .init(state.blockProj), tokens: b,
            encoder: enc)
        try encodeRMSBF16(
            x: state.blockProj, residual: state.blockResidual, weight: layer.postAttnNorm,
            out: state.blockNormed, rows: b, hasResidual: true, enc: enc)
        try ops.gemv(
            layer.gateProj, x: .init(state.blockNormed), y: .init(state.blockGate), tokens: b,
            encoder: enc)
        try ops.gemv(
            layer.upProj, x: .init(state.blockNormed), y: .init(state.blockUp), tokens: b,
            encoder: enc)

        var siluCount = UInt32(b * cfg.intermediateSize)
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(try pipeline("draft_silu_mul"))
        enc.setBuffer(state.blockGate, offset: 0, index: 0)
        enc.setBuffer(state.blockUp, offset: 0, index: 1)
        enc.setBuffer(state.blockAct, offset: 0, index: 2)
        enc.setBytes(&siluCount, length: 4, index: 3)
        enc.dispatchThreads(
            MTLSize(width: Int(siluCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        try ops.gemv(
            layer.downProj, x: .init(state.blockAct), y: .init(state.blockMLPOut), tokens: b,
            encoder: enc)
    }

    private func runMarkovChain(anchorToken: Int) throws -> [Int] {
        let cfg = weights.config
        let h = cfg.hiddenSize
        let r = cfg.markovRank
        let nSpec = cfg.numSpeculativeTokens
        memset(state.rnnStateA.contents(), 0, r * 4)
        var prev = anchorToken
        var drafted: [Int] = []
        drafted.reserveCapacity(nSpec)

        let headPtr = state.headHidden.contents().bindMemory(
            to: Float.self, capacity: cfg.blockSize * h)
        let fRnnPtr = state.fRnn.contents().bindMemory(to: Float.self, capacity: h)
        let w1Ptr = (weights.markovW1.operand.buffer.contents() + weights.markovW1.operand.offset)
            .bindMemory(to: UInt16.self, capacity: weights.markovW1.rows * r)
        let rnnInPtr = state.rnnInput.contents().bindMemory(to: Float.self, capacity: 2 * r + 2 * h)

        // rnnInput layout matches vLLM Inkling2MarkovHead: [state(r), markov_w1[prev](r), hidden(h), f_rnn(h)]
        memcpy(rnnInPtr + 2 * r + h, fRnnPtr, h * 4)

        for step in 0..<nSpec {
            let curStateBuf = (step % 2 == 0) ? state.rnnStateA : state.rnnStateB
            let nextStateBuf = (step % 2 == 0) ? state.rnnStateB : state.rnnStateA
            let statePtr = curStateBuf.contents().bindMemory(to: Float.self, capacity: r)
            memcpy(rnnInPtr, statePtr, r * 4)
            let clampedPrev = max(0, min(prev, weights.markovW1.rows - 1))
            let rowStart = clampedPrev * r
            for c in 0..<r {
                rnnInPtr[r + c] = Float(bitPattern: UInt32(w1Ptr[rowStart + c]) << 16)
            }
            // Rows 1..<blockSize hold the nSpec draft positions (row 0 is anchor).
            let stepRowBase = (step + 1) * h
            memcpy(rnnInPtr + 2 * r, headPtr + stepRowBase, h * 4)

            guard let cmd = context.queue.makeCommandBuffer(),
                let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
            else { throw MetalError.commandEncoderFailed }

            try ops.gemv(
                weights.rnnProjWeight, x: .init(state.rnnInput), y: .init(state.rnnRaw), tokens: 1,
                encoder: enc)
            var rawWidth = UInt32(3 * r)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try pipeline("draft_add_bf16_bias"))
            enc.setBuffer(state.rnnRaw, offset: 0, index: 0)
            enc.setBuffer(
                weights.rnnProjBias.operand.buffer, offset: weights.rnnProjBias.operand.offset,
                index: 1)
            enc.setBytes(&rawWidth, length: 4, index: 2)
            enc.dispatchThreads(
                MTLSize(width: 3 * r, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            var rankU32 = UInt32(r)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try pipeline("draft_markov_gate"))
            enc.setBuffer(state.rnnRaw, offset: 0, index: 0)
            enc.setBuffer(curStateBuf, offset: 0, index: 1)
            enc.setBuffer(nextStateBuf, offset: 0, index: 2)
            enc.setBuffer(state.rnnLatent, offset: 0, index: 3)
            enc.setBytes(&rankU32, length: 4, index: 4)
            enc.dispatchThreads(
                MTLSize(width: r, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, r), height: 1, depth: 1))

            try ops.gemv(
                weights.hiddenOutWeight, x: .init(state.rnnLatent), y: .init(state.hiddenDelta),
                tokens: 1, encoder: enc)
            var hU32 = UInt32(h)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try pipeline("draft_add_bf16_bias"))
            enc.setBuffer(state.hiddenDelta, offset: 0, index: 0)
            enc.setBuffer(
                weights.hiddenOutBias.operand.buffer, offset: weights.hiddenOutBias.operand.offset,
                index: 1)
            enc.setBytes(&hU32, length: 4, index: 2)
            enc.dispatchThreads(
                MTLSize(width: h, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            if let err = cmd.error { throw err }

            let deltaPtr = state.hiddenDelta.contents().bindMemory(to: Float.self, capacity: h)
            let stepPtr = state.stepHidden.contents().bindMemory(to: Float.self, capacity: h)
            for c in 0..<h { stepPtr[c] = headPtr[stepRowBase + c] + deltaPtr[c] }

            guard let cmd2 = context.queue.makeCommandBuffer(),
                let enc2 = cmd2.makeComputeCommandEncoder(dispatchType: .concurrent)
            else { throw MetalError.commandEncoderFailed }
            try ops.gemv(
                targetWeights.output, x: .init(state.stepHidden), y: .init(state.draftLogits),
                tokens: 1, encoder: enc2)
            try ops.argmax(
                logits: .init(state.draftLogits), count: manifest.vocabulary,
                partials: .init(state.argmaxPartials), groupCount: InklingQ4State.argmaxGroups,
                token: .init(state.draftNextToken), encoder: enc2)
            enc2.endEncoding()
            cmd2.commit()
            cmd2.waitUntilCompleted()
            if let err = cmd2.error { throw err }

            prev = Int(state.draftNextToken.contents().load(as: Int32.self))
            drafted.append(prev)
        }
        return drafted
    }

    private func encodeRMSBF16(
        x: MTLBuffer, residual: MTLBuffer, weight: InklingQ4Tensor, out: MTLBuffer,
        rows: Int, hasResidual: Bool, enc: MTLComputeCommandEncoder
    ) throws {
        var args = RMSBF16Args(
            width: UInt32(weights.config.hiddenSize),
            hasResidual: hasResidual ? 1 : 0,
            epsilon: weights.config.rmsEpsilon)
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(try pipeline("draft_rms_norm_bf16"))
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(residual, offset: 0, index: 1)
        enc.setBuffer(weight.operand.buffer, offset: weight.operand.offset, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.setBytes(&args, length: MemoryLayout<RMSBF16Args>.stride, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func encodeQKNormRoPE(
        vec: MTLBuffer, weight: InklingQ4Tensor, heads: Int,
        basePosition: Int, tokens: Int, enc: MTLComputeCommandEncoder
    ) throws {
        let cfg = weights.config
        var args = QKNormRoPEArgs(
            heads: UInt32(heads), headDim: UInt32(cfg.headDim),
            basePosition: UInt32(basePosition), hasNorm: 1,
            epsilon: cfg.rmsEpsilon, ropeTheta: cfg.ropeTheta)
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(try pipeline("draft_qk_norm_rope"))
        enc.setBuffer(vec, offset: 0, index: 0)
        enc.setBuffer(weight.operand.buffer, offset: weight.operand.offset, index: 1)
        enc.setBytes(&args, length: MemoryLayout<QKNormRoPEArgs>.stride, index: 2)
        enc.dispatchThreadgroups(
            MTLSize(width: tokens * heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: cfg.headDim / 2, height: 1, depth: 1))
    }

    func encodeCaptureTap(
        src: Operand, width: Int, taps: Int, tapIndex: Int, tokens: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        var widthU32 = UInt32(width)
        var tapsU32 = UInt32(taps)
        var tapIdxU32 = UInt32(tapIndex)
        encoder.memoryBarrier(scope: .buffers)
        encoder.setComputePipelineState(try pipeline("draft_capture_tap"))
        encoder.setBuffer(src.buffer, offset: src.offset, index: 0)
        encoder.setBuffer(state.targetTaps, offset: 0, index: 1)
        encoder.setBytes(&widthU32, length: 4, index: 2)
        encoder.setBytes(&tapsU32, length: 4, index: 3)
        encoder.setBytes(&tapIdxU32, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: width, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, width), height: 1, depth: 1))
    }

    func encodeConvSnapshot(
        x: Operand, convState: Operand, routeWeights: Operand?, visit: Int,
        channels: Int, slots: Int, tokens: Int, channelOffset: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        var args = ConvSnapshotArgs(
            channels: UInt32(channels), slots: UInt32(slots), tokens: UInt32(tokens),
            snapStrideFloats: UInt32(state.convSnapshotStrideFloats),
            channelOffset: UInt32(channelOffset))
        encoder.memoryBarrier(scope: .buffers)
        encoder.setComputePipelineState(try pipeline("draft_conv_snapshots"))
        encoder.setBuffer(x.buffer, offset: x.offset, index: 0)
        encoder.setBuffer(convState.buffer, offset: convState.offset, index: 1)
        let rw = routeWeights ?? convState
        encoder.setBuffer(rw.buffer, offset: rw.offset, index: 2)
        encoder.setBuffer(state.convSnapshots[visit], offset: 0, index: 3)
        encoder.setBytes(&args, length: MemoryLayout<ConvSnapshotArgs>.stride, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: channels, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, channels), height: 1, depth: 1))
    }
}
