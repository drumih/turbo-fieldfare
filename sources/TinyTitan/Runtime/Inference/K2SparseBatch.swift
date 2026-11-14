import Foundation
import Metal
import TinyTitanCore

/// One sparse K2 layer over a prefill chunk.
///
/// The token path (`K2SparseLayerProbe`) runs one position at a time, so the
/// q/k/gate/o projections and the shared expert read their weights again for
/// every token. This batches everything that does not route per token:
///
///   * grouped norm, q/k/gate and o projections, the softplus gate, both
///     residuals and the post-attention norm run over the chunk;
///   * the MoVA router and values stay per token (the router picks four of 64
///     value experts per hidden state) but are encoded into the same command
///     buffer, reading their hidden row by offset;
///   * RoPE and causal attention are per token by nature (positions advance,
///     one query each);
///   * MoE router and shared expert are per token — one `K2MoEStage` per slot
///     so all the router ids survive until the fetch phase. The routed experts
///     are fetched as a group's union and dispatched per token against it.
///
/// The MoE fetch is still the slow part; batching the projections is what this
/// stage can do without changing routing.
final class K2SparseBatch {
    private let model: Model
    private let config: ArchConfig
    private let layer: Int
    private let maxSequence: Int
    let maxTokens: Int

    private let rms: RMSNorm
    private let rope: RoPE
    private let attention: Attention
    private let elementwise: Elementwise
    private let gemm4: AffineQuantGEMM
    private let gemm8: AffineQuantGEMM
    private let gemv4: AffineQuantGEMV
    private let gemv8: AffineQuantGEMV
    private let mova: MoVAValueStage
    private let vExperts: [MoVAValueStage.ExpertWeights]
    private let moeStages: [K2MoEStage]
    private let sharedWeights: K2MoEStage.ExpertWeights
    private let experts: K2SparseExpertDispatch

    private let normed: MTLBuffer
    private let q: MTLBuffer
    private let k: MTLBuffer
    private let gate: MTLBuffer
    private let attn: MTLBuffer
    private let oOut: MTLBuffer
    private let h1: MTLBuffer
    private let normed2: MTLBuffer
    private let vChunk: MTLBuffer
    private let moeOut: MTLBuffer
    private let kCache: MTLBuffer
    private let vCache: MTLBuffer

    init(
        model: Model, context: MetalContext, layer: Int,
        maxSequence: Int, maxTokens: Int,
        kCache sharedKCache: MTLBuffer? = nil,
        vCache sharedVCache: MTLBuffer? = nil
    ) throws {
        precondition(maxSequence > 0 && maxTokens > 0)
        self.model = model
        self.config = model.config
        self.layer = layer
        self.maxSequence = maxSequence
        self.maxTokens = maxTokens
        guard !config.denseLayerIndices.contains(layer) else {
            throw ModelError.internalInconsistency(
                detail: "layer \(layer) is dense, not a K2 sparse layer")
        }
        let cfg = config
        let hidden = cfg.hiddenSize
        let queryDim = cfg.numHeads * cfg.fullHeadDim
        let kvDim = cfg.numFullKVHeads * cfg.fullHeadDim
        let vBits = try Self.weightBits(
            try model.k2VExpert(layer: layer, expert: 0),
            rows: kvDim, columns: hidden)
        self.rms = try RMSNorm(context: context)
        self.rope = try RoPE(context: context)
        self.attention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.gemm4 = try AffineQuantGEMM(context: context, weightBits: 4)
        self.gemm8 = try AffineQuantGEMM(context: context, weightBits: 8)
        self.gemv4 = try AffineQuantGEMV(context: context, weightBits: 4)
        self.gemv8 = try AffineQuantGEMV(context: context, weightBits: 8)
        self.mova = try MoVAValueStage(
            context: context, weightBits: vBits,
            hiddenSize: hidden, valueDim: kvDim,
            maxExperts: cfg.mova.numExperts)
        self.vExperts = try (0..<cfg.mova.numExperts).map { expert in
            let view = try model.k2VExpert(layer: layer, expert: expert)
            return MoVAValueStage.ExpertWeights(
                weights: view.buffer, weightsOffset: Int(view.offset),
                scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                biases: view.buffer, biasesOffset: Int(view.biasOffset))
        }
        let routedEntry = try model.packedExpertsLayout.expert(
            layer: layer, expert: 0)
        let routedBits = routedEntry.subTensors["gate"]?.bits ?? 4
        self.moeStages = try (0..<maxTokens).map { _ in
            try K2MoEStage(
                context: context, weightBits: routedBits,
                hiddenSize: hidden, expertWidth: cfg.moeIntermediateSize,
                numExperts: cfg.numExperts, topK: cfg.topKExperts)
        }
        self.sharedWeights = K2MoEStage.ExpertWeights(
            gate: try model.k2SharedGate(layer: layer),
            up: try model.k2SharedUp(layer: layer),
            down: try model.k2SharedDown(layer: layer))

        func buffer(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: max(count * MemoryLayout<UInt16>.stride, 2),
                    options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        self.normed = try buffer(maxTokens * hidden, "k2.sparseBatch.normed")
        self.q = try buffer(maxTokens * queryDim, "k2.sparseBatch.q")
        self.k = try buffer(maxTokens * kvDim, "k2.sparseBatch.k")
        self.gate = try buffer(maxTokens * queryDim, "k2.sparseBatch.gate")
        self.attn = try buffer(maxTokens * queryDim, "k2.sparseBatch.attn")
        self.oOut = try buffer(maxTokens * hidden, "k2.sparseBatch.oOut")
        self.h1 = try buffer(maxTokens * hidden, "k2.sparseBatch.h1")
        self.normed2 = try buffer(maxTokens * hidden, "k2.sparseBatch.normed2")
        self.vChunk = try buffer(maxTokens * kvDim, "k2.sparseBatch.v")
        self.moeOut = try buffer(maxTokens * hidden, "k2.sparseBatch.moeOut")
        if let sharedKCache {
            self.kCache = sharedKCache
        } else {
            self.kCache = try buffer(maxSequence * kvDim, "k2.sparseBatch.kCache")
        }
        if let sharedVCache {
            self.vCache = sharedVCache
        } else {
            self.vCache = try buffer(maxSequence * kvDim, "k2.sparseBatch.vCache")
        }
        self.experts = try K2SparseExpertDispatch(
            model: model, layer: layer, hiddenSize: hidden,
            stages: moeStages, normed2: normed2, moeOut: moeOut)
    }

    private static func weightBits(
        _ view: TensorView, rows: Int, columns: Int
    ) throws -> Int {
        let bits = Int(view.length) * 8 / (rows * columns)
        guard bits == 4 || bits == 8 else {
            throw ModelError.internalInconsistency(
                detail: "K2 value expert is \(bits)-bit; expected 4 or 8")
        }
        return bits
    }

    /// Run `tokenCount` consecutive positions starting at `basePosition`.
    func encode(
        queue: MTLCommandQueue,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        basePosition: Int, tokenCount: Int
    ) async throws {
        guard tokenCount > 0, tokenCount <= maxTokens,
            basePosition >= 0, basePosition + tokenCount <= maxSequence
        else {
            throw ModelError.internalInconsistency(
                detail:
                    "K2 sparse batch shape \(tokenCount)@\(basePosition) exceeds \(maxTokens)/\(maxSequence)"
            )
        }
        try await projectionPhase(
            queue: queue, hidden: hidden, hiddenOffset: hiddenOffset,
            basePosition: basePosition, tokenCount: tokenCount)
        try await experts.encode(queue: queue, tokenCount: tokenCount)
        try await residualPhase(
            queue: queue, out: out, outOffset: outOffset,
            tokenCount: tokenCount)
    }

    /// Everything up to and including each slot's MoE router and shared
    /// expert, in one command buffer.
    private func projectionPhase(
        queue: MTLCommandQueue,
        hidden: MTLBuffer, hiddenOffset: Int,
        basePosition: Int, tokenCount: Int
    ) async throws {
        guard let cb = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try encodeAttentionPhase(
            cb, hidden: hidden, hiddenOffset: hiddenOffset,
            basePosition: basePosition, tokenCount: tokenCount)
        try encodeRouters(cb, tokenCount: tokenCount)
        cb.commit()
        await cb.completed()
        if let error = cb.error {
            throw ModelError.internalInconsistency(
                detail: "K2 sparse batch projection phase failed: \(error)")
        }
    }

    /// The norm and q/k/gate projections over the chunk, then per token the
    /// MoVA values, RoPE, the cache write and cached attention; the gate,
    /// o_proj, both residuals and the post-attention norm close the phase.
    private func encodeAttentionPhase(
        _ cb: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int,
        basePosition: Int, tokenCount: Int
    ) throws {
        let d = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let groups = max(1, config.normGroups)
        let half = MemoryLayout<UInt16>.stride
        let inputNorm = try model.inputNorm(layer: layer)
        try rms.encodeBF16WGrouped(
            commandBuffer: cb,
            x: hidden, xOffset: hiddenOffset,
            weight: inputNorm.buffer, weightOffset: Int(inputNorm.offset),
            out: normed,
            groupDim: UInt32(d / groups), numGroups: groups,
            eps: 1e-6, tokens: tokenCount)
        try projection(
            commandBuffer: cb, view: try model.qProj(layer: layer),
            x: normed, y: q, rows: queryDim, columns: d, tokens: tokenCount)
        try projection(
            commandBuffer: cb, view: try model.kProj(layer: layer),
            x: normed, y: k, rows: kvDim, columns: d, tokens: tokenCount)
        try projection(
            commandBuffer: cb, view: try model.k2AttentionGate(layer: layer),
            x: normed, y: gate, rows: queryDim, columns: d,
            tokens: tokenCount)
        try encodeTokens(
            cb, basePosition: basePosition, tokenCount: tokenCount,
            queryDim: queryDim, kvDim: kvDim, half: half)
        try elementwise.encodeSoftplusGateMul(
            commandBuffer: cb, out: attn, gate: gate,
            count: tokenCount * queryDim)
        try projection(
            commandBuffer: cb, view: try model.oProj(layer: layer),
            x: attn, y: oOut, rows: d, columns: queryDim,
            tokens: tokenCount)
        try copy(
            cb, from: hidden, fromOffset: hiddenOffset,
            to: h1, count: tokenCount * d)
        try elementwise.encodeResidualAdd(
            commandBuffer: cb, hidden: h1, delta: oOut,
            count: tokenCount * d)
        let postNorm = try model.postAttnNorm(layer: layer)
        try rms.encodeBF16WGrouped(
            commandBuffer: cb,
            x: h1,
            weight: postNorm.buffer, weightOffset: Int(postNorm.offset),
            out: normed2,
            groupDim: UInt32(d / groups), numGroups: groups,
            eps: 1e-6, tokens: tokenCount)
    }

    /// The per-token part of the attention phase: MoVA router and values,
    /// RoPE at that token's position, the K/V write, cached attention.
    private func encodeTokens(
        _ cb: MTLCommandBuffer,
        basePosition: Int, tokenCount: Int,
        queryDim: Int, kvDim: Int, half: Int
    ) throws {
        let d = config.hiddenSize
        let rotaryDim = UInt32(
            (Double(config.fullHeadDim) * config.partialRotaryFactor).rounded())
        let vRouter = try model.k2VRouter(layer: layer)
        let vRouterBias = try model.k2VRouterBias(layer: layer)
        for token in 0..<tokenCount {
            let position = UInt32(basePosition + token)
            try mova.encodeRouter(
                commandBuffer: cb, hidden: normed,
                hiddenOffset: token * d * half,
                routerWeight: vRouter.buffer,
                routerWeightOffset: Int(vRouter.offset),
                routerBias: vRouterBias.buffer,
                routerBiasOffset: Int(vRouterBias.offset),
                numExperts: config.mova.numExperts,
                scaling: Float(config.mova.routerScaling))
            try mova.encodeRoutedValues(
                commandBuffer: cb, hidden: normed,
                hiddenOffset: token * d * half, experts: vExperts,
                out: vChunk, outOffset: token * kvDim * half)
            try rope.encodeNeoxSubdim(
                commandBuffer: cb, data: q,
                dataOffset: token * queryDim * half,
                position: position, headDim: UInt32(config.fullHeadDim),
                numHeads: UInt32(config.numHeads), rotaryDim: rotaryDim,
                theta: Float(config.fullRopeTheta))
            try rope.encodeNeoxSubdim(
                commandBuffer: cb, data: k,
                dataOffset: token * kvDim * half,
                position: position, headDim: UInt32(config.fullHeadDim),
                numHeads: UInt32(config.numFullKVHeads), rotaryDim: rotaryDim,
                theta: Float(config.fullRopeTheta))
            try copy(
                cb, from: k, fromOffset: token * kvDim * half,
                to: kCache, toOffset: (basePosition + token) * kvDim * half,
                count: kvDim)
            try copy(
                cb, from: vChunk, fromOffset: token * kvDim * half,
                to: vCache, toOffset: (basePosition + token) * kvDim * half,
                count: kvDim)
            try attention.encodeFull(
                commandBuffer: cb,
                q: q, qOffset: token * queryDim * half,
                k: kCache, v: vCache,
                out: attn, outOffset: token * queryDim * half,
                headDim: UInt32(config.fullHeadDim),
                numQHeads: UInt32(config.numHeads),
                numKVHeads: UInt32(config.numFullKVHeads),
                seqLen: UInt32(basePosition + token + 1),
                scale: Float(config.attentionScale))
        }
    }

    /// The MoE router and shared expert, per token, into the same buffer; the
    /// routed pass is added later, after its experts are streamed.
    private func encodeRouters(
        _ cb: MTLCommandBuffer, tokenCount: Int
    ) throws {
        let d = config.hiddenSize
        let half = MemoryLayout<UInt16>.stride
        let moeRouter = try model.k2MoeRouter(layer: layer)
        let moeBias = try model.k2MoeRouterBias(layer: layer)
        for token in 0..<tokenCount {
            let stage = moeStages[token]
            try stage.encodeRouter(
                commandBuffer: cb, hidden: normed2,
                hiddenOffset: token * d * half,
                routerWeight: moeRouter.buffer,
                routerWeightOffset: Int(moeRouter.offset),
                routerBias: moeBias.buffer,
                routerBiasOffset: Int(moeBias.offset),
                scaling: Float(config.routerScaling))
            try stage.encodeSharedExpert(
                commandBuffer: cb, hidden: normed2,
                hiddenOffset: token * d * half, shared: sharedWeights,
                out: moeOut, outOffset: token * d * half)
        }
    }

    /// out = h1 + (shared + routed), over the chunk.
    private func residualPhase(
        queue: MTLCommandQueue,
        out: MTLBuffer, outOffset: Int, tokenCount: Int
    ) async throws {
        let d = config.hiddenSize
        guard let cb = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try copy(
            cb, from: h1, to: out, toOffset: outOffset, count: tokenCount * d)
        try elementwise.encodeResidualAdd(
            commandBuffer: cb, hidden: out, hiddenOffset: outOffset,
            delta: moeOut, count: tokenCount * d)
        cb.commit()
        await cb.completed()
        if let error = cb.error {
            throw ModelError.internalInconsistency(
                detail: "K2 sparse batch residual phase failed: \(error)")
        }
    }

    /// One projection over the chunk, with the GEMM when the shape tiles and
    /// the per-token GEMV otherwise (the toy geometry is too short).
    private func projection(
        commandBuffer: MTLCommandBuffer,
        view: TensorView,
        x: MTLBuffer, y: MTLBuffer,
        rows: Int, columns: Int, tokens: Int
    ) throws {
        let half = MemoryLayout<UInt16>.stride
        if view.dtype != 1 {
            let bits = Int(view.length) * 8 / (rows * columns)
            if bits == 4 || bits == 8,
                AffineQuantGEMM.canEncode(
                    rows: rows, columns: columns, tokens: tokens,
                    weightBits: bits)
            {
                let gemm = bits == 8 ? gemm8 : gemm4
                try gemm.encode(
                    commandBuffer: commandBuffer,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: x, y: y, rows: UInt32(rows), columns: UInt32(columns),
                    tokens: UInt32(tokens))
                return
            }
        }
        let bits = Int(view.length) * 8 / (rows * columns)
        guard bits == 4 || bits == 8 else {
            throw ModelError.internalInconsistency(
                detail: "K2 sparse batch needs quantized projections")
        }
        let gemv = bits == 8 ? gemv8 : gemv4
        for token in 0..<tokens {
            try gemv.encode(
                commandBuffer: commandBuffer,
                weights: view.buffer, weightsOffset: Int(view.offset),
                scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                biases: view.buffer, biasesOffset: Int(view.biasOffset),
                x: x, xOffset: token * columns * half,
                y: y, yOffset: token * rows * half,
                m: UInt32(rows), n: UInt32(columns))
        }
    }

    private func copy(
        _ commandBuffer: MTLCommandBuffer,
        from: MTLBuffer, fromOffset: Int = 0,
        to: MTLBuffer, toOffset: Int = 0, count: Int
    ) throws {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(
            from: from, sourceOffset: fromOffset,
            to: to, destinationOffset: toOffset,
            size: count * MemoryLayout<UInt16>.stride)
        blit.endEncoding()
    }
}
