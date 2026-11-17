import Foundation
import Metal
import TinyTitanCore

/// One dense K2 layer over a prefill chunk.
///
/// The token path (`K2DenseLayer`) runs one position at a time, so every
/// projection reads its weights again. This runs the chunk layer-major: the
/// grouped norm, the four attention projections and the three MLP projections
/// are `AffineQuantGEMM` calls over the whole chunk, and only the shapes that
/// are genuinely per-token — RoPE (positions advance by one) and attention
/// (causal, one query per token) — loop. Elementwise stages are flat
/// dispatches over the chunk's rows.
///
/// Projections whose shape does not fit the register-resident GEMM (very short
/// rows, or a bf16 view) fall back to per-token GEMV, so the same path serves
/// the toy geometry and an install.
final class K2DenseBatch {
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

    private let normed: MTLBuffer
    private let q: MTLBuffer
    private let k: MTLBuffer
    private let v: MTLBuffer
    private let gate: MTLBuffer
    private let attn: MTLBuffer
    private let oOut: MTLBuffer
    private let h1: MTLBuffer
    private let normed2: MTLBuffer
    private let mlpGate: MTLBuffer
    private let mlpUp: MTLBuffer
    private let act: MTLBuffer
    private let mlpOut: MTLBuffer
    /// Internal so a caller can confirm the batch is using the decoder
    /// layer's cache (and the parity test can assert it).
    let kCache: MTLBuffer
    let vCache: MTLBuffer

    /// `kCache`/`vCache` let a caller share the decoder layer's cache: a chunk
    /// prefill must write the rows a later decode step reads. Omitted, the
    /// batch allocates its own (what the parity test uses).
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
        guard config.denseLayerIndices.contains(layer) else {
            throw ModelError.internalInconsistency(
                detail: "layer \(layer) is not a dense K2 layer")
        }
        self.rms = try RMSNorm(context: context)
        self.rope = try RoPE(context: context)
        self.attention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.gemm4 = try AffineQuantGEMM(context: context, weightBits: 4)
        self.gemm8 = try AffineQuantGEMM(context: context, weightBits: 8)
        self.gemv4 = try AffineQuantGEMV(context: context, weightBits: 4)
        self.gemv8 = try AffineQuantGEMV(context: context, weightBits: 8)

        let d = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let intermediate = config.intermediateSize
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
        self.normed = try buffer(maxTokens * d, "k2.batch.normed")
        self.q = try buffer(maxTokens * queryDim, "k2.batch.q")
        self.k = try buffer(maxTokens * kvDim, "k2.batch.k")
        self.v = try buffer(maxTokens * kvDim, "k2.batch.v")
        self.gate = try buffer(maxTokens * queryDim, "k2.batch.gate")
        self.attn = try buffer(maxTokens * queryDim, "k2.batch.attn")
        self.oOut = try buffer(maxTokens * d, "k2.batch.oOut")
        self.h1 = try buffer(maxTokens * d, "k2.batch.h1")
        self.normed2 = try buffer(maxTokens * d, "k2.batch.normed2")
        self.mlpGate = try buffer(maxTokens * intermediate, "k2.batch.mlpGate")
        self.mlpUp = try buffer(maxTokens * intermediate, "k2.batch.mlpUp")
        self.act = try buffer(maxTokens * intermediate, "k2.batch.act")
        self.mlpOut = try buffer(maxTokens * d, "k2.batch.mlpOut")
        if let sharedKCache {
            self.kCache = sharedKCache
        } else {
            self.kCache = try buffer(maxSequence * kvDim, "k2.batch.kCache")
        }
        if let sharedVCache {
            self.vCache = sharedVCache
        } else {
            self.vCache = try buffer(maxSequence * kvDim, "k2.batch.vCache")
        }
    }

    /// Run `tokenCount` consecutive positions starting at `basePosition`.
    /// `hidden` and `out` are `[tokenCount][hidden]`, token-major.
    func encode(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        basePosition: Int,
        tokenCount: Int
    ) throws {
        guard tokenCount > 0, tokenCount <= maxTokens else {
            throw ModelError.internalInconsistency(
                detail: "K2 batch layer got \(tokenCount) tokens, capacity \(maxTokens)")
        }
        guard basePosition >= 0,
            basePosition + tokenCount <= maxSequence
        else {
            throw ModelError.internalInconsistency(
                detail:
                    "K2 batch layer positions \(basePosition)..\(basePosition + tokenCount - 1) exceed its \(maxSequence)-token cache"
            )
        }
        try attentionBlock(
            commandBuffer: commandBuffer,
            hidden: hidden, hiddenOffset: hiddenOffset,
            basePosition: basePosition, tokenCount: tokenCount)
        try mlpBlock(
            commandBuffer: commandBuffer,
            hidden: hidden, hiddenOffset: hiddenOffset,
            out: out, outOffset: outOffset, tokenCount: tokenCount)
    }

    /// Norm, the four attention projections, RoPE, the cache write, causal
    /// attention, the softplus gate, o_proj and the first residual. Leaves the
    /// result in `h1`.
    private func attentionBlock(
        commandBuffer: MTLCommandBuffer,
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
            commandBuffer: commandBuffer,
            x: hidden, xOffset: hiddenOffset,
            weight: inputNorm.buffer, weightOffset: Int(inputNorm.offset),
            out: normed,
            groupDim: UInt32(d / groups), numGroups: groups,
            eps: 1e-6, tokens: tokenCount)

        try projection(
            commandBuffer: commandBuffer, view: try model.qProj(layer: layer),
            x: normed, y: q, rows: queryDim, columns: d, tokens: tokenCount)
        try projection(
            commandBuffer: commandBuffer, view: try model.kProj(layer: layer),
            x: normed, y: k, rows: kvDim, columns: d, tokens: tokenCount)
        try projection(
            commandBuffer: commandBuffer, view: try model.vProj(layer: layer),
            x: normed, y: v, rows: kvDim, columns: d, tokens: tokenCount)
        try projection(
            commandBuffer: commandBuffer,
            view: try model.k2AttentionGate(layer: layer),
            x: normed, y: gate, rows: queryDim, columns: d,
            tokens: tokenCount)

        // Positions advance by one per token, which the scalar-position RoPE
        // cannot express for a chunk: one call per token, in place.
        let rotaryDim = UInt32(
            (Double(config.fullHeadDim) * config.partialRotaryFactor).rounded())
        for token in 0..<tokenCount {
            let position = UInt32(basePosition + token)
            try rope.encodeNeoxSubdim(
                commandBuffer: commandBuffer, data: q,
                dataOffset: token * queryDim * half,
                position: position, headDim: UInt32(config.fullHeadDim),
                numHeads: UInt32(config.numHeads), rotaryDim: rotaryDim,
                theta: Float(config.fullRopeTheta))
            try rope.encodeNeoxSubdim(
                commandBuffer: commandBuffer, data: k,
                dataOffset: token * kvDim * half,
                position: position, headDim: UInt32(config.fullHeadDim),
                numHeads: UInt32(config.numFullKVHeads), rotaryDim: rotaryDim,
                theta: Float(config.fullRopeTheta))
        }
        // The chunk's K/V rows are contiguous and land at consecutive cache
        // positions, so one blit each, not one per token.
        try copy(
            commandBuffer: commandBuffer, from: k, to: kCache,
            toOffset: basePosition * kvDim * half, count: tokenCount * kvDim)
        try copy(
            commandBuffer: commandBuffer, from: v, to: vCache,
            toOffset: basePosition * kvDim * half, count: tokenCount * kvDim)

        for token in 0..<tokenCount {
            try attention.encodeFull(
                commandBuffer: commandBuffer,
                q: q, qOffset: token * queryDim * half,
                k: kCache, v: vCache,
                out: attn, outOffset: token * queryDim * half,
                headDim: UInt32(config.fullHeadDim),
                numQHeads: UInt32(config.numHeads),
                numKVHeads: UInt32(config.numFullKVHeads),
                seqLen: UInt32(basePosition + token + 1),
                scale: Float(config.attentionScale))
        }
        try elementwise.encodeSoftplusGateMul(
            commandBuffer: commandBuffer, out: attn, gate: gate,
            count: tokenCount * queryDim)
        try projection(
            commandBuffer: commandBuffer, view: try model.oProj(layer: layer),
            x: attn, y: oOut, rows: d, columns: queryDim,
            tokens: tokenCount)

        try copy(
            commandBuffer: commandBuffer, from: hidden,
            fromOffset: hiddenOffset, to: h1, count: tokenCount * d)
        try elementwise.encodeResidualAdd(
            commandBuffer: commandBuffer, hidden: h1, delta: oOut,
            count: tokenCount * d)
    }

    /// Post-attention norm, the SwiGLU MLP over the chunk, and the second
    /// residual into `out`.
    private func mlpBlock(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int,
        out: MTLBuffer, outOffset: Int, tokenCount: Int
    ) throws {
        let d = config.hiddenSize
        let intermediate = config.intermediateSize
        let groups = max(1, config.normGroups)

        let postNorm = try model.postAttnNorm(layer: layer)
        try rms.encodeBF16WGrouped(
            commandBuffer: commandBuffer,
            x: h1,
            weight: postNorm.buffer, weightOffset: Int(postNorm.offset),
            out: normed2,
            groupDim: UInt32(d / groups), numGroups: groups,
            eps: 1e-6, tokens: tokenCount)
        try projection(
            commandBuffer: commandBuffer,
            view: try model.k2DenseGate(layer: layer),
            x: normed2, y: mlpGate, rows: intermediate, columns: d,
            tokens: tokenCount)
        try projection(
            commandBuffer: commandBuffer,
            view: try model.k2DenseUp(layer: layer),
            x: normed2, y: mlpUp, rows: intermediate, columns: d,
            tokens: tokenCount)
        try elementwise.encodeSiluMul(
            commandBuffer: commandBuffer, gate: mlpGate, up: mlpUp,
            out: act, count: tokenCount * intermediate)
        try projection(
            commandBuffer: commandBuffer,
            view: try model.k2DenseDown(layer: layer),
            x: act, y: mlpOut, rows: d, columns: intermediate,
            tokens: tokenCount)

        try copy(
            commandBuffer: commandBuffer, from: h1, to: out,
            toOffset: outOffset, count: tokenCount * d)
        try elementwise.encodeResidualAdd(
            commandBuffer: commandBuffer, hidden: out, hiddenOffset: outOffset,
            delta: mlpOut, count: tokenCount * d)
    }

    /// One projection over the chunk. A quantized view whose shape fits the
    /// register-resident GEMM uses it; anything else (a bf16 view, or a row too
    /// short to tile) falls back to the per-token GEMV, which keeps the toy
    /// geometry and any unquantized projection on the same code path.
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
        guard view.dtype != 1 else {
            throw ModelError.internalInconsistency(
                detail:
                    "K2 batch prefill has no bf16 projection path (layer \(layer)); quantized installs only"
            )
        }
        let bits = Int(view.length) * 8 / (rows * columns)
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
        commandBuffer: MTLCommandBuffer,
        from: MTLBuffer, fromOffset: Int = 0,
        to: MTLBuffer, toOffset: Int = 0,
        count: Int
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
