import Foundation
import Metal
import TinyTitanCore

/// One K2-Horizon dense layer on a decode step: grouped norm, GQA with the
/// separate softplus gate, RoPE, then the SwiGLU MLP and both residuals.
///
/// It owns a plain FP16 K/V cache for one sequence (`maxSequence` positions).
/// That is the probe-sized form of what the serving graph will use: enough to
/// check a multi-token decode against a reference, without the runner's
/// paged/quantized cache manager.
final class K2DenseLayer {
    private let config: ArchConfig
    private let model: Model
    private let rms: RMSNorm
    private let attention: Attention
    private let rope: RoPE
    private let elementwise: Elementwise
    private let gemv4: AffineQuantGEMV
    private let gemv8: AffineQuantGEMV
    private let bf16: BF16GEMV

    private let normed: MTLBuffer
    private let q: MTLBuffer
    private let k: MTLBuffer
    private let v: MTLBuffer
    private let gate: MTLBuffer
    /// The plain FP16 K/V cache, shared with the chunked path (`K2DenseBatch`)
    /// so a prefill chunk and a decode step read and write the same rows. A
    /// batch that allocates its own cache would silently diverge from the
    /// layer's decode state.
    let kCache: MTLBuffer
    let vCache: MTLBuffer
    private let maxSequence: Int
    private let attn: MTLBuffer
    private let oOut: MTLBuffer
    private let h1: MTLBuffer
    private let normed2: MTLBuffer
    private let mlpGate: MTLBuffer
    private let mlpUp: MTLBuffer
    private let act: MTLBuffer
    private let mlpOut: MTLBuffer

    let layer: Int

    init(
        model: Model, context: MetalContext, layer: Int,
        maxSequence: Int = 1
    ) throws {
        precondition(maxSequence > 0, "a layer needs at least one KV position")
        self.model = model
        self.config = model.config
        self.layer = layer
        self.maxSequence = maxSequence
        guard config.denseLayerIndices.contains(layer) else {
            throw ModelError.internalInconsistency(
                detail: "layer \(layer) is not a dense K2 layer")
        }
        self.rms = try RMSNorm(context: context)
        self.attention = try Attention(context: context)
        self.rope = try RoPE(context: context)
        self.elementwise = try Elementwise(context: context)
        self.gemv4 = try AffineQuantGEMV(context: context, weightBits: 4)
        self.gemv8 = try AffineQuantGEMV(context: context, weightBits: 8)
        self.bf16 = try BF16GEMV(context: context)

        let hidden = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let ffn = config.intermediateSize
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
        self.normed = try buffer(hidden, "k2.normed")
        self.q = try buffer(queryDim, "k2.q")
        self.k = try buffer(kvDim, "k2.k")
        self.v = try buffer(kvDim, "k2.v")
        self.kCache = try buffer(maxSequence * kvDim, "k2.kCache")
        self.vCache = try buffer(maxSequence * kvDim, "k2.vCache")
        self.gate = try buffer(queryDim, "k2.gate")
        self.attn = try buffer(queryDim, "k2.attn")
        self.oOut = try buffer(hidden, "k2.o")
        self.h1 = try buffer(hidden, "k2.h1")
        self.normed2 = try buffer(hidden, "k2.normed2")
        self.mlpGate = try buffer(ffn, "k2.mlpGate")
        self.mlpUp = try buffer(ffn, "k2.mlpUp")
        self.act = try buffer(ffn, "k2.act")
        self.mlpOut = try buffer(hidden, "k2.mlpOut")
    }

    /// Encode the layer from `hidden` into `out` (both half, one token).
    func encode(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        position: UInt32
    ) throws {
        guard Int(position) < maxSequence else {
            throw ModelError.internalInconsistency(
                detail: "K2 layer position \(position) exceeds its \(maxSequence)-token cache")
        }
        let d = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let groups = max(1, config.normGroups)

        let inputNorm = try model.inputNorm(layer: layer)
        try rms.encodeBF16WGrouped(
            commandBuffer: commandBuffer,
            x: hidden, xOffset: hiddenOffset,
            weight: inputNorm.buffer, weightOffset: Int(inputNorm.offset),
            out: normed,
            groupDim: UInt32(d / groups), numGroups: groups,
            eps: 1e-6)

        try gemv(
            commandBuffer: commandBuffer,
            view: try model.qProj(layer: layer), x: normed,
            y: q, rows: queryDim, columns: d)
        try gemv(
            commandBuffer: commandBuffer,
            view: try model.kProj(layer: layer), x: normed,
            y: k, rows: kvDim, columns: d)
        try gemv(
            commandBuffer: commandBuffer,
            view: try model.vProj(layer: layer), x: normed,
            y: v, rows: kvDim, columns: d)
        try gemv(
            commandBuffer: commandBuffer,
            view: try model.k2AttentionGate(layer: layer), x: normed,
            y: gate, rows: queryDim, columns: d)

        let rotaryDim = UInt32(
            (Double(config.fullHeadDim) * config.partialRotaryFactor).rounded())
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: q,
            position: position, headDim: UInt32(config.fullHeadDim),
            numHeads: UInt32(config.numHeads), rotaryDim: rotaryDim,
            theta: Float(config.fullRopeTheta))
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: k,
            position: position, headDim: UInt32(config.fullHeadDim),
            numHeads: UInt32(config.numFullKVHeads), rotaryDim: rotaryDim,
            theta: Float(config.fullRopeTheta))

        let cacheBytes = Int(position) * kvDim * MemoryLayout<UInt16>.stride
        try copy(
            commandBuffer: commandBuffer, from: k, to: kCache,
            toOffset: cacheBytes, count: kvDim)
        try copy(
            commandBuffer: commandBuffer, from: v, to: vCache,
            toOffset: cacheBytes, count: kvDim)

        try attention.encodeFull(
            commandBuffer: commandBuffer,
            q: q, k: kCache, v: vCache, out: attn,
            headDim: UInt32(config.fullHeadDim),
            numQHeads: UInt32(config.numHeads),
            numKVHeads: UInt32(config.numFullKVHeads),
            seqLen: position + 1, scale: Float(config.attentionScale))
        try elementwise.encodeSoftplusGateMul(
            commandBuffer: commandBuffer, out: attn, gate: gate,
            count: queryDim)
        try gemv(
            commandBuffer: commandBuffer,
            view: try model.oProj(layer: layer), x: attn,
            y: oOut, rows: d, columns: queryDim)

        try copy(
            commandBuffer: commandBuffer,
            from: hidden, fromOffset: hiddenOffset,
            to: h1, count: d)
        try elementwise.encodeResidualAdd(
            commandBuffer: commandBuffer, hidden: h1, delta: oOut, count: d)

        let postNorm = try model.postAttnNorm(layer: layer)
        try rms.encodeBF16WGrouped(
            commandBuffer: commandBuffer,
            x: h1,
            weight: postNorm.buffer, weightOffset: Int(postNorm.offset),
            out: normed2,
            groupDim: UInt32(d / groups), numGroups: groups,
            eps: 1e-6)
        try gemv(
            commandBuffer: commandBuffer,
            view: try model.k2DenseGate(layer: layer), x: normed2,
            y: mlpGate, rows: config.intermediateSize, columns: d)
        try gemv(
            commandBuffer: commandBuffer,
            view: try model.k2DenseUp(layer: layer), x: normed2,
            y: mlpUp, rows: config.intermediateSize, columns: d)
        try elementwise.encodeSiluMul(
            commandBuffer: commandBuffer,
            gate: mlpGate, up: mlpUp, out: act,
            count: config.intermediateSize)
        try gemv(
            commandBuffer: commandBuffer,
            view: try model.k2DenseDown(layer: layer), x: act,
            y: mlpOut, rows: d, columns: config.intermediateSize)

        try copy(
            commandBuffer: commandBuffer, from: h1, to: out,
            toOffset: outOffset, count: d)
        try elementwise.encodeResidualAdd(
            commandBuffer: commandBuffer,
            hidden: out, hiddenOffset: outOffset,
            delta: mlpOut, count: d)
    }

    private func gemv(
        commandBuffer: MTLCommandBuffer,
        view: TensorView,
        x: MTLBuffer,
        y: MTLBuffer,
        rows: Int,
        columns: Int
    ) throws {
        if view.dtype == 1 {
            try bf16.encode(
                commandBuffer: commandBuffer,
                weights: view.buffer, weightsOffset: Int(view.offset),
                x: x, y: y,
                m: UInt32(rows), n: UInt32(columns))
            return
        }
        let bits = Int(view.length) * 8 / (rows * columns)
        let encoder = bits == 8 ? gemv8 : gemv4
        try encoder.encode(
            commandBuffer: commandBuffer,
            weights: view.buffer, weightsOffset: Int(view.offset),
            scales: view.buffer, scalesOffset: Int(view.scaleOffset),
            biases: view.buffer, biasesOffset: Int(view.biasOffset),
            x: x, y: y, m: UInt32(rows), n: UInt32(columns))
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
