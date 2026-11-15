import Foundation
import Metal
import TinyTitanCore

/// One Laguna-S-2.1 layer on a decode step.
///
/// Layer 0 is dense (a 12,288-wide SwiGLU MLP); layers 1-47 route to 256
/// experts at top-10 and always run the 1,024-wide shared expert. Every layer
/// has the same attention *shape family* but not the same width: the 12 global
/// layers run 48 query heads, the 36 sliding-window layers 72, both against 8
/// KV heads of width 128. The per-layer count comes from
/// `config.perLayerHeadGeometry.headsPerLayer`, never from `config.numHeads`.
///
/// The K/V cache is a plain FP16 buffer for one sequence (the probe-sized form
/// of the serving graph), and the routed experts are fetched between command
/// buffers because the router's ids have to come back to the CPU.
final class LagunaLayer {
    /// The residual stream is kept divided by this factor so the checkpoint's
    /// massive activations (layer 46 reaches ~1,200) fit in FP16. RMSNorm is
    /// scale-invariant apart from its epsilon, so every sublayer reads the
    /// same normalised input; only the residual additions carry 1/scale.
    static let residualScale: Float = 1.0 / 32.0

    private let config: ArchConfig
    private let model: Model
    private let layer: Int
    private let heads: Int
    private let queryDim: Int
    private let kvDim: Int
    private let isSliding: Bool
    private let isDense: Bool
    private let maxSequence: Int
    private let rope: RoPE
    private let rotaryDim: UInt32
    private let rms: RMSNorm
    private let attention: Attention
    private let elementwise: Elementwise
    private let gemv4: AffineQuantGEMV
    private let gemv8: AffineQuantGEMV
    private let bf16: BF16GEMV
    private let moe: K2MoEStage?

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
    private let kCache: MTLBuffer
    private let vCache: MTLBuffer

    init(
        model: Model, context: MetalContext, layer: Int,
        maxSequence: Int
    ) throws {
        let config = model.config
        guard config.perLayerHeadGeometry.headsPerLayer.indices.contains(layer) else {
            throw ModelError.internalInconsistency(
                detail: "Laguna layer \(layer) has no declared head count")
        }
        self.model = model
        self.config = config
        self.layer = layer
        self.heads = config.perLayerHeadGeometry.headsPerLayer[layer]
        self.queryDim = heads * config.fullHeadDim
        self.kvDim = config.numFullKVHeads * config.fullHeadDim
        self.isSliding = config.fullAttentionLayerMask[layer] == 0
        self.isDense = config.denseLayerIndices.contains(layer)
        self.maxSequence = maxSequence
        if isSliding {
            self.rope = try RoPE(context: context)
            self.rotaryDim = UInt32(
                (Double(config.fullHeadDim) * config.perLayerHeadGeometry.swaPartialRotaryFactor)
                    .rounded())
        } else {
            let yarn = YaRNRoPEParameters(
                headDim: config.fullHeadDim,
                partialRotaryFactor: config.partialRotaryFactor,
                theta: config.fullRopeTheta,
                targetContextTokens: 1_048_576,
                originalContextTokens: config.perLayerHeadGeometry.yarnOriginalMax,
                betaFast: config.perLayerHeadGeometry.yarnBetaFast,
                betaSlow: config.perLayerHeadGeometry.yarnBetaSlow)
            self.rope = try RoPE(context: context, yarn: yarn)
            self.rotaryDim = UInt32(
                (Double(config.fullHeadDim) * config.partialRotaryFactor).rounded())
        }
        self.rms = try RMSNorm(context: context)
        self.attention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.gemv4 = try AffineQuantGEMV(context: context, weightBits: 4)
        self.gemv8 = try AffineQuantGEMV(context: context, weightBits: 8)
        self.bf16 = try BF16GEMV(context: context)

        if isDense {
            self.moe = nil
        } else {
            let entry = try model.packedExpertsLayout.expert(layer: layer, expert: 0)
            let bits = entry.subTensors["gate"]?.bits ?? 4
            self.moe = try K2MoEStage(
                context: context, weightBits: bits,
                hiddenSize: config.hiddenSize,
                expertWidth: config.moeIntermediateSize,
                numExperts: config.numExperts, topK: config.topKExperts)
        }

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
        let hidden = config.hiddenSize
        let ffn = isDense ? config.intermediateSize : config.moeIntermediateSize
        self.normed = try buffer(hidden, "laguna.normed")
        self.q = try buffer(queryDim, "laguna.q")
        self.k = try buffer(kvDim, "laguna.k")
        self.v = try buffer(kvDim, "laguna.v")
        self.gate = try buffer(heads, "laguna.gate")
        self.attn = try buffer(queryDim, "laguna.attn")
        self.oOut = try buffer(hidden, "laguna.o")
        self.h1 = try buffer(hidden, "laguna.h1")
        self.normed2 = try buffer(hidden, "laguna.normed2")
        self.mlpGate = try buffer(ffn, "laguna.mlpGate")
        self.mlpUp = try buffer(ffn, "laguna.mlpUp")
        self.act = try buffer(ffn, "laguna.act")
        self.mlpOut = try buffer(hidden, "laguna.mlpOut")
        self.kCache = try buffer(maxSequence * kvDim, "laguna.kCache")
        self.vCache = try buffer(maxSequence * kvDim, "laguna.vCache")
    }

    func run(
        queue: MTLCommandQueue,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        position: Int
    ) async throws {
        precondition(position < maxSequence)
        guard let first = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try encodeAttention(
            first, hidden: hidden, hiddenOffset: hiddenOffset,
            position: position)
        if isDense {
            try encodeDenseMLP(first)
            try copy(first, from: h1, to: out, toOffset: outOffset, count: config.hiddenSize)
            try elementwise.encodeResidualAddScaled(
                commandBuffer: first, hidden: out, hiddenOffset: outOffset,
                delta: mlpOut, count: config.hiddenSize,
                factor: Self.residualScale)
            first.commit()
            await first.completed()
            dumpLayerBuffers(position: position)
            return
        }
        guard let stage = moe else {
            throw ModelError.internalInconsistency(detail: "sparse layer without a MoE stage")
        }
        let router = try model.resident(name: LagunaTensors.moeRouter(layer))
        let bias = try model.lagunaRouterBias(layer: layer)
        try stage.encodeRouter(
            commandBuffer: first, hidden: normed2,
            routerWeight: router.buffer,
            routerWeightOffset: Int(router.offset),
            routerBias: bias.buffer,
            routerBiasOffset: Int(bias.offset),
            scaling: Float(config.routerScaling))
        first.commit()
        await first.completed()
        dumpLayerBuffers(position: position)

        let ids = stage.selectedExpertIDs()
        guard let plan = try model.planRoutedExperts(layer: layer, experts: ids) else {
            throw ModelError.internalInconsistency(
                detail: "Laguna could not plan expert fetches for layer \(layer)")
        }
        let blobs = try await model.fetchRoutedExperts(plan: plan)
        var experts: [K2MoEStage.ExpertWeights] = []
        for (index, id) in plan.experts.enumerated() {
            let entry = try model.packedExpertsLayout.expert(layer: layer, expert: id)
            experts.append(
                try K2MoEStage.expertWeights(blob: blobs[index], entry: entry))
        }
        let shared = K2MoEStage.ExpertWeights(
            gate: try model.lagunaSharedGate(layer: layer),
            up: try model.lagunaSharedUp(layer: layer),
            down: try model.lagunaSharedDown(layer: layer))
        guard let second = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try stage.encodeExperts(
            commandBuffer: second, hidden: normed2,
            experts: experts, shared: shared, out: mlpOut,
            downInputScale: Self.residualScale)
        try copy(second, from: h1, to: out, toOffset: outOffset, count: config.hiddenSize)
        // The MoE output was computed through a 1/residualScale input to the
        // down projection, so it is already in the scaled residual's units.
        try elementwise.encodeResidualAddScaled(
            commandBuffer: second, hidden: out, hiddenOffset: outOffset,
            delta: mlpOut, count: config.hiddenSize, factor: 1)
        second.commit()
        await second.completed()
        dumpLayerBuffers(position: position)
    }

    /// Norm, q/k/v/g, QK norms, RoPE, cached windowed/full attention, the
    /// per-head gate, o_proj and the attention residual plus post-norm.
    private func encodeAttention(
        _ commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int,
        position: Int
    ) throws {
        let d = config.hiddenSize
        let inputNorm = try model.inputNorm(layer: layer)
        try rms.encodeBF16WGrouped(
            commandBuffer: commandBuffer,
            x: hidden, xOffset: hiddenOffset,
            weight: inputNorm.buffer, weightOffset: Int(inputNorm.offset),
            out: normed,
            groupDim: UInt32(d / max(1, config.normGroups)),
            numGroups: max(1, config.normGroups),
            eps: 1e-6)
        try gemv(
            commandBuffer, try model.qProj(layer: layer),
            x: normed, y: q, rows: queryDim, columns: d)
        try gemv(
            commandBuffer, try model.kProj(layer: layer),
            x: normed, y: k, rows: kvDim, columns: d)
        try gemv(
            commandBuffer, try model.vProj(layer: layer),
            x: normed, y: v, rows: kvDim, columns: d)
        try gemv(
            commandBuffer, try model.lagunaAttentionGate(layer: layer),
            x: normed, y: gate, rows: heads, columns: d)

        if config.perLayerHeadGeometry.qkNorm {
            let qNorm = try model.qNorm(layer: layer)
            let kNorm = try model.kNorm(layer: layer)
            try rms.encodeBF16WPerHead(
                commandBuffer: commandBuffer, x: q,
                weight: qNorm.buffer, weightOffset: Int(qNorm.offset),
                out: q, headDim: UInt32(config.fullHeadDim),
                numHeads: heads, eps: 1e-6)
            try rms.encodeBF16WPerHead(
                commandBuffer: commandBuffer, x: k,
                weight: kNorm.buffer, weightOffset: Int(kNorm.offset),
                out: k, headDim: UInt32(config.fullHeadDim),
                numHeads: config.numFullKVHeads, eps: 1e-6)
        }

        let theta = Float(isSliding ? config.ropeTheta : config.fullRopeTheta)
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: q, position: UInt32(position),
            headDim: UInt32(config.fullHeadDim), numHeads: UInt32(heads),
            rotaryDim: rotaryDim, theta: theta)
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: k, position: UInt32(position),
            headDim: UInt32(config.fullHeadDim),
            numHeads: UInt32(config.numFullKVHeads),
            rotaryDim: rotaryDim, theta: theta)

        let cacheBytes = position * kvDim * MemoryLayout<UInt16>.stride
        try copy(commandBuffer, from: k, to: kCache, toOffset: cacheBytes, count: kvDim)
        try copy(commandBuffer, from: v, to: vCache, toOffset: cacheBytes, count: kvDim)

        if isSliding {
            try attention.encodeSWA(
                commandBuffer: commandBuffer,
                q: q, k: kCache, v: vCache, out: attn,
                headDim: UInt32(config.fullHeadDim),
                numQHeads: UInt32(heads),
                numKVHeads: UInt32(config.numFullKVHeads),
                seqLen: UInt32(position + 1),
                window: UInt32(config.slidingWindow),
                scale: Float(config.attentionScale))
        } else {
            try attention.encodeFull(
                commandBuffer: commandBuffer,
                q: q, k: kCache, v: vCache, out: attn,
                headDim: UInt32(config.fullHeadDim),
                numQHeads: UInt32(heads),
                numKVHeads: UInt32(config.numFullKVHeads),
                seqLen: UInt32(position + 1),
                scale: Float(config.attentionScale))
        }
        if config.perLayerHeadGeometry.gatePerHead {
            try elementwise.encodeSoftplusGatePerHeadMul(
                commandBuffer: commandBuffer, out: attn, gate: gate,
                count: queryDim, headDim: config.fullHeadDim)
        } else {
            try elementwise.encodeSoftplusGateMul(
                commandBuffer: commandBuffer, out: attn, gate: gate,
                count: queryDim)
        }
        try gemv(
            commandBuffer, try model.oProj(layer: layer),
            x: attn, y: oOut, rows: d, columns: queryDim)
        try copy(
            commandBuffer, from: hidden, fromOffset: hiddenOffset,
            to: h1, count: d)
        try elementwise.encodeResidualAddScaled(
            commandBuffer: commandBuffer, hidden: h1, delta: oOut, count: d,
            factor: Self.residualScale)

        let postNorm = try model.postAttnNorm(layer: layer)
        try rms.encodeBF16WGrouped(
            commandBuffer: commandBuffer,
            x: h1, weight: postNorm.buffer, weightOffset: Int(postNorm.offset),
            out: normed2,
            groupDim: UInt32(d / max(1, config.normGroups)),
            numGroups: max(1, config.normGroups),
            eps: 1e-6)
    }

    /// Debug hook: dump layer 0's intermediates at one position so an
    /// independent reference can be compared stage by stage.
    private func dumpLayerBuffers(position: Int) {
        guard
            let value = ProcessInfo.processInfo.environment["TINYTITAN_LAGUNA_LAYER_DUMP"],
            let dumpLayer = Int(value), dumpLayer == layer
        else { return }
        let directory = URL(
            fileURLWithPath: "/tmp/laguna-layer\(layer)-pos\(position)")
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        func write(_ name: String, _ buffer: MTLBuffer, _ count: Int) {
            let data = Data(
                bytes: buffer.contents(),
                count: count * MemoryLayout<UInt16>.stride)
            try? data.write(to: directory.appendingPathComponent(name + ".bin"))
        }
        write("q", q, queryDim)
        write("k", k, kvDim)
        write("v", v, kvDim)
        write("gate", gate, heads)
        write("attn", attn, queryDim)
        write("o", oOut, config.hiddenSize)
        write("h1", h1, config.hiddenSize)
        write("normed2", normed2, config.hiddenSize)
        write("mlpOut", mlpOut, config.hiddenSize)
    }

    private func encodeDenseMLP(_ commandBuffer: MTLCommandBuffer) throws {
        let d = config.hiddenSize
        try gemv(
            commandBuffer, try model.lagunaDenseGate(layer: layer),
            x: normed2, y: mlpGate, rows: config.intermediateSize, columns: d)
        try gemv(
            commandBuffer, try model.lagunaDenseUp(layer: layer),
            x: normed2, y: mlpUp, rows: config.intermediateSize, columns: d)
        try elementwise.encodeSiluMul(
            commandBuffer: commandBuffer, gate: mlpGate, up: mlpUp, out: act,
            count: config.intermediateSize)
        try gemv(
            commandBuffer, try model.lagunaDenseDown(layer: layer),
            x: act, y: mlpOut, rows: d, columns: config.intermediateSize)
    }

    private func gemv(
        _ commandBuffer: MTLCommandBuffer,
        _ view: TensorView, x: MTLBuffer, y: MTLBuffer,
        rows: Int, columns: Int
    ) throws {
        if view.dtype == 1 {
            try bf16.encode(
                commandBuffer: commandBuffer,
                weights: view.buffer, weightsOffset: Int(view.offset),
                x: x, y: y, m: UInt32(rows), n: UInt32(columns))
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
        _ commandBuffer: MTLCommandBuffer,
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
