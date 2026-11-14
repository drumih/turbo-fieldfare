import Foundation
import Metal
import TinyTitanCore

/// K2-Horizon's sparse expert block: bf16 router, sigmoid selection, then the
/// selected SwiGLU experts and the always-on shared expert.
///
/// The routed experts are separate GEMV dispatches over different blobs, so
/// the stage owns one gate/up/act/down scratch slot per selected expert and a
/// final pass applies the router weights and adds the result to the shared
/// expert's output. That is the resident form; a streaming caller hands the
/// same `ExpertWeights` descriptors for whichever blobs are in memory.
final class K2MoEStage {
    struct ExpertWeights {
        let gate: TensorView
        let up: TensorView
        let down: TensorView
    }

    /// Build the three per-role views for one packed expert blob.
    ///
    /// `fetchRoutedExperts` returns one view at the blob's start; the layout's
    /// sub-tensor table says where each projection and its BF16 companions
    /// live inside it.
    static func expertWeights(
        blob: TensorView, entry: ExpertEntry
    ) throws -> ExpertWeights {
        func view(_ role: String) throws -> TensorView {
            guard let sub = entry.subTensors[role],
                let scales = entry.subTensors["\(role)_scales"],
                let biases = entry.subTensors["\(role)_biases"]
            else {
                throw ModelError.indexCorrupt(
                    detail: "expert \(entry.expert) is missing role \(role)")
            }
            let dtype: UInt8 = sub.dtype == "BF16" ? 1 : 0
            return TensorView(
                buffer: blob.buffer,
                offset: blob.offset + sub.offset,
                length: sub.size,
                scaleOffset: blob.offset + scales.offset,
                scaleLength: scales.size,
                biasOffset: blob.offset + biases.offset,
                biasLength: biases.size,
                shape: (
                    sub.shape.first ?? 0,
                    sub.shape.count > 1 ? sub.shape[1] : 0, 0, 0
                ),
                dtype: dtype)
        }
        return ExpertWeights(
            gate: try view("gate"), up: try view("up"), down: try view("down"))
    }

    private let hiddenSize: Int
    private let expertWidth: Int
    private let numExperts: Int
    private let topK: Int
    private let bf16: BF16GEMV
    private let gemv: AffineQuantGEMV
    private let selection: K2RouterSelection
    private let elementwise: Elementwise
    private let combinePSO: MTLComputePipelineState
    /// The shape-specific fused pass: one dispatch for every selected expert's
    /// gate/up/SiLU and one for their down projections, reading the experts
    /// through argument buffers. The per-expert form above stays as the
    /// reference path and the fallback.
    private let fusedGateUpPSO: MTLComputePipelineState
    private let fusedDownPSO: MTLComputePipelineState
    private let fusedArgEncoder: MTLArgumentEncoder
    private let gateArgs: MTLBuffer
    private let upArgs: MTLBuffer
    private let downArgs: MTLBuffer
    private let fusedActs: MTLBuffer
    private let logits: MTLBuffer
    private let indices: MTLBuffer
    private let weights: MTLBuffer
    private let gateScratch: MTLBuffer
    private let upScratch: MTLBuffer
    private let actScratch: MTLBuffer
    private let downScratch: MTLBuffer
    private let sharedGate: MTLBuffer
    private let sharedUp: MTLBuffer
    private let sharedAct: MTLBuffer

    init(
        context: MetalContext,
        weightBits: Int,
        hiddenSize: Int,
        expertWidth: Int,
        numExperts: Int,
        topK: Int
    ) throws {
        precondition(hiddenSize > 0 && hiddenSize % 64 == 0)
        precondition(expertWidth > 0)
        precondition(numExperts > 0 && topK > 0 && topK <= min(16, numExperts))
        self.hiddenSize = hiddenSize
        self.expertWidth = expertWidth
        self.numExperts = numExperts
        self.topK = topK
        self.bf16 = try BF16GEMV(context: context)
        self.gemv = try AffineQuantGEMV(context: context, weightBits: weightBits)
        self.selection = try K2RouterSelection(context: context)
        self.elementwise = try Elementwise(context: context)
        self.combinePSO = try context.pipeline("k2_moe_accumulate_fp16")
        // 4-bit is the kernel's default; only 8-bit needs the constant.
        let fusedConstants: [MetalFunctionConstant] =
            weightBits == 4
            ? []
            : [MetalFunctionConstant(index: 120, value: .uint32(UInt32(weightBits)))]
        self.fusedGateUpPSO = try context.pipeline(
            "k2_moe_gate_up_act", constants: fusedConstants)
        self.fusedDownPSO = try context.pipeline(
            "k2_moe_down", constants: fusedConstants)
        guard
            let fusedFunction = context.library.makeFunction(
                name: "k2_moe_gate_up_act")
        else {
            throw MetalError.missingFunction("k2_moe_gate_up_act")
        }
        self.fusedArgEncoder = fusedFunction.makeArgumentEncoder(bufferIndex: 0)

        func buffer(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: max(count * MemoryLayout<UInt16>.stride, 4),
                    options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        func rawBuffer(_ length: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: max(length, 4), options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        self.logits = try rawBuffer(
            numExperts * MemoryLayout<Float>.stride, "k2.moe.logits")
        self.indices = try rawBuffer(
            topK * MemoryLayout<UInt32>.stride, "k2.moe.indices")
        self.weights = try buffer(topK, "k2.moe.weights")
        self.gateScratch = try buffer(topK * expertWidth, "k2.moe.gate")
        self.upScratch = try buffer(topK * expertWidth, "k2.moe.up")
        self.actScratch = try buffer(topK * expertWidth, "k2.moe.act")
        self.downScratch = try buffer(topK * hiddenSize, "k2.moe.down")
        self.sharedGate = try buffer(expertWidth, "k2.moe.sharedGate")
        self.sharedUp = try buffer(expertWidth, "k2.moe.sharedUp")
        self.sharedAct = try buffer(expertWidth, "k2.moe.sharedAct")
        self.fusedActs = try buffer(topK * expertWidth, "k2.moe.fusedActs")
        let argumentLength = fusedArgEncoder.encodedLength
        self.gateArgs = try rawBuffer(argumentLength, "k2.moe.gateArgs")
        self.upArgs = try rawBuffer(argumentLength, "k2.moe.upArgs")
        self.downArgs = try rawBuffer(argumentLength, "k2.moe.downArgs")
    }

    /// The router GEMV (float logits) and selection. `hidden` is the post-norm
    /// block input.
    func encodeRouter(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        routerWeight: MTLBuffer, routerWeightOffset: Int = 0,
        routerBias: MTLBuffer, routerBiasOffset: Int = 0,
        scaling: Float
    ) throws {
        try bf16.encodeFloat(
            commandBuffer: commandBuffer,
            weights: routerWeight, weightsOffset: routerWeightOffset,
            x: hidden, xOffset: hiddenOffset,
            y: logits, m: UInt32(numExperts), n: UInt32(hiddenSize))
        try selection.encode(
            commandBuffer: commandBuffer,
            logits: logits,
            bias: routerBias, biasOffset: routerBiasOffset,
            outIndices: indices, outWeights: weights,
            numExperts: numExperts, topK: topK, scaling: scaling)
    }

    /// Meaningful after the router command buffer completed.
    func selectedExpertIDs() -> [Int] {
        let base = indices.contents()
        return (0..<topK).map {
            Int(
                base.load(
                    fromByteOffset: $0 * MemoryLayout<UInt32>.stride,
                    as: UInt32.self))
        }
    }

    /// The shared expert then the selected routed experts, summed into `out`.
    /// `experts` is in selection order (slot order), which is the order a
    /// streaming caller fetches them in. The command buffer that encoded
    /// `encodeRouter` must have completed.
    func encodeExperts(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        experts: [ExpertWeights],
        shared: ExpertWeights?,
        out: MTLBuffer, outOffset: Int = 0,
        downInputScale: Float = 1
    ) throws {
        if let shared {
            try encodeSwiGLU(
                commandBuffer: commandBuffer,
                weights: shared,
                x: hidden, xOffset: hiddenOffset,
                gate: sharedGate, up: sharedUp, act: sharedAct,
                y: out, yOffset: outOffset,
                rows: hiddenSize, width: expertWidth,
                downInputScale: downInputScale)
        }

        guard experts.count == topK else {
            throw ModelError.internalInconsistency(
                detail: "K2 MoE selected \(topK) experts, but \(experts.count) are bound")
        }
        for slot in 0..<topK {
            try encodeSwiGLU(
                commandBuffer: commandBuffer,
                weights: experts[slot],
                x: hidden, xOffset: hiddenOffset,
                gate: gateScratch, up: upScratch, act: actScratch,
                y: downScratch,
                yOffset: slot * hiddenSize * MemoryLayout<UInt16>.stride,
                rows: hiddenSize, width: expertWidth,
                downInputScale: downInputScale)
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(combinePSO)
        encoder.setBuffer(downScratch, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        var count = UInt32(hiddenSize)
        var slots = UInt32(topK)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&slots, length: MemoryLayout<UInt32>.stride, index: 4)
        let width = min(combinePSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: hiddenSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// The shared expert alone, into `out`. The sparse layer encodes this in
    /// its pre-fetch command buffer so the post-fetch work is only the routed
    /// experts; `encodeFusedExperts` then adds them to `out` in place.
    func encodeSharedExpert(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        shared: ExpertWeights,
        out: MTLBuffer, outOffset: Int = 0
    ) throws {
        try encodeSwiGLU(
            commandBuffer: commandBuffer,
            weights: shared,
            x: hidden, xOffset: hiddenOffset,
            gate: sharedGate, up: sharedUp, act: sharedAct,
            y: out, yOffset: outOffset,
            rows: hiddenSize, width: expertWidth, downInputScale: 1)
    }

    /// The same block as `encodeExperts`, dispatched as two shape-specific
    /// passes: every selected expert's gate/up/SiLU in one dispatch and every
    /// down projection plus the router-weighted sum in another. The experts are
    /// bound through argument buffers, so the whole top-k set costs two
    /// encoders instead of four per expert.
    ///
    /// `out` receives the shared expert first; the down pass then adds the
    /// routed sum to `out` in place. The caller adds the layer residual as
    /// before.
    func encodeFusedExperts(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        experts: [ExpertWeights],
        shared: ExpertWeights?,
        out: MTLBuffer, outOffset: Int = 0
    ) throws {
        if let shared {
            try encodeSwiGLU(
                commandBuffer: commandBuffer,
                weights: shared,
                x: hidden, xOffset: hiddenOffset,
                gate: sharedGate, up: sharedUp, act: sharedAct,
                y: out, yOffset: outOffset,
                rows: hiddenSize, width: expertWidth, downInputScale: 1)
        }
        guard experts.count == topK else {
            throw ModelError.internalInconsistency(
                detail: "K2 MoE selected \(topK) experts, but \(experts.count) are bound")
        }
        encodeArguments(experts)

        guard let gateUp = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        gateUp.setComputePipelineState(fusedGateUpPSO)
        gateUp.setBuffer(gateArgs, offset: 0, index: 0)
        gateUp.setBuffer(upArgs, offset: 0, index: 1)
        gateUp.setBuffer(hidden, offset: hiddenOffset, index: 2)
        gateUp.setBuffer(fusedActs, offset: 0, index: 3)
        var d = UInt32(hiddenSize)
        var f = UInt32(expertWidth)
        var k = UInt32(topK)
        gateUp.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 4)
        gateUp.setBytes(&f, length: MemoryLayout<UInt32>.stride, index: 5)
        gateUp.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 6)
        for expert in experts {
            gateUp.useResource(expert.gate.buffer, usage: .read)
            gateUp.useResource(expert.up.buffer, usage: .read)
        }
        gateUp.dispatchThreadgroups(
            MTLSize(width: (topK * expertWidth + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        gateUp.endEncoding()

        guard let down = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        down.setComputePipelineState(fusedDownPSO)
        down.setBuffer(downArgs, offset: 0, index: 0)
        down.setBuffer(fusedActs, offset: 0, index: 1)
        down.setBuffer(weights, offset: 0, index: 2)
        down.setBuffer(out, offset: outOffset, index: 3)
        down.setBuffer(out, offset: outOffset, index: 4)
        down.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 5)
        down.setBytes(&f, length: MemoryLayout<UInt32>.stride, index: 6)
        down.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 7)
        for expert in experts {
            down.useResource(expert.down.buffer, usage: .read)
        }
        down.dispatchThreadgroups(
            MTLSize(width: (hiddenSize + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        down.endEncoding()
    }

    /// `K2ExpertBuffers`'s pointer-array extent in moe.metal. The encoder's
    /// argument indices are laid out as weights, then scales, then biases, each
    /// `maxStreamedExperts` wide; this must stay in step with the shader.
    private static let maxStreamedExperts = 16

    /// Point the three argument buffers at the selected experts.
    private func encodeArguments(_ experts: [ExpertWeights]) {
        let roles: [(MTLBuffer, KeyPath<ExpertWeights, TensorView>)] = [
            (gateArgs, \ExpertWeights.gate),
            (upArgs, \ExpertWeights.up),
            (downArgs, \ExpertWeights.down),
        ]
        for (args, role) in roles {
            fusedArgEncoder.setArgumentBuffer(args, offset: 0)
            for (slot, expert) in experts.enumerated() {
                let view = expert[keyPath: role]
                let plane = Self.maxStreamedExperts
                fusedArgEncoder.setBuffer(
                    view.buffer, offset: Int(view.offset), index: slot)
                fusedArgEncoder.setBuffer(
                    view.buffer, offset: Int(view.scaleOffset),
                    index: plane + slot)
                fusedArgEncoder.setBuffer(
                    view.buffer, offset: Int(view.biasOffset),
                    index: 2 * plane + slot)
            }
        }
    }

    private func encodeSwiGLU(
        commandBuffer: MTLCommandBuffer,
        weights: ExpertWeights,
        x: MTLBuffer, xOffset: Int,
        gate: MTLBuffer, up: MTLBuffer, act: MTLBuffer,
        y: MTLBuffer, yOffset: Int,
        rows: Int, width: Int,
        downInputScale: Float
    ) throws {
        try gemv(
            weights.gate, commandBuffer: commandBuffer, x: x, xOffset: xOffset, y: gate,
            rows: width, columns: hiddenSize)
        try gemv(
            weights.up, commandBuffer: commandBuffer, x: x, xOffset: xOffset, y: up, rows: width,
            columns: hiddenSize)
        try elementwise.encodeSiluMul(
            commandBuffer: commandBuffer,
            gate: gate, up: up, out: act, count: width)
        if downInputScale != 1 {
            try elementwise.encodeScale(
                commandBuffer: commandBuffer, x: act, out: act,
                count: width, factor: downInputScale)
        }
        try gemv(
            weights.down, commandBuffer: commandBuffer, x: act, y: y, yOffset: yOffset, rows: rows,
            columns: width)
    }

    private func gemv(
        _ view: TensorView,
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        rows: Int, columns: Int
    ) throws {
        if view.dtype == 1 {
            throw ModelError.internalInconsistency(
                detail: "K2 MoE expects quantized expert projections")
        }
        try gemv.encode(
            commandBuffer: commandBuffer,
            weights: view.buffer, weightsOffset: Int(view.offset),
            scales: view.buffer, scalesOffset: Int(view.scaleOffset),
            biases: view.buffer, biasesOffset: Int(view.biasOffset),
            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
            m: UInt32(rows), n: UInt32(columns))
    }
}
