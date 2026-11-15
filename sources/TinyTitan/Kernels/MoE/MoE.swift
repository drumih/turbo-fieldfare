import Foundation
import Metal
import TinyTitanCore

@frozen
public struct MoEExpertOffsets {
    public var gateWOff: UInt32
    public var gateSOff: UInt32
    public var gateBOff: UInt32
    public var upWOff: UInt32
    public var upSOff: UInt32
    public var upBOff: UInt32
    public var downWOff: UInt32
    public var downSOff: UInt32
    public var downBOff: UInt32

    public init(
        gateWOff: UInt32, gateSOff: UInt32, gateBOff: UInt32,
        upWOff: UInt32, upSOff: UInt32, upBOff: UInt32,
        downWOff: UInt32, downSOff: UInt32, downBOff: UInt32
    ) {
        self.gateWOff = gateWOff
        self.gateSOff = gateSOff
        self.gateBOff = gateBOff
        self.upWOff = upWOff
        self.upSOff = upSOff
        self.upBOff = upBOff
        self.downWOff = downWOff
        self.downSOff = downSOff
        self.downBOff = downBOff
    }
}

final class MoE {
    /// The largest expert count `encodeRouter` accepts, and therefore the size
    /// the router-logits scratch has to cover.
    ///
    /// The allocation and `encodeRouter`'s guard are two halves of one
    /// contract and must move together. They did not: the guard was raised to
    /// 512 "for Qwen3.8-Flash-Next" with a comment calling the old 256 "a
    /// conservative guard rather than a width limit", while the scratch stayed
    /// 256 floats. At 512 experts the router GEMV writes and the selector reads
    /// 2048 bytes through a 1024-byte `MTLBuffer` — out of bounds relative to
    /// the buffer's declared length, and non-faulting only because the driver's
    /// allocation is page-granular. That is undefined behaviour that happens to
    /// work, on a pinned model, on every layer and every token.
    static let maxRouterExperts = 512
    /// Experts one routed dispatch serves — the architecture's `topKExperts`,
    /// supplied at init. It sizes the routed argument buffer and every
    /// per-dispatch validation; nothing may assume the literal 8 (the
    /// Qwen3.5-MoE value) because Qwen3.8-Flash-Next routes top-10. The Metal
    /// side is currently compiled for k = 8 (`moe_phase2_reduce_k8` and the
    /// argument-buffer encoder length); init refuses anything else so a new
    /// k arrives as explicit kernel work, never as silent misexecution.
    let maxStreamedExperts: Int

    let realDecodeD: UInt32
    let realDecodeF: UInt32
    /// The k baked into the specialized pipelines. This was a constant 8 --
    /// the value every shipping model used -- which made the specialization
    /// silently wrong for a family that shares those hidden dimensions but
    /// routes to a different number of experts: phase 1 simply never wrote
    /// the slots past 8, and the reduce summed zeros for them. It is the
    /// model's own top-k now, and `useRealDecodeConstants` checks it.
    let realDecodeTopK: UInt32
    let realDecodeNumExperts: UInt32

    let routerGemvPSO: MTLComputePipelineState
    let routerGemvSpecializedPSO: MTLComputePipelineState
    let routerSelectK8PSO: MTLComputePipelineState
    let routerSelectK8SpecializedPSO: MTLComputePipelineState
    /// Used when top-k is not 8. The k8 kernel stays the golden path.
    let routerSelectKNPSO: MTLComputePipelineState
    /// One-simdgroup top-k for k != 8, same order as the serial kernel.
    /// Default on: both Qwen3.8 goldens are byte-identical with it, and it
    /// takes the router pair (real + next-layer probe) from 11.0 to 3.8
    /// ms/token because the serial kernel's insertion sort over 512 logits
    /// was the router's cost, not the GEMV. TINYTITAN_ROUTER_TOPK_SIMD=0 restores
    /// the serial kernel.
    let routerSelectKNSimdPSO: MTLComputePipelineState?
    let routerTopKSimd: Bool
    let residencyClassifyPSO: MTLComputePipelineState
    let routerLogits: MTLBuffer
    let phase1U16PSO: MTLComputePipelineState
    let phase1U16SpecializedPSO: MTLComputePipelineState
    let phase1SubsetU16PSO: MTLComputePipelineState
    let phase1SubsetU16SpecializedPSO: MTLComputePipelineState
    let phase2ReduceK8PSO: MTLComputePipelineState
    let phase2ReduceK8SpecializedPSO: MTLComputePipelineState
    /// Used when top-k is not 8; the k8 kernels stay the golden path.
    let phase2ReduceKNPSO: MTLComputePipelineState
    let routedArgEncoder: MTLArgumentEncoder
    let reusableRoutedArgBuffer: MTLBuffer
    let alwaysReadyIOStatus: MTLBuffer

    /// `specializedD`/`specializedF`/`specializedNumExperts` describe the
    /// production shape this instance specializes for (the specialized
    /// defaults 2816/704/128 predate Qwen-only support; Qwen 3.6 passes
    /// 2048/512/256). `siluActivation` selects the expert FFN activation
    /// (false = gelu_pytorch_tanh, true = silu).
    init(
        context: MetalContext,
        routerTopKSimd: Bool = true,
        siluActivation: Bool = false,
        routedWeightBits: Int = 4,
        routerWeightBits: Int = 8,
        eventGatedIO: Bool = false,
        specializedD: UInt32 = 2816,
        specializedF: UInt32 = 704,
        specializedNumExperts: UInt32 = 128,
        topKExperts: Int = 8
    ) throws {
        self.routerTopKSimd = routerTopKSimd
        self.realDecodeD = specializedD
        self.realDecodeF = specializedF
        self.realDecodeTopK = UInt32(topKExperts)
        self.realDecodeNumExperts = specializedNumExperts
        // 16 is the argument buffer's blob-array extent (kMaxStreamedExperts
        // in moe.metal). Router select and phase-2 reduce both have k != 8
        // variants; see docs/qwen38-flash-next-port.md.
        //
        // `0` is a real value: a dense model has no routed experts, so it has
        // no router to score and no routed FFN to run. The runtime is still
        // constructed -- the argument buffers are per-runner scratch and the
        // pipelines are built once here -- and every routed stage is skipped
        // when the model has no experts, so this specialization is never
        // dispatched. The shader's own arrays are sized by the fixed
        // `kMaxStreamedExperts` rather than by this constant, so a zero needs
        // no kernel variant.
        precondition(
            (0...16).contains(topKExperts),
            "top-\(topKExperts) exceeds the routed argument buffer's "
                + "expert slots (16)")
        self.maxStreamedExperts = topKExperts
        precondition([2, 4, 8].contains(routedWeightBits))
        // 16 means the router is unquantized bf16; the shader branches on it.
        precondition([4, 8, 16].contains(routerWeightBits))
        let activationConstants: [MetalFunctionConstant] =
            siluActivation
            ? [MetalFunctionConstant(index: 4, value: .bool(true))]
            : []
        let weightConstants =
            routedWeightBits == 4
            ? []
            : [
                MetalFunctionConstant(index: 5, value: .uint32(UInt32(routedWeightBits)))
            ]
        let ioConstants = [MetalFunctionConstant(index: 6, value: .bool(eventGatedIO))]
        let moeConstants: [MetalFunctionConstant] =
            [
                MetalFunctionConstant(index: 0, value: .uint32(specializedD)),
                MetalFunctionConstant(index: 1, value: .uint32(specializedF)),
                MetalFunctionConstant(index: 2, value: .uint32(realDecodeTopK)),
                MetalFunctionConstant(index: 3, value: .bool(true)),
            ] + activationConstants + weightConstants + ioConstants
        let routerConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 40, value: .uint32(specializedNumExperts)),
            MetalFunctionConstant(index: 41, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 42, value: .uint32(realDecodeTopK)),
            MetalFunctionConstant(index: 43, value: .bool(true)),
            MetalFunctionConstant(index: 44, value: .uint32(UInt32(routerWeightBits))),
        ]
        let routerName = "router_gemv_r4"
        self.routerGemvPSO = try context.pipeline(
            routerName,
            constants: [
                MetalFunctionConstant(
                    index: 44,
                    value: .uint32(UInt32(routerWeightBits)))
            ],
            maxTotalThreadsPerThreadgroup: 512)
        self.routerGemvSpecializedPSO = try context.pipeline(
            routerName,
            constants: routerConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.routerSelectK8PSO = try context.pipeline("router_topk_select_k8")
        self.routerSelectK8SpecializedPSO = try context.pipeline(
            "router_topk_select_k8",
            constants: routerConstants)
        self.routerSelectKNPSO = try context.pipeline("router_topk_select_kn")
        self.routerSelectKNSimdPSO = try? context.pipeline("router_topk_select_kn_simd")
        self.residencyClassifyPSO = try context.pipeline("moe_classify_expert_residency")
        let phase1Name =
            routedWeightBits == 4
            ? "moe_phase1_gate_up_act_u16load" : "moe_affine_phase1_gate_up_act"
        let phase1SubsetName =
            routedWeightBits == 4
            ? "moe_phase1_gate_up_act_subset_u16load" : "moe_affine_phase1_gate_up_act_subset"
        let phase2Name =
            routedWeightBits == 4
            ? "moe_phase2_down_reduce_k8" : "moe_affine_phase2_down_reduce_k8"
        self.phase1U16PSO = try context.pipeline(
            phase1Name, constants: activationConstants + weightConstants + ioConstants)
        self.phase1U16SpecializedPSO = try context.pipeline(
            phase1Name,
            constants: moeConstants)
        self.phase1SubsetU16PSO = try context.pipeline(
            phase1SubsetName, constants: activationConstants + weightConstants + ioConstants)
        self.phase1SubsetU16SpecializedPSO = try context.pipeline(
            phase1SubsetName,
            constants: moeConstants)
        self.phase2ReduceK8PSO = try context.pipeline(
            phase2Name, constants: weightConstants + ioConstants)
        let phase2KNName =
            routedWeightBits == 4
            ? "moe_phase2_down_reduce_kn" : "moe_affine_phase2_down_reduce_kn"
        self.phase2ReduceKNPSO = try context.pipeline(
            phase2KNName, constants: weightConstants + ioConstants)
        self.phase2ReduceK8SpecializedPSO = try context.pipeline(
            phase2Name,
            constants: moeConstants)

        guard
            let logits = context.device.makeBuffer(
                length: Int(Self.maxRouterExperts) * MemoryLayout<Float>.stride,
                options: .storageModeShared),
            let readyStatus = context.device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
            let phase1Function = context.library.makeFunction(name: phase1Name)
        else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        readyStatus.contents().storeBytes(of: UInt32(1), as: UInt32.self)
        self.alwaysReadyIOStatus = readyStatus
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard
            let reusable = context.device.makeBuffer(
                length: routedArgEncoder.encodedLength,
                options: .storageModeShared)
        else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    func encodeRouter(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        effectiveScale: MTLBuffer, effectiveScaleOffset: Int = 0,
        perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
        outIndices: MTLBuffer, outIndicesOffset: Int = 0,
        outWeights: MTLBuffer, outWeightsOffset: Int = 0,
        numExperts: UInt32,
        d: UInt32,
        topK: UInt32
    ) throws {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        // Expert ids are UInt32 end to end (ExpertResidencyTable) and the
        // kernels read num_experts dynamically, so the guard is a real width
        // limit on the scratch, not a conservative one: see `maxRouterExperts`.
        precondition(
            numExperts <= Self.maxRouterExperts,
            "encodeRouter: numExperts \(numExperts) exceeds the router-logits scratch (\(Self.maxRouterExperts))"
        )
        // The scratch is sized for the maximum above, but this is the check that
        // would have caught the two drifting apart in the first place.
        precondition(
            routerLogits.length >= Int(numExperts) * MemoryLayout<Float>.stride,
            "encodeRouter: routerLogits must cover [numExperts] Float (allocated \(routerLogits.length) bytes, need \(Int(numExperts) * MemoryLayout<Float>.stride))"
        )
        precondition(topK == UInt32(maxStreamedExperts))
        // K16: `router_gemv_r4` multiplies every hidden element by
        // `effective_scale[idx]` and `router_topk_select_k8` multiplies every
        // weight by `per_expert_scale[expert]`. Qwen 3.6 has no router scale
        // tensors, so the runner synthesizes 1.0-filled buffers for both —
        // they must always be supplied with at least the addressed element
        // count, never nil/undersized, or the kernels read out of bounds.
        precondition(
            effectiveScale.length >= Int(d) * MemoryLayout<UInt16>.stride,
            "encodeRouter: effectiveScale must cover [d] BF16 (runner synthesizes a 1.0 buffer for Qwen; the kernel always reads it)"
        )
        precondition(
            perExpertScale.length >= Int(numExperts) * MemoryLayout<UInt16>.stride,
            "encodeRouter: perExpertScale must cover [numExperts] BF16 (runner synthesizes a 1.0 buffer for Qwen; router_topk_select_k8 always dereferences it)"
        )

        var expertCount = numExperts
        var dimension = d
        let useSpecialized =
            numExperts == realDecodeNumExperts
            && d == realDecodeD
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(
            useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 3)
        encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
        encoder.setBuffer(routerLogits, offset: 0, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()

        guard let selector = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        // The one-simdgroup selector serves every k, k == 8 included: the k8
        // kernel is the same single-thread insertion sort, and the simd
        // kernel reproduces its order exactly (checked byte-identical on the
        // AgentWorld goldens, top-8 over 256 experts).
        let usesSimdSelector = routerTopKSimd && expertCount <= 1024 && routerSelectKNSimdPSO != nil
        let selectPipeline: MTLComputePipelineState
        if usesSimdSelector, let simd = routerSelectKNSimdPSO {
            selectPipeline = simd
        } else if maxStreamedExperts == 8 {
            selectPipeline = useSpecialized ? routerSelectK8SpecializedPSO : routerSelectK8PSO
        } else {
            selectPipeline = routerSelectKNPSO
        }
        selector.setComputePipelineState(selectPipeline)
        selector.setBuffer(routerLogits, offset: 0, index: 0)
        selector.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
        selector.setBuffer(outIndices, offset: outIndicesOffset, index: 2)
        selector.setBuffer(outWeights, offset: outWeightsOffset, index: 3)
        selector.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        if maxStreamedExperts != 8 || usesSimdSelector {
            var k = UInt32(maxStreamedExperts)
            selector.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 5)
        }
        selector.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        selector.endEncoding()
    }

    func makeRoutedArgumentBuffer(
        routedBlobs: [MTLBuffer],
        topK: UInt32,
        routedBufferOffsets: [Int]? = nil
    ) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs, topK: topK)
        guard
            let buffer = routedBlobs.first?.device.makeBuffer(
                length: routedArgEncoder.encodedLength,
                options: .storageModeShared)
        else {
            return nil
        }
        encodeRoutedArgumentBuffer(
            buffer, routedBlobs: routedBlobs,
            routedBufferOffsets: routedBufferOffsets)
        return buffer
    }

    func encodeResidencyClassification(
        commandBuffer: MTLCommandBuffer,
        topKIndices: MTLBuffer,
        residencyTable: MTLBuffer,
        hitCount: MTLBuffer,
        hitPositions: MTLBuffer,
        missCount: MTLBuffer,
        missPositions: MTLBuffer,
        missExperts: MTLBuffer,
        resolvedSlots: MTLBuffer,
        resolvedGenerations: MTLBuffer,
        topK: UInt32,
        numExperts: UInt32
    ) throws {
        precondition(topK <= UInt32(maxStreamedExperts))
        var topKValue = topK
        var expertCount = numExperts
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(residencyClassifyPSO)
        encoder.setBuffer(topKIndices, offset: 0, index: 0)
        encoder.setBuffer(residencyTable, offset: 0, index: 1)
        encoder.setBuffer(hitCount, offset: 0, index: 2)
        encoder.setBuffer(hitPositions, offset: 0, index: 3)
        encoder.setBuffer(missCount, offset: 0, index: 4)
        encoder.setBuffer(missPositions, offset: 0, index: 5)
        encoder.setBuffer(missExperts, offset: 0, index: 6)
        encoder.setBuffer(resolvedSlots, offset: 0, index: 7)
        encoder.setBuffer(resolvedGenerations, offset: 0, index: 8)
        encoder.setBytes(&topKValue, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// An argument buffer with no views encoded yet, for callers that
    /// re-encode per use via `writeRoutedArgumentBuffer`.
    func makeEmptyRoutedArgumentBuffer(device: MTLDevice) -> MTLBuffer? {
        device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared)
    }

    /// Re-encode the views of an argument buffer created by
    /// `makeRoutedArgumentBuffer`. The caller owns the hazard: the buffer must
    /// not be rewritten while a committed command still reads it.
    func writeRoutedArgumentBuffer(
        _ buffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        topK: UInt32,
        routedBufferOffsets: [Int]? = nil
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(
            buffer, routedBlobs: routedBlobs,
            routedBufferOffsets: routedBufferOffsets)
    }

    func makeReusedRoutedArgumentBuffer(
        routedBlobs: [MTLBuffer],
        topK: UInt32,
        routedBufferOffsets: [Int]? = nil
    ) -> MTLBuffer {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(
            reusableRoutedArgBuffer, routedBlobs: routedBlobs,
            routedBufferOffsets: routedBufferOffsets)
        return reusableRoutedArgBuffer
    }

}
