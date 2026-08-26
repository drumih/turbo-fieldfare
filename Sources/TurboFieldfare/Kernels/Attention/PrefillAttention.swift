import Foundation
import Metal

enum PrefillAttentionLayerKind: Sendable, Equatable {
    case full
    case slidingWindow
}

struct PrefillAttentionParams: Sendable, Equatable {
    var startPosition: UInt32
    var queryCount: UInt32
    var headDim: UInt32
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var kvValidCount: UInt32
    var slidingWindow: UInt32
    var kvTokenStrideElements: UInt32
    var qTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float
    var bidirectionalBlockStart: UInt32
    var bidirectionalBlockEnd: UInt32

    init(startPosition: UInt32,
                queryCount: UInt32,
                headDim: UInt32,
                numQHeads: UInt32,
                numKVHeads: UInt32,
                kvValidCount: UInt32,
                slidingWindow: UInt32,
                kvTokenStrideElements: UInt32,
                qTokenStrideElements: UInt32,
                oTokenStrideElements: UInt32,
                scale: Float,
                bidirectionalBlockStart: UInt32 = 0,
                bidirectionalBlockEnd: UInt32 = 0) {
        self.startPosition = startPosition
        self.queryCount = queryCount
        self.headDim = headDim
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.kvValidCount = kvValidCount
        self.slidingWindow = slidingWindow
        self.kvTokenStrideElements = kvTokenStrideElements
        self.qTokenStrideElements = qTokenStrideElements
        self.oTokenStrideElements = oTokenStrideElements
        self.scale = scale
        self.bidirectionalBlockStart = bidirectionalBlockStart
        self.bidirectionalBlockEnd = bidirectionalBlockEnd
    }
}


final class PrefillAttention {
    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoParamsSmoke: MTLComputePipelineState
    private let psoFullTensorOps2DValidityV2: MTLComputePipelineState?

    var tensorOps2DValidityV2Available: Bool {
        psoFullTensorOps2DValidityV2 != nil
    }

    convenience init(context: MetalContext) throws {
        try self.init(context: context, simulatingMissingTensorOps: false)
    }

    /// Tests can force the production fallback even on a host where the MSL 4
    /// TensorOps pipeline builds.
    init(context: MetalContext, simulatingMissingTensorOps: Bool) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        self.psoParamsSmoke = try context.pipeline("prefill_attention_params_smoke")
        if simulatingMissingTensorOps {
            self.psoFullTensorOps2DValidityV2 = nil
        } else {
            do {
                self.psoFullTensorOps2DValidityV2 = try context.pipeline(
                    "attention_prefill_full_tensorops_2d_validity_v2")
            } catch {
                self.psoFullTensorOps2DValidityV2 = nil
                FileHandle.standardError.write(Data(
                    ("PrefillAttention: TensorOps 2D pipeline unavailable; "
                     + "using causal-tiled fallback: \(error)\n").utf8))
            }
        }
    }

    func encodeCausal(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int = 0,
                             k: MTLBuffer, kOffset: Int = 0,
                             v: MTLBuffer, vOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             params: PrefillAttentionParams,
                             kvRingCapacity: UInt32 = 0,
                             layerKind: PrefillAttentionLayerKind = .full,
                             path: RuntimePrefillAttentionPath = .causalTiled) {
        var effectiveParams = params
        // Only sliding-window layers make an image block bidirectional;
        // full-attention layers stay causal. Zeroed here as well as at the
        // call site so a caller cannot widen visibility by mistake.
        if layerKind == .full {
            effectiveParams.bidirectionalBlockStart = 0
            effectiveParams.bidirectionalBlockEnd = 0
        }
        validate(effectiveParams)

        let requestsTensorOps = path == .fullTensorOps2DPreferred
            || path == .fullTensorOps2DValidityV2
        // TensorOps starts its key loop at zero and ignores slidingWindow, so
        // these are visibility guards rather than shape optimizations.
        let windowNeverClips = effectiveParams.slidingWindow == 0
            || effectiveParams.slidingWindow >= effectiveParams.kvValidCount
        let tensorOpsShape = requestsTensorOps
            && layerKind == .full
            && kvRingCapacity == 0
            && windowNeverClips
            && effectiveParams.headDim == 512
            && effectiveParams.numQHeads == 16
            && effectiveParams.numKVHeads == 2
            && effectiveParams.scale == 1.0
        let tensorOpsPipeline = tensorOpsShape ? psoFullTensorOps2DValidityV2 : nil
        let useTensorOps = tensorOpsPipeline != nil
        let pipeline: MTLComputePipelineState
        if let tensorOpsPipeline {
            pipeline = tensorOpsPipeline
        } else if tensorOpsShape && path == .fullTensorOps2DValidityV2 {
            preconditionFailure(
                "TensorOps 2D prefill attention pipeline is unavailable on this Metal stack")
        } else {
            // Explicit mode also falls back for incompatible shapes. Benchmark
            // fixtures must use 512/16/2 to prove that TensorOps ran.
            pipeline = causalTiledPipeline(kvRingCapacity: kvRingCapacity)
        }
        let headDim = Int(effectiveParams.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount = useTensorOps
            ? 128
            : roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth)
        precondition(threadCount <= pipeline.maxTotalThreadsPerThreadgroup,
                     "tiled prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = effectiveParams
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        let groups = useTensorOps
            ? MTLSize(width: Int(effectiveParams.queryCount),
                      height: Int(effectiveParams.numQHeads) / 8,
                      depth: 1)
            : MTLSize(width: Int(effectiveParams.queryCount),
                      height: Int(effectiveParams.numQHeads),
                      depth: 1)
        enc.dispatchThreadgroups(
            groups,
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        enc.endEncoding()
    }


    private func validate(_ params: PrefillAttentionParams) {
        precondition(params.headDim > 0, "headDim must be positive")
        precondition(params.queryCount > 0, "queryCount must be positive")
        precondition(params.numQHeads > 0, "numQHeads must be positive")
        precondition(params.numKVHeads > 0, "numKVHeads must be positive")
        precondition(params.numQHeads % params.numKVHeads == 0,
                     "numQHeads must be divisible by numKVHeads")
        precondition(params.qTokenStrideElements >= params.numQHeads * params.headDim,
                     "q token stride is too small")
        precondition(params.oTokenStrideElements >= params.numQHeads * params.headDim,
                     "output token stride is too small")
        precondition(params.kvTokenStrideElements >= params.numKVHeads * params.headDim,
                     "KV token stride is too small")
        precondition(params.startPosition + params.queryCount <= params.kvValidCount,
                     "kvValidCount must include all in-flight query rows")
        precondition(params.bidirectionalBlockStart <= params.bidirectionalBlockEnd,
                     "bidirectional block range is invalid")
        precondition(params.bidirectionalBlockEnd <= params.kvValidCount,
                     "bidirectional block exceeds valid KV rows")
    }


    private func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }


    /// Reads every field back through the MSL struct. `PrefillAttentionParams`
    /// is mirrored by hand in `prefill.metal`, and a field added on one side
    /// only shifts every later field silently.
    func encodeParamsSmoke(commandBuffer: MTLCommandBuffer,
                           params: PrefillAttentionParams,
                           out: MTLBuffer) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoParamsSmoke)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: 13, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 13, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func causalTiledPipeline(kvRingCapacity: UInt32) -> MTLComputePipelineState {
        guard kvRingCapacity > 0 else { return psoCausalTiled }
        do {
            return try context.pipeline(
                "attention_prefill_causal_tiled",
                constants: [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity))])
        } catch {
            preconditionFailure("failed to build FP16 KV ring prefill attention pipeline: \(error)")
        }
    }
}
