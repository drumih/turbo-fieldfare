import Foundation
import Metal

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
                scale: Float) {
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
    }
}


final class PrefillAttention {
    // The causal tiled fallback runs one threadgroup per query/head pair. On
    // pre-Apple10 GPUs, a full-attention dispatch beyond 4K can exceed macOS's
    // interactive GPU-work limit even though the allocation is small. Since
    // work grows with both query rows and visible KV length, shrink row batches
    // as the context grows while preserving the optimized Apple10 path.
    static let longContextThreshold = 4_096
    static let longContextQueryKVPairBudget = 24_576
    static let longContextMaximumQueryRowsPerEncoder = 8

    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoFullTensorOps2DValidityV2: MTLComputePipelineState?

    init(context: MetalContext) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        self.psoFullTensorOps2DValidityV2 = context.device.supportsFamily(.apple10)
            ? try? context.pipeline("attention_prefill_full_tensorops_2d_validity_v2")
            : nil
    }

    func encodeCausal(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int = 0,
                             k: MTLBuffer, kOffset: Int = 0,
                             v: MTLBuffer, vOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             params: PrefillAttentionParams,
                             kvRingCapacity: UInt32 = 0,
                             path: RuntimePrefillAttentionPath = .causalTiled,
                             watchdogProtectionEnabled: Bool = true) {
        validate(params)

        let requestsTensorOps = path == .fullTensorOps2DPreferred
            || path == .fullTensorOps2DValidityV2
        // The pinned model uses 512/16/2 only for full attention; its
        // sliding-window layers use 256/16/8. A future model that reuses this
        // shape for sliding attention must add a full-visibility check here.
        let tensorOpsShape = requestsTensorOps
            && kvRingCapacity == 0
            && params.headDim == 512
            && params.numQHeads == 16
            && params.numKVHeads == 2
            && params.scale == 1.0
        let tensorOpsPipeline = tensorOpsShape ? psoFullTensorOps2DValidityV2 : nil
        let useTensorOps = tensorOpsPipeline != nil
        let pipeline: MTLComputePipelineState
        if let tensorOpsPipeline {
            pipeline = tensorOpsPipeline
        } else if tensorOpsShape && path == .fullTensorOps2DValidityV2 {
            preconditionFailure(
                "TensorOps 2D prefill attention requires Apple10 MPP tensor support")
        } else {
            // Explicit mode also falls back for incompatible shapes. Benchmark
            // fixtures must use 512/16/2 to prove that TensorOps ran.
            pipeline = causalTiledPipeline(kvRingCapacity: kvRingCapacity)
        }
        let headDim = Int(params.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount = useTensorOps
            ? 128
            : roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth)
        precondition(threadCount <= pipeline.maxTotalThreadsPerThreadgroup,
                     "tiled prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")

        let fullAttentionShape = kvRingCapacity == 0
            && params.headDim == 512
            && params.numQHeads == 16
            && params.numKVHeads == 2
        let spans = Self.querySpans(queryCount: Int(params.queryCount),
                                    kvValidCount: Int(params.kvValidCount),
                                    fullAttentionShape: fullAttentionShape,
                                    useTensorOps: useTensorOps,
                                    watchdogProtectionEnabled: watchdogProtectionEnabled)
        for span in spans {
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
            enc.label = "prefill.attention queries=\(span.lowerBound)..<\(span.upperBound) kv=\(params.kvValidCount)"
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(
                q,
                offset: qOffset
                    + span.lowerBound * Int(params.qTokenStrideElements)
                    * MemoryLayout<Float16>.stride,
                index: 0)
            enc.setBuffer(k, offset: kOffset, index: 1)
            enc.setBuffer(v, offset: vOffset, index: 2)
            enc.setBuffer(
                out,
                offset: outOffset
                    + span.lowerBound * Int(params.oTokenStrideElements)
                    * MemoryLayout<Float16>.stride,
                index: 3)
            var batchParams = params
            batchParams.startPosition += UInt32(span.lowerBound)
            batchParams.queryCount = UInt32(span.count)
            enc.setBytes(
                &batchParams,
                length: MemoryLayout<PrefillAttentionParams>.stride,
                index: 4)
            let groups = useTensorOps
                ? MTLSize(width: span.count,
                          height: Int(params.numQHeads) / 8,
                          depth: 1)
                : MTLSize(width: span.count,
                          height: Int(params.numQHeads),
                          depth: 1)
            enc.dispatchThreadgroups(
                groups,
                threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
            enc.endEncoding()
        }
    }

    static func querySpans(queryCount: Int,
                           kvValidCount: Int,
                           fullAttentionShape: Bool,
                           useTensorOps: Bool,
                           watchdogProtectionEnabled: Bool = true) -> [Range<Int>] {
        precondition(queryCount > 0, "queryCount must be positive")
        let rows: Int
        if watchdogProtectionEnabled
            && !useTensorOps
            && fullAttentionShape
            && kvValidCount > longContextThreshold {
            rows = max(
                1,
                min(
                    longContextMaximumQueryRowsPerEncoder,
                    longContextQueryKVPairBudget / kvValidCount))
        } else {
            rows = queryCount
        }
        return stride(from: 0, to: queryCount, by: rows).map {
            $0..<min($0 + rows, queryCount)
        }
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
    }


    private func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
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
