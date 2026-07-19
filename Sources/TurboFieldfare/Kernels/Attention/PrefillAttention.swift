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

struct PrefillTurboQuantAttentionParams: Sendable, Equatable {
    var startPosition: UInt32
    var queryCount: UInt32
    var headDim: UInt32
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var kvValidCount: UInt32
    var slidingWindow: UInt32
    var qTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float
    var layer: UInt32
    var rotationSeed: UInt32
    var keyBytesPerHead: UInt32
    var keyPackedOffset: UInt32
    var keyScaleOffset: UInt32
    var valueBytesPerHead: UInt32
    var valuePackedOffset: UInt32
    var valueScaleOffset: UInt32

    init(prefill: PrefillAttentionParams,
                layer: UInt32,
                rotationSeed: UInt32,
                keyLayout: TurboQuantKVRoleLayout,
                valueLayout: TurboQuantKVRoleLayout) {
        self.startPosition = prefill.startPosition
        self.queryCount = prefill.queryCount
        self.headDim = prefill.headDim
        self.numQHeads = prefill.numQHeads
        self.numKVHeads = prefill.numKVHeads
        self.kvValidCount = prefill.kvValidCount
        self.slidingWindow = prefill.slidingWindow
        self.qTokenStrideElements = prefill.qTokenStrideElements
        self.oTokenStrideElements = prefill.oTokenStrideElements
        self.scale = prefill.scale
        self.layer = layer
        self.rotationSeed = rotationSeed
        self.keyBytesPerHead = UInt32(keyLayout.bytesPerHead)
        self.keyPackedOffset = UInt32(keyLayout.packedOffsetPerHead)
        self.keyScaleOffset = UInt32(keyLayout.scaleOffsetPerHead)
        self.valueBytesPerHead = UInt32(valueLayout.bytesPerHead)
        self.valuePackedOffset = UInt32(valueLayout.packedOffsetPerHead)
        self.valueScaleOffset = UInt32(valueLayout.scaleOffsetPerHead)
    }
}

final class PrefillAttention {
    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoCausalTurboQuantTiled: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        self.psoCausalTurboQuantTiled = try context.pipeline("attention_prefill_turboquant_causal_tiled")
    }

    func encodeCausal(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int = 0,
                             k: MTLBuffer, kOffset: Int = 0,
                             v: MTLBuffer, vOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             params: PrefillAttentionParams,
                             kvRingCapacity: UInt32 = 0) {
        validate(params)

        let pipeline = causalTiledPipeline(kvRingCapacity: kvRingCapacity)
        let headDim = Int(params.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount = roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth)
        precondition(threadCount <= pipeline.maxTotalThreadsPerThreadgroup,
                     "tiled prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(params.queryCount), height: Int(params.numQHeads), depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeTurboQuantCausal(commandBuffer: MTLCommandBuffer,
                                       q: MTLBuffer, qOffset: Int = 0,
                                       keyCache: MTLBuffer, keyCacheOffset: Int = 0,
                                       valueCache: MTLBuffer, valueCacheOffset: Int = 0,
                                       out: MTLBuffer, outOffset: Int = 0,
                                       params: PrefillTurboQuantAttentionParams) {
        validateTurboQuant(params)

        let threadWidth = max(1, psoCausalTurboQuantTiled.threadExecutionWidth)
        let threadCount = roundUp(max(threadWidth, min(Int(params.headDim), 256)),
                                  toMultipleOf: threadWidth)
        precondition(threadCount <= psoCausalTurboQuantTiled.maxTotalThreadsPerThreadgroup,
                     "TurboQuant prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoCausalTurboQuantTiled)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(keyCache, offset: keyCacheOffset, index: 1)
        enc.setBuffer(valueCache, offset: valueCacheOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillTurboQuantAttentionParams>.stride, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(params.queryCount), height: Int(params.numQHeads), depth: 1),
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
    }

    private func validateTurboQuant(_ params: PrefillTurboQuantAttentionParams) {
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
        precondition(params.keyBytesPerHead > 0, "key bytes per head must be positive")
        precondition(params.valueBytesPerHead > 0, "value bytes per head must be positive")
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
