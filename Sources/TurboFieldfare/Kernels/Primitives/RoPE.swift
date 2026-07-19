import Metal

/// NeoX RoPE variants used by Gemma 4 attention.
final class RoPE {
    private let defaultNeox: MTLComputePipelineState
    private let proportionalNeox: MTLComputePipelineState
    private let defaultNeoxSWAQ: MTLComputePipelineState
    private let defaultNeoxSWAK: MTLComputePipelineState
    private let proportionalNeoxFullQ: MTLComputePipelineState
    private let proportionalNeoxFullK: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.defaultNeox = try context.pipeline("rope_default_neox")
        self.proportionalNeox = try context.pipeline("rope_proportional_neox")
        self.defaultNeoxSWAQ = try Self.specializedPipeline(
            context, "rope_default_neox", headDim: 256, numHeads: 16)
        self.defaultNeoxSWAK = try Self.specializedPipeline(
            context, "rope_default_neox", headDim: 256, numHeads: 8)
        self.proportionalNeoxFullQ = try Self.specializedPipeline(
            context, "rope_proportional_neox",
            headDim: 512, numHeads: 16, rotatedPairs: 64)
        self.proportionalNeoxFullK = try Self.specializedPipeline(
            context, "rope_proportional_neox",
            headDim: 512, numHeads: 2, rotatedPairs: 64)
    }

    func encodeDefaultNeox(commandBuffer: MTLCommandBuffer,
                                  data: MTLBuffer,
                                  dataOffset: Int = 0,
                                  position: UInt32,
                                  headDim: UInt32,
                                  numHeads: UInt32,
                                  numTokens: UInt32 = 1,
                                  theta: Float = 10_000.0) {
        precondition(headDim.isMultiple(of: 2), "head_dim must be even")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let pipeline = defaultPipeline(headDim: headDim, numHeads: numHeads)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(data, offset: dataOffset, index: 0)
        var positionValue = position
        var headDimValue = headDim
        var numHeadsValue = numHeads
        var thetaValue = theta
        encoder.setBytes(&positionValue, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setBytes(&headDimValue, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&numHeadsValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.size, index: 4)
        dispatch(encoder: encoder,
                 pipeline: pipeline,
                 pairs: Int(headDim) / 2,
                 heads: Int(numHeads),
                 tokens: Int(numTokens))
        encoder.endEncoding()
    }

    func encodeProportionalNeox(commandBuffer: MTLCommandBuffer,
                                       data: MTLBuffer,
                                       dataOffset: Int = 0,
                                       position: UInt32,
                                       headDim: UInt32,
                                       numHeads: UInt32,
                                       rotatedPairs: UInt32,
                                       numTokens: UInt32 = 1,
                                       theta: Float = 1_000_000.0) {
        precondition(headDim.isMultiple(of: 2), "head_dim must be even")
        precondition(rotatedPairs * 2 <= headDim,
                     "rotatedPairs * 2 must not exceed head_dim")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let pipeline = proportionalPipeline(
            headDim: headDim,
            numHeads: numHeads,
            rotatedPairs: rotatedPairs)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(data, offset: dataOffset, index: 0)
        var positionValue = position
        var headDimValue = headDim
        var numHeadsValue = numHeads
        var thetaValue = theta
        var rotatedPairsValue = rotatedPairs
        encoder.setBytes(&positionValue, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setBytes(&headDimValue, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&numHeadsValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.size, index: 4)
        encoder.setBytes(&rotatedPairsValue, length: MemoryLayout<UInt32>.size, index: 5)
        dispatch(encoder: encoder,
                 pipeline: pipeline,
                 pairs: Int(headDim) / 2,
                 heads: Int(numHeads),
                 tokens: Int(numTokens))
        encoder.endEncoding()
    }

    private static func specializedPipeline(_ context: MetalContext,
                                            _ name: String,
                                            headDim: UInt32,
                                            numHeads: UInt32,
                                            rotatedPairs: UInt32 = 0) throws
        -> MTLComputePipelineState {
        try context.pipeline(
            name,
            constants: [
                MetalFunctionConstant(index: 50, value: .uint32(headDim)),
                MetalFunctionConstant(index: 51, value: .uint32(numHeads)),
                MetalFunctionConstant(index: 52, value: .uint32(rotatedPairs)),
                MetalFunctionConstant(index: 53, value: .bool(true)),
            ])
    }

    private func defaultPipeline(headDim: UInt32,
                                 numHeads: UInt32) -> MTLComputePipelineState {
        if headDim == 256 && numHeads == 16 { return defaultNeoxSWAQ }
        if headDim == 256 && numHeads == 8 { return defaultNeoxSWAK }
        return defaultNeox
    }

    private func proportionalPipeline(headDim: UInt32,
                                      numHeads: UInt32,
                                      rotatedPairs: UInt32) -> MTLComputePipelineState {
        if headDim == 512 && rotatedPairs == 64 && numHeads == 16 {
            return proportionalNeoxFullQ
        }
        if headDim == 512 && rotatedPairs == 64 && numHeads == 2 {
            return proportionalNeoxFullK
        }
        return proportionalNeox
    }

    private func dispatch(encoder: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          pairs: Int,
                          heads: Int,
                          tokens: Int) {
        let width = min(pairs, Int(pipeline.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(
            MTLSize(width: pairs, height: heads, depth: tokens),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
