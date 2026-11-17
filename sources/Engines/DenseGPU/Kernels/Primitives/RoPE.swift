import Metal
import TinyTitanCore

/// NeoX RoPE variants used by Qwen 3.6 attention.
final class RoPE {
    private let defaultNeox: MTLComputePipelineState
    private let proportionalNeox: MTLComputePipelineState
    private let neoxSubdim: MTLComputePipelineState
    private let stridedNeoxSubdim: MTLComputePipelineState
    private let defaultNeoxSWAQ: MTLComputePipelineState
    private let defaultNeoxSWAK: MTLComputePipelineState
    private let proportionalNeoxFullQ: MTLComputePipelineState
    private let proportionalNeoxFullK: MTLComputePipelineState
    private let yarnNeoxSubdim: MTLComputePipelineState
    private let yarnInverseFrequencies: MTLBuffer?
    private let yarnAttentionFactor: Float

    init(context: MetalContext, yarn: YaRNRoPEParameters? = nil) throws {
        self.defaultNeox = try context.pipeline("rope_default_neox")
        self.proportionalNeox = try context.pipeline("rope_proportional_neox")
        self.neoxSubdim = try context.pipeline("rope_neox_subdim")
        self.stridedNeoxSubdim = try context.pipeline("rope_neox_subdim_strided")
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
        self.yarnNeoxSubdim = try context.pipeline("rope_yarn_neox_subdim")
        self.yarnAttentionFactor = yarn?.attentionFactor ?? 1
        if let yarn {
            guard
                let buffer = yarn.inverseFrequencies.withUnsafeBytes({ bytes -> MTLBuffer? in
                    guard let baseAddress = bytes.baseAddress else { return nil }
                    return context.device.makeBuffer(
                        bytes: baseAddress, length: bytes.count,
                        options: .storageModeShared)
                })
            else {
                throw MetalError.bufferAllocationFailed("YaRN inverse frequencies")
            }
            self.yarnInverseFrequencies = buffer
        } else {
            self.yarnInverseFrequencies = nil
        }
    }

    /// Qwen-style partial RoPE: rotation confined to the first `rotaryDim`
    /// elements of each head, frequency divisor = rotaryDim.
    /// NeoX sub-dim rope where row `i` sits at `position + i * stride`.
    /// Used for the indexer's pooled blocks, which advance by the compression
    /// ratio rather than by one token.
    func encodeNeoxSubdimStrided(
        commandBuffer: MTLCommandBuffer,
        data: MTLBuffer,
        dataOffset: Int = 0,
        position: UInt32,
        headDim: UInt32,
        numHeads: UInt32,
        rotaryDim: UInt32,
        numTokens: UInt32,
        stride: UInt32,
        theta: Float
    ) throws {
        precondition(rotaryDim.isMultiple(of: 2), "rotary_dim must be even")
        precondition(rotaryDim <= headDim, "rotary_dim must not exceed head_dim")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(stridedNeoxSubdim)
        enc.setBuffer(data, offset: dataOffset, index: 0)
        var positionValue = position
        var headDimValue = headDim
        var numHeadsValue = numHeads
        var thetaValue = theta
        var rotaryValue = rotaryDim
        var strideValue = stride
        enc.setBytes(&positionValue, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&headDimValue, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&numHeadsValue, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&thetaValue, length: MemoryLayout<Float>.size, index: 4)
        enc.setBytes(&rotaryValue, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&strideValue, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreads(
            MTLSize(
                width: Int(rotaryDim / 2), height: Int(numHeads),
                depth: Int(numTokens)),
            threadsPerThreadgroup: MTLSize(
                width: min(
                    Int(rotaryDim / 2),
                    stridedNeoxSubdim.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeNeoxSubdim(
        commandBuffer: MTLCommandBuffer,
        data: MTLBuffer,
        dataOffset: Int = 0,
        position: UInt32,
        headDim: UInt32,
        numHeads: UInt32,
        rotaryDim: UInt32,
        numTokens: UInt32 = 1,
        theta: Float
    ) throws {
        precondition(rotaryDim.isMultiple(of: 2), "rotary_dim must be even")
        precondition(rotaryDim <= headDim, "rotary_dim must not exceed head_dim")
        if let frequencies = yarnInverseFrequencies {
            try encodeYaRNNeoxSubdim(
                commandBuffer: commandBuffer, data: data,
                dataOffset: dataOffset, position: position,
                headDim: headDim, numHeads: numHeads,
                rotaryDim: rotaryDim, numTokens: numTokens,
                frequencies: frequencies)
            return
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(neoxSubdim)
        encoder.setBuffer(data, offset: dataOffset, index: 0)
        var positionValue = position
        var headDimValue = headDim
        var numHeadsValue = numHeads
        var thetaValue = theta
        var rotaryDimValue = rotaryDim
        encoder.setBytes(&positionValue, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setBytes(&headDimValue, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&numHeadsValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.size, index: 4)
        encoder.setBytes(&rotaryDimValue, length: MemoryLayout<UInt32>.size, index: 5)
        dispatch(
            encoder: encoder,
            pipeline: neoxSubdim,
            pairs: Int(rotaryDim) / 2,
            heads: Int(numHeads),
            tokens: Int(numTokens))
        encoder.endEncoding()
    }

    private func encodeYaRNNeoxSubdim(
        commandBuffer: MTLCommandBuffer,
        data: MTLBuffer,
        dataOffset: Int,
        position: UInt32,
        headDim: UInt32,
        numHeads: UInt32,
        rotaryDim: UInt32,
        numTokens: UInt32,
        frequencies: MTLBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(yarnNeoxSubdim)
        encoder.setBuffer(data, offset: dataOffset, index: 0)
        encoder.setBuffer(frequencies, offset: 0, index: 1)
        var pos = position
        var dim = headDim
        var heads = numHeads
        var rotary = rotaryDim
        var magnitude = yarnAttentionFactor
        encoder.setBytes(&pos, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&rotary, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&magnitude, length: MemoryLayout<Float>.size, index: 6)
        dispatch(
            encoder: encoder, pipeline: yarnNeoxSubdim,
            pairs: Int(rotaryDim) / 2, heads: Int(numHeads),
            tokens: Int(numTokens))
        encoder.endEncoding()
    }

    func encodeDefaultNeox(
        commandBuffer: MTLCommandBuffer,
        data: MTLBuffer,
        dataOffset: Int = 0,
        position: UInt32,
        headDim: UInt32,
        numHeads: UInt32,
        numTokens: UInt32 = 1,
        theta: Float = 10_000.0
    ) throws {
        precondition(headDim.isMultiple(of: 2), "head_dim must be even")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
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
        dispatch(
            encoder: encoder,
            pipeline: pipeline,
            pairs: Int(headDim) / 2,
            heads: Int(numHeads),
            tokens: Int(numTokens))
        encoder.endEncoding()
    }

    func encodeProportionalNeox(
        commandBuffer: MTLCommandBuffer,
        data: MTLBuffer,
        dataOffset: Int = 0,
        position: UInt32,
        headDim: UInt32,
        numHeads: UInt32,
        rotatedPairs: UInt32,
        numTokens: UInt32 = 1,
        theta: Float = 1_000_000.0
    ) throws {
        precondition(headDim.isMultiple(of: 2), "head_dim must be even")
        precondition(
            rotatedPairs * 2 <= headDim,
            "rotatedPairs * 2 must not exceed head_dim")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
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
        // K4: dispatch the active pair count, not headDim/2. The kernel guards
        // `pair >= active_pairs` so the old headDim/2 dispatch was only
        // wasteful, not wrong — dispatching the true count keeps the guard
        // meaningful and avoids launching idle threads.
        dispatch(
            encoder: encoder,
            pipeline: pipeline,
            pairs: Int(rotatedPairs),
            heads: Int(numHeads),
            tokens: Int(numTokens))
        encoder.endEncoding()
    }

    private static func specializedPipeline(
        _ context: MetalContext,
        _ name: String,
        headDim: UInt32,
        numHeads: UInt32,
        rotatedPairs: UInt32 = 0
    ) throws
        -> MTLComputePipelineState
    {
        try context.pipeline(
            name,
            constants: [
                MetalFunctionConstant(index: 50, value: .uint32(headDim)),
                MetalFunctionConstant(index: 51, value: .uint32(numHeads)),
                MetalFunctionConstant(index: 52, value: .uint32(rotatedPairs)),
                MetalFunctionConstant(index: 53, value: .bool(true)),
            ])
    }

    private func defaultPipeline(
        headDim: UInt32,
        numHeads: UInt32
    ) -> MTLComputePipelineState {
        if headDim == 256 && numHeads == 16 { return defaultNeoxSWAQ }
        if headDim == 256 && numHeads == 8 { return defaultNeoxSWAK }
        return defaultNeox
    }

    private func proportionalPipeline(
        headDim: UInt32,
        numHeads: UInt32,
        rotatedPairs: UInt32
    ) -> MTLComputePipelineState {
        if headDim == 512 && rotatedPairs == 64 && numHeads == 16 {
            return proportionalNeoxFullQ
        }
        if headDim == 512 && rotatedPairs == 64 && numHeads == 2 {
            return proportionalNeoxFullK
        }
        return proportionalNeox
    }

    private func dispatch(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        pairs: Int,
        heads: Int,
        tokens: Int
    ) {
        let width = min(pairs, Int(pipeline.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreads(
            MTLSize(width: pairs, height: heads, depth: tokens),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
