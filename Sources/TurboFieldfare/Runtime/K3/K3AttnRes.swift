import Foundation
import Metal

/// K3 AttnRes (attention residuals). One encode per call site: scores the
/// ≤9 block vectors + prefix through the fused `norm.weight ⊙ proj.weight`
/// vector, fp32 softmax, and writes the weighted sum of the raw vectors.
/// The output REPLACES the stream (no residual add); with zero blocks it is
/// the prefix bit-exactly.
final class K3AttnRes {
    static let maxBlocks = 9

    private static let realHidden: UInt32 = 7168
    private static let realConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 82, value: .uint32(realHidden)),
        MetalFunctionConstant(index: 84, value: .bool(true)),
    ]

    private let pipeline: MTLComputePipelineState
    private let specializedPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = K3MetalLibrary.shared
        self.pipeline = try library.pipeline(
            device: context.device, name: "k3_attnres")
        self.specializedPipeline = try library.pipeline(
            device: context.device, name: "k3_attnres",
            constants: Self.realConstants)
    }

    /// `blocks` is the fp16 slab [numBlocks][hidden] (pass any valid buffer
    /// when numBlocks == 0 — it is not read), `prefix` fp16 [hidden],
    /// `scoreVector` fp32 [hidden] (norm.weight ⊙ proj.weight, fused at
    /// load), `out` fp16 [hidden].
    func encode(commandBuffer: MTLCommandBuffer,
                blocks: MTLBuffer, blocksOffset: Int = 0,
                prefix: MTLBuffer, prefixOffset: Int = 0,
                scoreVector: MTLBuffer, scoreVectorOffset: Int = 0,
                out: MTLBuffer, outOffset: Int = 0,
                hidden: UInt32, numBlocks: UInt32, eps: Float) {
        precondition(hidden > 0)
        precondition(numBlocks <= UInt32(Self.maxBlocks),
                     "numBlocks \(numBlocks) exceeds kernel staging (\(Self.maxBlocks))")
        var h = hidden
        var b = numBlocks
        var e = eps
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            hidden == Self.realHidden ? specializedPipeline : pipeline)
        encoder.setBuffer(blocks, offset: blocksOffset, index: 0)
        encoder.setBuffer(prefix, offset: prefixOffset, index: 1)
        encoder.setBuffer(scoreVector, offset: scoreVectorOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        encoder.setBytes(&h, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&b, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&e, length: MemoryLayout<Float>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
