import Foundation
import Metal

/// GPU dequantisation of Q6_K blocks (256 values, 210 bytes) to bf16: the
/// embedding, the output head and the attention projections.
final class DequantQ6K {
    private let pso: MTLComputePipelineState

    static let blockElements = 256
    static let blockBytes = 210

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_q6k_bf16")
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        blocks: MTLBuffer,
        blocksOffset: Int = 0,
        count: Int,
        out: MTLBuffer,
        outOffset: Int = 0
    ) throws {
        precondition(
            count % Self.blockElements == 0,
            "Q6_K decodes whole blocks of \(Self.blockElements)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(blocks, offset: blocksOffset, index: 0)
        encoder.setBuffer(out, offset: outOffset, index: 1)
        var blockCount = UInt32(count / Self.blockElements)
        encoder.setBytes(&blockCount, length: MemoryLayout<UInt32>.size, index: 2)
        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        encoder.dispatchThreads(
            MTLSize(width: Int(blockCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
