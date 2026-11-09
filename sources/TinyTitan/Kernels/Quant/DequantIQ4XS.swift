import Foundation
import Metal

/// GPU dequantisation of IQ4_XS blocks (256 values, 136 bytes) to bf16: layer
/// 47's gate/up experts. The nibble table is `kvalues_iq4nl`, shared with the
/// IQ4_NL kernel.
final class DequantIQ4XS {
    private let pso: MTLComputePipelineState

    static let blockElements = 256
    static let blockBytes = 136

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_iq4xs_bf16")
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
            "IQ4_XS decodes whole blocks of \(Self.blockElements)")
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
