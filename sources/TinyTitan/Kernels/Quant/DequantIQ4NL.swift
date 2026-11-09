import Foundation
import Metal

/// GPU dequantisation of IQ4_NL blocks to bf16.
///
/// The as-is GGUF path keeps the checkpoint in its own block format and hands
/// the existing bf16 GEMV and gather kernels a decoded tile. This is the first
/// of those decoders: the down-expert matrices and the PLE table are IQ4_NL, and
/// `dequant_iq4nl_bf16` is checked element-for-element against `GGUFDequant`'s
/// CPU decoder, which is itself bit-exact against ggml.
final class DequantIQ4NL {
    private let pso: MTLComputePipelineState

    /// IQ4_NL block geometry: 32 values in 18 bytes.
    static let blockElements = 32
    static let blockBytes = 18

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_iq4nl_bf16")
    }

    /// Decode `count` values (a multiple of 32) from `blocks` into `out`.
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
            "IQ4_NL decodes whole blocks of \(Self.blockElements)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(blocks, offset: blocksOffset, index: 0)
        encoder.setBuffer(out, offset: outOffset, index: 1)
        var blocks = UInt32(count / Self.blockElements)
        encoder.setBytes(&blocks, length: MemoryLayout<UInt32>.size, index: 2)
        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        encoder.dispatchThreads(
            MTLSize(width: Int(blocks), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
