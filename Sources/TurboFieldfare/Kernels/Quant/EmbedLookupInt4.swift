import Foundation
import Metal

/// 4-bit affine embedding lookup with fused output scale.
///
/// Gemma 4 ships a tied 4-bit MLX-affine embedding table (`embed_tokens.weight`
/// is `U32`/packed 4-bit with `weightBits=4` in the manifest). The same table
/// also drives the lm_head GEMV via `DequantInt4GEMV` over the transposed
/// access pattern. The `outScale` parameter fuses the `sqrt(hidden_size)`
/// post-embedding scale so the per-token dequant + scale
/// is one pass.
final class EmbedLookupInt4 {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("embed_lookup_int4")
    }

    /// Encodes the lookup. `table`, `scales`, `biases` typically live inside
    /// one resident blob — pass that buffer with the per-region offsets.
    /// Pass `outScale = 1.0` to write the raw dequantized row.
    func encode(commandBuffer: MTLCommandBuffer,
                       table:  MTLBuffer, tableOffset:  Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       out:    MTLBuffer, outOffset: Int = 0,
                       tokenId: UInt32,
                       d: UInt32,
                       outScale: Float) {
        precondition(d % UInt32(Quantization.groupSize) == 0,
                     "D must be a multiple of \(Quantization.groupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table,  offset: tableOffset,  index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(out,    offset: outOffset,    index: 3)
        var tokenVar = tokenId
        var dVar     = d
        var sVar     = outScale
        enc.setBytes(&tokenVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&sVar,     length: MemoryLayout<Float>.size,  index: 6)

        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: Int(d), height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}
