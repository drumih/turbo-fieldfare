import Foundation
import Metal

/// Gathers token-embedding rows from the checkpoint's Q6_K table and decodes
/// them to bf16. The table is 521 MB in blocks and is never materialised.
package final class GGUFEmbeddingGather {
    private let pso: MTLComputePipelineState
    package static let rowElements = 2_560
    package static let blockBytes = 210

    package init(context: MetalContext) throws {
        self.pso = try context.pipeline("gguf_embed_gather_q6k_bf16")
    }

    package func encode(
        commandBuffer: MTLCommandBuffer,
        table: MTLBuffer,
        tableOffset: Int = 0,
        rows: MTLBuffer,
        count: Int,
        out: MTLBuffer,
        outOffset: Int = 0
    ) throws {
        guard count > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(rows, offset: 0, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        var n = UInt32(count)
        var width = UInt32(Self.rowElements)
        encoder.setBytes(&n, length: 4, index: 3)
        encoder.setBytes(&width, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(pso.maxTotalThreadsPerThreadgroup), 256),
                height: 1, depth: 1))
        encoder.endEncoding()
    }
}
