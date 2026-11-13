import Foundation
import Metal

/// Embedding lookup from a table kept at the checkpoint's own bf16.
///
/// The bf16 counterpart of `EmbedLookupInt4` (one row, decode) and
/// `PrefillEmbedLookupInt4` (a chunk of rows, prefill). An install that keeps
/// `embed_tokens` unquantized has no scales or biases beside it, so handing the
/// table to either affine lookup would read the bf16 bytes as packed nibbles and
/// the bytes after the row as scales -- a plausible-looking hidden state with
/// nothing to do with the token. The runner picks this one from the tensor's
/// dtype (`Model.embeddingWeightBits == 16`), never from the slot.
final class EmbedLookupBF16 {
    private let rowPipeline: MTLComputePipelineState
    private let blockPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.rowPipeline = try context.pipeline("embed_lookup_bf16")
        self.blockPipeline = try context.pipeline("prefill_embed_lookup_bf16_block")
    }

    /// One token's row into `out` (`d` halfs), scaled by `outScale`. A token
    /// at or past `vocab` writes zeros, as the affine lookups do.
    func encode(
        commandBuffer: MTLCommandBuffer,
        table: MTLBuffer, tableOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        tokenId: UInt32,
        d: UInt32,
        outScale: Float,
        vocab: UInt32
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(rowPipeline)
        enc.setBuffer(table, offset: tableOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var tokenVar = tokenId
        var dVar = d
        var scaleVar = outScale
        var vocabVar = vocab
        enc.setBytes(&tokenVar, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 4)
        enc.setBytes(&vocabVar, length: MemoryLayout<UInt32>.size, index: 5)
        let threads = min(Int(rowPipeline.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `t` tokens' rows into `out`, token-major (`[t, d]` halfs).
    func encodeBlock(
        commandBuffer: MTLCommandBuffer,
        table: MTLBuffer, tableOffset: Int = 0,
        tokens: MTLBuffer, tokensOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        t: UInt32,
        d: UInt32,
        outScale: Float,
        vocab: UInt32
    ) throws {
        guard t > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(blockPipeline)
        enc.setBuffer(table, offset: tableOffset, index: 0)
        enc.setBuffer(tokens, offset: tokensOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var tVar = t
        var dVar = d
        var scaleVar = outScale
        var vocabVar = vocab
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 5)
        enc.setBytes(&vocabVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreads(
            MTLSize(width: Int(d), height: Int(t), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}
