import Foundation
import Metal

/// Gathers arbitrary rows of the PLE table, which stays in its GGUF IQ4_NL
/// blocks, and decodes them to bf16.
///
/// The table is 51.2 G weights (28.8 GB on disk, 102.4 GB as bf16), so it is
/// never materialised: a token asks for a few hundred rows and only those are
/// decoded.
final class PLEGatherIQ4NL {
    private let pso: MTLComputePipelineState

    static let rowElements = 160
    static let rowBytes = (160 / 32) * 18

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("ple_gather_iq4nl_bf16")
    }

    func encode(
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
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 3)
        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
