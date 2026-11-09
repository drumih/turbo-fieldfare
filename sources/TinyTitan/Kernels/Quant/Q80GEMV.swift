import Foundation
import Metal

/// Fused Q8_0 dequantise-and-GEMV: shared-expert down projections and the
/// Q8_0 half of the shared gate/up.
final class Q80GEMV {
    private let pso: MTLComputePipelineState
    static let blockElements = 32
    static let blockBytes = 34

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("q8_0_gemv_simd")
    }

    func encode(
        commandBuffer: MTLCommandBuffer, weights: MTLBuffer, weightsOffset: Int = 0,
        x: MTLBuffer, y: MTLBuffer, rows: Int, columns: Int
    ) throws {
        precondition(columns % Self.blockElements == 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(x, offset: 0, index: 1)
        encoder.setBuffer(y, offset: 0, index: 2)
        var m = UInt32(rows)
        var n = UInt32(columns)
        encoder.setBytes(&m, length: 4, index: 3)
        encoder.setBytes(&n, length: 4, index: 4)
        let threadgroups = (rows + 7) / 8
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
