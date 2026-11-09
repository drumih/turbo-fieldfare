import Foundation
import Metal

/// Fused IQ4_NL dequantise-and-GEMV: the down projection and every expert path
/// that stores IQ4_NL, read in place.
final class IQ4NLGEMV {
    private let pso: MTLComputePipelineState

    static let blockElements = 32
    static let blockBytes = 18

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("iq4nl_gemv_simd")
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        x: MTLBuffer,
        y: MTLBuffer,
        rows: Int,
        columns: Int
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
        encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
        let lanes = 32
        let rowsPerThreadgroup = 8
        let threadgroups = (rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: lanes * rowsPerThreadgroup, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
