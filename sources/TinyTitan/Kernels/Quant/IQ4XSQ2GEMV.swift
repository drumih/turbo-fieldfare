import Foundation
import Metal

/// Fused GEMV for the two layer-exception expert formats: IQ4_XS (layer 47's
/// gate/up) and Q2_0 (layers 8/10/13 down).
final class IQ4XSQ2GEMV {
    private let iq4xs: MTLComputePipelineState
    private let q2: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.iq4xs = try context.pipeline("iq4xs_gemv_simd")
        self.q2 = try context.pipeline("q2_0_gemv_simd")
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        x: MTLBuffer,
        y: MTLBuffer,
        rows: Int,
        columns: Int,
        type: GGUFType
    ) throws {
        let pso: MTLComputePipelineState
        switch type {
        case .iq4XS: pso = iq4xs
        case .q2_0: pso = q2
        default:
            throw MetalError.invalidState("no fused GEMV for \(type.name)")
        }
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
        let rowsPerThreadgroup = 8
        let threadgroups = (rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * rowsPerThreadgroup, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
