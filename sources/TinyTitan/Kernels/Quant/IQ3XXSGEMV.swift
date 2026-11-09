import Foundation
import Metal

/// Fused IQ3_XXS dequantise-and-GEMV: the gate/up matrices of the layers that
/// use this format, read in place.
final class IQ3XXSGEMV {
    private let pso: MTLComputePipelineState
    private let grid: MTLBuffer
    private let signs: MTLBuffer

    static let blockElements = 256
    static let blockBytes = 98

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("iq3xxs_gemv_simd")
        guard
            let gridBuffer = Self.makeBuffer(GGUFIQTables.iq3xxsGrid, device: context.device),
            let signBuffer = Self.makeBuffer(GGUFIQTables.ksigns, device: context.device)
        else {
            throw MetalError.bufferAllocationFailed("IQ3_XXS GEMV tables")
        }
        self.grid = gridBuffer
        self.signs = signBuffer
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
        encoder.setBuffer(grid, offset: 0, index: 1)
        encoder.setBuffer(signs, offset: 0, index: 2)
        encoder.setBuffer(x, offset: 0, index: 3)
        encoder.setBuffer(y, offset: 0, index: 4)
        var m = UInt32(rows)
        var n = UInt32(columns)
        encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 6)
        let rowsPerThreadgroup = 8
        let threadgroups = (rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * rowsPerThreadgroup, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func makeBuffer<T>(_ values: [T], device: MTLDevice) -> MTLBuffer? {
        values.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base, length: buffer.count * MemoryLayout<T>.stride,
                options: .storageModeShared)
        }
    }
}
