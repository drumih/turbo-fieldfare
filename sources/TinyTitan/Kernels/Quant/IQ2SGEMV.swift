import Foundation
import Metal

/// Fused IQ2_S dequantise-and-GEMV: reads the gate/up expert matrices in their
/// GGUF block form and writes half results, so the as-is path never materialises
/// a decoded matrix.
final class IQ2SGEMV {
    private let pso: MTLComputePipelineState
    private let grid: MTLBuffer

    static let blockElements = 256
    static let blockBytes = 82

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("iq2s_gemv_simd")
        guard let gridBuffer = Self.makeBuffer(GGUFIQTables.iq2sGrid, device: context.device)
        else {
            throw MetalError.bufferAllocationFailed("IQ2_S GEMV grid")
        }
        self.grid = gridBuffer
    }

    /// `rows` x `columns` matrix in IQ2_S blocks; `x` has `columns` halves.
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
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(y, offset: 0, index: 3)
        var m = UInt32(rows)
        var n = UInt32(columns)
        encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 5)
        let lanes = 32
        let rowsPerThreadgroup = 8
        let threadgroups = (rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: lanes * rowsPerThreadgroup, height: 1, depth: 1))
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
