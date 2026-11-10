import Foundation
import Metal

/// Batched IQ2_S GEMM: `Y[t][m] = sum_k A[t][k] * W[m][k]` with the weight in
/// IQ2_S blocks. Rows are 256-element block aligned, which is every IQ2_S
/// tensor in the checkpoint (the 640-wide down projections use 32/64-element
/// formats instead).
final class IQ2SGemm {
    private let pso: MTLComputePipelineState
    private let grid: MTLBuffer

    static let blockElements = 256
    static let tokenTile = 128

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("iq2s_gemm_bf16")
        guard let gridBuffer = Self.makeBuffer(GGUFIQTables.iq2sGrid, device: context.device)
        else {
            throw MetalError.bufferAllocationFailed("IQ2_S GEMM grid")
        }
        self.grid = gridBuffer
    }

    private static func makeBuffer<T>(_ values: [T], device: MTLDevice) -> MTLBuffer? {
        values.withUnsafeBytes { raw in
            guard let base = raw.baseAddress, !values.isEmpty else { return nil }
            return device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared)
        }
    }

    /// `weights` is `rows` x `columns` in IQ2_S blocks, `x` is tokens x columns
    /// in half, `out` is tokens x rows in bf16.
    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        x: MTLBuffer,
        out: MTLBuffer,
        rows: UInt32,
        columns: UInt32,
        tokens: UInt32
    ) throws {
        precondition(
            columns % UInt32(Self.blockElements) == 0,
            "IQ2_S GEMM needs a 256-element-aligned width, got \(columns)")
        precondition(tokens > 0, "IQ2_S GEMM needs at least one token")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(grid, offset: 0, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        var m = rows
        var k = columns
        var t = tokens
        encoder.setBytes(&m, length: 4, index: 4)
        encoder.setBytes(&k, length: 4, index: 5)
        encoder.setBytes(&t, length: 4, index: 6)
        let tokenBlocks = (Int(tokens) + Self.tokenTile - 1) / Self.tokenTile
        let threadgroups = MTLSize(
            width: Int(rows) * tokenBlocks, height: 1, depth: 1)
        encoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: MTLSize(width: Self.tokenTile, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
