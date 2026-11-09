import Foundation
import Metal

/// GPU dequantisation of IQ2_S blocks to bf16: the gate/up expert format on
/// most layers. Block is 82 bytes; the 1,024-entry grid is ggml's and is
/// uploaded once from `GGUFIQTables`.
final class DequantIQ2S {
    private let pso: MTLComputePipelineState
    private let grid: MTLBuffer

    static let blockElements = 256
    static let blockBytes = 82

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_iq2s_bf16")
        guard let gridBuffer = Self.makeBuffer(GGUFIQTables.iq2sGrid, device: context.device)
        else {
            throw MetalError.bufferAllocationFailed("IQ2_S grid")
        }
        self.grid = gridBuffer
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        blocks: MTLBuffer,
        blocksOffset: Int = 0,
        count: Int,
        out: MTLBuffer,
        outOffset: Int = 0
    ) throws {
        precondition(
            count % Self.blockElements == 0,
            "IQ2_S decodes whole blocks of \(Self.blockElements)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(blocks, offset: blocksOffset, index: 0)
        encoder.setBuffer(grid, offset: 0, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        var blockCount = UInt32(count / Self.blockElements)
        encoder.setBytes(&blockCount, length: MemoryLayout<UInt32>.size, index: 3)
        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        encoder.dispatchThreads(
            MTLSize(width: Int(blockCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Upload a table. `baseAddress` is checked rather than forced: the tables
    /// are non-empty by construction, but a future empty one should be a thrown
    /// error, not a crash.
    private static func makeBuffer<T>(_ values: [T], device: MTLDevice) -> MTLBuffer? {
        values.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return nil }
            return device.makeBuffer(
                bytes: base, length: buffer.count * MemoryLayout<T>.stride,
                options: .storageModeShared)
        }
    }
}
