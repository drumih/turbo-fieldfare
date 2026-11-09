import Foundation
import Metal

/// GPU dequantisation of IQ3_XXS blocks to bf16.
///
/// Second of the as-is decoders. The block is 98 bytes: an fp16 scale, 64 grid
/// indices and 32 scale/sign bytes. The grid and sign tables are ggml's own,
/// uploaded once from `GGUFIQTables`, so the shader and the CPU decoder share a
/// single source for them.
final class DequantIQ3XXS {
    private let pso: MTLComputePipelineState
    private let grid: MTLBuffer
    private let signs: MTLBuffer

    static let blockElements = 256
    static let blockBytes = 98

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_iq3xxs_bf16")
        guard
            let gridBuffer = Self.makeBuffer(GGUFIQTables.iq3xxsGrid, device: context.device),
            let signBuffer = Self.makeBuffer(GGUFIQTables.ksigns, device: context.device)
        else {
            throw MetalError.bufferAllocationFailed("IQ3_XXS tables")
        }
        self.grid = gridBuffer
        self.signs = signBuffer
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
            "IQ3_XXS decodes whole blocks of \(Self.blockElements)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(blocks, offset: blocksOffset, index: 0)
        encoder.setBuffer(grid, offset: 0, index: 1)
        encoder.setBuffer(signs, offset: 0, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var blockCount = UInt32(count / Self.blockElements)
        encoder.setBytes(&blockCount, length: MemoryLayout<UInt32>.size, index: 4)
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
