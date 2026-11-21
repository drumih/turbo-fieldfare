import Foundation
import Metal
import TinyTitanCore

/// Fused Q2_K / Q4_K GEMV. Reads GGUF blocks in place; used by the Inkling2
/// GGUF path and microbenchmarked independently of the decoder.
final class GGUFKQuantGEMV {
    private let q2k: MTLComputePipelineState
    private let q4k: MTLComputePipelineState
    private let context: MetalContext

    init(context: MetalContext) throws {
        self.context = context
        self.q2k = try context.pipeline("q2k_gemv")
        self.q4k = try context.pipeline("q4k_gemv")
    }

    func gemv(
        weights: [UInt8], x: [Float], rows: Int, columns: Int, type: GGUFType
    ) throws -> [Float] {
        guard columns > 0, columns % 256 == 0, rows > 0, x.count == columns else {
            throw GGUFDequantError.byteCountNotABlockMultiple(
                type: type.name, bytes: weights.count, blockBytes: Int(type.blockBytes))
        }
        let expected: Int
        switch type {
        case .q2K: expected = rows * (columns / 256) * 84
        case .q4K: expected = rows * (columns / 256) * 144
        default: throw GGUFDequantError.unsupportedType(type.name)
        }
        guard weights.count == expected else {
            throw GGUFDequantError.byteCountNotABlockMultiple(
                type: type.name, bytes: weights.count, blockBytes: Int(type.blockBytes))
        }
        guard
            let weightBuffer = context.device.makeBuffer(
                bytes: weights, length: weights.count, options: .storageModeShared),
            let xBuffer = context.device.makeBuffer(
                bytes: x, length: x.count * 4, options: .storageModeShared),
            let yBuffer = context.device.makeBuffer(
                length: rows * 4, options: .storageModeShared),
            let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw MetalError.bufferAllocationFailed("K-quant GEMV") }
        encoder.setComputePipelineState(type == .q2K ? q2k : q4k)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(xBuffer, offset: 0, index: 1)
        encoder.setBuffer(yBuffer, offset: 0, index: 2)
        var rowCount = UInt32(rows)
        var columnCount = UInt32(columns)
        encoder.setBytes(&rowCount, length: 4, index: 3)
        encoder.setBytes(&columnCount, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = yBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: pointer, count: rows))
    }
}
