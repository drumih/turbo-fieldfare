import Foundation
import Metal
import TinyTitanCore

/// GPU GEMV for one packed routed expert. The packed bytes are uploaded as-is;
/// dequantization happens inside the kernel, so no full-precision expert is
/// ever materialized.
/// unchecked-invariant: the Metal device/queue/pipelines are immutable and are
/// used from the single model thread; command buffers are created per call.
public final class Inkling2GPUExpert: @unchecked Sendable {
    private let context: MetalContext
    private let nvfp4: MTLComputePipelineState
    private let mxfp8: MTLComputePipelineState
    private let bf16Float: MTLComputePipelineState
    private let bf16FloatX: MTLComputePipelineState

    public init() throws {
        self.context = try MetalContext()
        self.nvfp4 = try context.pipeline("nvfp4_gemv")
        self.mxfp8 = try context.pipeline("mxfp8_gemv")
        self.bf16Float = try context.pipeline("bf16_gemv_float_simd")
        self.bf16FloatX = try context.pipeline("bf16_gemv_floatx")
    }

    public func gemvBF16(
        weights: Data, x: [Float], rows: Int, columns: Int
    ) throws -> [Float] {
        precondition(x.count == columns)
        guard
            let weightsBuffer = context.device.makeBuffer(
                bytes: [UInt8](weights), length: weights.count,
                options: .storageModeShared),
            let xBuffer = context.device.makeBuffer(
                bytes: x, length: x.count * 4, options: .storageModeShared),
            let yBuffer = context.device.makeBuffer(
                length: rows * 4, options: .storageModeShared),
            let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw MetalError.bufferAllocationFailed("inkling dense gemv") }
        encoder.setComputePipelineState(bf16FloatX)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
        encoder.setBuffer(xBuffer, offset: 0, index: 1)
        encoder.setBuffer(yBuffer, offset: 0, index: 2)
        var rowsValue = UInt32(rows)
        var columnsValue = UInt32(columns)
        encoder.setBytes(&rowsValue, length: 4, index: 3)
        encoder.setBytes(&columnsValue, length: 4, index: 4)
        let groups = (rows + 7) / 8
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = yBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: pointer, count: rows))
    }

    public func logits(weights: Data, hidden: [Float], rows: Int, columns: Int)
        throws -> [Float]
    {
        precondition(hidden.count == columns)
        let halves = hidden.map { Float16($0) }
        guard
            let weightsBuffer = context.device.makeBuffer(
                bytes: [UInt8](weights), length: weights.count,
                options: .storageModeShared),
            let xBuffer = context.device.makeBuffer(
                bytes: halves, length: halves.count * 2,
                options: .storageModeShared),
            let yBuffer = context.device.makeBuffer(
                length: rows * 4, options: .storageModeShared),
            let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw MetalError.bufferAllocationFailed("inkling logits") }
        encoder.setComputePipelineState(bf16Float)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 0)
        encoder.setBuffer(xBuffer, offset: 0, index: 1)
        encoder.setBuffer(yBuffer, offset: 0, index: 2)
        var rowsValue = UInt32(rows)
        var columnsValue = UInt32(columns)
        encoder.setBytes(&rowsValue, length: 4, index: 3)
        encoder.setBytes(&columnsValue, length: 4, index: 4)
        let groups = (rows + 7) / 8
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = yBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: pointer, count: rows))
    }

    public func gemv(
        _ expert: Inkling2PackedExpert, x: [Float], rows: Int, columns: Int,
        w13: Bool
    ) throws -> [Float] {
        let packed = w13 ? expert.w13 : expert.w2
        let scales = w13 ? expert.w13Scales : expert.w2Scales
        let outer = w13 ? expert.w13Outer : expert.w2Outer
        precondition(x.count == columns)
        let pipeline = expert.format == .nvfp4 ? nvfp4 : mxfp8
        guard
            let packedBuffer = context.device.makeBuffer(
                bytes: [UInt8](packed), length: packed.count, options: .storageModeShared),
            let scaleBuffer = context.device.makeBuffer(
                bytes: [UInt8](scales), length: scales.count, options: .storageModeShared),
            let xBuffer = context.device.makeBuffer(
                bytes: x, length: x.count * 4, options: .storageModeShared),
            let yBuffer = context.device.makeBuffer(
                length: rows * 4, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("inkling expert") }
        var rowsValue = UInt32(rows)
        var columnsValue = UInt32(columns)
        guard let commandBuffer = context.queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(packedBuffer, offset: 0, index: 0)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
        if expert.format == .nvfp4 {
            var outerValue = outer
            let outerBuffer = context.device.makeBuffer(
                bytes: &outerValue, length: 4, options: .storageModeShared)
            guard let outerBuffer else { throw MetalError.bufferAllocationFailed("outer") }
            encoder.setBuffer(outerBuffer, offset: 0, index: 2)
            encoder.setBuffer(xBuffer, offset: 0, index: 3)
            encoder.setBuffer(yBuffer, offset: 0, index: 4)
            encoder.setBytes(&rowsValue, length: 4, index: 5)
            encoder.setBytes(&columnsValue, length: 4, index: 6)
        } else {
            encoder.setBuffer(xBuffer, offset: 0, index: 2)
            encoder.setBuffer(yBuffer, offset: 0, index: 3)
            encoder.setBytes(&rowsValue, length: 4, index: 4)
            encoder.setBytes(&columnsValue, length: 4, index: 5)
        }
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = yBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: pointer, count: rows))
    }
}
