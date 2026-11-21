import Foundation
import Metal
import TinyTitanCore

extension Inkling2GPUExpert {
    public func gemv(
        _ expert: Inkling2PackedExpert, x: [Float], rows: Int, columns: Int,
        w13: Bool
    ) throws -> [Float] {
        let packed = w13 ? expert.w13 : expert.w2
        let scales = w13 ? expert.w13Scales : expert.w2Scales
        let outer = w13 ? expert.w13Outer : expert.w2Outer
        precondition(x.count == columns)
        let pipeline = expert.format == .nvfp4 ? nvfp4 : mxfp8
        let packedBuffer = try upload(
            packed, cache: &packedCache, bytes: &packedBytes, label: "expert packed")
        let scaleBuffer = try upload(
            scales, cache: &scaleCache, bytes: &scaleBytes, label: "expert scales")
        let xBuffer = try upload(
            x, cache: &xCache, bytes: &xBytes, label: "expert x")
        let yBuffer = try output(
            bytes: rows * 4, cache: &yCache, bytes: &yBytes, label: "expert y")
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
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let pointer = yBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: pointer, count: rows))
    }

    /// One command buffer for a whole routed-expert stage. Each expert keeps
    /// its own packed/scales/output buffers; the activation vector is shared.
    public func gemvBatch(
        _ experts: [Inkling2PackedExpert], xs: [[Float]], rows: Int,
        columns: Int, w13: Bool
    ) throws -> [[Float]] {
        precondition(experts.count == xs.count)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        var outputs: [(MTLBuffer, MTLBuffer)] = []
        for (expert, x) in zip(experts, xs) {
            let buffers = try expertBuffers(expert)
            guard
                let xBuffer = context.device.makeBuffer(
                    bytes: x, length: x.count * 4, options: .storageModeShared),
                let yBuffer = context.device.makeBuffer(
                    length: rows * 4, options: .storageModeShared),
                let encoder = commandBuffer.makeComputeCommandEncoder()
            else { throw MetalError.bufferAllocationFailed("expert batch") }
            let pipeline = expert.format == .nvfp4 ? nvfp4 : mxfp8
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(
                w13 ? buffers.w13Packed : buffers.w2Packed, offset: 0, index: 0)
            encoder.setBuffer(
                w13 ? buffers.w13Scales : buffers.w2Scales, offset: 0, index: 1)
            var rowsValue = UInt32(rows)
            var columnsValue = UInt32(columns)
            if expert.format == .nvfp4 {
                encoder.setBuffer(
                    w13 ? buffers.w13Outer : buffers.w2Outer, offset: 0, index: 2)
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
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            outputs.append((xBuffer, yBuffer))
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return outputs.map { _, yBuffer in
            Array(
                UnsafeBufferPointer(
                    start: yBuffer.contents().bindMemory(to: Float.self, capacity: rows),
                    count: rows))
        }
    }

}
