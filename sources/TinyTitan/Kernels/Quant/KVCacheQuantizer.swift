import Metal
import TinyTitanCore

/// Affine per-group quantization used by the K/V cache. Each token row is
/// independently quantized in groups of 64 values so appends never rewrite
/// history and prompt snapshots remain simple byte copies.
final class KVCacheQuantizer {
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline("kv_cache_quantize_affine")
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        source: MTLBuffer,
        sourceOffset: Int = 0,
        sourceTokenStrideElements: Int,
        destination: KVView,
        tokenCount: Int,
        elementCount: Int
    ) throws {
        precondition(
            destination.precision.isQuantized,
            "KV quantizer requires 4-bit or 8-bit destination")
        precondition(tokenCount > 0, "tokenCount must be positive")
        precondition(elementCount > 0, "elementCount must be positive")
        precondition(
            sourceTokenStrideElements >= elementCount,
            "source token stride is too small")
        // The kernel derives each element's index from `thread_position.x`
        // (`flat = group * group_size + lid`), so the dispatch width *is* the
        // group size, and its two-entry partial arrays assume exactly two SIMD
        // groups. Both come from `quantizationGroupSize` today; asserting it here
        // means a KV layout with any other group size fails loudly instead of
        // quantizing every group from the wrong elements.
        precondition(
            destination.groupSize == KVCacheManager.quantizationGroupSize,
            "KV quantizer dispatches \(KVCacheManager.quantizationGroupSize) "
                + "threads but the destination uses groups of "
                + "\(destination.groupSize)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: sourceOffset, index: 0)
        encoder.setBuffer(destination.buffer, offset: destination.offset, index: 1)
        var sourceStride = UInt32(sourceTokenStrideElements)
        var destinationStride = UInt32(destination.stride)
        var valuesBytes = UInt32(destination.valueBytes)
        var elements = UInt32(elementCount)
        var bits = UInt32(destination.precision.rawValue)
        var groupSize = UInt32(destination.groupSize)
        encoder.setBytes(&sourceStride, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&destinationStride, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&valuesBytes, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&elements, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&bits, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&groupSize, length: MemoryLayout<UInt32>.size, index: 7)
        let groups = (elementCount + destination.groupSize - 1) / destination.groupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: tokenCount, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: KVCacheManager.quantizationGroupSize,
                height: 1, depth: 1))
        encoder.endEncoding()
    }
}
