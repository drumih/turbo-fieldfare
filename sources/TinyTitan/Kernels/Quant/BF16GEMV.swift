import Metal
import TinyTitanCore

/// GEMV over a weight matrix kept at the checkpoint's own bf16.
///
/// Its reason to exist is the 8-bit build: promoting a family to bf16 removes
/// the last of its quantization error, and for the small families that is
/// nearly free -- the seven promoted ones are 2% of the active parameters per
/// token and cost ~108 MB resident. Decode reads resident weights from RAM,
/// not from SSD, so the cost lands where there is bandwidth to spare.
///
/// Shares the affine kernels' launch geometry so `SlotGEMV` can pick between
/// them per tensor without the caller knowing which it got.
final class BF16GEMV {
    private let pipeline: MTLComputePipelineState
    private let floatPipeline: MTLComputePipelineState
    private let multiRowPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline(
            "bf16_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        self.floatPipeline = try context.pipeline(
            "bf16_gemv_float_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        self.multiRowPipeline = try context.pipeline(
            "bf16_gemv_multirow_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        precondition(
            n.isMultiple(of: 64),
            "bf16 GEMV expects a column count that is a multiple of 64")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(y, offset: yOffset, index: 2)
        var rows = m
        var columns = n
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 4)
        let rowsPerThreadgroup = 8
        let groups = (Int(m) + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * rowsPerThreadgroup,
                height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// A prompt's worth of rows in one dispatch: one threadgroup per (row
    /// block, token).
    ///
    /// Amortises launches only -- the bytes are the same as calling `encode`
    /// once per token -- and exists because a prefill chunk issues one dispatch
    /// per token per projection: ~180k for a 1500-token prompt on this family.
    /// Strides are in elements, so the caller can serve a chunk stored
    /// token-major.
    func encodeMultiRow(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32,
        tokens: Int,
        xStrideElements: Int,
        yStrideElements: Int
    ) throws {
        precondition(
            BF16PrefillDispatch.isEncodable(columns: Int(n)),
            "bf16 multirow GEMV expects a positive column count that is a multiple of 64")
        guard let grid = BF16PrefillDispatch.grid(rows: Int(m), tokens: tokens),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            return
        }
        encoder.setComputePipelineState(multiRowPipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(y, offset: yOffset, index: 2)
        var rows = m
        var columns = n
        var xStride = UInt32(xStrideElements)
        var yStride = UInt32(yStrideElements)
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&xStride, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&yStride, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: grid.rowBlocks, height: grid.tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: grid.threadsPerThreadgroup, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// `y` is `[m]` float: router logits that feed a selection, where rounding
    /// to half before the comparison could change the chosen experts.
    func encodeFloat(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        precondition(
            n.isMultiple(of: 64),
            "bf16 GEMV expects a column count that is a multiple of 64")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(floatPipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(y, offset: yOffset, index: 2)
        var rows = m
        var columns = n
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 4)
        let rowsPerThreadgroup = 8
        let groups = (Int(m) + rowsPerThreadgroup - 1) / rowsPerThreadgroup
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * rowsPerThreadgroup,
                height: 1, depth: 1))
        encoder.endEncoding()
    }
}
