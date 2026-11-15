import Metal
import TinyTitanCore

/// K27: shared two-row GEMV encode used by the MTP adapter in both
/// `AffineQuantGEMV` and `DequantInt4GEMV` (previously copy-pasted). The
/// two-row kernels share the buffer layout and dispatch; only the pipeline
/// selection and the alignment preconditions differ, and those stay at the
/// call sites.
func encodeTwoRowsShared(
    commandBuffer: MTLCommandBuffer,
    pipeline: MTLComputePipelineState,
    weights: MTLBuffer, weightsOffset: Int,
    scales: MTLBuffer, scalesOffset: Int,
    biases: MTLBuffer, biasesOffset: Int,
    x: MTLBuffer, xOffset: Int,
    y: MTLBuffer, yOffset: Int,
    m: UInt32, n: UInt32
) throws {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw MetalError.commandEncoderFailed
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(weights, offset: weightsOffset, index: 0)
    encoder.setBuffer(scales, offset: scalesOffset, index: 1)
    encoder.setBuffer(biases, offset: biasesOffset, index: 2)
    encoder.setBuffer(x, offset: xOffset, index: 3)
    encoder.setBuffer(y, offset: yOffset, index: 4)
    var rows = m
    var columns = n
    encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 5)
    encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 6)
    let rowsPerThreadgroup = 8
    encoder.dispatchThreadgroups(
        MTLSize(
            width: (Int(m) + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(
            width: 32 * rowsPerThreadgroup,
            height: 1, depth: 1))
    encoder.endEncoding()
}

/// Four-row GEMV encode. Same buffer layout and dispatch as the two-row
/// helper; only the kernel differs (it accumulates four rows per weight read).
func encodeFourRowsShared(
    commandBuffer: MTLCommandBuffer,
    pipeline: MTLComputePipelineState,
    weights: MTLBuffer, weightsOffset: Int,
    scales: MTLBuffer, scalesOffset: Int,
    biases: MTLBuffer, biasesOffset: Int,
    x: MTLBuffer, xOffset: Int,
    y: MTLBuffer, yOffset: Int,
    m: UInt32, n: UInt32
) throws {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw MetalError.commandEncoderFailed
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(weights, offset: weightsOffset, index: 0)
    encoder.setBuffer(scales, offset: scalesOffset, index: 1)
    encoder.setBuffer(biases, offset: biasesOffset, index: 2)
    encoder.setBuffer(x, offset: xOffset, index: 3)
    encoder.setBuffer(y, offset: yOffset, index: 4)
    var rows = m
    var columns = n
    encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 5)
    encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 6)
    let rowsPerThreadgroup = 8
    encoder.dispatchThreadgroups(
        MTLSize(
            width: (Int(m) + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(
            width: 32 * rowsPerThreadgroup,
            height: 1, depth: 1))
    encoder.endEncoding()
}

final class AffineQuantGEMV {
    let weightBits: Int
    private let pipeline: MTLComputePipelineState
    private let twoRowPipeline: MTLComputePipelineState
    private let fourRowPipeline: MTLComputePipelineState

    init(context: MetalContext, weightBits: Int) throws {
        precondition([2, 4, 8].contains(weightBits))
        self.weightBits = weightBits
        self.pipeline = try context.pipeline(
            "affine_quant_gemv_simd",
            constants: [
                MetalFunctionConstant(
                    index: 100,
                    value: .uint32(UInt32(weightBits)))
            ],
            maxTotalThreadsPerThreadgroup: 256)
        self.twoRowPipeline = try context.pipeline(
            "affine_quant_gemv2_simd",
            constants: [
                MetalFunctionConstant(
                    index: 100,
                    value: .uint32(UInt32(weightBits)))
            ],
            maxTotalThreadsPerThreadgroup: 256)
        self.fourRowPipeline = try context.pipeline(
            "affine_quant_gemv4_simd",
            constants: [
                MetalFunctionConstant(
                    index: 100,
                    value: .uint32(UInt32(weightBits)))
            ],
            maxTotalThreadsPerThreadgroup: 256)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        precondition(n.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(weightsOffset.isMultiple(of: MemoryLayout<UInt32>.alignment))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var rows = m
        var columns = n
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 6)
        let rowsPerThreadgroup = 8
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (Int(m) + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * rowsPerThreadgroup,
                height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeTwoRows(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        precondition(n.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(weightsOffset.isMultiple(of: MemoryLayout<UInt32>.alignment))
        try encodeTwoRowsShared(
            commandBuffer: commandBuffer,
            pipeline: twoRowPipeline,
            weights: weights, weightsOffset: weightsOffset,
            scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset,
            x: x, xOffset: xOffset,
            y: y, yOffset: yOffset,
            m: m, n: n)
    }

    /// Four activation rows per weight read. `y` is `[4, m]`, row-major.
    func encodeFourRows(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        precondition(n.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(weightsOffset.isMultiple(of: MemoryLayout<UInt32>.alignment))
        try encodeFourRowsShared(
            commandBuffer: commandBuffer,
            pipeline: fourRowPipeline,
            weights: weights, weightsOffset: weightsOffset,
            scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset,
            x: x, xOffset: xOffset,
            y: y, yOffset: yOffset,
            m: m, n: n)
    }
}

final class AffineQuantEmbeddingLookup {
    let weightBits: Int
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext, weightBits: Int) throws {
        precondition([2, 4, 8].contains(weightBits))
        self.weightBits = weightBits
        self.pipeline = try context.pipeline(
            "affine_quant_embedding_lookup",
            constants: [
                MetalFunctionConstant(
                    index: 100,
                    value: .uint32(UInt32(weightBits)))
            ])
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        table: MTLBuffer, tableOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        tokenId: UInt32, d: UInt32, outScale: Float,
        vocab: UInt32
    ) throws {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(tableOffset.isMultiple(of: MemoryLayout<UInt32>.alignment))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var token = tokenId
        var dimension = d
        var scale = outScale
        var vocabCount = vocab
        encoder.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 6)
        encoder.setBytes(&vocabCount, length: MemoryLayout<UInt32>.size, index: 7)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: width,
                height: 1, depth: 1))
        encoder.endEncoding()
    }
}
