import Foundation
import Metal
import MetalPerformanceShaders

final class Gemma4VisionKernels {
    private let device: MTLDevice
    private let convertBF16: MTLComputePipelineState
    private let addPositions: MTLComputePipelineState
    private let rmsWeightedRows: MTLComputePipelineState
    private let rmsNoScaleRows: MTLComputePipelineState
    private let rmsWeightedHeads: MTLComputePipelineState
    private let rmsNoScaleHeads: MTLComputePipelineState
    private let rope2D: MTLComputePipelineState
    private let attention: MTLComputePipelineState
    private let geluMultiply: MTLComputePipelineState
    private let residualAdd: MTLComputePipelineState
    private let poolStandardize: MTLComputePipelineState
    private let sanitizeFinite: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.device = context.device
        let library = try MetalContext.moduleLibrary(device: context.device, module: "vision")
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw MetalError.missingFunction(name)
            }
            return try context.device.makeComputePipelineState(function: function)
        }
        self.convertBF16 = try pipeline("vision_bf16_to_f16")
        self.addPositions = try pipeline("vision_add_axial_position_embedding")
        self.rmsWeightedRows = try pipeline("vision_rmsnorm_bf16_rows")
        self.rmsNoScaleRows = try pipeline("vision_rmsnorm_no_scale_rows")
        self.rmsWeightedHeads = try pipeline("vision_rmsnorm_bf16_heads")
        self.rmsNoScaleHeads = try pipeline("vision_rmsnorm_no_scale_heads")
        self.rope2D = try pipeline("vision_rope_2d_qk")
        self.attention = try pipeline("vision_attention_noncausal_online")
        self.geluMultiply = try pipeline("vision_gelu_tanh_multiply")
        self.residualAdd = try pipeline("vision_residual_add")
        self.poolStandardize = try pipeline("vision_pool_and_standardize")
        self.sanitizeFinite = try pipeline("vision_sanitize_finite")
    }

    func encodeDenseBF16(commandBuffer: MTLCommandBuffer,
                         input: MTLBuffer,
                         inputOffset: Int = 0,
                         weight: MTLBuffer,
                         weightOffset: Int,
                         convertedWeight: MTLBuffer,
                         output: MTLBuffer,
                         outputOffset: Int = 0,
                         rows: Int,
                         outputColumns: Int,
                         inputColumns: Int) {
        precondition(rows > 0 && outputColumns > 0 && inputColumns > 0)
        let weightElements = outputColumns * inputColumns
        precondition(convertedWeight.length >= weightElements * 2)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convertBF16)
        encoder.setBuffer(weight, offset: weightOffset, index: 0)
        encoder.setBuffer(convertedWeight, offset: 0, index: 1)
        var elementCount = UInt32(weightElements)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        dispatch1D(encoder, pipeline: convertBF16, count: weightElements)
        encoder.endEncoding()

        let leftDescriptor = MPSMatrixDescriptor(
            rows: rows,
            columns: inputColumns,
            rowBytes: inputColumns * MemoryLayout<Float16>.stride,
            dataType: .float16)
        let rightDescriptor = MPSMatrixDescriptor(
            rows: outputColumns,
            columns: inputColumns,
            rowBytes: inputColumns * MemoryLayout<Float16>.stride,
            dataType: .float16)
        let outputDescriptor = MPSMatrixDescriptor(
            rows: rows,
            columns: outputColumns,
            rowBytes: outputColumns * MemoryLayout<Float16>.stride,
            dataType: .float16)
        let left = MPSMatrix(buffer: input, offset: inputOffset, descriptor: leftDescriptor)
        let right = MPSMatrix(buffer: convertedWeight, offset: 0, descriptor: rightDescriptor)
        let result = MPSMatrix(buffer: output, offset: outputOffset, descriptor: outputDescriptor)
        let multiplication = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: rows,
            resultColumns: outputColumns,
            interiorColumns: inputColumns,
            alpha: 1,
            beta: 0)
        multiplication.encode(commandBuffer: commandBuffer,
                              leftMatrix: left,
                              rightMatrix: right,
                              resultMatrix: result)
    }

    func encodeAddPositions(commandBuffer: MTLCommandBuffer,
                            hidden: MTLBuffer,
                            positions: MTLBuffer,
                            patchCount: Int,
                            patchColumns: Int,
                            hiddenSize: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(addPositions)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(positions, offset: 0, index: 1)
        var p = UInt32(patchCount), pc = UInt32(patchColumns), d = UInt32(hiddenSize)
        encoder.setBytes(&p, length: 4, index: 2)
        encoder.setBytes(&pc, length: 4, index: 3)
        encoder.setBytes(&d, length: 4, index: 4)
        encoder.dispatchThreads(MTLSize(width: hiddenSize, height: patchCount, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
    }

    func encodeRMSRows(commandBuffer: MTLCommandBuffer,
                       input: MTLBuffer, inputOffset: Int = 0,
                       weight: MTLBuffer?, weightOffset: Int = 0,
                       output: MTLBuffer, outputOffset: Int = 0,
                       rows: Int, width: Int, eps: Float = 1e-6) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let pipeline = weight == nil ? rmsNoScaleRows : rmsWeightedRows
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        if let weight {
            encoder.setBuffer(weight, offset: weightOffset, index: 1)
            encoder.setBuffer(output, offset: outputOffset, index: 2)
            var r = UInt32(rows), w = UInt32(width), e = eps
            encoder.setBytes(&r, length: 4, index: 3)
            encoder.setBytes(&w, length: 4, index: 4)
            encoder.setBytes(&e, length: 4, index: 5)
        } else {
            encoder.setBuffer(output, offset: outputOffset, index: 1)
            var r = UInt32(rows), w = UInt32(width), e = eps
            encoder.setBytes(&r, length: 4, index: 2)
            encoder.setBytes(&w, length: 4, index: 3)
            encoder.setBytes(&e, length: 4, index: 4)
        }
        let threads = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRMSHeads(commandBuffer: MTLCommandBuffer,
                        input: MTLBuffer,
                        weight: MTLBuffer?, weightOffset: Int = 0,
                        output: MTLBuffer,
                        tokens: Int, heads: Int, headSize: Int,
                        eps: Float = 1e-6) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let pipeline = weight == nil ? rmsNoScaleHeads : rmsWeightedHeads
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        if let weight {
            encoder.setBuffer(weight, offset: weightOffset, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            var t = UInt32(tokens), h = UInt32(heads), d = UInt32(headSize), e = eps
            encoder.setBytes(&t, length: 4, index: 3)
            encoder.setBytes(&h, length: 4, index: 4)
            encoder.setBytes(&d, length: 4, index: 5)
            encoder.setBytes(&e, length: 4, index: 6)
        } else {
            encoder.setBuffer(output, offset: 0, index: 1)
            var t = UInt32(tokens), h = UInt32(heads), d = UInt32(headSize), e = eps
            encoder.setBytes(&t, length: 4, index: 2)
            encoder.setBytes(&h, length: 4, index: 3)
            encoder.setBytes(&d, length: 4, index: 4)
            encoder.setBytes(&e, length: 4, index: 5)
        }
        let threads = min(pipeline.maxTotalThreadsPerThreadgroup, 96)
        encoder.dispatchThreadgroups(MTLSize(width: tokens, height: heads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoPE(commandBuffer: MTLCommandBuffer,
                    query: MTLBuffer, key: MTLBuffer,
                    tokens: Int, patchColumns: Int,
                    heads: Int, headSize: Int, theta: Float = 100) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(rope2D)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(key, offset: 0, index: 1)
        var t = UInt32(tokens), pc = UInt32(patchColumns), h = UInt32(heads), d = UInt32(headSize), th = theta
        encoder.setBytes(&t, length: 4, index: 2)
        encoder.setBytes(&pc, length: 4, index: 3)
        encoder.setBytes(&h, length: 4, index: 4)
        encoder.setBytes(&d, length: 4, index: 5)
        encoder.setBytes(&th, length: 4, index: 6)
        dispatch1D(encoder, pipeline: rope2D, count: tokens * heads * headSize / 2)
        encoder.endEncoding()
    }

    func encodeAttention(commandBuffer: MTLCommandBuffer,
                         query: MTLBuffer, key: MTLBuffer, value: MTLBuffer,
                         output: MTLBuffer,
                         tokens: Int, heads: Int, headSize: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(attention)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(key, offset: 0, index: 1)
        encoder.setBuffer(value, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var t = UInt32(tokens), h = UInt32(heads), d = UInt32(headSize)
        encoder.setBytes(&t, length: 4, index: 4)
        encoder.setBytes(&h, length: 4, index: 5)
        encoder.setBytes(&d, length: 4, index: 6)
        let width = min(attention.maxTotalThreadsPerThreadgroup, 96)
        encoder.dispatchThreadgroups(MTLSize(width: tokens, height: heads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeGELUMultiply(commandBuffer: MTLCommandBuffer,
                            gate: MTLBuffer, up: MTLBuffer, count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(geluMultiply)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        var c = UInt32(count)
        encoder.setBytes(&c, length: 4, index: 2)
        dispatch1D(encoder, pipeline: geluMultiply, count: count)
        encoder.endEncoding()
    }

    func encodeResidualAdd(commandBuffer: MTLCommandBuffer,
                           hidden: MTLBuffer, hiddenOffset: Int = 0,
                           branch: MTLBuffer, branchOffset: Int = 0,
                           count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(residualAdd)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 0)
        encoder.setBuffer(branch, offset: branchOffset, index: 1)
        var c = UInt32(count)
        encoder.setBytes(&c, length: 4, index: 2)
        dispatch1D(encoder, pipeline: residualAdd, count: count)
        encoder.endEncoding()
    }

    /// The reference vision tower uses BF16 activations.  This bounded FP16
    /// implementation must saturate a temporary activation before a following
    /// norm can encounter an infinity (where `inf * 0` would become NaN).
    func encodeSanitize(commandBuffer: MTLCommandBuffer,
                        buffer: MTLBuffer,
                        offset: Int = 0,
                        count: Int) {
        guard count > 0, let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(sanitizeFinite)
        encoder.setBuffer(buffer, offset: offset, index: 0)
        var c = UInt32(count)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 1)
        dispatch1D(encoder, pipeline: sanitizeFinite, count: count)
        encoder.endEncoding()
    }

    func encodePool(commandBuffer: MTLCommandBuffer,
                    hidden: MTLBuffer,
                    standardBias: MTLBuffer, biasOffset: Int,
                    standardScale: MTLBuffer, scaleOffset: Int,
                    output: MTLBuffer,
                    patchColumns: Int, patchRows: Int, hiddenSize: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(poolStandardize)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(standardBias, offset: biasOffset, index: 1)
        encoder.setBuffer(standardScale, offset: scaleOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var pc = UInt32(patchColumns), pr = UInt32(patchRows), d = UInt32(hiddenSize)
        encoder.setBytes(&pc, length: 4, index: 4)
        encoder.setBytes(&pr, length: 4, index: 5)
        encoder.setBytes(&d, length: 4, index: 6)
        let softTokens = patchColumns / 3 * (patchRows / 3)
        let threads = min(poolStandardize.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreadgroups(MTLSize(width: softTokens, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func dispatch1D(_ encoder: MTLComputeCommandEncoder,
                            pipeline: MTLComputePipelineState,
                            count: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
