import Foundation
import Metal

/// Small elementwise kernels used by the Qwen 3.6 layer graph: the
/// full-attention output gate, the shared-expert scalar gate, and the plain
/// pre-norm residual add (architectures without Gemma's fused sandwich tail).
final class Elementwise {
    private let sigmoidGateMulPSO: MTLComputePipelineState
    private let sigmoidScalarMulPSO: MTLComputePipelineState
    private let residualAddPSO: MTLComputePipelineState
    private let splitQGatePSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.sigmoidGateMulPSO = try context.pipeline("sigmoid_gate_mul_fp16")
        self.sigmoidScalarMulPSO = try context.pipeline("sigmoid_scalar_mul_fp16")
        self.residualAddPSO = try context.pipeline("residual_add_fp16")
        self.splitQGatePSO = try context.pipeline("split_q_gate_fp16")
    }

    /// packed [H, 2D] per-head [query ; gate] → q [H, D], gate [H, D].
    /// `rows` > 1 processes consecutive token rows (packed stride 2*H*D,
    /// output strides H*D).
    func encodeSplitQGate(commandBuffer: MTLCommandBuffer,
                          packed: MTLBuffer, packedOffset: Int = 0,
                          q: MTLBuffer, qOffset: Int = 0,
                          gate: MTLBuffer, gateOffset: Int = 0,
                          heads: Int, dim: Int, rows: Int = 1) {
        let rowElems = heads * dim
        for row in 0..<rows {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(splitQGatePSO)
            encoder.setBuffer(packed, offset: packedOffset + row * 2 * rowElems * 2, index: 0)
            encoder.setBuffer(q, offset: qOffset + row * rowElems * 2, index: 1)
            encoder.setBuffer(gate, offset: gateOffset + row * rowElems * 2, index: 2)
            var headCount = UInt32(heads)
            var headDim = UInt32(dim)
            encoder.setBytes(&headCount, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&headDim, length: MemoryLayout<UInt32>.size, index: 4)
            dispatch(encoder, pipeline: splitQGatePSO, threads: rowElems)
            encoder.endEncoding()
        }
    }

    /// out[i] *= sigmoid(gate[i])
    func encodeSigmoidGateMul(commandBuffer: MTLCommandBuffer,
                              out: MTLBuffer, outOffset: Int = 0,
                              gate: MTLBuffer, gateOffset: Int = 0,
                              count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(sigmoidGateMulPSO)
        encoder.setBuffer(out, offset: outOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: sigmoidGateMulPSO, threads: count)
        encoder.endEncoding()
    }

    /// y[i] *= sigmoid(gate[0])
    func encodeSigmoidScalarMul(commandBuffer: MTLCommandBuffer,
                                y: MTLBuffer, yOffset: Int = 0,
                                gate: MTLBuffer, gateOffset: Int = 0,
                                count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(sigmoidScalarMulPSO)
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: sigmoidScalarMulPSO, threads: count)
        encoder.endEncoding()
    }

    /// hidden[i] += delta[i]
    func encodeResidualAdd(commandBuffer: MTLCommandBuffer,
                           hidden: MTLBuffer, hiddenOffset: Int = 0,
                           delta: MTLBuffer, deltaOffset: Int = 0,
                           count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(residualAddPSO)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 0)
        encoder.setBuffer(delta, offset: deltaOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: residualAddPSO, threads: count)
        encoder.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          threads: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
