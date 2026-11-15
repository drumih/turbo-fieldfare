import Foundation
import Metal
import TinyTitanCore

/// Small elementwise kernels used by the Qwen 3.6 layer graph: the
/// full-attention output gate, the shared-expert scalar gate, and the plain
/// pre-norm residual add (architectures without a fused sandwich tail).
final class Elementwise {
    private let sigmoidGateMulPSO: MTLComputePipelineState
    private let softplusGateMulPSO: MTLComputePipelineState
    private let softplusGatePerHeadPSO: MTLComputePipelineState
    private let scalePSO: MTLComputePipelineState
    private let residualAddScaledPSO: MTLComputePipelineState
    private let sigmoidScalarMulPSO: MTLComputePipelineState
    private let residualAddPSO: MTLComputePipelineState
    private let splitQGatePSO: MTLComputePipelineState
    private let concatRowsPSO: MTLComputePipelineState
    // Hyper-connection (Gated Residual) stream plumbing.
    let hcMixReducePSO: MTLComputePipelineState
    let hcInjectPSO: MTLComputePipelineState
    private let sigmoidPSO: MTLComputePipelineState
    private let siluPSO: MTLComputePipelineState
    private let siluMulPSO: MTLComputePipelineState
    // PLE (n-gram) block.
    let pleScorePSO: MTLComputePipelineState
    let pleGatePSO: MTLComputePipelineState
    let pleBroadcastPSO: MTLComputePipelineState
    let pleConvPSO: MTLComputePipelineState
    let hcBroadcastPSO: MTLComputePipelineState
    let hcExpandPSO: MTLComputePipelineState
    let hcMeanPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.sigmoidGateMulPSO = try context.pipeline("sigmoid_gate_mul_fp16")
        self.softplusGateMulPSO = try context.pipeline("softplus_gate_mul_fp16")
        self.softplusGatePerHeadPSO = try context.pipeline(
            "softplus_gate_per_head_mul_fp16")
        self.scalePSO = try context.pipeline("scale_fp16")
        self.residualAddScaledPSO = try context.pipeline("residual_add_scaled_fp16")
        self.sigmoidScalarMulPSO = try context.pipeline("sigmoid_scalar_mul_fp16")
        self.residualAddPSO = try context.pipeline("residual_add_fp16")
        self.splitQGatePSO = try context.pipeline("split_q_gate_fp16")
        self.concatRowsPSO = try context.pipeline("concat_rows_fp16")
        self.hcMixReducePSO = try context.pipeline("hc_stream_mix_reduce_fp16")
        self.hcInjectPSO = try context.pipeline("hc_stream_inject_fp16")
        self.sigmoidPSO = try context.pipeline("sigmoid_fp16")
        self.siluPSO = try context.pipeline("silu_fp16")
        self.siluMulPSO = try context.pipeline("silu_mul_fp16")
        self.pleScorePSO = try context.pipeline("ple_stream_score_fp16")
        self.pleGatePSO = try context.pipeline("ple_signed_sqrt_gate_fp16")
        self.pleBroadcastPSO = try context.pipeline("ple_broadcast_scale_fp16")
        self.pleConvPSO = try context.pipeline("ple_dilated_depthwise_conv_fp16")
        self.hcBroadcastPSO = try context.pipeline("hc_stream_broadcast_fp16")
        self.hcExpandPSO = try context.pipeline("hc_stream_expand_fp16")
        hcMeanPSO = try context.pipeline("hc_stream_mean_fp16")
    }

    private func encodeUnary(
        _ pso: MTLComputePipelineState,
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int,
        out: MTLBuffer, outOffset: Int,
        count: Int,
        inScale: Float,
        outScale: Float?
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var n = UInt32(count)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 2)
        var inS = inScale
        enc.setBytes(&inS, length: MemoryLayout<Float>.size, index: 3)
        if var outS = outScale {
            enc.setBytes(&outS, length: MemoryLayout<Float>.size, index: 4)
        }
        let w = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// out = sigmoid(x). Standalone, unlike `sigmoid_gate_mul` which folds the
    /// gate into an existing value.
    func encodeSigmoid(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        count: Int,
        inScale: Float = 1, outScale: Float = 1
    ) throws {
        try encodeUnary(
            sigmoidPSO, commandBuffer: commandBuffer,
            x: x, xOffset: xOffset, out: out, outOffset: outOffset,
            count: count, inScale: inScale, outScale: outScale)
    }

    /// out = silu(x) = x * sigmoid(x), where `silu_mul` fuses a second operand
    /// the hyper-connection low-rank path does not have.
    func encodeSilu(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        count: Int, inScale: Float = 1
    ) throws {
        try encodeUnary(
            siluPSO, commandBuffer: commandBuffer,
            x: x, xOffset: xOffset, out: out, outOffset: outOffset,
            count: count, inScale: inScale, outScale: nil)
    }

    /// out = silu(gate) * up, the SwiGLU activation K2's dense and shared
    /// experts use. The Qwen families get the same arithmetic inside their
    /// fused expert kernels; this entry point is for callers that run the two
    /// projections as separate GEMVs.
    func encodeSiluMul(
        commandBuffer: MTLCommandBuffer,
        gate: MTLBuffer, gateOffset: Int = 0,
        up: MTLBuffer, upOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        count: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(siluMulPSO)
        encoder.setBuffer(gate, offset: gateOffset, index: 0)
        encoder.setBuffer(up, offset: upOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch(encoder, pipeline: siluMulPSO, threads: count)
        encoder.endEncoding()
    }

    /// packed [H, 2D] per-head [query ; gate] → q [H, D], gate [H, D].
    /// `rows` > 1 processes consecutive token rows (packed stride 2*H*D,
    /// output strides H*D).
    ///
    /// K5: all rows are dispatched from ONE encoder (a per-row dispatch loop
    /// with per-row buffer offsets) instead of creating one encoder per row.
    func encodeSplitQGate(
        commandBuffer: MTLCommandBuffer,
        packed: MTLBuffer, packedOffset: Int = 0,
        q: MTLBuffer, qOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        heads: Int, dim: Int, rows: Int = 1
    ) throws {
        let rowElems = heads * dim
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(splitQGatePSO)
        var headCount = UInt32(heads)
        var headDim = UInt32(dim)
        encoder.setBytes(&headCount, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&headDim, length: MemoryLayout<UInt32>.size, index: 4)
        for row in 0..<rows {
            encoder.setBuffer(packed, offset: packedOffset + row * 2 * rowElems * 2, index: 0)
            encoder.setBuffer(q, offset: qOffset + row * rowElems * 2, index: 1)
            encoder.setBuffer(gate, offset: gateOffset + row * rowElems * 2, index: 2)
            dispatch(encoder, pipeline: splitQGatePSO, threads: rowElems)
        }
        encoder.endEncoding()
    }

    /// out[i] *= sigmoid(gate[i])
    func encodeSigmoidGateMul(
        commandBuffer: MTLCommandBuffer,
        out: MTLBuffer, outOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        count: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(sigmoidGateMulPSO)
        encoder.setBuffer(out, offset: outOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: sigmoidGateMulPSO, threads: count)
        encoder.endEncoding()
    }

    /// out[i] *= softplus(gate[i]; beta = ln 2) — K2-Horizon's attention gate.
    func encodeSoftplusGateMul(
        commandBuffer: MTLCommandBuffer,
        out: MTLBuffer, outOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        count: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(softplusGateMulPSO)
        encoder.setBuffer(out, offset: outOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: softplusGateMulPSO, threads: count)
        encoder.endEncoding()
    }

    /// out[i] *= softplus(gate[head(i)]) — Laguna's per-head attention gate.
    func encodeSoftplusGatePerHeadMul(
        commandBuffer: MTLCommandBuffer,
        out: MTLBuffer, outOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        count: Int,
        headDim: Int
    ) throws {
        precondition(count % headDim == 0, "count must cover whole heads")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(softplusGatePerHeadPSO)
        encoder.setBuffer(out, offset: outOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var elementCount = UInt32(count)
        var width = UInt32(headDim)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch(encoder, pipeline: softplusGatePerHeadPSO, threads: count)
        encoder.endEncoding()
    }

    /// y[i] *= sigmoid(gate[0])
    func encodeSigmoidScalarMul(
        commandBuffer: MTLCommandBuffer,
        y: MTLBuffer, yOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        count: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(sigmoidScalarMulPSO)
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: sigmoidScalarMulPSO, threads: count)
        encoder.endEncoding()
    }

    /// out[i] = x[i] * factor
    func encodeScale(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        count: Int, factor: Float
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(scalePSO)
        encoder.setBuffer(out, offset: outOffset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        var elementCount = UInt32(count)
        var scale = factor
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 3)
        dispatch(encoder, pipeline: scalePSO, threads: count)
        encoder.endEncoding()
    }

    /// hidden[i] += factor * delta[i] — a scaled residual add.
    func encodeResidualAddScaled(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        delta: MTLBuffer, deltaOffset: Int = 0,
        count: Int, factor: Float
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(residualAddScaledPSO)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 0)
        encoder.setBuffer(delta, offset: deltaOffset, index: 1)
        var elementCount = UInt32(count)
        var scale = factor
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 3)
        dispatch(encoder, pipeline: residualAddScaledPSO, threads: count)
        encoder.endEncoding()
    }

    /// hidden[i] += delta[i]
    func encodeResidualAdd(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        delta: MTLBuffer, deltaOffset: Int = 0,
        count: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(residualAddPSO)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 0)
        encoder.setBuffer(delta, offset: deltaOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: residualAddPSO, threads: count)
        encoder.endEncoding()
    }

    func encodeConcatRows(
        commandBuffer: MTLCommandBuffer,
        lhs: MTLBuffer,
        rhs: MTLBuffer,
        out: MTLBuffer,
        rows: Int,
        dim: Int
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(concatRowsPSO)
        encoder.setBuffer(lhs, offset: 0, index: 0)
        encoder.setBuffer(rhs, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        var rowCount = UInt32(rows)
        var dimension = UInt32(dim)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch(encoder, pipeline: concatRowsPSO, threads: rows * dim)
        encoder.endEncoding()
    }

    private func dispatch(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        threads: Int
    ) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
