import Foundation
import Metal
import TinyTitanCore

/// Prefill-only wrapper over the int8 shared expert.
///
/// The decode path calls the fused single-token GEMV once per row. That is the
/// right shape for one token and the wrong one for a 4K chunk: 2,505 rows means
/// 2,505 dispatches per layer, each re-reading the same int8 weights, measured
/// at 9.12 s of GPU over forty layers. The batched path below uses the prefill
/// affine matmul for gate, up and down — the shared expert's weights are int8
/// rows and its scales/biases are bf16 per group of 64, which is exactly that
/// kernel's layout — with one elementwise activation between them.
final class PrefillSharedExpert {
    private let shared: SharedExpertRuntime
    private let qmm: MTLComputePipelineState?
    private let act: MTLComputePipelineState?
    /// K5: the capability-gated MPP tensor-ops variant of the same matmul.
    /// Nil when the block is built for a test that wants the scalar path only;
    /// a non-nil candidate can still be unavailable (policy fallback), which
    /// `ffnTensorOpsSelection` reports.
    let tensorOps: MPPPrefillInt4QMM?

    init(
        context: MetalContext,
        weightBits: Int = 8,
        siluActivation: Bool = false,
        tensorOps: MPPPrefillInt4QMM? = nil
    ) throws {
        self.shared = try SharedExpertRuntime(
            context: context,
            weightBits: weightBits,
            siluActivation: siluActivation)
        self.tensorOps = tensorOps
        // Function constant 78 is FC_PREFILL_AFFINE_BITS; the kernel defaults
        // to 4 when it is unset, and the shared expert is int8.
        self.qmm = try? context.pipeline(
            "prefill_dequant_affine_qmm_f16_block",
            constants: [
                MetalFunctionConstant(
                    index: 78, value: .uint32(UInt32(weightBits)))
            ])
        // Function constant 77 is FC_PREFILL_ACT_SILU, and the shader defaults
        // to gelu when it is unset; this model's shared expert is SwiGLU.
        self.act = try? context.pipeline(
            "prefill_shared_act_block",
            constants: [
                MetalFunctionConstant(index: 77, value: .bool(siluActivation))
            ])
    }

    func encodeBlock(
        commandBuffer cb: MTLCommandBuffer,
        x: MTLBuffer,
        xOffset: Int = 0,
        y: MTLBuffer,
        yOffset: Int = 0,
        gate: SharedExpertInt8Proj,
        up: SharedExpertInt8Proj,
        down: SharedExpertInt8Proj,
        scratchGate: MTLBuffer,
        scratchGateOffset: Int = 0,
        scratchUp: MTLBuffer,
        scratchUpOffset: Int = 0,
        scratchAct: MTLBuffer,
        scratchActOffset: Int = 0,
        queryCount: Int,
        d: Int,
        intermediate: Int,
        xStrideElements: Int,
        yStrideElements: Int
    ) throws {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(d > 0, "d must be positive")
        precondition(intermediate > 0, "intermediate must be positive")
        precondition(xStrideElements >= d, "x stride is too small")
        precondition(yStrideElements >= d, "y stride is too small")
        guard gate.rows == UInt32(intermediate), gate.cols == UInt32(d),
            up.rows == UInt32(intermediate), up.cols == UInt32(d),
            down.rows == UInt32(d), down.cols == UInt32(intermediate)
        else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "expected gate/up=(\(intermediate),\(d)) down=(\(d),\(intermediate))")
        }

        // The batched kernel indexes X and Y by their logical strides, so it is
        // only correct when those strides are exactly K and N.
        // A promoted (bf16) shared expert carries no scales or biases, and the
        // batched kernel reads packed affine weights; fall to the per-row path,
        // which dispatches on the projection's dtype.
        let batched =
            FusedKernelPolicy.mayUseBatchedSharedExpert(
                dTypes: [gate.dtype, up.dtype, down.dtype])
            && ProcessInfo.processInfo.environment["TINYTITAN_PREFILL_SHARED_GEMM"] != "0"
            && queryCount >= 8
            && gate.cols % 64 == 0 && down.cols % 64 == 0
            && xStrideElements == Int(gate.cols)
            && yStrideElements == Int(down.rows)
            && qmm != nil && act != nil
        if batched {
            let tensorOpsSelection = ffnTensorOpsSelection(
                queryCount: queryCount,
                d: Int(gate.cols),
                intermediate: Int(gate.rows))
            if tensorOpsSelection?.selected == true, let tensorOps {
                try encodeBatchedTensorOps(
                    cb: cb, x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                    gate: gate, up: up, down: down,
                    scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                    scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                    scratchAct: scratchAct, scratchActOffset: scratchActOffset,
                    queryCount: queryCount, tensorOps: tensorOps)
            } else {
                try encodeBatched(
                    cb: cb, x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                    gate: gate, up: up, down: down,
                    scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                    scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                    scratchAct: scratchAct, scratchActOffset: scratchActOffset,
                    queryCount: queryCount)
            }
            return
        }

        let halfBytes = MemoryLayout<Float16>.stride
        for row in 0..<queryCount {
            try shared.encode(
                commandBuffer: cb,
                x: x,
                xOffset: xOffset + row * xStrideElements * halfBytes,
                gate: gate,
                up: up,
                down: down,
                y: y,
                yOffset: yOffset + row * yStrideElements * halfBytes,
                scratchGate: scratchGate,
                scratchGateOffset: scratchGateOffset,
                scratchUp: scratchUp,
                scratchUpOffset: scratchUpOffset,
                scratchAct: scratchAct,
                scratchActOffset: scratchActOffset)
        }
    }

    /// The policy's answer for the FFN's three matmuls, for the startup log.
    /// Nil only when the block was built without a candidate at all; a
    /// candidate that fell back still reports its fallback reason.
    var tensorOpsDecision: TensorOpsDecision? { tensorOps?.tensorOpsDecision }

    /// Whether the tensor-ops variant serves this block's gate, up and down
    /// matmuls. All three must qualify: the activation between them reads the
    /// same fp16 scratch, so a mixed path would only make the fallback harder
    /// to reason about. Nil when the block has no candidate.
    func ffnTensorOpsSelection(
        queryCount: Int, d: Int, intermediate: Int
    ) -> MPPPrefillInt4QMM.Selection? {
        guard let tensorOps else { return nil }
        // The batched path already proved X and Y are contiguous, and gate/up
        // write [rows, intermediate] and down writes [rows, d] scratch.
        let shapes = [
            MPPPrefillInt4QMM.Shape(
                tokenCount: queryCount, rows: intermediate, columns: d,
                weightBits: tensorOps.weightBits,
                xStrideElements: d, yStrideElements: intermediate),
            MPPPrefillInt4QMM.Shape(
                tokenCount: queryCount, rows: intermediate, columns: d,
                weightBits: tensorOps.weightBits,
                xStrideElements: d, yStrideElements: intermediate),
            MPPPrefillInt4QMM.Shape(
                tokenCount: queryCount, rows: d, columns: intermediate,
                weightBits: tensorOps.weightBits,
                xStrideElements: intermediate, yStrideElements: d),
        ]
        for shape in shapes {
            let selection = tensorOps.select(shape)
            if !selection.selected { return selection }
        }
        return MPPPrefillInt4QMM.Selection(
            selected: true,
            reason: "MPP gate/up/down, \(queryCount) rows, "
                + "\(tensorOps.weightBits)-bit")
    }

    private func encodeBatched(
        cb: MTLCommandBuffer,
        x: MTLBuffer,
        xOffset: Int,
        y: MTLBuffer,
        yOffset: Int,
        gate: SharedExpertInt8Proj,
        up: SharedExpertInt8Proj,
        down: SharedExpertInt8Proj,
        scratchGate: MTLBuffer,
        scratchGateOffset: Int,
        scratchUp: MTLBuffer,
        scratchUpOffset: Int,
        scratchAct: MTLBuffer,
        scratchActOffset: Int,
        queryCount: Int
    ) throws {
        try encodeQMM(
            cb: cb,
            weights: gate.weights, weightsOffset: gate.weightsOffset,
            scales: gate.scales, scalesOffset: gate.scalesOffset,
            biases: gate.biases, biasesOffset: gate.biasesOffset,
            input: x, inputOffset: xOffset,
            output: scratchGate, outputOffset: scratchGateOffset,
            tokenCount: queryCount, rows: Int(gate.rows), columns: Int(gate.cols))
        try encodeQMM(
            cb: cb,
            weights: up.weights, weightsOffset: up.weightsOffset,
            scales: up.scales, scalesOffset: up.scalesOffset,
            biases: up.biases, biasesOffset: up.biasesOffset,
            input: x, inputOffset: xOffset,
            output: scratchUp, outputOffset: scratchUpOffset,
            tokenCount: queryCount, rows: Int(up.rows), columns: Int(up.cols))
        try encodeActivation(
            cb: cb,
            gate: scratchGate, gateOffset: scratchGateOffset,
            up: scratchUp, upOffset: scratchUpOffset,
            out: scratchAct, outOffset: scratchActOffset,
            count: queryCount * Int(gate.rows))
        try encodeQMM(
            cb: cb,
            weights: down.weights, weightsOffset: down.weightsOffset,
            scales: down.scales, scalesOffset: down.scalesOffset,
            biases: down.biases, biasesOffset: down.biasesOffset,
            input: scratchAct, inputOffset: scratchActOffset,
            output: y, outputOffset: yOffset,
            tokenCount: queryCount, rows: Int(down.rows), columns: Int(down.cols))
    }

    /// K5: the same three matmuls on the MPP tensor-ops kernel. `select`
    /// already accepted all three shapes, so a refusal here is an internal
    /// inconsistency, not a shape to fall back from: the command encoder for
    /// gate/up may already have been recorded.
    private func encodeBatchedTensorOps(
        cb: MTLCommandBuffer,
        x: MTLBuffer,
        xOffset: Int,
        y: MTLBuffer,
        yOffset: Int,
        gate: SharedExpertInt8Proj,
        up: SharedExpertInt8Proj,
        down: SharedExpertInt8Proj,
        scratchGate: MTLBuffer,
        scratchGateOffset: Int,
        scratchUp: MTLBuffer,
        scratchUpOffset: Int,
        scratchAct: MTLBuffer,
        scratchActOffset: Int,
        queryCount: Int,
        tensorOps: MPPPrefillInt4QMM
    ) throws {
        try encodeTensorOpsMatmul(
            cb: cb, tensorOps: tensorOps,
            projection: gate,
            input: x, inputOffset: xOffset,
            output: scratchGate, outputOffset: scratchGateOffset,
            tokenCount: queryCount)
        try encodeTensorOpsMatmul(
            cb: cb, tensorOps: tensorOps,
            projection: up,
            input: x, inputOffset: xOffset,
            output: scratchUp, outputOffset: scratchUpOffset,
            tokenCount: queryCount)
        try encodeActivation(
            cb: cb,
            gate: scratchGate, gateOffset: scratchGateOffset,
            up: scratchUp, upOffset: scratchUpOffset,
            out: scratchAct, outOffset: scratchActOffset,
            count: queryCount * Int(gate.rows))
        try encodeTensorOpsMatmul(
            cb: cb, tensorOps: tensorOps,
            projection: down,
            input: scratchAct, inputOffset: scratchActOffset,
            output: y, outputOffset: yOffset,
            tokenCount: queryCount)
    }

    private func encodeTensorOpsMatmul(
        cb: MTLCommandBuffer,
        tensorOps: MPPPrefillInt4QMM,
        projection: SharedExpertInt8Proj,
        input: MTLBuffer,
        inputOffset: Int,
        output: MTLBuffer,
        outputOffset: Int,
        tokenCount: Int
    ) throws {
        let rows = Int(projection.rows)
        let columns = Int(projection.cols)
        let path = try tensorOps.encode(
            commandBuffer: cb,
            weights: projection.weights,
            weightsOffset: projection.weightsOffset,
            scales: projection.scales,
            scalesOffset: projection.scalesOffset,
            biases: projection.biases,
            biasesOffset: projection.biasesOffset,
            x: input,
            xOffset: inputOffset,
            y: output,
            yOffset: outputOffset,
            m: tokenCount,
            n: rows,
            k: columns,
            xStrideElements: columns,
            yStrideElements: rows)
        guard path == .affineThreadgroupF16 else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "MPP tensor-ops matmul declined a shape the block selected")
        }
    }

    private func encodeQMM(
        cb: MTLCommandBuffer,
        weights: MTLBuffer,
        weightsOffset: Int,
        scales: MTLBuffer,
        scalesOffset: Int,
        biases: MTLBuffer,
        biasesOffset: Int,
        input: MTLBuffer,
        inputOffset: Int,
        output: MTLBuffer,
        outputOffset: Int,
        tokenCount: Int,
        rows: Int,
        columns: Int
    ) throws {
        guard let qmm else { throw SharedExpertInt8Error.dimensionMismatch("no qmm") }
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw SharedExpertInt8Error.dimensionMismatch("no encoder")
        }
        enc.setComputePipelineState(qmm)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(input, offset: inputOffset, index: 3)
        enc.setBuffer(output, offset: outputOffset, index: 4)
        var t = UInt32(tokenCount)
        var n = UInt32(rows)
        var k = UInt32(columns)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: (tokenCount + 7) / 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
    }

    private func encodeActivation(
        cb: MTLCommandBuffer,
        gate: MTLBuffer,
        gateOffset: Int,
        up: MTLBuffer,
        upOffset: Int,
        out: MTLBuffer,
        outOffset: Int,
        count: Int
    ) throws {
        guard let act else { throw SharedExpertInt8Error.dimensionMismatch("no activation") }
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw SharedExpertInt8Error.dimensionMismatch("no encoder")
        }
        enc.setComputePipelineState(act)
        enc.setBuffer(gate, offset: gateOffset, index: 0)
        enc.setBuffer(up, offset: upOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var c = UInt32(count)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        let threads = 256
        enc.dispatchThreadgroups(
            MTLSize(width: (count + threads - 1) / threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
