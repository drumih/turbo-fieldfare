import Foundation
import Metal

public enum SharedExpertError: Error, CustomStringConvertible {
    case unsupportedWeightBits(Int)
    case dimensionMismatch(String)
    case scratchTooSmall(String)

    public var description: String {
        switch self {
        case .unsupportedWeightBits(let bits):
            return "SharedExpert unsupported weight bits: \(bits)"
        case .dimensionMismatch(let detail):
            return "SharedExpert dimension mismatch: \(detail)"
        case .scratchTooSmall(let detail):
            return "SharedExpert scratch too small: \(detail)"
        }
    }
}

public final class SharedExpertInt4 {
    private let int4: DequantInt4GEMV
    private let geluMulPSO: MTLComputePipelineState

    public init(context: MetalContext, siluActivation: Bool = false) throws {
        self.int4 = try DequantInt4GEMV(context: context)
        self.geluMulPSO = try context.pipeline(
            siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16")
    }

    public func encode(
        commandBuffer cb: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        gate: SharedExpertProjection,
        up: SharedExpertProjection,
        down: SharedExpertProjection,
        y: MTLBuffer, yOffset: Int = 0,
        scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
        scratchUp: MTLBuffer, scratchUpOffset: Int = 0,
        scratchAct: MTLBuffer, scratchActOffset: Int = 0
    ) throws {
        guard gate.rows == up.rows, gate.cols == up.cols,
            down.rows == gate.cols, down.cols == gate.rows
        else {
            throw SharedExpertError.dimensionMismatch(
                "gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols)) down=(\(down.rows),\(down.cols))"
            )
        }
        let intermediate = Int(gate.rows)
        let required = intermediate * MemoryLayout<Float16>.stride
        guard scratchGateOffset >= 0, scratchGateOffset + required <= scratchGate.length,
            scratchUpOffset >= 0, scratchUpOffset + required <= scratchUp.length,
            scratchActOffset >= 0, scratchActOffset + required <= scratchAct.length
        else {
            throw SharedExpertError.scratchTooSmall(
                "need \(required) bytes per intermediate buffer")
        }
        let inputBytes = Int(gate.cols) * MemoryLayout<Float16>.stride
        let outputBytes = Int(down.rows) * MemoryLayout<Float16>.stride
        guard xOffset >= 0, xOffset + inputBytes <= x.length else {
            throw SharedExpertError.scratchTooSmall("input range exceeds x buffer")
        }
        guard yOffset >= 0, yOffset + outputBytes <= y.length else {
            throw SharedExpertError.scratchTooSmall("output range exceeds y buffer")
        }

        try int4.encode(
            commandBuffer: cb,
            weights: gate.weights, weightsOffset: gate.weightsOffset,
            scales: gate.scales, scalesOffset: gate.scalesOffset,
            biases: gate.biases, biasesOffset: gate.biasesOffset,
            x: x, xOffset: xOffset,
            y: scratchGate, yOffset: scratchGateOffset,
            m: gate.rows, n: gate.cols)
        try int4.encode(
            commandBuffer: cb,
            weights: up.weights, weightsOffset: up.weightsOffset,
            scales: up.scales, scalesOffset: up.scalesOffset,
            biases: up.biases, biasesOffset: up.biasesOffset,
            x: x, xOffset: xOffset,
            y: scratchUp, yOffset: scratchUpOffset,
            m: up.rows, n: up.cols)

        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw SharedExpertError.dimensionMismatch("encoder alloc failed")
        }
        encoder.setComputePipelineState(geluMulPSO)
        encoder.setBuffer(scratchGate, offset: scratchGateOffset, index: 0)
        encoder.setBuffer(scratchUp, offset: scratchUpOffset, index: 1)
        encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 2)
        var count = UInt32(intermediate)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(geluMulPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: intermediate, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()

        try int4.encode(
            commandBuffer: cb,
            weights: down.weights, weightsOffset: down.weightsOffset,
            scales: down.scales, scalesOffset: down.scalesOffset,
            biases: down.biases, biasesOffset: down.biasesOffset,
            x: scratchAct, xOffset: scratchActOffset,
            y: y, yOffset: yOffset,
            m: down.rows, n: down.cols)
    }
}

/// The shared expert at the checkpoint's own bf16.
///
/// A promoted projection carries no scales or biases, so the affine kernels
/// cannot read it; the three matvecs go through `BF16GEMV` and the activation
/// runs on the same fp16 scratch and pipeline the affine path uses, which is
/// what keeps one activation kernel for both.
public final class BF16SharedExpert {
    private let gemv: BF16GEMV
    private let geluMulPSO: MTLComputePipelineState

    init(context: MetalContext, siluActivation: Bool = false) throws {
        self.gemv = try BF16GEMV(context: context)
        self.geluMulPSO = try context.pipeline(
            siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16")
    }

    func encode(
        commandBuffer cb: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        gate: SharedExpertProjection,
        up: SharedExpertProjection,
        down: SharedExpertProjection,
        y: MTLBuffer, yOffset: Int = 0,
        scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
        scratchUp: MTLBuffer, scratchUpOffset: Int = 0,
        scratchAct: MTLBuffer, scratchActOffset: Int = 0
    ) throws {
        guard gate.rows == up.rows, gate.cols == up.cols,
            down.rows == gate.cols, down.cols == gate.rows
        else {
            throw SharedExpertError.dimensionMismatch(
                "gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols)) down=(\(down.rows),\(down.cols))"
            )
        }
        try gemv.encode(
            commandBuffer: cb,
            weights: gate.weights, weightsOffset: gate.weightsOffset,
            x: x, xOffset: xOffset,
            y: scratchGate, yOffset: scratchGateOffset,
            m: gate.rows, n: gate.cols)
        try gemv.encode(
            commandBuffer: cb,
            weights: up.weights, weightsOffset: up.weightsOffset,
            x: x, xOffset: xOffset,
            y: scratchUp, yOffset: scratchUpOffset,
            m: up.rows, n: up.cols)
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw SharedExpertError.dimensionMismatch("encoder alloc failed")
        }
        encoder.setComputePipelineState(geluMulPSO)
        encoder.setBuffer(scratchGate, offset: scratchGateOffset, index: 0)
        encoder.setBuffer(scratchUp, offset: scratchUpOffset, index: 1)
        encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 2)
        var count = gate.rows
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(geluMulPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: Int(gate.rows), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        try gemv.encode(
            commandBuffer: cb,
            weights: down.weights, weightsOffset: down.weightsOffset,
            x: scratchAct, xOffset: scratchActOffset,
            y: y, yOffset: yOffset,
            m: down.rows, n: down.cols)
    }
}

public final class SharedExpertRuntime {
    private enum Implementation {
        case int4(SharedExpertInt4)
        case int8(SharedExpertInt8)
    }

    private let implementation: Implementation
    /// Built beside the affine path, because a promoted shared expert sits
    /// inside a slot that may still say 4 or 8 bits: the tensor decides.
    private let bf16: BF16SharedExpert
    /// The multi-row form (`encodeRows`): decode's GEMVs per row, weights read
    /// once, and the same activation kernel the single-row paths dispatch.
    private let multiRow: MultiRowGEMV
    private let activationPSO: MTLComputePipelineState
    public let weightBits: Int

    public init(
        context: MetalContext, weightBits: Int,
        siluActivation: Bool = false
    ) throws {
        self.weightBits = weightBits
        self.bf16 = try BF16SharedExpert(
            context: context, siluActivation: siluActivation)
        self.multiRow = MultiRowGEMV(context: context)
        self.activationPSO = try context.pipeline(
            siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16")
        // Only 4 and 8. A 6-bit arm used to route here and compose affine GEMVs,
        // but `AffineQuantGEMV` asserts `[4, 8]`, so it aborted the process
        // instead of failing the load. 6-bit is a withdrawn format and no width
        // outside `[4, 8]` can reach this constructor: a slot is refused at
        // manifest read (`ManifestReader.validateQuant`) and a per-tensor
        // override is refused while decoding (`SSDAIManifestQuantV1.init(from:)`).
        switch weightBits {
        case 4:
            self.implementation = .int4(
                try SharedExpertInt4(
                    context: context, siluActivation: siluActivation))
        case 8:
            self.implementation = .int8(
                try SharedExpertInt8(
                    context: context, siluActivation: siluActivation))
        default: throw SharedExpertError.unsupportedWeightBits(weightBits)
        }
    }

    public func encode(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        gate: SharedExpertProjection,
        up: SharedExpertProjection,
        down: SharedExpertProjection,
        y: MTLBuffer, yOffset: Int = 0,
        scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
        scratchUp: MTLBuffer, scratchUpOffset: Int = 0,
        scratchAct: MTLBuffer, scratchActOffset: Int = 0
    ) throws {
        if gate.dtype == 1 {
            try bf16.encode(
                commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                scratchAct: scratchAct, scratchActOffset: scratchActOffset)
            return
        }
        switch implementation {
        case .int4(let runtime):
            try runtime.encode(
                commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        case .int8(let runtime):
            try runtime.encode(
                commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        }
    }

    /// The multi-row GEMV format `encodeRows` reads a projection of `dtype`
    /// with, or nil where it has no multi-row form (the 8-bit path).
    func multiRowFormat(dtype: Int) -> MultiRowGEMV.Format? {
        if dtype == 1 { return .bf16 }
        if case .int4 = implementation { return .int4 }
        return nil
    }

    /// Compiles `encodeRows`'s pipelines for every row count in `rows`.
    func prewarmRows(dtype: Int, hidden: Int, intermediate: Int, rows: ClosedRange<Int>) throws {
        guard let format = multiRowFormat(dtype: dtype) else { return }
        try multiRow.prewarm(formats: [format, format], columns: hidden, rows: rows)
        try multiRow.prewarm(formats: [format], columns: intermediate, rows: rows)
    }

    /// `encode` for `rows` activation rows at once, weights read once: row `r`
    /// reads `x[xOffset + r * cols]` and writes `y[yOffset + r * yRowStride]`
    /// with the bytes `encode` gives that row alone. The scratch buffers hold
    /// `rows * intermediate` values each.
    ///
    /// Returns false, encoding nothing, for the 8-bit path: its fused
    /// gate/up/activation kernel has no multi-row form yet, so the caller keeps
    /// one `encode` per row there.
    func encodeRows(
        commandBuffer cb: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0, rows: Int,
        gate: SharedExpertProjection,
        up: SharedExpertProjection,
        down: SharedExpertProjection,
        y: MTLBuffer, yOffset: Int = 0, yRowStride: Int,
        scratchGate: MTLBuffer, scratchUp: MTLBuffer, scratchAct: MTLBuffer
    ) throws -> Bool {
        guard let format = multiRowFormat(dtype: gate.dtype) else { return false }
        guard gate.rows == up.rows, gate.cols == up.cols,
            down.rows == gate.cols, down.cols == gate.rows
        else {
            throw SharedExpertError.dimensionMismatch(
                "gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols)) down=(\(down.rows),\(down.cols))"
            )
        }
        let intermediate = Int(gate.rows)
        let hidden = Int(gate.cols)
        let required = rows * intermediate * MemoryLayout<Float16>.stride
        guard required <= scratchGate.length, required <= scratchUp.length,
            required <= scratchAct.length
        else {
            throw SharedExpertError.scratchTooSmall(
                "need \(required) bytes per intermediate buffer")
        }
        func segment(_ p: SharedExpertProjection, _ out: MTLBuffer, _ offset: Int, _ stride: Int)
            -> MultiRowGEMV.Segment
        {
            .init(
                weights: p.weights, weightsOffset: p.weightsOffset,
                scales: p.scales, scalesOffset: p.scalesOffset,
                biases: p.biases, biasesOffset: p.biasesOffset,
                format: format, rows: Int(p.rows),
                y: out, yOffset: offset, yRowStride: stride)
        }
        try multiRow.encode(
            commandBuffer: cb,
            segments: [
                segment(gate, scratchGate, 0, intermediate),
                segment(up, scratchUp, 0, intermediate),
            ],
            x: x, xOffset: xOffset, xRowStride: hidden,
            activationRows: rows, columns: hidden)
        // Elementwise, so all rows' activations are one dispatch.
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw SharedExpertError.dimensionMismatch("encoder alloc failed")
        }
        encoder.setComputePipelineState(activationPSO)
        encoder.setBuffer(scratchGate, offset: 0, index: 0)
        encoder.setBuffer(scratchUp, offset: 0, index: 1)
        encoder.setBuffer(scratchAct, offset: 0, index: 2)
        var count = UInt32(rows * intermediate)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(activationPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: rows * intermediate, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        try multiRow.encode(
            commandBuffer: cb,
            segments: [segment(down, y, yOffset, yRowStride)],
            x: scratchAct, xRowStride: intermediate,
            activationRows: rows, columns: intermediate)
        return true
    }
}
