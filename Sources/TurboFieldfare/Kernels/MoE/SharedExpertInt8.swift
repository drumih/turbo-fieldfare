import Foundation
import Metal

/// Per-projection 8-bit affine weights for the Gemma 4 shared (dense) MLP.
///
/// The Gemma 4 26B-A4B manifest stores `mlp.gate/up/down_proj.weight` at 8-bit
/// MLX affine (group=64, BF16 scale + BF16 bias). Shapes:
///   gate: [F=2112, D=2816]   up:   [F=2112, D=2816]   down: [D=2816, F=2112]
public struct SharedExpertProjection {
    public let weights: MTLBuffer
    public let scales:  MTLBuffer
    public let biases:  MTLBuffer
    public let weightsOffset: Int
    public let scalesOffset:  Int
    public let biasesOffset:  Int
    public let rows: UInt32
    public let cols: UInt32

    public init(weights: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                weightsOffset: Int = 0, scalesOffset: Int = 0, biasesOffset: Int = 0,
                rows: UInt32, cols: UInt32) {
        self.weights       = weights
        self.scales        = scales
        self.biases        = biases
        self.weightsOffset = weightsOffset
        self.scalesOffset  = scalesOffset
        self.biasesOffset  = biasesOffset
        self.rows          = rows
        self.cols          = cols
    }
}

public typealias SharedExpertInt8Proj = SharedExpertProjection

enum SharedExpertInt8Error: Error, CustomStringConvertible {
    case dimensionMismatch(String)
    case scratchTooSmall(String)

    public var description: String {
        switch self {
        case .dimensionMismatch(let s): return "SharedExpertInt8 dimension mismatch: \(s)"
        case .scratchTooSmall(let s):   return "SharedExpertInt8 scratch too small: \(s)"
        }
    }
}

/// Standalone 8-bit dense MLP — the "shared expert" branch of Gemma 4's
/// parallel MoE. Runs on the same hidden as the routed
/// branch, in parallel; their outputs are summed downstream.
///
///     y = down(gelu_pytorch_tanh(gate(x)) * up(x))
///
/// The gate, up, and activation work is fused. The caller provides one FP16
/// activation buffer of length F, which can be reused across layers.
final class SharedExpertInt8 {
    private let int8: DequantInt8GEMV
    private let fusedGateUpActPSO: MTLComputePipelineState
    private let specializedFusedGateUpActPSO: MTLComputePipelineState?

    init(context: MetalContext) throws {
        self.int8 = try DequantInt8GEMV(context: context)
        self.fusedGateUpActPSO = try context.pipeline("shared_int8_gate_up_act_simd")
        self.specializedFusedGateUpActPSO = try? context.pipeline(
            "shared_int8_gate_up_act_simd",
            constants: [
                MetalFunctionConstant(index: 70, value: .uint32(2112)),
                MetalFunctionConstant(index: 71, value: .uint32(2816)),
                MetalFunctionConstant(index: 72, value: .bool(true)),
                MetalFunctionConstant(index: 73, value: .uint32(8)),
            ])
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertInt8Proj,
                       up:   SharedExpertInt8Proj,
                       down: SharedExpertInt8Proj,
                       y: MTLBuffer, yOffset: Int = 0,
                       scratchAct:  MTLBuffer, scratchActOffset:  Int = 0) throws {
        guard gate.rows == up.rows, gate.cols == up.cols else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "gate/up shapes differ: gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols))")
        }
        guard down.rows == gate.cols, down.cols == gate.rows else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "down expected (\(gate.cols),\(gate.rows)), got (\(down.rows),\(down.cols))")
        }
        let needBytes = Int(gate.rows) * MemoryLayout<Float16>.size
        guard scratchActOffset >= 0, scratchActOffset + needBytes <= scratchAct.length else {
            throw SharedExpertInt8Error.scratchTooSmall(
                "scratchAct offset \(scratchActOffset) + needed \(needBytes) exceeds length \(scratchAct.length)")
        }
        let inputBytes = Int(gate.cols) * MemoryLayout<Float16>.size
        let outputBytes = Int(down.rows) * MemoryLayout<Float16>.size
        guard xOffset >= 0, xOffset + inputBytes <= x.length else {
            throw SharedExpertInt8Error.scratchTooSmall(
                "x offset \(xOffset) + needed \(inputBytes) exceeds length \(x.length)")
        }
        guard yOffset >= 0, yOffset + outputBytes <= y.length else {
            throw SharedExpertInt8Error.scratchTooSmall(
                "y offset \(yOffset) + needed \(outputBytes) exceeds length \(y.length)")
        }

        try encodePhase1(commandBuffer: cb,
                         x: x,
                         xOffset: xOffset,
                         gate: gate,
                         up: up,
                         scratchAct: scratchAct,
                         scratchActOffset: scratchActOffset)

        try encodeDown(commandBuffer: cb,
                       down: down,
                       y: y,
                       yOffset: yOffset,
                       scratchAct: scratchAct,
                       scratchActOffset: scratchActOffset)
    }

    func encodePhase1(commandBuffer cb: MTLCommandBuffer,
                             x: MTLBuffer,
                             xOffset: Int = 0,
                             gate: SharedExpertInt8Proj,
                             up: SharedExpertInt8Proj,
                             scratchAct: MTLBuffer,
                             scratchActOffset: Int = 0) throws {
        guard gate.rows == up.rows, gate.cols == up.cols else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "gate/up shapes differ: gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols))")
        }
        let needBytes = Int(gate.rows) * MemoryLayout<Float16>.size
        guard scratchActOffset >= 0, scratchActOffset + needBytes <= scratchAct.length else {
            throw SharedExpertInt8Error.scratchTooSmall(
                "scratchAct offset \(scratchActOffset) + needed \(needBytes) exceeds length \(scratchAct.length)")
        }
        let inputBytes = Int(gate.cols) * MemoryLayout<Float16>.size
        guard xOffset >= 0, xOffset + inputBytes <= x.length else {
            throw SharedExpertInt8Error.scratchTooSmall(
                "x offset \(xOffset) + needed \(inputBytes) exceeds length \(x.length)")
        }

        try encodeFusedGateUpAct(commandBuffer: cb,
                                 x: x,
                                 xOffset: xOffset,
                                 gate: gate,
                                 up: up,
                                 scratchAct: scratchAct,
                                 scratchActOffset: scratchActOffset)
    }

    func encodeDown(commandBuffer cb: MTLCommandBuffer,
                           down: SharedExpertInt8Proj,
                           y: MTLBuffer,
                           yOffset: Int = 0,
                           scratchAct: MTLBuffer,
                           scratchActOffset: Int = 0) throws {
        let inputBytes = Int(down.cols) * MemoryLayout<Float16>.size
        let outputBytes = Int(down.rows) * MemoryLayout<Float16>.size
        guard scratchActOffset >= 0, scratchActOffset + inputBytes <= scratchAct.length else {
            throw SharedExpertInt8Error.scratchTooSmall(
                "scratchAct offset \(scratchActOffset) + needed \(inputBytes) exceeds length \(scratchAct.length)")
        }
        guard yOffset >= 0, yOffset + outputBytes <= y.length else {
            throw SharedExpertInt8Error.scratchTooSmall(
                "y offset \(yOffset) + needed \(outputBytes) exceeds length \(y.length)")
        }
        int8.encode(commandBuffer: cb,
                    weights: down.weights, weightsOffset: down.weightsOffset,
                    scales:  down.scales,  scalesOffset:  down.scalesOffset,
                    biases:  down.biases,  biasesOffset:  down.biasesOffset,
                    x: scratchAct, xOffset: scratchActOffset,
                    y: y, yOffset: yOffset,
                    m: down.rows, n: down.cols)
    }

    private func encodeFusedGateUpAct(commandBuffer cb: MTLCommandBuffer,
                                      x: MTLBuffer,
                                      xOffset: Int,
                                      gate: SharedExpertInt8Proj,
                                      up: SharedExpertInt8Proj,
                                      scratchAct: MTLBuffer,
                                      scratchActOffset: Int) throws {
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw SharedExpertInt8Error.dimensionMismatch("encoder alloc failed")
        }
        enc.setComputePipelineState(
            (gate.rows == 2112 && gate.cols == 2816 ? specializedFusedGateUpActPSO : nil)
            ?? fusedGateUpActPSO)
        enc.setBuffer(gate.weights, offset: gate.weightsOffset, index: 0)
        enc.setBuffer(gate.scales,  offset: gate.scalesOffset,  index: 1)
        enc.setBuffer(gate.biases,  offset: gate.biasesOffset,  index: 2)
        enc.setBuffer(up.weights,   offset: up.weightsOffset,   index: 3)
        enc.setBuffer(up.scales,    offset: up.scalesOffset,    index: 4)
        enc.setBuffer(up.biases,    offset: up.biasesOffset,    index: 5)
        enc.setBuffer(x, offset: xOffset, index: 6)
        enc.setBuffer(scratchAct, offset: scratchActOffset, index: 7)
        var rows = gate.rows
        var cols = gate.cols
        enc.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&cols, length: MemoryLayout<UInt32>.size, index: 9)

        let rowsPerTG = 8
        let groups = (Int(gate.rows) + rowsPerTG - 1) / rowsPerTG
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32 * rowsPerTG,
                                                                height: 1,
                                                                depth: 1))
        enc.endEncoding()
    }
}
