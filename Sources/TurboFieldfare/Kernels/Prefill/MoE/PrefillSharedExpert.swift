import Foundation
import Metal

final class PrefillSharedExpert {
    private let shared: SharedExpertRuntime

    init(context: MetalContext, weightBits: Int = 8) throws {
        self.shared = try SharedExpertRuntime(context: context, weightBits: weightBits)
    }

    func encodeBlock(commandBuffer cb: MTLCommandBuffer,
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
                            yStrideElements: Int) throws {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(d > 0, "d must be positive")
        precondition(intermediate > 0, "intermediate must be positive")
        precondition(xStrideElements >= d, "x stride is too small")
        precondition(yStrideElements >= d, "y stride is too small")
        guard gate.rows == UInt32(intermediate), gate.cols == UInt32(d),
              up.rows == UInt32(intermediate), up.cols == UInt32(d),
              down.rows == UInt32(d), down.cols == UInt32(intermediate) else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "expected gate/up=(\(intermediate),\(d)) down=(\(d),\(intermediate))")
        }

        let halfBytes = MemoryLayout<Float16>.stride
        for row in 0..<queryCount {
            try shared.encode(commandBuffer: cb,
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
}
