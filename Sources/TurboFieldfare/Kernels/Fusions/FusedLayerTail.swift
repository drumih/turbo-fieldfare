import Foundation
import Metal

/// Single-kernel Gemma 4 decoder-layer tail.
///
/// Equivalent to:
///   h2 = rmsnorm_bf16w(h2, post_feedforward_layernorm_2)
///   h12 = h1 + h2
///   h12 = rmsnorm_bf16w(h12, post_feedforward_layernorm)
///   hidden = (hidden + h12) * layer_scalar
final class FusedLayerTail {
    private let pso: MTLComputePipelineState
    private let specializedPSO: MTLComputePipelineState?

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("fused_layer_tail")
        self.specializedPSO = try? context.pipeline(
            "fused_layer_tail",
            constants: [
                MetalFunctionConstant(index: 80, value: .uint32(2816)),
                MetalFunctionConstant(index: 86, value: .bool(true)),
            ])
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                       h2: MTLBuffer,
                       h1: MTLBuffer,
                       hidden: MTLBuffer,
                       postFFN2Weight: MTLBuffer,
                       postFFN2WeightOffset: Int = 0,
                       postFFNWeight: MTLBuffer,
                       postFFNWeightOffset: Int = 0,
                       d: UInt32,
                       eps: Float,
                       layerScalar: Float) {
        precondition(d <= 4096,
                     "D > 4096 exceeds the fused-tail threadgroup scratch (kFusedMaxD)")
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState((d == 2816 ? specializedPSO : nil) ?? pso)
        enc.setBuffer(h2,             offset: 0,                    index: 0)
        enc.setBuffer(h1,             offset: 0,                    index: 1)
        enc.setBuffer(hidden,         offset: 0,                    index: 2)
        enc.setBuffer(postFFN2Weight, offset: postFFN2WeightOffset, index: 3)
        enc.setBuffer(postFFNWeight,  offset: postFFNWeightOffset,  index: 4)
        var dVar = d
        var epsVar = eps
        var scaleVar = layerScalar
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&epsVar,   length: MemoryLayout<Float>.size,  index: 6)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size,  index: 7)

        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
