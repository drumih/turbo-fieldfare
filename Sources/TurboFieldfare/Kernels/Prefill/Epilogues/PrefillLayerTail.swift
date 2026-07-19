import Foundation
import Metal

final class PrefillLayerTail {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("prefill_layer_tail_block")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       h2: MTLBuffer,
                       h2Offset: Int = 0,
                       h1: MTLBuffer,
                       h1Offset: Int = 0,
                       hidden: MTLBuffer,
                       hiddenOffset: Int = 0,
                       postFFN2Weight: MTLBuffer,
                       postFFN2WeightOffset: Int = 0,
                       postFFNWeight: MTLBuffer,
                       postFFNWeightOffset: Int = 0,
                       queryCount: UInt32,
                       d: UInt32,
                       h2StrideElements: UInt32,
                       h1StrideElements: UInt32,
                       hiddenStrideElements: UInt32,
                       eps: Float,
                       layerScalar: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(d > 0 && d <= 4096, "D must be in 1...4096")
        precondition(h2StrideElements >= d, "h2 stride is too small")
        precondition(h1StrideElements >= d, "h1 stride is too small")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(h2, offset: h2Offset, index: 0)
        enc.setBuffer(h1, offset: h1Offset, index: 1)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 2)
        enc.setBuffer(postFFN2Weight, offset: postFFN2WeightOffset, index: 3)
        enc.setBuffer(postFFNWeight, offset: postFFNWeightOffset, index: 4)
        var tVar = queryCount
        var dVar = d
        var h2Stride = h2StrideElements
        var h1Stride = h1StrideElements
        var hiddenStride = hiddenStrideElements
        var epsVar = eps
        var scaleVar = layerScalar
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&h2Stride, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&h1Stride, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&hiddenStride, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 10)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 11)

        let threads = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(queryCount), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
