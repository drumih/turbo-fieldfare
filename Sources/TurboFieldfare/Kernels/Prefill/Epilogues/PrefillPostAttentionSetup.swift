import Foundation
import Metal

final class PrefillPostAttentionSetup {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("prefill_post_attn_setup_block")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       hidden: MTLBuffer,
                       hiddenOffset: Int = 0,
                       attn: MTLBuffer,
                       attnOffset: Int = 0,
                       denseX: MTLBuffer,
                       denseXOffset: Int = 0,
                       routedX: MTLBuffer,
                       routedXOffset: Int = 0,
                       routerX: MTLBuffer,
                       routerXOffset: Int = 0,
                       postAttentionWeight: MTLBuffer,
                       postAttentionWeightOffset: Int = 0,
                       preFFNWeight: MTLBuffer,
                       preFFNWeightOffset: Int = 0,
                       preFFN2Weight: MTLBuffer,
                       preFFN2WeightOffset: Int = 0,
                       queryCount: UInt32,
                       d: UInt32,
                       hiddenStrideElements: UInt32,
                       attnStrideElements: UInt32,
                       denseStrideElements: UInt32,
                       routedStrideElements: UInt32,
                       routerStrideElements: UInt32,
                       eps: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(d > 0 && d <= 4096, "D must be in 1...4096")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        precondition(attnStrideElements >= d, "attention stride is too small")
        precondition(denseStrideElements >= d, "dense stride is too small")
        precondition(routedStrideElements >= d, "routed stride is too small")
        precondition(routerStrideElements >= d, "router stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
        enc.setBuffer(attn, offset: attnOffset, index: 1)
        enc.setBuffer(denseX, offset: denseXOffset, index: 2)
        enc.setBuffer(routedX, offset: routedXOffset, index: 3)
        enc.setBuffer(routerX, offset: routerXOffset, index: 4)
        enc.setBuffer(postAttentionWeight, offset: postAttentionWeightOffset, index: 5)
        enc.setBuffer(preFFNWeight, offset: preFFNWeightOffset, index: 6)
        enc.setBuffer(preFFN2Weight, offset: preFFN2WeightOffset, index: 7)
        var tVar = queryCount
        var dVar = d
        var hiddenStride = hiddenStrideElements
        var attnStride = attnStrideElements
        var denseStride = denseStrideElements
        var routedStride = routedStrideElements
        var routerStride = routerStrideElements
        var epsVar = eps
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&hiddenStride, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&attnStride, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&denseStride, length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBytes(&routedStride, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&routerStride, length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 15)

        let threads = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(queryCount), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
