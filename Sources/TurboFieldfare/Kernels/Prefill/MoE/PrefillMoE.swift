import Foundation
import Metal

final class PrefillMoE {
    private let reducePSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.reducePSO = try context.pipeline("prefill_moe_reduce_token_major")
    }

    func encodeReduceTokenMajor(commandBuffer: MTLCommandBuffer,
                                       routePartials: MTLBuffer,
                                       routePartialsOffset: Int = 0,
                                       routeWeights: MTLBuffer,
                                       routeWeightsOffset: Int = 0,
                                       h2: MTLBuffer,
                                       h2Offset: Int = 0,
                                       queryCount: UInt32,
                                       topK: UInt32,
                                       d: UInt32) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(topK > 0, "topK must be positive")
        precondition(d > 0, "D must be positive")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(reducePSO)
        enc.setBuffer(routePartials, offset: routePartialsOffset, index: 0)
        enc.setBuffer(routeWeights, offset: routeWeightsOffset, index: 1)
        enc.setBuffer(h2, offset: h2Offset, index: 2)
        var tVar = queryCount
        var topKVar = topK
        var dVar = d
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&topKVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreads(MTLSize(width: Int(d), height: Int(queryCount), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }

}
