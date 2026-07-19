import Foundation
import Metal

final class PrefillPerHeadNorm {
    private let psoBF16W: MTLComputePipelineState
    private let psoNoScale: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoBF16W = try context.pipeline("prefill_rmsnorm_bf16w_perhead_block")
        self.psoNoScale = try context.pipeline("prefill_rmsnorm_no_scale_perhead_block")
    }

    func encodeBF16W(commandBuffer: MTLCommandBuffer,
                            x: MTLBuffer,
                            xOffset: Int = 0,
                            weight: MTLBuffer,
                            weightOffset: Int = 0,
                            out: MTLBuffer,
                            outOffset: Int = 0,
                            queryCount: UInt32,
                            headDim: UInt32,
                            numHeads: UInt32,
                            tokenStrideElements: UInt32,
                            eps: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim > 0, "headDim must be positive")
        precondition(numHeads > 0, "numHeads must be positive")
        precondition(tokenStrideElements >= headDim * numHeads,
                     "token stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoBF16W)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var t = queryCount
        var hd = headDim
        var heads = numHeads
        var stride = tokenStrideElements
        var epsVar = eps
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 7)
        let threads = min(psoBF16W.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: Int(queryCount), depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeNoScale(commandBuffer: MTLCommandBuffer,
                              x: MTLBuffer,
                              xOffset: Int = 0,
                              out: MTLBuffer,
                              outOffset: Int = 0,
                              queryCount: UInt32,
                              headDim: UInt32,
                              numHeads: UInt32,
                              tokenStrideElements: UInt32,
                              eps: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim > 0, "headDim must be positive")
        precondition(numHeads > 0, "numHeads must be positive")
        precondition(tokenStrideElements >= headDim * numHeads,
                     "token stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoNoScale)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var t = queryCount
        var hd = headDim
        var heads = numHeads
        var stride = tokenStrideElements
        var epsVar = eps
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 6)
        let threads = min(psoNoScale.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: Int(queryCount), depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
