import Foundation
import Metal

final class PrefillQKVEpilogue {
    private let perHeadNorm: PrefillPerHeadNorm
    private let rope: PrefillRoPE

    init(context: MetalContext) throws {
        self.perHeadNorm = try PrefillPerHeadNorm(context: context)
        self.rope = try PrefillRoPE(context: context)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       q: MTLBuffer,
                       qOffset: Int = 0,
                       k: MTLBuffer,
                       kOffset: Int = 0,
                       v: MTLBuffer,
                       vOffset: Int = 0,
                       qWeight: MTLBuffer,
                       qWeightOffset: Int = 0,
                       kWeight: MTLBuffer,
                       kWeightOffset: Int = 0,
                       startPosition: UInt32,
                       queryCount: UInt32,
                       headDim: UInt32,
                       numQHeads: UInt32,
                       numKVHeads: UInt32,
                       qTokenStrideElements: UInt32,
                       kvTokenStrideElements: UInt32,
                       theta: Float,
                       rotatedPairs: UInt32,
                       eps: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim > 0 && headDim % 2 == 0, "headDim must be positive and even")
        precondition(numQHeads > 0, "numQHeads must be positive")
        precondition(numKVHeads > 0, "numKVHeads must be positive")
        precondition(rotatedPairs > 0, "rotatedPairs must be positive")
        precondition(rotatedPairs * 2 <= headDim, "rotatedPairs must fit inside one head")
        precondition(qTokenStrideElements >= numQHeads * headDim,
                     "Q token stride is too small")
        precondition(kvTokenStrideElements >= numKVHeads * headDim,
                     "KV token stride is too small")

        perHeadNorm.encodeBF16W(commandBuffer: commandBuffer,
                                x: q,
                                xOffset: qOffset,
                                weight: qWeight,
                                weightOffset: qWeightOffset,
                                out: q,
                                outOffset: qOffset,
                                queryCount: queryCount,
                                headDim: headDim,
                                numHeads: numQHeads,
                                tokenStrideElements: qTokenStrideElements,
                                eps: eps)
        perHeadNorm.encodeBF16W(commandBuffer: commandBuffer,
                                x: k,
                                xOffset: kOffset,
                                weight: kWeight,
                                weightOffset: kWeightOffset,
                                out: k,
                                outOffset: kOffset,
                                queryCount: queryCount,
                                headDim: headDim,
                                numHeads: numKVHeads,
                                tokenStrideElements: kvTokenStrideElements,
                                eps: eps)
        perHeadNorm.encodeNoScale(commandBuffer: commandBuffer,
                                  x: v,
                                  xOffset: vOffset,
                                  out: v,
                                  outOffset: vOffset,
                                  queryCount: queryCount,
                                  headDim: headDim,
                                  numHeads: numKVHeads,
                                  tokenStrideElements: kvTokenStrideElements,
                                  eps: eps)

        if rotatedPairs * 2 == headDim {
            rope.encodeDefaultNeox(commandBuffer: commandBuffer,
                                   data: q,
                                   dataOffset: qOffset,
                                   startPosition: startPosition,
                                   queryCount: queryCount,
                                   headDim: headDim,
                                   numHeads: numQHeads,
                                   tokenStrideElements: qTokenStrideElements,
                                   theta: theta)
            rope.encodeDefaultNeox(commandBuffer: commandBuffer,
                                   data: k,
                                   dataOffset: kOffset,
                                   startPosition: startPosition,
                                   queryCount: queryCount,
                                   headDim: headDim,
                                   numHeads: numKVHeads,
                                   tokenStrideElements: kvTokenStrideElements,
                                   theta: theta)
        } else {
            rope.encodeProportionalNeox(commandBuffer: commandBuffer,
                                        data: q,
                                        dataOffset: qOffset,
                                        startPosition: startPosition,
                                        queryCount: queryCount,
                                        headDim: headDim,
                                        numHeads: numQHeads,
                                        rotatedPairs: rotatedPairs,
                                        tokenStrideElements: qTokenStrideElements,
                                        theta: theta)
            rope.encodeProportionalNeox(commandBuffer: commandBuffer,
                                        data: k,
                                        dataOffset: kOffset,
                                        startPosition: startPosition,
                                        queryCount: queryCount,
                                        headDim: headDim,
                                        numHeads: numKVHeads,
                                        rotatedPairs: rotatedPairs,
                                        tokenStrideElements: kvTokenStrideElements,
                                        theta: theta)
        }
    }
}
