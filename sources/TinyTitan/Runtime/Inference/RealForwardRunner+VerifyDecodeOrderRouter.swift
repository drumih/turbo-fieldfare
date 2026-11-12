import Foundation
import Metal

/// Decode-order routing for the wide-verify pass (docs/plan-wide-verify.md,
/// task 3).
extension RealForwardRunner {
    /// Routes the verify rows. In decode-order mode each row runs decode's own
    /// router GEMV and top-k selection in turn; they share one logits scratch,
    /// so the rows stay in order on `cb`. The batched prefill router reduces in
    /// another order, which moves routing weights by a few ULP. The identity
    /// bisect saw that as the first `mlp_out` difference (probe run 9).
    func encodeVerifyRouter(
        cb: MTLCommandBuffer, layer L: Int, routerView: TensorView,
        scratch: PrefillChunkScratchBuffers, rows t: Int, hiddenSize D: Int,
        perExpertScale: (buffer: any MTLBuffer, offset: Int)
    ) throws {
        let topK = cfg.topKExperts
        guard decodeOrderVerify else {
            try prefillRouter.encodeBlock(
                commandBuffer: cb,
                weights: routerView.buffer, weightsOffset: Int(routerView.offset),
                scales: routerView.buffer, scalesOffset: Int(routerView.scaleOffset),
                biases: routerView.buffer, biasesOffset: Int(routerView.biasOffset),
                hidden: scratch.routedX,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale.buffer,
                perExpertScaleOffset: perExpertScale.offset,
                outIndices: scratch.routeIDs, outWeights: scratch.routeWeights,
                queryCount: UInt32(t), numExperts: UInt32(cfg.numExperts),
                d: UInt32(D), topK: UInt32(topK), hiddenStrideElements: UInt32(D))
            return
        }
        let half = MemoryLayout<Float16>.stride
        for row in 0..<t {
            try moe.encodeRouter(
                commandBuffer: cb,
                weights: routerView.buffer, weightsOffset: Int(routerView.offset),
                scales: routerView.buffer, scalesOffset: Int(routerView.scaleOffset),
                biases: routerView.buffer, biasesOffset: Int(routerView.biasOffset),
                hidden: scratch.routedX, hiddenOffset: row * D * half,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale.buffer,
                perExpertScaleOffset: perExpertScale.offset,
                outIndices: scratch.routeIDs,
                outIndicesOffset: row * topK * MemoryLayout<UInt32>.stride,
                outWeights: scratch.routeWeights,
                outWeightsOffset: row * topK * half,
                numExperts: UInt32(cfg.numExperts), d: UInt32(D), topK: UInt32(topK))
        }
    }
}
