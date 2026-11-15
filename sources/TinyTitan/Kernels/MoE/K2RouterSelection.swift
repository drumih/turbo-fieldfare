import Foundation
import Metal
import TinyTitanCore

/// K2-Horizon's router selection: sigmoid scores, a selection bias, top-k
/// renormalization and the family's scaling factor.
///
/// The Qwen selector in `MoE` is a softmax over the top-k logits with an
/// optional per-expert scale; K2's router adds its bias only to the ranking,
/// takes the weights from the un-biased sigmoid scores, normalizes over the
/// selected set and multiplies by `router_scaling_factor` (2.5). Those are
/// different arithmetic, not a flag on the same one, so it has its own kernel.
/// The same selection serves the 100-expert MoE (top-8) and the 64-expert MoVA
/// value router (top-4); both pass their counts in.
final class K2RouterSelection {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("k2_router_topk_select_kn")
    }

    /// Encode the selection. `logits` is `numExperts` floats (raw router
    /// output), `bias` is `numExperts` BF16 values, and the outputs are
    /// `topK` expert ids and BF16 weights.
    func encode(
        commandBuffer: MTLCommandBuffer,
        logits: MTLBuffer, logitsOffset: Int = 0,
        bias: MTLBuffer, biasOffset: Int = 0,
        outIndices: MTLBuffer, outIndicesOffset: Int = 0,
        outWeights: MTLBuffer, outWeightsOffset: Int = 0,
        numExperts: Int,
        topK: Int,
        scaling: Float
    ) throws {
        precondition(numExperts > 0, "a router with no experts has nothing to select")
        precondition((1...16).contains(topK), "top-\(topK) is outside the 16-slot bound")
        precondition(
            logits.length - logitsOffset >= numExperts * MemoryLayout<Float>.stride,
            "logits buffer must cover \(numExperts) Float")
        precondition(
            bias.length - biasOffset >= numExperts * MemoryLayout<UInt16>.stride,
            "bias buffer must cover \(numExperts) BF16")
        precondition(
            outIndices.length - outIndicesOffset >= topK * MemoryLayout<UInt32>.stride,
            "outIndices must cover \(topK) UInt32")
        precondition(
            outWeights.length - outWeightsOffset >= topK * MemoryLayout<UInt16>.stride,
            "outWeights must cover \(topK) BF16")

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(logits, offset: logitsOffset, index: 0)
        encoder.setBuffer(bias, offset: biasOffset, index: 1)
        encoder.setBuffer(outIndices, offset: outIndicesOffset, index: 2)
        encoder.setBuffer(outWeights, offset: outWeightsOffset, index: 3)
        var experts = UInt32(numExperts)
        var k = UInt32(topK)
        var scale = scaling
        encoder.setBytes(&experts, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 6)
        // One threadgroup, one working thread: the scan is serial, as the CPU
        // reference's insertion order is. 32 satisfies the SIMD width without
        // giving the idle lanes anything to do.
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
