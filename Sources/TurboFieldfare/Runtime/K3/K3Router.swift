import Foundation
import Metal

/// K3 FP32 sigmoid router. The GPU kernel computes `scores = sigmoid(W_gate
/// . x)` (and optionally `scores + e_score_correction_bias`) into small
/// shared readback buffers — 896 floats is 3.5 KB, so there is deliberately
/// no GPU top-k. Selection and renormalization run on the CPU in
/// `selectTopK`, exactly as `KimiMoEGate` does:
///
///   keys    = sigmoid(logits) + bias          (bias only affects selection)
///   weights = gather(sigmoid(logits), topK)   (UNBIASED scores)
///   weights /= sum(weights) + 1e-20; weights *= routed_scaling_factor (1.0)
final class K3Router {
    private static let rowsPerThreadgroup = 4
    private static let realDecodeNumExperts: UInt32 = 896
    private static let realDecodeD: UInt32 = 7168
    private static let realDecodeConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 57, value: .uint32(realDecodeNumExperts)),
        MetalFunctionConstant(index: 58, value: .uint32(realDecodeD)),
        MetalFunctionConstant(index: 59, value: .bool(true)),
    ]

    private let pipeline: MTLComputePipelineState
    private let specializedPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = K3MetalLibrary.shared
        self.pipeline = try library.pipeline(
            device: context.device, name: "k3_router_gemv_f32",
            maxTotalThreadsPerThreadgroup: 512)
        self.specializedPipeline = try library.pipeline(
            device: context.device, name: "k3_router_gemv_f32",
            constants: Self.realDecodeConstants,
            maxTotalThreadsPerThreadgroup: 512)
    }

    /// Encodes the sigmoid router GEMV. `gate` is FP32 row-major
    /// `[numExperts, d]`, `x` is the FP32 upcast hidden state `[d]`, `bias`
    /// is `e_score_correction_bias` `[numExperts]`. `scores` receives the raw
    /// sigmoid scores; when `scoresBiased` is non-nil it additionally
    /// receives `scores + bias` (selection keys) — pass nil to skip.
    func encodeScores(commandBuffer: MTLCommandBuffer,
                      gate: MTLBuffer, gateOffset: Int = 0,
                      x: MTLBuffer, xOffset: Int = 0,
                      bias: MTLBuffer, biasOffset: Int = 0,
                      scores: MTLBuffer, scoresOffset: Int = 0,
                      scoresBiased: MTLBuffer? = nil,
                      scoresBiasedOffset: Int = 0,
                      numExperts: UInt32, d: UInt32) {
        precondition(numExperts > 0)
        precondition(d > 0)
        precondition(gateOffset % MemoryLayout<Float>.stride == 0)
        precondition(xOffset % MemoryLayout<Float>.stride == 0)
        precondition(biasOffset % MemoryLayout<Float>.stride == 0)
        precondition(scoresOffset % MemoryLayout<Float>.stride == 0)
        var expertCount = numExperts
        var dimension = d
        var writeBiased: UInt32 = scoresBiased != nil ? 1 : 0
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            numExperts == Self.realDecodeNumExperts && d == Self.realDecodeD
                ? specializedPipeline : pipeline)
        encoder.setBuffer(gate, offset: gateOffset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(scores, offset: scoresOffset, index: 2)
        encoder.setBuffer(bias, offset: biasOffset, index: 3)
        // The kernel requires a valid binding even when the biased output is
        // disabled; reuse the scores buffer as a dummy sink.
        encoder.setBuffer(scoresBiased ?? scores,
                          offset: scoresBiased != nil ? scoresBiasedOffset : scoresOffset,
                          index: 4)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&writeBiased, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(numExperts) + Self.rowsPerThreadgroup - 1)
                        / Self.rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// CPU top-k selection + renormalize, mirroring `KimiMoEGate.forward`
    /// for `num_expert_group == 1` (grouped top-k is a no-op on K3).
    /// Returns indices in descending selection-key order (ties keep the
    /// lower expert index first, the deterministic rule the CPU reference
    /// uses) and the corresponding renormalized unbiased weights.
    static func selectTopK(scores: [Float],
                           bias: [Float],
                           topK: Int,
                           scalingFactor: Float = 1.0)
        -> (indices: [UInt32], weights: [Float])
    {
        precondition(scores.count == bias.count)
        precondition(topK >= 1 && topK <= scores.count)
        let count = scores.count
        var keys = [Float](repeating: 0, count: count)
        for i in 0..<count {
            keys[i] = scores[i] + bias[i]
        }
        let order = (0..<count).sorted { a, b in
            if keys[a] != keys[b] { return keys[a] > keys[b] }
            return a < b
        }
        var indices = [UInt32](repeating: 0, count: topK)
        var weights = [Float](repeating: 0, count: topK)
        var sum: Float = 0
        for slot in 0..<topK {
            let expert = order[slot]
            indices[slot] = UInt32(expert)
            weights[slot] = scores[expert]
            sum += scores[expert]
        }
        let denominator = sum + 1e-20
        for slot in 0..<topK {
            weights[slot] = weights[slot] / denominator * scalingFactor
        }
        return (indices, weights)
    }
}
