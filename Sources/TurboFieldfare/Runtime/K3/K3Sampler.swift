import Foundation
import Metal

/// GPU-resident sampler for Kimi K3: vocab 163,840, FP32 logits, NO softcap.
/// Mirrors the house `Sampler` exactly minus the softcap stage — policy order
/// is repetition penalty (host, in place) → softmax → top-p → top-k →
/// temperature draw, greedy is the GPU argmax path, and a fixed `seed` is
/// reproducible across token positions (same per-position splitmix64 mixing
/// as the house, via `Sampler.seedFor`).
///
/// `logits` is the fp32 [vocab] output of `K3LMHeadGEMV` in a
/// `.storageModeShared` buffer (the penalty path edits it in place); `probs`
/// is a preallocated fp32 [vocab] scratch; `outToken` holds one UInt32 read
/// after the command buffer completes.
final class K3Sampler {
    private let softmaxPSO: MTLComputePipelineState
    private let samplePSO: MTLComputePipelineState
    let vocab: Int

    init(context: MetalContext, vocab: Int = 163_840) throws {
        let library = K3MetalLibrary.shared
        self.softmaxPSO = try library.pipeline(
            device: context.device, name: "k3_softmax_f32",
            maxTotalThreadsPerThreadgroup: 256)
        self.samplePSO = try library.pipeline(
            device: context.device, name: "k3_sample_f32",
            maxTotalThreadsPerThreadgroup: 256)
        self.vocab = vocab
    }

    /// Encode the sampler onto `commandBuffer`; returns the path taken.
    @discardableResult
    func sample(commandBuffer: MTLCommandBuffer,
                logits: MTLBuffer,
                probs: MTLBuffer,
                history: [Int32],
                config: GenerationConfig,
                position: Int,
                outToken: MTLBuffer) -> SamplePath {
        let v = UInt32(vocab)

        let appliedPenalty = config.repetitionPenalty != 1.0 && !history.isEmpty
        if appliedPenalty {
            applyRepetitionPenaltyInPlace(logits: logits,
                                          history: history,
                                          penalty: config.repetitionPenalty)
        }

        encodeSoftmax(commandBuffer: commandBuffer, logits: logits, probs: probs, v: v)

        let isGreedy = config.temperature == 0
        let seed = Sampler.seedFor(config: config, position: position)
        encodeSample(commandBuffer: commandBuffer,
                     probs: probs, outToken: outToken, v: v,
                     temperature: isGreedy ? 0.0 : config.temperature,
                     topK: UInt32(config.topK ?? 0),
                     topP: config.topP ?? 1.0,
                     seed: seed,
                     position: UInt32(position))

        if appliedPenalty { return .hostPenalty }
        return isGreedy ? .greedyGPU : .gpuSampled
    }

    // MARK: - Kernel encodes

    private func encodeSoftmax(commandBuffer: MTLCommandBuffer,
                               logits: MTLBuffer,
                               probs: MTLBuffer,
                               v: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(softmaxPSO)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(probs, offset: 0, index: 1)
        var vVar = v
        encoder.setBytes(&vVar, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256,
                                                               height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeSample(commandBuffer: MTLCommandBuffer,
                              probs: MTLBuffer,
                              outToken: MTLBuffer,
                              v: UInt32,
                              temperature: Float,
                              topK: UInt32,
                              topP: Float,
                              seed: UInt64,
                              position: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(samplePSO)
        encoder.setBuffer(probs, offset: 0, index: 0)
        encoder.setBuffer(outToken, offset: 0, index: 1)
        var vVar = v
        var tVar = temperature
        var kVar = topK
        var pVar = topP
        var sVar = seed
        var posVar = position
        encoder.setBytes(&vVar, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&tVar, length: MemoryLayout<Float>.stride, index: 3)
        encoder.setBytes(&kVar, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&pVar, length: MemoryLayout<Float>.stride, index: 5)
        encoder.setBytes(&sVar, length: MemoryLayout<UInt64>.stride, index: 6)
        encoder.setBytes(&posVar, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256,
                                                               height: 1, depth: 1))
        encoder.endEncoding()
    }

    // MARK: - Repetition penalty (host, in place)

    /// HF convention, identical to the house pass but without the softcap
    /// round-trip (K3 logits are uncapped fp32): for each unique token id in
    /// `history`, a positive logit is divided by `penalty`, a negative one
    /// multiplied.
    private func applyRepetitionPenaltyInPlace(logits: MTLBuffer,
                                               history: [Int32],
                                               penalty: Float) {
        let ptr = logits.contents().bindMemory(to: Float.self, capacity: vocab)
        var seen = Set<Int32>()
        seen.reserveCapacity(history.count)
        for id in history {
            guard id >= 0 && Int(id) < vocab, seen.insert(id).inserted else { continue }
            let i = Int(id)
            let z = ptr[i]
            ptr[i] = z > 0 ? z / penalty : z * penalty
        }
    }
}
