import Metal
import Testing
@testable import TurboFieldfare

@Suite struct SampleTopK64Tests {
    @Test func truncationDefaultsDoNotDisableGreedyEligibility() {
        let config = GenerationConfig(temperature: 0, topK: 64, topP: 0.95)
        #expect(config.isPureGreedy)
    }

    @Test func generationConfigRejectsSamplerStatesTheKernelCannotHonor() throws {
        #expect(throws: GeneratorError.self) {
            try GenerationConfig(temperature: 1, topK: 257, topP: 0.95).validate()
        }
        #expect(throws: GeneratorError.self) {
            try GenerationConfig(temperature: 1, topK: nil, topP: 0.95).validate()
        }
        try GenerationConfig(temperature: 0, topK: nil, topP: 0.95).validate()
    }

    private final class Rig {
        let context: MetalContext
        let current: Sample
        let candidate: SampleTopK64
        let probs: MTLBuffer
        let currentOutput: MTLBuffer
        let candidateOutput: MTLBuffer
        let vocab: Int

        init(vocab: Int) throws {
            self.context = try MetalContext()
            self.current = try Sample(context: context)
            self.candidate = try SampleTopK64(context: context, vocab: vocab)
            self.vocab = vocab
            guard let probs = context.device.makeBuffer(
                      length: vocab * MemoryLayout<Float16>.stride,
                      options: .storageModeShared),
                  let currentOutput = context.device.makeBuffer(
                      length: MemoryLayout<UInt32>.stride,
                      options: .storageModeShared),
                  let candidateOutput = context.device.makeBuffer(
                      length: MemoryLayout<UInt32>.stride,
                      options: .storageModeShared)
            else {
                throw MetalError.noDevice
            }
            self.probs = probs
            self.currentOutput = currentOutput
            self.candidateOutput = candidateOutput
        }

        func write(_ values: (Int) -> Float) {
            let ptr = probs.contents().bindMemory(to: Float16.self, capacity: vocab)
            for i in 0..<vocab {
                ptr[i] = Float16(values(i))
            }
        }

        func draw(seed: UInt64,
                  temperature: Float = 1.0,
                  topP: Float) -> (current: UInt32, candidate: UInt32) {
            let cb = context.queue.makeCommandBuffer()!
            current.encode(commandBuffer: cb,
                           probs: probs,
                           outToken: currentOutput,
                           v: UInt32(vocab),
                           temperature: temperature,
                           topK: 64,
                           topP: topP,
                           seed: seed)
            candidate.encode(commandBuffer: cb,
                             probs: probs,
                             outToken: candidateOutput,
                             temperature: temperature,
                             topP: topP,
                             seed: seed)
            cb.commit()
            cb.waitUntilCompleted()
            return (currentOutput.contents().load(as: UInt32.self),
                    candidateOutput.contents().load(as: UInt32.self))
        }
    }

    @Test func productionVocabularyMatchesCurrentSampler() throws {
        let rig = try Rig(vocab: 262_144)
        #expect(rig.candidate.scratchBytes == 139_264)
        rig.write { i in
            let mixed = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407
            return Float(UInt32(mixed >> 40) + 1) * (1.0 / 16_777_217.0)
        }

        for temperature: Float in [0.7, 0.85, 1.0] {
            for seed: UInt64 in [1, 2, 0x1234_5678_9ABC_DEF0, UInt64.max] {
                let result = rig.draw(seed: seed, temperature: temperature, topP: 0.95)
                #expect(result.candidate == result.current,
                        "temperature \(temperature), seed \(seed): candidate \(result.candidate), current \(result.current)")
            }
        }
    }

    @Test func tiesAndPartialTailMatchCurrentSampler() throws {
        let rig = try Rig(vocab: 1_003)
        rig.write { _ in 1.0 }

        for seed in UInt64(1)...UInt64(8) {
            let result = rig.draw(seed: seed, topP: 0.95)
            #expect(result.candidate == result.current,
                    "seed \(seed): candidate \(result.candidate), current \(result.current)")
            #expect(result.candidate < 64)
        }
    }

    @Test func topPUsesFullVocabularyMassBeforeTopK() throws {
        let rig = try Rig(vocab: 1_003)
        rig.write { _ in 1.0 / 1_003.0 }

        // The full-distribution 0.95 nucleus is much wider than 64 tokens, so
        // mlx-lm's Top-P-then-Top-K chain leaves all Top-64 entries eligible.
        // The previous Top-K-renormalize-then-Top-P order kept only 61.
        var sawLastThree = false
        for seed in UInt64(1)...UInt64(256) {
            let result = rig.draw(seed: seed, topP: 0.95)
            #expect(result.candidate == result.current)
            #expect(result.candidate < 64)
            if result.candidate >= 61 { sawLastThree = true }
        }
        #expect(sawLastThree, "Top-P incorrectly truncated the renormalized Top-64 set")
    }
}
