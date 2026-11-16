import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct SampleTopK64Tests {
    @Test func generationConfigUsesCanonicalSamplingDefaults() throws {
        let config = GenerationConfig()
        #expect(config.temperature == 0.6)
        #expect(config.topK == 20)
        #expect(config.topP == 0.95)
        #expect(config.presencePenalty == 0)
        try config.validate()
    }

    @Test func aPresencePenaltyNowValidates() throws {
        // Qwen3.8's published instruct row uses 1.5, and this used to throw:
        // nonzero presence penalties were refused outright.
        try GenerationConfig(presencePenalty: 1.5).validate()
    }

    @Test func aNonzeroMinPIsRefusedRatherThanIgnored() {
        #expect(throws: GeneratorError.self) {
            try GenerationConfig(minP: 0.1).validate()
        }
    }

    @Test func qwen38PublishesBothModeRows() {
        let thinking = GenerationDefaults.forFamily(.qwen38flash, thinking: true)
        #expect(thinking.temperature == 1.0)
        #expect(thinking.topP == 0.95)
        #expect(thinking.topK == 20)
        #expect(thinking.minP == 0)
        #expect(thinking.presencePenalty == 0)

        let instruct = GenerationDefaults.forFamily(.qwen38flash, thinking: false)
        #expect(instruct.temperature == 0.7)
        #expect(instruct.topP == 0.80)
        #expect(instruct.topK == 20)
        #expect(instruct.minP == 0)
        #expect(instruct.presencePenalty == 1.5)

        // The mode-less accessor keeps the card's default, which is thinking.
        #expect(GenerationDefaults.forFamily(.qwen38flash) == thinking)
        #expect(GenerationDefaults.forFamily(.qwen38flashMTP, thinking: false) == instruct)
    }

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

    @Test func samplerPathControlDefaultsToTiledAndFailsClosed() throws {
        #expect(try RuntimeSamplerPath.environmentValue([:]) == .tiled)
        #expect(
            try RuntimeSamplerPath.environmentValue(
                ["TINYTITAN_SAMPLER_PATH": "tiled"]) == .tiled)
        #expect(
            try RuntimeSamplerPath.environmentValue(
                ["TINYTITAN_SAMPLER_PATH": "generic"]) == .generic)
        #expect(throws: GeneratorError.self) {
            try RuntimeSamplerPath.environmentValue(["TINYTITAN_SAMPLER_PATH": "fast"])
        }
        #expect(throws: GeneratorError.self) {
            try RuntimeSamplerPath.environmentValue(["TINYTITAN_SAMPLER_PATH": ""])
        }
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
            guard
                let probs = context.device.makeBuffer(
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

        func draw(
            seed: UInt64,
            temperature: Float = 1.0,
            topP: Float,
            topK: Int = 64
        ) throws -> (current: UInt32, candidate: UInt32) {
            let cb = try #require(context.queue.makeCommandBuffer())
            try current.encode(
                commandBuffer: cb,
                probs: probs,
                outToken: currentOutput,
                v: UInt32(vocab),
                temperature: temperature,
                topK: UInt32(topK),
                topP: topP,
                seed: seed)
            try candidate.encode(
                commandBuffer: cb,
                probs: probs,
                outToken: candidateOutput,
                temperature: temperature,
                topP: topP,
                seed: seed,
                topK: UInt32(topK))
            cb.commit()
            cb.waitUntilCompleted()
            return (
                currentOutput.contents().load(as: UInt32.self),
                candidateOutput.contents().load(as: UInt32.self)
            )
        }
    }

    @Test func productionVocabularyMatchesCurrentSampler() throws {
        let rig = try Rig(vocab: 262_144)
        #expect(rig.candidate.scratchBytes == 139_264)
        rig.write { i in
            let mixed = UInt64(i) &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(UInt32(mixed >> 40) + 1) * (1.0 / 16_777_217.0)
        }

        for temperature: Float in [0.7, 0.85, 1.0] {
            for seed: UInt64 in [1, 2, 0x1234_5678_9ABC_DEF0, UInt64.max] {
                let result = try rig.draw(seed: seed, temperature: temperature, topP: 0.95)
                #expect(
                    result.candidate == result.current,
                    "temperature \(temperature), seed \(seed): candidate \(result.candidate), current \(result.current)"
                )
            }
        }
    }

    @Test func tiesAndPartialTailMatchCurrentSampler() throws {
        let rig = try Rig(vocab: 1_003)
        rig.write { _ in 1.0 }

        for seed in UInt64(1)...UInt64(8) {
            let result = try rig.draw(seed: seed, topP: 0.95)
            #expect(
                result.candidate == result.current,
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
            let result = try rig.draw(seed: seed, topP: 0.95)
            #expect(result.candidate == result.current)
            #expect(result.candidate < 64)
            if result.candidate >= 61 { sawLastThree = true }
        }
        #expect(sawLastThree, "Top-P incorrectly truncated the renormalized Top-64 set")
    }

    /// The tiled reduction serves every k in 1...64, not only 64.
    ///
    /// Stage 1 keeps the top 64 of each tile, so the top k of the vocabulary
    /// is a strict subset of its output for any k <= 64 and the final stage
    /// only has to cut there. This is the claim that lets the production
    /// default (Top-K 20) leave the generic k-full-vocabulary-passes kernel,
    /// so it is pinned against that kernel rather than against a golden list.
    @Test func everyKUpToSixtyFourMatchesTheGenericSampler() throws {
        let rig = try Rig(vocab: 262_144)
        rig.write { i in
            let mixed = UInt64(i) &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(UInt32(mixed >> 40) + 1) * (1.0 / 16_777_217.0)
        }

        for topK in [1, 2, 7, 20, 33, 63, 64] {
            for temperature: Float in [0.6, 1.0] {
                for seed: UInt64 in [1, 42, 0x1234_5678_9ABC_DEF0] {
                    let result = try rig.draw(
                        seed: seed,
                        temperature: temperature,
                        topP: 0.95,
                        topK: topK)
                    #expect(
                        result.candidate == result.current,
                        "k \(topK), t \(temperature), seed \(seed): candidate \(result.candidate) current \(result.current)"
                    )
                }
            }
        }
    }

    /// Top-K must cut before the Top-P scan, not after.
    ///
    /// A flat distribution puts the 0.95 nucleus far beyond rank 64, so
    /// Top-P never fires and both orders agree. This uses a distribution whose
    /// nucleus lands *inside* the top 64, which is the only shape that can
    /// tell the two orders apart: cutting after Top-P would keep the whole
    /// nucleus at small k, where the generic kernel keeps exactly k.
    @Test func topKCutsBeforeTopPWhenTheNucleusIsNarrow() throws {
        let rig = try Rig(vocab: 1_003)
        // Geometric decay: the top ~30 entries already carry >95% of the mass.
        rig.write { i in i < 128 ? Float(pow(0.85, Double(i))) : 0 }

        for topK in [3, 8, 20, 64] {
            for seed: UInt64 in [1, 5, 99, 4_242] {
                let result = try rig.draw(
                    seed: seed,
                    temperature: 1.0,
                    topP: 0.95,
                    topK: topK)
                #expect(
                    result.candidate == result.current,
                    "k \(topK), seed \(seed): candidate \(result.candidate) current \(result.current)"
                )
                #expect(
                    result.candidate < UInt32(topK),
                    "k \(topK) selected index \(result.candidate), outside the top k of a decreasing distribution"
                )
            }
        }
    }
}
