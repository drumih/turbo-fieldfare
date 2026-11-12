import Foundation
import Metal
import Testing

@testable import TinyTitan

/// Wide verify in the real decode loop (task 11) on the qwen toy runner: the
/// same prompt generated with wide verify off and on must emit the same
/// tokens and leave the same KV/GDN state, while drafts are both accepted and
/// rejected along the way.
@Suite(.serialized) struct WideVerifyDecodeTests {
    /// A pass always pays under this model, so every match is verified at the
    /// widest W the drafter offers.
    static let eager = WideVerifyDecodeOptions(
        costModel: WideVerifyCostModel(
            scalarMilliseconds: 100, passBaseMilliseconds: 0, passRowMilliseconds: 1))

    struct Run {
        var tokens: [Int32] = []
        var result: RawDecodeResult?
        var state: InferenceStateSnapshot?
    }

    final class Harness {
        let directory: URL
        let context: MetalContext
        let runner: RealForwardRunner
        let tokenizer: GFTokenizer
        let scratch: RawCompletionScratch

        init(fusedHead: Bool) async throws {
            directory = try QwenToySynthetic.write(weightBits: 4, variedLinearAttention: true)
            context = try MetalContext()
            let model = try Model.load(
                directoryURL: directory, device: context.device, expecting: .qwenToy())
            runner = try RealForwardRunner(
                model: model, context: context, maxContext: 1536, slots: 1,
                runtimeConfiguration: try RuntimeConfiguration(forceLogitsHead: !fusedHead))
            tokenizer = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
            scratch = try RawCompletionScratch(context: context, vocab: runner.cfg.vocabSize)
        }

        deinit { try? FileManager.default.removeItem(at: directory) }

        func run(
            _ prompt: [Int32], config: GenerationConfig, wide: WideVerifyDecodeOptions?
        ) async throws -> Run {
            var run = Run()
            let result = try await runRawCompletion(
                producer: runner, tokenizer: tokenizer, promptIds: prompt, config: config,
                context: context, scratch: scratch, wideVerify: wide
            ) { progress in
                if case .token(_, let id, _) = progress { run.tokens.append(id) }
            }
            run.result = result
            run.state = try runner.captureInferenceState()
            return run
        }
    }

    /// Repeated n-grams. The toy's greedy output is token 0 over and over, so
    /// the phrase puts a 0 after its fourth token: the prompt ends on that
    /// prefix, the first output token completes it, and the drafter proposes
    /// the phrase's tail (rejected); after that the output's own repeats
    /// draft 0s (accepted).
    static let prompt: [Int32] = {
        let phrase: [Int32] = [11, 12, 13, 14, 0, 16, 17, 18]
        return [1] + phrase + [40, 41] + phrase + [50] + phrase.prefix(4)
    }()

    /// The toy's logits are nearly flat over its low ids, so sampling it
    /// rarely repeats itself, and a 1.5 presence penalty pushes every id it
    /// has seen out of a small top-k. A prompt holding every vocabulary id
    /// once puts all ids under the penalty from the start: the penalty still
    /// runs on every row (and on the drafts each row accepts), and top-2
    /// sampling then repeats often enough for drafts to be accepted and
    /// rejected.
    static func sampledPrompt(vocab: Int, tail: [Int32] = []) -> [Int32] {
        [1] + (0..<Int32(vocab)) + prompt.dropFirst() + tail
    }

    static func sampled(seed: UInt64) -> GenerationConfig {
        GenerationConfig(
            maxNewTokens: 64, temperature: 0.7, topK: 2, topP: 0.95,
            presencePenalty: 1.5, seed: seed)
    }

    private func expectIdentical(_ scalar: Run, _ wide: Run) throws {
        let s = try #require(scalar.result)
        let w = try #require(wide.result)
        #expect(wide.tokens == scalar.tokens)
        #expect(w.newTokens == s.newTokens)
        #expect(w.reason == s.reason)
        #expect(w.kvPosition == s.kvPosition)
        #expect(w.kvBackedTokenIDs == s.kvBackedTokenIDs)
        #expect(w.uncommittedBoundaryTokenIDs == s.uncommittedBoundaryTokenIDs)
        let a = try #require(scalar.state)
        let b = try #require(wide.state)
        #expect(a.descriptor == b.descriptor)
        let first = a.payload.indices.first { a.payload[$0] != b.payload[$0] }
        #expect(
            a.payload == b.payload,
            "first differing state byte \(first.map(String.init) ?? "-") of \(a.payload.count)")
    }

    private func expectDraftsJudged(_ run: Run) throws {
        let stats = try #require(run.result?.wideVerify)
        #expect(stats.passes > 0, "\(stats.summary) \(run.tokens)")
        #expect(stats.acceptedDrafts > 0, "\(stats.summary) \(run.tokens)")
        #expect(stats.rejectingPasses > 0, "\(stats.summary) \(run.tokens)")
    }

    @Test(arguments: [true, false])
    func greedyMatchesScalar(fusedHead: Bool) async throws {
        let harness = try await Harness(fusedHead: fusedHead)
        let config = GenerationConfig(
            maxNewTokens: 48, temperature: 0, topK: nil, topP: nil, presencePenalty: 0)
        let scalar = try await harness.run(Self.prompt, config: config, wide: nil)
        let wide = try await harness.run(Self.prompt, config: config, wide: Self.eager)
        #expect(scalar.result?.wideVerify == nil)
        try expectIdentical(scalar, wide)
        try expectDraftsJudged(wide)
    }

    @Test func seededSamplingWithPresencePenaltyMatchesScalar() async throws {
        let harness = try await Harness(fusedHead: false)
        let config = Self.sampled(seed: 1234)
        let prompt = Self.sampledPrompt(vocab: harness.runner.cfg.vocabSize)
        let scalar = try await harness.run(prompt, config: config, wide: nil)
        let wide = try await harness.run(prompt, config: config, wide: Self.eager)
        try expectIdentical(scalar, wide)
        try expectDraftsJudged(wide)
    }

    /// A stop token emitted inside a pass's accepted run (not its last
    /// emitted token) stops where scalar decode stops, and the rows the pass
    /// fed past it are rolled back: same tokens, same KV/GDN state.
    @Test func stopTokenInsideAcceptedRunStopsLikeScalar() async throws {
        let harness = try await Harness(fusedHead: false)
        let vocab = harness.runner.cfg.vocabSize
        // The token besides 0 that top-2 sampling keeps choosing.
        let free = try await harness.run(
            Self.sampledPrompt(vocab: vocab), config: Self.sampled(seed: 1), wide: nil)
        let other = try #require(free.tokens.first { $0 != 0 })
        // The prompt ends on "7 8 9"; when the first sample is 0, "7 8 9 0"
        // matches the tail's earlier copy and drafts `other 0 0`.
        let prompt = Self.sampledPrompt(
            vocab: vocab, tail: [7, 8, 9, 0, other, 0, 0, other, 7, 8, 9])
        var cutInsideARun = 0
        for seed: UInt64 in 1...16 where cutInsideARun == 0 {
            var config = Self.sampled(seed: seed)
            config.extraStopTokens = [other]
            let scalar = try await harness.run(prompt, config: config, wide: nil)
            let wide = try await harness.run(prompt, config: config, wide: Self.eager)
            #expect(scalar.result?.reason == .eos)
            try expectIdentical(scalar, wide)
            cutInsideARun += wide.result?.wideVerify?.passesCutAtStop ?? 0
        }
        #expect(cutInsideARun > 0, "no stop token landed inside an accepted run")
    }

    /// The server's switch: off unless `TINYTITAN_WIDE_VERIFY=1`.
    @Test func optionsFromEnvironment() {
        #expect(WideVerifyDecodeOptions.fromEnvironment([:]) == nil)
        #expect(WideVerifyDecodeOptions.fromEnvironment(["TINYTITAN_WIDE_VERIFY": "0"]) == nil)
        let on = WideVerifyDecodeOptions.fromEnvironment([
            "TINYTITAN_WIDE_VERIFY": "1", "TINYTITAN_WIDE_VERIFY_ROWS": "4",
        ])
        #expect(on?.maxRows == 4)
    }
}
