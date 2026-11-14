import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
import TinyTitanCore

@testable import TinyTitan

/// Structured output end to end through the real decode loop: the mask is
/// applied by the real sampler on the real Metal path, the loop advances the
/// real grammar, and what comes out is checked as JSON.
///
/// The fixture tokenizer is a real ChatML tokenizer with a one-byte vocabulary,
/// so the grammar meets real token ids without a model being loaded.
@Suite struct JSONConstraintLoopTests {
    private struct Collected {
        var tokens: [(index: Int, id: Int32, delta: String)] = []
        var text: String { tokens.map(\.delta).joined() }
    }

    private func fixture() async throws -> GFTokenizer {
        try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    }

    private func schema() -> JSONSchemaNode {
        .object(properties: ["a": .scalar([.integer])], required: ["a"], additional: false)
    }

    /// The document the script wants to write, and the token it *prefers* at
    /// each of those positions. The preferred token is never legal there, so a
    /// response that followed the logits would not be JSON.
    private let document = ["{", "\"", "a", "\"", ":", "1", "}"]
    private let bait = "x"

    /// Run the loop over a script that always prefers the bait token.
    private func run(constraint: JSONConstraint?) async throws -> (Collected, RawDecodeResult) {
        let context = try MetalContext()
        let tokenizer = try await fixture()
        let prompt = tokenizer.encode("go", addBOS: true)
        let baitID = try #require(tokenizer.encode(bait, addBOS: false).first)
        // The logits the first sample reads are the ones the *last* prefill
        // call wrote, so the document script starts at that call's index.
        var built: [ScriptedLogitProducer.Step] = Array(
            repeating: .argmax(tokenizer.eosID), count: max(0, prompt.count - 1))
        for text in document {
            let allowedID = try #require(tokenizer.encode(text, addBOS: false).first)
            built.append(
                .vector(
                    sparse(
                        vocab: tokenizer.vocabSize,
                        high: [(baitID, 30), (allowedID, 20)])))
        }
        let script = built
        let producer = ScriptedLogitProducer(vocabSize: tokenizer.vocabSize) { _, call in
            call < script.count ? script[call] : .argmax(tokenizer.eosID)
        }
        let config = GenerationConfig(
            maxNewTokens: 16, temperature: 0,
            topK: nil, topP: nil, constraint: constraint)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        let collected = Collected()
        let box = Collector(collected)
        let result = try await runRawCompletion(
            producer: producer, tokenizer: tokenizer,
            promptIds: prompt, config: config,
            context: context, scratch: scratch,
            prefillConfig: .off
        ) { progress in
            if case .token(let index, let id, let delta) = progress {
                box.append((index, id, delta))
            }
        }
        return (box.value, result)
    }

    /// A logits row that is -30 everywhere except the named ids.
    private func sparse(vocab: Int, high: [(Int32, Float)]) -> [Float] {
        var values = [Float](repeating: -30, count: vocab)
        for (id, value) in high where Int(id) < vocab { values[Int(id)] = value }
        return values
    }

    /// unchecked-invariant: the loop publishes from one task and the test reads
    /// after it returns.
    private final class Collector: @unchecked Sendable {
        private var storage = Collected()
        init(_ initial: Collected) { storage = initial }
        func append(_ token: (Int, Int32, String)) { storage.tokens.append(token) }
        var value: Collected { storage }
    }

    @Test func theMaskOverridesTheModelsPreference() async throws {
        let tokenizer = try await fixture()
        let constraint = JSONConstraint(
            table: JSONTokenTable(tokenizer: tokenizer),
            node: schema())
        let (collected, result) = try await run(constraint: constraint)
        #expect(collected.text == #"{"a":1}"#)
        #expect(result.reason == .endOfTurn || result.reason == .eos)
        #expect(!collected.text.contains(bait))
        let parsed = try JSONSerialization.jsonObject(with: Data(collected.text.utf8))
        #expect((parsed as? [String: Any])?["a"] as? Int == 1)
        #expect(constraint.isComplete)
        #expect(constraint.violationCount == 0)
    }

    /// The control: the same script with no constraint writes exactly what the
    /// logits asked for, which is not JSON. Without this the test above would
    /// pass even if the mask were never consulted.
    @Test func withoutAConstraintTheSameScriptWritesTheBait() async throws {
        let (collected, _) = try await run(constraint: nil)
        #expect(collected.text.hasPrefix(bait))
        #expect(throws: (any Error).self) {
            _ = try JSONSerialization.jsonObject(with: Data(collected.text.utf8))
        }
    }
}
