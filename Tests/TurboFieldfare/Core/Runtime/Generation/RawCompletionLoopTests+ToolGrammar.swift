import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// The tool-call constraint inside the real decode loop: the fallback recovers
/// a derailed call, an in-grammar generation is bit-identical with the filter
/// off, and the two grammars refuse to share a generation.
extension RawCompletionLoopTests {

    /// Writes one scripted logit distribution per decode step, so a test can
    /// put an illegal token at the top and a legal one just below it and watch
    /// the grammar fallback choose.
    final class WeightedLogitProducer: LogitProducer, @unchecked Sendable {
        let vocabSize: Int
        private let steps: [[Int32: Float]]
        private let firstDecodeCall: Int
        private var calls = 0

        init(vocabSize: Int, promptTokens: Int, steps: [[Int32: Float]]) {
            self.vocabSize = vocabSize
            self.steps = steps
            self.firstDecodeCall = promptTokens - 1
        }

        func reset() { calls = 0 }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            let index = min(max(calls - firstDecodeCall, 0), steps.count - 1)
            calls += 1
            let pointer = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for i in 0..<vocabSize { pointer[i] = Float16(-30.0) }
            for (id, weight) in steps[index] { pointer[Int(id)] = Float16(weight) }
        }
    }

    private struct ToolRun {
        var ids: [Int32] = []
        var text = ""
        var events: [StructuredAssistantEvent] = []
        var result: RawDecodeResult
    }

    private func runToolLoop(steps: [[Int32: Float]],
                             tools: Set<String>,
                             constrained: Bool,
                             maxNewTokens: Int = 32) async throws -> ToolRun {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let producer = WeightedLogitProducer(vocabSize: tokenizer.vocabSize,
                                             promptTokens: promptIDs.count,
                                             steps: steps)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        let filter = constrained
            ? ToolCallTokenFilter(tokenizer: tokenizer, allowedNames: tools)
            : nil
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer,
                                                 allowedTools: tools,
                                                 idGenerator: { "call_fixed" })
        var ids: [Int32] = []
        var text = ""
        var events: [StructuredAssistantEvent] = []
        var decodeError: Error?
        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: maxNewTokens, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off,
            toolFilter: filter) { progress in
                guard case .token(_, let id, let delta) = progress else { return }
                ids.append(id)
                text += delta
                guard decodeError == nil else { return }
                do {
                    events += try decoder.consume(tokenID: id,
                                                  delta: delta,
                                                  validatedBody: filter?.closedCallBody)
                } catch {
                    decodeError = error
                }
        }
        if let decodeError { throw decodeError }
        return ToolRun(ids: ids, text: text, events: events, result: result)
    }

    /// The Gemma pieces of `call:read{path:<|"|>/tmp<|"|>}`, plus the markers.
    private static let inGrammarCall: [Int32] =
        [48, 6639, 236787, 1399, 236782, 2337, 236787, 52, 236786, 11935, 52, 236783, 49, 50]

    @Test func toolGrammarRecoversFromACanonicalJSONDerail() async throws {
        // Step 4 wants `{"` — the quoted-key drift behind the upstream report.
        // `{` sits just below it, so the fallback has somewhere to land.
        var steps = Self.inGrammarCall.map { [$0: Float(30)] }
        steps[4] = [14937: 30, 236782: 20]
        let run = try await runToolLoop(steps: steps, tools: ["read"], constrained: true)

        #expect(run.result.toolGrammarVetoes == 1)
        #expect(run.result.toolNameVetoes == 0)
        #expect(run.result.reason == .toolCalls)
        #expect(run.ids == Array(Self.inGrammarCall.dropLast()))
        guard case .toolCall(let call)? = run.events.first else {
            Issue.record("no tool call decoded: \(run.events)")
            return
        }
        #expect(call.name == "read")
        #expect(call.arguments == .object(["path": .string("/tmp")]))
    }

    /// Without the constraint the same derail is exactly the upstream failure:
    /// the region closes on a quoted key and the parser refuses it.
    @Test func theSameDerailIsUndecodableWithoutTheConstraint() async throws {
        var steps = Self.inGrammarCall.map { [$0: Float(30)] }
        steps[4] = [14937: 30, 236782: 20]
        await #expect(throws: GemmaToolCallParserError.malformed) {
            try await runToolLoop(steps: steps, tools: ["read"], constrained: false)
        }
    }

    @Test func inGrammarGenerationIsBitIdenticalWithTheFilterOff() async throws {
        let steps = Self.inGrammarCall.map { [$0: Float(30)] }
        let constrained = try await runToolLoop(steps: steps, tools: ["read"], constrained: true)
        let plain = try await runToolLoop(steps: steps, tools: ["read"], constrained: false)

        #expect(constrained.ids == plain.ids)
        #expect(constrained.text == plain.text)
        #expect(constrained.result.reason == plain.result.reason)
        #expect(constrained.result.newTokens == plain.result.newTokens)
        #expect(constrained.result.toolGrammarVetoes == 0)
        #expect(plain.result.toolGrammarVetoes == 0)
        #expect(constrained.events == plain.events)
    }

    /// An undeclared tool is reported apart from ordinary vetoes: it says the
    /// prompt and the declared set disagree, not that the grammar is working.
    @Test func undeclaredToolNameIsVetoedAndCounted() async throws {
        var steps = Self.inGrammarCall.map { [$0: Float(30)] }
        // `write` outranks `read`, but only `read` is declared.
        steps[3] = [4374: 30, 1399: 20]
        let run = try await runToolLoop(steps: steps, tools: ["read"], constrained: true)
        #expect(run.result.toolNameVetoes >= 1)
        #expect(run.result.toolGrammarVetoes == run.result.toolNameVetoes)
        guard case .toolCall(let call)? = run.events.first else {
            Issue.record("no tool call decoded: \(run.events)")
            return
        }
        #expect(call.name == "read")
    }

    @Test func forceJSONAndTheToolGrammarCannotShareAGeneration() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let producer = WeightedLogitProducer(vocabSize: tokenizer.vocabSize,
                                             promptTokens: promptIDs.count,
                                             steps: [[tokenizer.eosID: 30]])
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        await #expect(throws: GeneratorError.self) {
            _ = try await runRawCompletion(
                producer: producer,
                tokenizer: tokenizer,
                promptIds: promptIDs,
                config: GenerationConfig(maxNewTokens: 4, temperature: 0, forceJSON: true),
                context: context,
                scratch: scratch,
                prefillConfig: .off,
                toolFilter: ToolCallTokenFilter(tokenizer: tokenizer,
                                                allowedNames: ["read"])) { _ in }
        }
    }
}
