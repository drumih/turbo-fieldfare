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
        var tails: [String] = []
        var events: [StructuredAssistantEvent] = []
        /// The server's `truncatedByBudget` predicate: the generation ended
        /// with a `<|tool_call>` region the decoder never got to close.
        var openToolRegion = false
        var result: RawDecodeResult

        var decodedContent: String {
            events.reduce(into: "") { text, event in
                if case .content(let delta) = event { text += delta }
            }
        }
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
        var tails: [String] = []
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
                if case .tail(let tail) = progress { tails.append(tail) }
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
        return ToolRun(ids: ids,
                       text: text,
                       tails: tails,
                       events: events,
                       openToolRegion: decoder.hasOpenToolRegion,
                       result: result)
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

    /// The budget can land in the middle of a multi-byte scalar of a string
    /// argument. The detokenizer holds those byte-fallback pieces back until a
    /// stop, and the flush would then hand them over as ordinary text — the
    /// `é` tail of a path arriving as assistant content the model never wrote
    /// as prose, past the decoder that suppresses everything else in the
    /// region.
    @Test func aCutArgumentDoesNotFlushItsBytesAsContent() async throws {
        let tokenizer = try await GFTokenizer.load()
        let eAcute = try ["<0xC3>", "<0xA9>"].map {
            Int32(try #require(tokenizer.tokenizer.convertTokenToId($0)))
        }
        // `Sure.` + `<|tool_call>call:read{path:<|"|>` + `/tmp` + `é` spelled
        // one raw byte at a time, and then the budget runs out.
        let scripted = tokenizer.encode("Sure.", addBOS: false)
            + Array(Self.inGrammarCall.prefix(10))
            + eAcute
        let run = try await runToolLoop(steps: scripted.map { [$0: Float(30)] },
                                        tools: ["read"],
                                        constrained: true,
                                        maxNewTokens: scripted.count)

        #expect(run.result.reason == .maxTokens)
        #expect(run.openToolRegion)
        #expect(run.decodedContent == "Sure.")
        #expect(run.tails.isEmpty)

        // Not vacuous: the same two pieces through a bare detokenizer are the
        // `é` that used to reach the caller.
        var detok = GFDetokenizer(tokenizer: tokenizer)
        for id in scripted { _ = detok.push(id) }
        #expect(detok.flush().hasSuffix("é"))
    }

    /// At the 256 KiB region cap the grammar has no move left: every token
    /// overflows the cap and `<tool_call|>` cannot close an argument object
    /// that is still open. Ending the generation there is the honest report —
    /// the region is abandoned exactly like one cut by the token budget — and
    /// it keeps the request off the 500 path the fallback used to take.
    @Test func theRegionByteCapEndsTheGenerationInsteadOfFailingIt() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let head: Int32 = 900
        let filler: Int32 = 901
        let headBody = #"call:read{path:<|"|>"#
        let chunk = (GemmaToolCallParser.maximumBytes - headBody.utf8.count) / 4
        #expect(headBody.utf8.count + 4 * chunk == GemmaToolCallParser.maximumBytes)
        // Only these three pieces spell bytes, so at the cap the fallback scan
        // has nowhere to go — the state the real vocabulary reaches by writing
        // a quarter-megabyte string argument.
        let pieces: [Int32: String] = [
            tokenizer.escapeTokenID: #"<|"|>"#,
            head: #"call:read{path:"#,
            filler: String(repeating: "a", count: chunk),
        ]
        let filter = ToolCallTokenFilter(pieceLookup: { pieces[$0] },
                                         markers: ToolCallMarkerIDs(tokenizer: tokenizer),
                                         allowedNames: ["read"])
        let scripted: [Int32] = [tokenizer.toolCallStartID, head, tokenizer.escapeTokenID]
            + Array(repeating: filler, count: 4)
        let producer = WeightedLogitProducer(vocabSize: tokenizer.vocabSize,
                                             promptTokens: promptIDs.count,
                                             steps: scripted.map { [$0: Float(30)] })
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        var tails: [String] = []
        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 64, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off,
            toolFilter: filter) { progress in
                if case .tail(let tail) = progress { tails.append(tail) }
        }

        #expect(result.reason == .maxTokens)
        #expect(result.newTokens == scripted.count)
        #expect(result.newTokens < 64)
        #expect(result.toolGrammarVetoes == 1)
        #expect(filter.isInsideCall)
        #expect(tails.isEmpty)
        // Nothing was left dangling for the prompt cache to resume from either.
        #expect(result.uncommittedBoundaryTokenIDs.isEmpty)
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
