import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
@testable import TinyTitanCore

@testable import TinyTitan

extension RawCompletionLoopTests {
    @Test func stopsOnEOS() async throws {
        let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let idA = try #require(tok.encode("a", addBOS: false).first)
        let idB = try #require(tok.encode("b", addBOS: false).first)
        let (collected, result) = try await runLoop(
            seq: [idA, idB], end: tok.eosID,
            config: GenerationConfig(maxNewTokens: 50, temperature: 0))
        #expect(result.reason == .eos)
        #expect(result.newTokens == 3)
        #expect(collected.tokens.map(\.1) == [idA, idB])
    }

    @Test func stopsOnEndOfTurn() async throws {
        let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let idA = try #require(tok.encode("a", addBOS: false).first)
        let (_, result) = try await runLoop(
            seq: [idA], end: tok.endOfTurnID,
            config: GenerationConfig(maxNewTokens: 50, temperature: 0))
        #expect(result.reason == .endOfTurn)
    }

    @Test func stopsOnMaxTokensAndCountsExactly() async throws {
        let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let idA = try #require(tok.encode("a", addBOS: false).first)
        let (collected, result) = try await runLoop(
            seq: [idA, idA], end: idA,
            config: GenerationConfig(maxNewTokens: 5, temperature: 0))
        #expect(result.reason == .maxTokens)
        #expect(result.newTokens == 5)
        #expect(collected.tokens.count == 5)
        #expect(collected.tokens.map(\.0) == [0, 1, 2, 3, 4])
    }

    /// `StopReason.toolCalls` used to be decided by comparing the *stop* token to
    /// `toolResponseID`, which is never in `stopTokenIDs`, so this returned
    /// `.endOfTurn` and three callers stayed dead: tool-round prompt-cache reuse,
    /// the `orphanToolResponse` diagnostic and the `rawStop` string. The assertion
    /// fails by name on that code.
    @Test func stopsOnToolCallsAfterACompleteCall() async throws {
        let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let idA = try #require(tok.encode("a", addBOS: false).first)
        let (_, result) = try await runLoop(
            seq: [idA, tok.toolCallStartID, tok.toolCallEndID], end: tok.endOfTurnID,
            config: GenerationConfig(maxNewTokens: 50, temperature: 0))
        #expect(result.reason == .toolCalls)
    }

    /// A truncated call is not a tool turn: the model was cut off mid-call, so
    /// nothing parsed and `.endOfTurn` is the honest reason.
    @Test func anUnterminatedToolCallStaysEndOfTurn() async throws {
        let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let idA = try #require(tok.encode("a", addBOS: false).first)
        let (_, result) = try await runLoop(
            seq: [idA, tok.toolCallStartID], end: tok.endOfTurnID,
            config: GenerationConfig(maxNewTokens: 50, temperature: 0))
        #expect(result.reason == .endOfTurn)
    }

    /// A closing marker on its own must not make the turn a tool call, or the
    /// orphan diagnostic would fire on a turn that never opened one.
    @Test func aStrayToolCallCloserStaysEndOfTurn() async throws {
        let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let (_, result) = try await runLoop(
            seq: [tok.toolCallEndID], end: tok.endOfTurnID,
            config: GenerationConfig(maxNewTokens: 50, temperature: 0))
        #expect(result.reason == .endOfTurn)
    }

    /// The consumer's side of the sampler's in-range contract. An out-of-range id
    /// has been observed intermittently under full-suite GPU load and never
    /// identified; a token id indexes the embedding table and extends the KV
    /// history, so using one is silent corruption. It is a named error now, with
    /// the id in it -- which is what makes the next occurrence identify itself
    /// rather than corrupt something.
    @Test func anOutOfRangeSampledIdIsReported() throws {
        #expect(try validatedToken(41, vocab: 42) == 41)

        #expect(throws: GeneratorError.self) {
            _ = try validatedToken(42, vocab: 42)
        }
        #expect(throws: GeneratorError.self) {
            _ = try validatedToken(0xFFFF_FFFF, vocab: 42)
        }
    }

    @Test func stopsOnStopString() async throws {
        let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let idA = try #require(tok.encode("a", addBOS: false).first)
        let textA = tok.decode([idA], skipSpecialTokens: true)
        let (_, result) = try await runLoop(
            seq: [idA, idA], end: idA,
            config: GenerationConfig(maxNewTokens: 50, temperature: 0, stopStrings: [textA]))
        #expect(result.reason == .stopString)
    }

    /// One NaN logit must not empty the row. Before `softcap_value` folded NaN to
    /// -inf, `tanh(NaN)` made every probability NaN, so the draw fell into the
    /// sampler's in-range fallback and the loop emitted an arbitrary id (id 1 in
    /// practice here) instead of `target` -- the entry carrying the large logit.
    /// With the fold the finite logits decide.
    @Test func aSingleNaNLogitDoesNotEmptyTheRow() async throws {
        let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        // An id this fixture can decode and that is not a stop token: the existing
        // stop tests use the same encoding as their payload. It must not be 0, or
        // it would collide with the NaN entry below.
        let target = try #require(tok.encode("a", addBOS: false).first)
        try #require(target != 0)
        let row: [Float] = {
            // ScriptedLogitProducer fills the rest of the row with -30, so only
            // entries 0...target are listed. `target` carries the only large logit,
            // so the assertion can tell "the finite logits decided" from "the
            // empty-row fallback answered".
            var values = [Float](repeating: -30, count: Int(target) + 1)
            values[0] = .nan
            values[Int(target)] = 30
            return values
        }()
        let (collected, result) = try await runLoop(
            // Not keyed on the call index: with `prefillConfig: .off` the producer is
            // called once per *prompt* token, and it is the last of those calls whose
            // logits the first sample reads.
            step: { _, _ -> ScriptedLogitProducer.Step in .vector(row) },
            config: GenerationConfig(maxNewTokens: 8, temperature: 0))
        #expect(collected.tokens.first?.1 == target)
        #expect(collected.tokens.count == 8)
        #expect(result.reason == .maxTokens)
    }

    /// The case the fold cannot rescue: every logit non-finite. The softmax has no
    /// finite value to work with, so the kernels keep their in-range fallback (the
    /// generation loop indexes the vocabulary and the KV with that id) and the row
    /// is *reported* instead. Fails on the old code by completing normally with a
    /// run of token 0s.
    @Test func aRowWithNoFiniteLogitIsReported() async throws {
        let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let nan = [Float](repeating: .nan, count: tok.vocabSize)
        await #expect(throws: GeneratorError.degenerateLogitsRow) {
            try await self.runLoop(
                step: { _, _ -> ScriptedLogitProducer.Step in .vector(nan) },
                config: GenerationConfig(maxNewTokens: 4, temperature: 0))
        }
    }

}
