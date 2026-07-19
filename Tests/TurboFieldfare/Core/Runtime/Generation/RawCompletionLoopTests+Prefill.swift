import Testing

@testable import TurboFieldfare

extension RawCompletionLoopTests {
    @Test func prefillProgressCoversEveryPromptToken() async throws {
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let promptIDs = tokenizer.encode("one two three", addBOS: true)
        let (collected, result) = try await runLoop(
            seq: [tokenA],
            end: tokenizer.eosID,
            prompt: "one two three",
            config: GenerationConfig(maxNewTokens: 4, temperature: 0))

        #expect(result.prefillTokens == promptIDs.count)
        #expect(collected.prefills.count == promptIDs.count)
        #expect(collected.prefills.last?.0 == promptIDs.count)
        #expect(collected.prefills.allSatisfy { $0.1 == promptIDs.count })
    }

    @Test func disabledChunkedPrefillUsesScalarReplay() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let promptIDs = tokenizer.encode("one two three", addBOS: true)
        let producer = CountingProducer(
            vocabSize: tokenizer.vocabSize,
            step: automaton([tokenA], end: tokenizer.eosID))
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        _ = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 4, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off) { _ in }

        #expect(producer.resetCalls == 1)
        #expect(producer.produceCalls > promptIDs.count)
    }
}
