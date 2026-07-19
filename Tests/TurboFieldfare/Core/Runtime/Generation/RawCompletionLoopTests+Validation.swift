import Testing

@testable import TurboFieldfare

extension RawCompletionLoopTests {
    @Test func rawCompletionRejectsContextOverflowBeforeReset() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let promptIDs = tokenizer.encode("one two three", addBOS: true)
        let producer = CountingProducer(
            vocabSize: tokenizer.vocabSize,
            maxContext: promptIDs.count,
            step: automaton([tokenA], end: tokenizer.eosID))
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        do {
            _ = try await runRawCompletion(
                producer: producer,
                tokenizer: tokenizer,
                promptIds: promptIDs,
                config: GenerationConfig(maxNewTokens: 1, temperature: 0),
                context: context,
                scratch: scratch,
                prefillConfig: .off) { _ in }
            Issue.record("expected context overflow")
        } catch let error as GeneratorError {
            guard case .contextOverflow(let prompt, let maxNew, let maxContext) = error else {
                Issue.record("unexpected GeneratorError \(error)")
                return
            }
            #expect(prompt == promptIDs.count)
            #expect(maxNew == 1)
            #expect(maxContext == promptIDs.count)
        }

        #expect(producer.resetCalls == 0)
        #expect(producer.produceCalls == 0)
    }

    @Test func rawCompletionRejectsZeroMaxNewBeforeReset() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let promptIDs = tokenizer.encode("one two three", addBOS: true)
        let producer = CountingProducer(
            vocabSize: tokenizer.vocabSize,
            step: automaton([tokenA], end: tokenizer.eosID))
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        do {
            _ = try await runRawCompletion(
                producer: producer,
                tokenizer: tokenizer,
                promptIds: promptIDs,
                config: GenerationConfig(maxNewTokens: 0, temperature: 0),
                context: context,
                scratch: scratch,
                prefillConfig: .off) { _ in }
            Issue.record("expected invalid generation config")
        } catch let error as GeneratorError {
            guard case .invalidGenerationConfig(let reason) = error else {
                Issue.record("unexpected GeneratorError \(error)")
                return
            }
            #expect(reason.contains("maxNewTokens"))
        }

        #expect(producer.resetCalls == 0)
        #expect(producer.produceCalls == 0)
    }

    @Test func rawCompletionRejectsEmptyPromptBeforeReset() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let producer = CountingProducer(
            vocabSize: tokenizer.vocabSize,
            step: automaton([tokenA], end: tokenizer.eosID))
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        do {
            _ = try await runRawCompletion(
                producer: producer,
                tokenizer: tokenizer,
                promptIds: [],
                config: GenerationConfig(maxNewTokens: 1, temperature: 0),
                context: context,
                scratch: scratch,
                prefillConfig: .off) { _ in }
            Issue.record("expected empty prompt rejection")
        } catch let error as GeneratorError {
            #expect(error == .emptyPrompt)
        }

        #expect(producer.resetCalls == 0)
        #expect(producer.produceCalls == 0)
    }
}
