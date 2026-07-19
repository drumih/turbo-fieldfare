import Testing

@testable import TurboFieldfare

extension RawCompletionLoopTests {
    @Test func chunkedModeRequiresChunkedProducer() async throws {
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
                config: GenerationConfig(maxNewTokens: 4, temperature: 0),
                context: context,
                scratch: scratch,
                prefillConfig: .production(chunkTokens: 32)) { _ in }
            Issue.record("expected chunked unsupported error")
        } catch let error as PrefillError {
            guard case .chunkedUnsupported(let reason) = error else {
                Issue.record("unexpected PrefillError \(error)")
                return
            }
            #expect(reason == PrefillError.chunkedRequiresChunkedRunnerReason)
        }

        #expect(producer.produceCalls == 0)
    }

    @Test func chunkedModeUsesChunkedRunnerEntryPoint() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let producer = ChunkedTestProducer(vocabSize: tokenizer.vocabSize, firstToken: tokenA)
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        var prefills: [(Int, Int)] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .production(chunkTokens: 32)) { progress in
                if case .prefill(let done, let total) = progress {
                    prefills.append((done, total))
                }
            }

        #expect(result.newTokens == 1)
        #expect(producer.chunkedCalls == 1)
        #expect(producer.produceCalls == 0)
        #expect(producer.lastOutputMode == .logits)
        #expect(producer.lastConfig == .production(chunkTokens: 32))
        #expect(prefills.count == 1)
        #expect(prefills.first?.0 == promptIDs.count)
        #expect(prefills.first?.1 == promptIDs.count)
    }

    @Test func chunkedLogitsSeedProducesFirstToken() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let producer = ChunkedTestProducer(vocabSize: tokenizer.vocabSize, firstToken: tokenA)
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        var tokens: [Int32] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .production(chunkTokens: 32)) { progress in
                if case .token(_, let id, _) = progress {
                    tokens.append(id)
                }
            }

        #expect(result.newTokens == 1)
        #expect(tokens == [tokenA])
        #expect(producer.chunkedCalls == 1)
        #expect(producer.produceCalls == 0)
        #expect(producer.lastOutputMode == .logits)
    }

    @Test func chunkedPrefillRejectsGreedySeedWhenLogitsRequested() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let producer = ChunkedTestProducer(
            vocabSize: tokenizer.vocabSize,
            firstToken: tokenA,
            seed: .greedyToken(UInt32(bitPattern: tokenA)))
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        do {
            _ = try await runRawCompletion(
                producer: producer,
                tokenizer: tokenizer,
                promptIds: promptIDs,
                config: GenerationConfig(maxNewTokens: 1, temperature: 0.7),
                context: context,
                scratch: scratch,
                prefillConfig: .production(chunkTokens: 32)) { _ in }
            Issue.record("expected unsupported chunked prefill seed")
        } catch let error as PrefillError {
            guard case .unsupportedPrefillSeed(let reason) = error else {
                Issue.record("unexpected PrefillError \(error)")
                return
            }
            #expect(reason.contains("RawCompletion chunked prefill requested logits"))
        }

        #expect(producer.chunkedCalls == 1)
        #expect(producer.produceCalls == 0)
        #expect(producer.lastOutputMode == .logits)
    }
}
