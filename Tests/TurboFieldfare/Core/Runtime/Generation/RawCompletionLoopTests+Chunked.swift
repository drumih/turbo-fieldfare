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
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let work = PrefillWorkDiagnostics(executionPath: .scalarFallback,
                          scalarForwardCount: promptIDs.count,
                          chunkPassCount: 0,
                          commandBufferCount: promptIDs.count * 4,
                          embeddingNanos: 1,
                          mixerNanos: 2,
                          moePrepareNanos: 3,
                          expertFetchNanos: 4,
                          routedMoENanos: 5,
                          moeReduceNanos: 6,
                          finalHeadNanos: 7)
        let producer = ChunkedTestProducer(vocabSize: tokenizer.vocabSize,
                           firstToken: tokenA,
                           work: work)
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
        #expect(result.qwenDecodeDiagnostics?.decodeStepCount == 0)
        #expect(producer.lastOutputMode == .logits)
        #expect(producer.lastConfig == .production(chunkTokens: 32))
        #expect(result.prefillWork == work)
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

    @Test func qwenAggregateCountsOnlyPostPrefillForwardSteps() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let diagnostics = QwenDecodeDiagnostics(
            wallNanos: 100,
            embeddingNanos: 10,
            layerNanos: 70,
            logitsNanos: 20,
            expertFetchNanos: 30,
            layerCount: 1,
            fullAttentionLayerCount: 1,
            deltaNetLayerCount: 0,
            commandBufferCount: 3,
            routerEvaluationCount: 1,
            routedExpertCount: 2,
            routedExpertCacheHitCount: 1,
            routedExpertCacheMissCount: 1,
            routedExpertEstimatedBytes: 64)
        let producer = ChunkedTestProducer(vocabSize: tokenizer.vocabSize,
                                           firstToken: tokenA,
                                           produceDiagnostics: diagnostics)
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 2, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .production(chunkTokens: 32)) { _ in }

        let aggregate = try #require(result.qwenDecodeDiagnostics)
        #expect(result.newTokens == 2)
        #expect(producer.chunkedCalls == 1)
        #expect(producer.produceCalls == 1)
        #expect(aggregate.decodeStepCount == 1)
        #expect(aggregate.forwardWallNanos == 100)
        #expect(aggregate.commandBufferSubmissionCount == 3)
        #expect(aggregate.samplingNanos > 0)
        #expect(aggregate.decodeLoopWallNanos >= aggregate.attributedWallNanos)
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
