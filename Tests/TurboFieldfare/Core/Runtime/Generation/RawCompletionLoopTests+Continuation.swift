import Metal
import Testing

@testable import TurboFieldfare

extension RawCompletionLoopTests {
    final class ContinuationProducer: ChunkedPrefillRunner, PromptStateSnapshotting,
        FusedGreedyLogitProducer, @unchecked Sendable
    {
        let vocabSize: Int
        private let terminalToken: Int32
        private(set) var continuationPosition: Int
        private(set) var resetCalls = 0
        private(set) var prepareCalls: [Int] = []
        private(set) var restoreCalls: [Int] = []
        private(set) var saveCalls = 0
        private(set) var prefillRanges: [Range<Int>] = []
        private var savedPosition: Int?
        private var savedGreedyToken: UInt32?
        let usesFusedGreedyHead: Bool
        private(set) var lastGreedyToken: UInt32

        init(vocabSize: Int,
             terminalToken: Int32,
             position: Int,
             usesFusedGreedyHead: Bool = false,
             greedyToken: UInt32 = 0) {
            self.vocabSize = vocabSize
            self.terminalToken = terminalToken
            self.continuationPosition = position
            self.usesFusedGreedyHead = usesFusedGreedyHead
            self.lastGreedyToken = greedyToken
        }

        func reset() {
            resetCalls += 1
            continuationPosition = 0
        }

        func prepareForContinuation(expectedPosition: Int) throws {
            guard continuationPosition == expectedPosition else {
                throw PrefillError.prefillCursorMismatch("test cursor mismatch")
            }
            prepareCalls.append(expectedPosition)
        }

        func savePromptState() {
            saveCalls += 1
            savedPosition = continuationPosition
            savedGreedyToken = lastGreedyToken
        }

        func restorePromptState(expectedPosition: Int) throws {
            guard savedPosition == expectedPosition else {
                throw PrefillError.prefillCursorMismatch("test snapshot cursor mismatch")
            }
            continuationPosition = expectedPosition
            lastGreedyToken = savedGreedyToken ?? 0
            restoreCalls.append(expectedPosition)
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            guard continuationPosition == position else {
                throw PrefillError.prefillCursorMismatch("test scalar cursor mismatch")
            }
            continuationPosition += 1
            writeTerminal(to: logits)
        }

        func prefillChunked(tokens: ArraySlice<Int32>,
                            startPosition: Int,
                            outputMode: PrefillOutputMode,
                            config: PrefillRuntimeConfig,
                            into logits: MTLBuffer,
                            onProgress: (Int) -> Void) async throws -> PrefillResult {
            guard continuationPosition == startPosition else {
                throw PrefillError.prefillCursorMismatch("test prefill cursor mismatch")
            }
            prefillRanges.append(startPosition..<(startPosition + tokens.count))
            continuationPosition += tokens.count
            onProgress(tokens.count)
            writeTerminal(to: logits)
            return PrefillResult(newPosition: continuationPosition,
                                 seed: .logitsWritten)
        }

        func writeTerminal(to logits: MTLBuffer) {
            let pointer = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for index in 0..<vocabSize { pointer[index] = -30 }
            pointer[Int(terminalToken)] = 30
        }
    }

    @Test func resumedChunkedPrefillUsesNonzeroStart() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("one two three four", addBOS: true)
        let cached = prompt.count - 1
        let producer = ContinuationProducer(
            vocabSize: tokenizer.vocabSize,
            terminalToken: tokenizer.eosID,
            position: cached)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        var progress: [(Int, Int)] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: prompt,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .defaultChunked,
            start: .resume(cachedPromptTokens: cached)
        ) { event in
            if case .prefill(let done, let total) = event {
                progress.append((done, total))
            }
        }

        #expect(producer.resetCalls == 0)
        #expect(producer.prepareCalls == [cached])
        #expect(producer.prefillRanges == [cached..<prompt.count])
        #expect(progress.last?.0 == prompt.count)
        #expect(progress.last?.1 == prompt.count)
        #expect(result.prefillTokens == prompt.count)
        #expect(result.cachedPromptTokens == cached)
        #expect(result.computedPrefillTokens == 1)
        #expect(result.kvPosition == prompt.count)
        #expect(result.kvBackedTokenIDs == prompt)
        #expect(result.uncommittedBoundaryTokenIDs == [tokenizer.eosID])
    }

    @Test func resumeRejectsInvalidCachedCountsBeforeMutatingProducer() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("one two", addBOS: true)
        let producer = ContinuationProducer(
            vocabSize: tokenizer.vocabSize,
            terminalToken: tokenizer.eosID,
            position: 0)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        for count in [0, prompt.count, prompt.count + 1] {
            await #expect(throws: GeneratorError.self) {
                _ = try await runRawCompletion(
                    producer: producer,
                    tokenizer: tokenizer,
                    promptIds: prompt,
                    config: GenerationConfig(maxNewTokens: 1, temperature: 0),
                    context: context,
                    scratch: scratch,
                    prefillConfig: .off,
                    start: .resume(cachedPromptTokens: count)
                ) { _ in }
            }
        }

        #expect(producer.resetCalls == 0)
        #expect(producer.prepareCalls.isEmpty)
        #expect(producer.prefillRanges.isEmpty)
    }

    @Test func exactReplaySkipsPrefillAndAccountsForWholePrompt() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("one two three", addBOS: true)
        let producer = ContinuationProducer(
            vocabSize: tokenizer.vocabSize,
            terminalToken: tokenizer.eosID,
            position: prompt.count)
        producer.savePromptState()
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        producer.writeTerminal(to: scratch.logits)

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: prompt,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .defaultChunked,
            start: .replay(cachedPromptTokens: prompt.count),
            capturePromptState: true
        ) { _ in }

        #expect(producer.resetCalls == 0)
        #expect(producer.restoreCalls == [prompt.count])
        #expect(producer.saveCalls == 2)
        #expect(producer.prefillRanges.isEmpty)
        #expect(result.cachedPromptTokens == prompt.count)
        #expect(result.computedPrefillTokens == 0)
        #expect(result.kvPosition == prompt.count)
        #expect(result.uncommittedBoundaryTokenIDs == [tokenizer.eosID])
        #expect(result.promptLogits == scratch.captureLogits())
    }

    @Test func exactReplayUsesRestoredFusedGreedySeed() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("one two three", addBOS: true)
        let greedyToken = tokenizer.encode("a", addBOS: false).first!
        let producer = ContinuationProducer(
            vocabSize: tokenizer.vocabSize,
            terminalToken: tokenizer.eosID,
            position: prompt.count,
            usesFusedGreedyHead: true,
            greedyToken: UInt32(bitPattern: greedyToken))
        producer.savePromptState()
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        producer.writeTerminal(to: scratch.logits)

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: prompt,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .defaultChunked,
            start: .replay(cachedPromptTokens: prompt.count)
        ) { _ in }

        #expect(producer.restoreCalls == [prompt.count])
        #expect(producer.prefillRanges.isEmpty)
        #expect(result.uncommittedBoundaryTokenIDs == [greedyToken])
    }

    @Test func replayRejectsPartialPromptBeforeMutatingProducer() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("one two three", addBOS: true)
        let producer = ContinuationProducer(
            vocabSize: tokenizer.vocabSize,
            terminalToken: tokenizer.eosID,
            position: prompt.count)
        producer.savePromptState()
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        await #expect(throws: GeneratorError.self) {
            _ = try await runRawCompletion(
                producer: producer,
                tokenizer: tokenizer,
                promptIds: prompt,
                config: GenerationConfig(maxNewTokens: 1, temperature: 0),
                context: context,
                scratch: scratch,
                prefillConfig: .defaultChunked,
                start: .replay(cachedPromptTokens: prompt.count - 1)
            ) { _ in }
        }

        #expect(producer.resetCalls == 0)
        #expect(producer.restoreCalls.isEmpty)
        #expect(producer.prefillRanges.isEmpty)
    }
}
