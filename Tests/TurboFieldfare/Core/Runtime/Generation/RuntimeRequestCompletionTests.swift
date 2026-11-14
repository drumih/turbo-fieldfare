import Foundation
import Metal
import Synchronization
import Testing
@testable import TurboFieldfare

@Suite struct RuntimeRequestCompletionTests {
    enum Failure: Error, Equatable { case operation, drain }

    /// No model or kernel-forward work: state is locked, while each request
    /// exclusively owns the supplied logits buffer, as in the production API.
    final class Producer: ChunkedPrefillRunner, RequestFinishingProducer {
        struct State {
            var events: [String] = []
            var operationError: (any Error)?
            var drainError: (any Error)?
        }
        let vocab: Int
        let token: Int32
        let state = Mutex(State())

        init(vocab: Int, token: Int32) { self.vocab = vocab; self.token = token }

        func reset() { state.withLock { $0.events.append("reset") } }
        func finishRequest() throws {
            let error = state.withLock { state in
                state.events.append("finish")
                // Standalone prefill's entry fence must succeed before the
                // controlled operation/final-drain failure is introduced.
                return state.events.contains("prefill") ? state.drainError : nil
            }
            if let error { throw error }
        }
        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            Issue.record("one-token chunked completion must not call scalar produce")
        }
        func prefillChunked(tokens: ArraySlice<Int32>, startPosition: Int,
                            outputMode: PrefillOutputMode, config: PrefillRuntimeConfig,
                            into logits: MTLBuffer, onProgress: (Int) -> Void) async throws -> PrefillResult {
            let error = state.withLock { state in
                state.events.append("prefill")
                return state.operationError
            }
            if let error { throw error }
            let values = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
            for index in 0..<vocab { values[index] = -30 }
            values[Int(token)] = 30
            onProgress(tokens.count)
            return PrefillResult(newPosition: startPosition + tokens.count, seed: .logitsWritten)
        }
    }

    @Test func completionFencesBeforeResetAndFinishesWithoutIntermediatePrefillDrain() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let producer = Producer(vocab: tokenizer.vocabSize, token: tokenizer.eosID)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        let result = try await runRawCompletion(
            producer: producer, tokenizer: tokenizer, promptIds: [1],
            config: GenerationConfig(maxNewTokens: 1, temperature: 0), context: context,
            scratch: scratch, prefillConfig: .defaultChunked) { _ in }
        #expect(result.newTokens == 1)
        #expect(producer.state.withLock { $0.events } == ["finish", "reset", "prefill", "finish"])
    }

    @Test func successfulCompletionCannotHideFinalDrainFailure() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let producer = Producer(vocab: tokenizer.vocabSize, token: tokenizer.eosID)
        producer.state.withLock { $0.drainError = Failure.drain }
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        do {
            _ = try await runRawCompletion(
                producer: producer, tokenizer: tokenizer, promptIds: [1],
                config: GenerationConfig(maxNewTokens: 1, temperature: 0), context: context,
                scratch: scratch, prefillConfig: .defaultChunked) { _ in }
            Issue.record("final drain failure must prevent success")
        } catch {
            #expect(error as? Failure == .drain)
        }
        #expect(producer.state.withLock { $0.events } == ["finish", "reset", "prefill", "finish"])
    }

    @Test func completionFailureDrainsBeforeResetAndChunkedRetry() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let producer = Producer(vocab: tokenizer.vocabSize, token: tokenizer.eosID)
        producer.state.withLock { $0.operationError = Failure.operation; $0.drainError = Failure.drain }
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        do {
            _ = try await runRawCompletion(
                producer: producer, tokenizer: tokenizer, promptIds: [1],
                config: GenerationConfig(maxNewTokens: 1, temperature: 0), context: context,
                scratch: scratch, prefillConfig: .defaultChunked) { _ in }
            Issue.record("operation and cleanup must fail")
        } catch {
            let combined = try #require(error as? RequestCompletionFailure)
            #expect(combined.primary as? Failure == .operation)
            #expect(combined.drain as? Failure == .drain)
            #expect(producer.state.withLock { $0.events.last } == "finish")
            try MultimodalConversationKVRecovery.restoreAfterFailure(
                positionBefore: 0, generationError: error, rewind: { _ in }, reset: producer.reset)
        }
        #expect(producer.state.withLock { $0.events } == ["finish", "reset", "prefill", "finish", "reset"])
        producer.state.withLock { $0.operationError = nil; $0.drainError = nil }
        _ = try await runRawCompletion(
            producer: producer, tokenizer: tokenizer, promptIds: [1],
            config: GenerationConfig(maxNewTokens: 1, temperature: 0), context: context,
            scratch: scratch, prefillConfig: .defaultChunked) { _ in }
        #expect(producer.state.withLock { Array($0.events.suffix(4)) } == ["finish", "reset", "prefill", "finish"])
    }

    @Test func completionCancellationRetainsDrainContextAndPrimaryClassification() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let producer = Producer(vocab: tokenizer.vocabSize, token: tokenizer.eosID)
        producer.state.withLock { $0.drainError = Failure.drain }
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        do {
            // Cancel the actual request task at the natural final-prefill
            // callback. RawCompletion's next Task.checkCancellation throws.
            let task = Task {
                try await runRawCompletion(
                    producer: producer, tokenizer: tokenizer, promptIds: [1],
                    config: GenerationConfig(maxNewTokens: 1, temperature: 0), context: context,
                    scratch: scratch, prefillConfig: .defaultChunked) { progress in
                        if case .prefill = progress {
                            withUnsafeCurrentTask { $0?.cancel() }
                        }
                    }
            }
            _ = try await task.value
            Issue.record("cancellation must propagate")
        } catch {
            #expect(requestPrimaryError(error) is CancellationError)
            #expect((error as? RequestCompletionFailure)?.drain as? Failure == .drain)
            #expect(producer.state.withLock { $0.events.last } == "finish")
        }
    }

    @Test func standaloneRestorePrefillFencesEntryAndFinishesOnSuccess() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let producer = Producer(vocab: tokenizer.vocabSize, token: tokenizer.eosID)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        _ = try await runRawPrefill(
            producer: producer, promptIds: [1], multimodalInput: nil,
            prefillConfig: .defaultChunked, start: .reset, outputMode: .logits,
            seedUse: .discarded, historyReserve: 1, scratch: scratch) { _ in }
        #expect(producer.state.withLock { $0.events } == ["finish", "reset", "prefill", "finish"])
    }

    @Test func standaloneRestorePrefillPreservesOperationAndDrainErrors() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let producer = Producer(vocab: tokenizer.vocabSize, token: tokenizer.eosID)
        producer.state.withLock { $0.operationError = Failure.operation; $0.drainError = Failure.drain }
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        do {
            _ = try await runRawPrefill(
                producer: producer, promptIds: [1], multimodalInput: nil,
                prefillConfig: .defaultChunked, start: .reset, outputMode: .logits,
                seedUse: .discarded, historyReserve: 1, scratch: scratch) { _ in }
            Issue.record("restore-prefill errors must propagate")
        } catch {
            #expect(requestPrimaryError(error) as? Failure == .operation)
            #expect((error as? RequestCompletionFailure)?.drain as? Failure == .drain)
        }
        #expect(producer.state.withLock { $0.events } == ["finish", "reset", "prefill", "finish"])
    }
}
