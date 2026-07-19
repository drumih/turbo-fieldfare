import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Shared raw-completion loop validation: stops, progress callbacks, tail
/// flush, and cancellation, via `ScriptedLogitProducer` (kernel-independent).
@Suite struct RawCompletionLoopTests {

    func automaton(_ seq: [Int32], end: Int32) -> @Sendable (Int32, Int) -> ScriptedLogitProducer.Step {
        let next: [Int32: Int32] = {
            var n: [Int32: Int32] = [:]
            for i in 0..<max(0, seq.count - 1) { n[seq[i]] = seq[i + 1] }
            if let last = seq.last { n[last] = end }
            return n
        }()
        let first = seq.first ?? end
        return { input, _ in .argmax(next[input] ?? first) }
    }

    struct Collected {
        var prefills: [(Int, Int)] = []
        var tokens: [(Int, Int32, String)] = []
        var tails: [String] = []
    }

    final class CountingProducer: LogitProducer, ContextWindowReporting, @unchecked Sendable {
        let vocabSize: Int
        let maxContext: Int
        private let step: @Sendable (Int32, Int) -> ScriptedLogitProducer.Step
        private var calls = 0
        private(set) var resetCalls = 0
        private(set) var produceCalls = 0

        init(vocabSize: Int,
             maxContext: Int = Int.max,
             step: @escaping @Sendable (Int32, Int) -> ScriptedLogitProducer.Step) {
            self.vocabSize = vocabSize
            self.maxContext = maxContext
            self.step = step
        }

        func reset() {
            resetCalls += 1
            calls = 0
            produceCalls = 0
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            produceCalls += 1
            let spec = step(token, calls)
            calls += 1
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            switch spec {
            case .argmax(let token):
                for i in 0..<vocabSize { ptr[i] = Float16(-30.0) }
                if Int(token) >= 0 && Int(token) < vocabSize {
                    ptr[Int(token)] = Float16(30.0)
                }
            case .vector(let values):
                for i in 0..<vocabSize {
                    ptr[i] = Float16(i < values.count ? values[i] : -30.0)
                }
            }
        }

    }

    final class ChunkedTestProducer: LogitProducer, ChunkedPrefillRunner, @unchecked Sendable {
        let vocabSize: Int
        private let firstToken: Int32
        private let seed: PrefillSeed
        private(set) var resetCalls = 0
        private(set) var produceCalls = 0
        private(set) var chunkedCalls = 0
        private(set) var lastOutputMode: PrefillOutputMode?
        private(set) var lastConfig: PrefillRuntimeConfig?

        init(vocabSize: Int,
             firstToken: Int32,
             seed: PrefillSeed = .logitsWritten) {
            self.vocabSize = vocabSize
            self.firstToken = firstToken
            self.seed = seed
        }

        func reset() {
            resetCalls += 1
            produceCalls = 0
            chunkedCalls = 0
            lastOutputMode = nil
            lastConfig = nil
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            produceCalls += 1
        }

        func prefillChunked(tokens: ArraySlice<Int32>,
                            startPosition: Int,
                            outputMode: PrefillOutputMode,
                            config: PrefillRuntimeConfig,
                            into logits: MTLBuffer,
                            onProgress: (Int) -> Void) async throws -> PrefillResult {
            chunkedCalls += 1
            lastOutputMode = outputMode
            lastConfig = config
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for i in 0..<vocabSize { ptr[i] = Float16(-30.0) }
            ptr[Int(firstToken)] = Float16(30.0)
            onProgress(tokens.count)
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: seed)
        }
    }

    func runLoop(seq: [Int32], end: Int32, prompt: String = "go",
                         config: GenerationConfig) async throws -> (Collected, RawDecodeResult) {
        let ctx = try MetalContext()
        let tok = try await GFTokenizer.load()
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize,
                                             step: automaton(seq, end: end))
        let promptIds = tok.encode(prompt, addBOS: true)
        let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)
        var collected = Collected()
        let result = try await runRawCompletion(producer: producer, tokenizer: tok,
                                                promptIds: promptIds, config: config,
                                                context: ctx, scratch: scratch,
                                                prefillConfig: .off) { progress in
            switch progress {
            case .prefill(let done, let total): collected.prefills.append((done, total))
            case .token(let index, let id, let delta): collected.tokens.append((index, id, delta))
            case .tail(let text): collected.tails.append(text)
            }
        }
        return (collected, result)
    }

}
