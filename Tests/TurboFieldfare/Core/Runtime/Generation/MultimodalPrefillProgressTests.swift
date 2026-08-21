import Metal
import Testing
@testable import TurboFieldfare

/// Progress reporting for a resumed multimodal prefill, and the two defaults it
/// travels with. All three are review findings whose only previous evidence was
/// that the build succeeded.
@Suite struct MultimodalPrefillProgressTests {
    /// A producer that prefills a multimodal suffix and reports suffix-local
    /// progress, exactly as `RealForwardRunner` does.
    final class MultimodalProducer: MultimodalPrefillRunner, ContinuableLogitProducer,
        @unchecked Sendable
    {
        let vocabSize: Int
        private let terminalToken: Int32
        private(set) var continuationPosition: Int

        init(vocabSize: Int, terminalToken: Int32, position: Int) {
            self.vocabSize = vocabSize
            self.terminalToken = terminalToken
            self.continuationPosition = position
        }

        func reset() { continuationPosition = 0 }

        func prepareForContinuation(expectedPosition: Int) throws {
            guard continuationPosition == expectedPosition else {
                throw PrefillError.prefillCursorMismatch("test cursor mismatch")
            }
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            guard continuationPosition == position else {
                throw PrefillError.prefillCursorMismatch("test scalar cursor mismatch")
            }
            continuationPosition += 1
            writeTerminal(to: logits)
        }

        func prefillMultimodal(input: MultimodalPrefillInput,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
            guard continuationPosition == startPosition else {
                throw PrefillError.prefillCursorMismatch("test multimodal cursor mismatch")
            }
            // Suffix-local counts, one per token, as the chunked runner reports.
            for done in 1...input.effectiveTokenIDs.count { onProgress(done) }
            continuationPosition += input.effectiveTokenIDs.count
            writeTerminal(to: logits)
            return PrefillResult(newPosition: continuationPosition, seed: .logitsWritten)
        }

        private func writeTerminal(to logits: MTLBuffer) {
            let pointer = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for index in 0..<vocabSize { pointer[index] = -30 }
            pointer[Int(terminalToken)] = 30
        }
    }

    private func makeInput(context: MetalContext,
                           tokenIDs: [Int32]) throws -> MultimodalPrefillInput {
        let hidden = VisionConfig().textHiddenSize
        let buffer = try #require(context.device.makeBuffer(
            length: 2 * hidden * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let features = VisionFeatures(
            buffer: buffer, tokenCount: 2, hiddenSize: hidden,
            gpuNanoseconds: 0, scratchBytes: 0,
            attentionVariant: .native72Q16, projectorPath: .fallback,
            expertResidencyTransition: nil, preprocessing: nil)
        var embedding = tokenIDs
        embedding[1] = 0
        embedding[2] = 0
        return try MultimodalPrefillInput(
            effectiveTokenIDs: tokenIDs,
            embeddingTokenIDs: embedding,
            imageSpans: [MultimodalImageSpan(tokenRange: 1..<3, features: features)])
    }

    /// A resumed image turn must report progress against the whole prompt. The
    /// producer counts only the suffix it prefills, so without the cached offset
    /// a 4-token image turn on a 5,000-token KV reported "1 of 5,004" and the
    /// bar restarted from zero on every turn.
    @Test func resumedMultimodalProgressCountsTheCachedPrefix() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let cached = 6
        let suffix: [Int32] = [40, 41, 42, 43]
        let promptIds: [Int32] = Array(repeating: 7, count: cached) + suffix
        let terminal = tokenizer.eosID
        let producer = MultimodalProducer(
            vocabSize: 64, terminalToken: terminal, position: cached)
        let scratch = try RawCompletionScratch(context: context, vocab: 64)
        var config = GenerationConfig(maxNewTokens: 2, temperature: 0)
        config.stopStrings = []

        var prefillProgress: [(done: Int, total: Int)] = []
        _ = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIds,
            multimodalInput: try makeInput(context: context, tokenIDs: suffix),
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: .production(chunkTokens: 32),
            start: .resume(cachedPromptTokens: cached)
        ) { progress in
            if case .prefill(let done, let total) = progress {
                prefillProgress.append((done, total))
            }
        }

        #expect(prefillProgress.map(\.done) == [7, 8, 9, 10],
                "progress ignored the \(cached) cached tokens: \(prefillProgress.map(\.done))")
        #expect(prefillProgress.allSatisfy { $0.total == promptIds.count })
        #expect(prefillProgress.last?.done == promptIds.count,
                "prefill finished below its own total")
        // Monotonic, and never above the total — a bar that goes backwards or
        // past the end is what the user actually sees.
        #expect(zip(prefillProgress, prefillProgress.dropFirst())
            .allSatisfy { $0.done < $1.done })
        #expect(prefillProgress.allSatisfy { $0.done <= $0.total })
    }

    /// Every entry point must default to the same residency. The conversation's
    /// default had drifted to `keepReady`, which pins ~1.1 GB of tower mappings
    /// for the life of the process while the CLI released them.
    @Test func everyEntryPointDefaultsToTheSameResidency() {
        #expect(VisionResidencyPolicy.defaultPolicy == .onDemand)
        #expect(VisionResidencyPolicy(rawValue: "keep-ready") == .keepReady,
                "the opt-in policy is still spelled keep-ready")
    }
}
