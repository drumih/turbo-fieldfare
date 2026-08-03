import Metal
import Testing

@testable import TurboFieldfare

private final class MultimodalTestProducer:
    LogitProducer, MultimodalChunkedPrefillRunner, @unchecked Sendable {
    let vocabSize: Int
    private let firstToken: Int32
    private(set) var resetCalls = 0
    private(set) var produceCalls = 0
    private(set) var textPrefillCalls = 0
    private(set) var multimodalPrefillCalls = 0
    private(set) var receivedRanges: [Range<Int>] = []
    private(set) var receivedConfig: PrefillRuntimeConfig?

    init(vocabSize: Int, firstToken: Int32) {
        self.vocabSize = vocabSize
        self.firstToken = firstToken
    }

    func reset() {
        resetCalls += 1
        produceCalls = 0
        textPrefillCalls = 0
        multimodalPrefillCalls = 0
        receivedRanges = []
        receivedConfig = nil
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
        textPrefillCalls += 1
        return try writeSeed(
            tokenCount: tokens.count,
            startPosition: startPosition,
            logits: logits,
            onProgress: onProgress)
    }

    func prefillMultimodal(tokens: ArraySlice<Int32>,
                           startPosition: Int,
                           input: MultimodalPrefillInput,
                           outputMode: PrefillOutputMode,
                           config: PrefillRuntimeConfig,
                           into logits: MTLBuffer,
                           onProgress: (Int) -> Void) async throws -> PrefillResult {
        multimodalPrefillCalls += 1
        receivedRanges = input.visionTokenRanges
        receivedConfig = config
        return try writeSeed(
            tokenCount: tokens.count,
            startPosition: startPosition,
            logits: logits,
            onProgress: onProgress)
    }

    private func writeSeed(tokenCount: Int,
                           startPosition: Int,
                           logits: MTLBuffer,
                           onProgress: (Int) -> Void) throws -> PrefillResult {
        let pointer = logits.contents().bindMemory(
            to: Float16.self,
            capacity: vocabSize)
        for index in 0..<vocabSize {
            pointer[index] = Float16(-30)
        }
        pointer[Int(firstToken)] = Float16(30)
        onProgress(tokenCount)
        return PrefillResult(
            newPosition: startPosition + tokenCount,
            seed: .logitsWritten)
    }
}

extension RawCompletionLoopTests {
    @Test func multimodalPromptForcesCorrectChunkedEntryPointWhenTextPrefillIsOff() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let prepared = try tokenizer.prepareChatPrompt([
            GFTokenizer.Message(
                role: .user,
                content: "Describe the image.",
                imageTokenCounts: [3]),
        ])
        let imageRange = try #require(prepared.imageSpans.first)
        let hiddenSize = 2_816
        let overlayBytes = imageRange.count * hiddenSize * MemoryLayout<Float16>.stride
        let overlayBuffer = try #require(context.device.makeBuffer(
            length: overlayBytes,
            options: .storageModeShared))
        let multimodal = MultimodalPrefillInput(embeddingOverlays: [
            PrefillEmbeddingOverlay(
                tokenRange: imageRange,
                buffer: overlayBuffer,
                hiddenSize: hiddenSize),
        ])
        let tokenA = try #require(tokenizer.encode("a", addBOS: false).first)
        let producer = MultimodalTestProducer(
            vocabSize: tokenizer.vocabSize,
            firstToken: tokenA)
        let scratch = try RawCompletionScratch(
            context: context,
            vocab: tokenizer.vocabSize)
        var progress: [(Int, Int)] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: prepared.tokenIDs,
            multimodalInput: multimodal,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off) { event in
                if case .prefill(let done, let total) = event {
                    progress.append((done, total))
                }
            }

        #expect(result.newTokens == 1)
        #expect(producer.multimodalPrefillCalls == 1)
        #expect(producer.textPrefillCalls == 0)
        #expect(producer.produceCalls == 0)
        #expect(producer.receivedRanges == [imageRange])
        #expect(producer.receivedConfig == .defaultChunked)
        #expect(progress.last?.0 == prepared.tokenIDs.count)
        #expect(progress.last?.1 == prepared.tokenIDs.count)
    }

    @Test func multimodalPromptRejectsOverlayOutsideImagePlaceholdersBeforeReset() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let promptIDs = tokenizer.encode("ordinary text", addBOS: true)
        let overlayBuffer = try #require(context.device.makeBuffer(
            length: 2_816 * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let multimodal = MultimodalPrefillInput(embeddingOverlays: [
            PrefillEmbeddingOverlay(
                tokenRange: 0..<1,
                buffer: overlayBuffer,
                hiddenSize: 2_816),
        ])
        let tokenA = try #require(tokenizer.encode("a", addBOS: false).first)
        let producer = MultimodalTestProducer(
            vocabSize: tokenizer.vocabSize,
            firstToken: tokenA)
        let scratch = try RawCompletionScratch(
            context: context,
            vocab: tokenizer.vocabSize)

        do {
            _ = try await runRawCompletion(
                producer: producer,
                tokenizer: tokenizer,
                promptIds: promptIDs,
                multimodalInput: multimodal,
                config: GenerationConfig(maxNewTokens: 1, temperature: 0),
                context: context,
                scratch: scratch,
                prefillConfig: .off) { _ in }
            Issue.record("expected image-placeholder alignment failure")
        } catch let error as PrefillError {
            guard case .chunkedUnsupported(let reason) = error else {
                Issue.record("unexpected PrefillError \(error)")
                return
            }
            #expect(reason.contains("does not align"))
        }

        #expect(producer.resetCalls == 0)
        #expect(producer.multimodalPrefillCalls == 0)
    }
}
