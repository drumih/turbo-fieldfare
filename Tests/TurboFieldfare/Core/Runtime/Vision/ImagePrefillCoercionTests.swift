import Testing

@testable import TurboFieldfare

/// Image spans are only served by the chunked multimodal path. Prefill mode is
/// a performance choice; whether images work at all is not, so an image turn is
/// coerced rather than refused after every image has already been encoded.
@Suite struct ImagePrefillCoercionTests {
    @Test func anImagePromptCoercesPrefillOffToChunked() {
        let off = PrefillRuntimeConfig.off
        #expect(!off.servesImagePrompt)

        let coerced = try? #require(off.coercedForImagePrompt())
        #expect(coerced?.mode == .chunked)
        #expect(coerced?.servesImagePrompt == true)
    }

    @Test func aChunkedConfigIsAlreadyFineAndIsNotRewritten() {
        let chunked = PrefillRuntimeConfig.defaultChunked
        #expect(chunked.servesImagePrompt)
        // nil means "already runs", not "cannot run".
        #expect(chunked.coercedForImagePrompt() == nil)
    }

    @Test func everySelectableChunkSizeStillServesImages() {
        for tokens in RuntimeConfiguration.allowedPrefillChunkTokens {
            let config = PrefillRuntimeConfig.production(chunkTokens: tokens)
            #expect(config.servesImagePrompt,
                    "chunk size \(tokens) must serve an image prompt")
            #expect(config.coercedForImagePrompt() == nil)
        }
    }

    /// The planner cuts text at the same clamp the scratch layout applies. A
    /// caller that asks for more must not hand the buffer a larger chunk than
    /// it was sized for.
    @Test func multimodalTextChunksNeverExceedTheScratchClamp() {
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: 1_000,
            imageRanges: [400..<680],
            chunkTokens: 4_096)
        let textItems = work.filter { !$0.isImage }
        #expect(!textItems.isEmpty)
        for item in textItems {
            #expect(item.range.count <= PrefillRuntimeConfig.maxChunkTokens)
        }
    }

    /// An image span is one indivisible block: its features are projected as a
    /// unit, so the planner must never split it even when it exceeds the clamp.
    @Test func anImageSpanPassesThroughWholeEvenWhenItExceedsTheClamp() {
        let span = 400..<680
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: 1_000, imageRanges: [span], chunkTokens: 128)
        let images = work.filter(\.isImage)
        #expect(images.count == 1)
        #expect(images.first?.range == span)
        #expect(span.count > PrefillRuntimeConfig.maxChunkTokens)
    }

    @Test func workItemsCoverTheWholePromptInOrder() {
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: 500, imageRanges: [10..<40, 200..<260], chunkTokens: 64)
        #expect(work.first?.range.lowerBound == 0)
        #expect(work.last?.range.upperBound == 500)
        for (previous, next) in zip(work, work.dropFirst()) {
            #expect(previous.range.upperBound == next.range.lowerBound)
        }
        #expect(work.filter(\.isImage).map(\.range) == [10..<40, 200..<260])
    }
}
