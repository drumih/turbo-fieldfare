import Metal
import Testing
@testable import TurboFieldfare

/// Resuming a multimodal prompt on a cached KV prefix is only safe if the tail
/// handed to the runner reconstructs the rendered prompt exactly. These pin that
/// contract; a rebasing mistake would silently shift image injection and corrupt
/// generation without failing anything else.
@Suite struct MultimodalSuffixPrefillTests {
    private func features(context: MetalContext, tokens: Int) throws -> VisionFeatures {
        let hidden = VisionConfig().textHiddenSize
        let buffer = try #require(context.device.makeBuffer(
            length: tokens * hidden * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        return VisionFeatures(
            buffer: buffer,
            tokenCount: tokens,
            hiddenSize: hidden,
            gpuNanoseconds: 0,
            scratchBytes: 0,
            attentionVariant: .mppTensorOps72,
            projectorPath: .affineThreadgroupBF16Apple10,
            expertResidencyTransition: nil,
            preprocessing: nil)
    }

    private func input(context: MetalContext,
                       total: Int,
                       spans: [(Range<Int>, Int)]) throws -> MultimodalPrefillInput {
        var effective = (0..<total).map { Int32(1_000 + $0) }
        var embedding = effective
        var built: [MultimodalImageSpan] = []
        for (range, tokens) in spans {
            for index in range {
                effective[index] = 258_880
                embedding[index] = 0
            }
            built.append(MultimodalImageSpan(
                tokenRange: range,
                features: try features(context: context, tokens: tokens)))
        }
        return try MultimodalPrefillInput(
            effectiveTokenIDs: effective,
            embeddingTokenIDs: embedding,
            imageSpans: built)
    }

    @Test func suffixReconstructsTheRenderedPromptExactly() throws {
        let context = try MetalContext()
        let full = try input(context: context, total: 120,
                             spans: [(40..<50, 10), (80..<95, 15)])
        for cached in [1, 20, 39, 40] {
            let suffix = try full.suffix(dropping: cached)
            #expect(Array(full.effectiveTokenIDs.dropFirst(cached))
                    == suffix.effectiveTokenIDs)
            #expect(Array(full.embeddingTokenIDs.dropFirst(cached))
                    == suffix.embeddingTokenIDs)
            // Every span must land on the same absolute tokens once the start
            // position is added back.
            #expect(suffix.imageSpans.count == full.imageSpans.count)
            for (original, rebased) in zip(full.imageSpans, suffix.imageSpans) {
                #expect(rebased.tokenRange.lowerBound + cached
                        == original.tokenRange.lowerBound)
                #expect(rebased.tokenRange.upperBound + cached
                        == original.tokenRange.upperBound)
                #expect(rebased.features.tokenCount == original.features.tokenCount)
            }
        }
    }

    @Test func spansAlreadyInsideTheCachedPrefixAreDropped() throws {
        let context = try MetalContext()
        let full = try input(context: context, total: 120,
                             spans: [(10..<20, 10), (80..<95, 15)])
        let suffix = try full.suffix(dropping: 60)
        #expect(suffix.imageSpans.count == 1)
        #expect(suffix.imageSpans[0].tokenRange == 20..<35)
        #expect(Array(full.effectiveTokenIDs.dropFirst(60))
                == suffix.effectiveTokenIDs)
    }

    /// A span crossing the boundary cannot be split: half its features are
    /// already in the KV. It must be refused, not silently truncated.
    @Test func spanStraddlingTheCachedBoundaryIsRefused() throws {
        let context = try MetalContext()
        let full = try input(context: context, total: 120,
                             spans: [(40..<50, 10)])
        #expect(throws: MultimodalPrefillInputError.self) {
            _ = try full.suffix(dropping: 45)
        }
    }

    @Test func outOfRangeCachedCountsAreRefused() throws {
        let context = try MetalContext()
        let full = try input(context: context, total: 60, spans: [(10..<20, 10)])
        #expect(throws: MultimodalPrefillInputError.self) {
            _ = try full.suffix(dropping: 60)
        }
        #expect(throws: MultimodalPrefillInputError.self) {
            _ = try full.suffix(dropping: -1)
        }
    }
}

/// The continuation bridge must produce the same tokens a full render produces
/// for the same turn, or the KV lineage silently diverges from the prompt.
@Suite struct MultimodalContinuationBridgeTests {
    @Test(arguments: [[280], [17], [280, 17]])
    func bridgeMatchesAFullRenderOfTheSameTurn(counts: [Int]) async throws {
        let tokenizer = try await GFTokenizer.load()
        var parts: [MultimodalContinuationPart] = [.text("look at ")]
        for _ in counts { parts.append(.image) }
        parts.append(.text(" and describe"))

        let bridge = try tokenizer.encodeMultimodalUserContinuation(
            textAndImages: parts, imageTokenCounts: counts)

        // What a full render of the same user turn expands to, built from the
        // renderer's own marker expansion rules.
        let markers = counts.map { _ in MultimodalPromptRenderer.placeholder }
            .joined(separator: "")
        let text = ("look at " + markers + " and describe")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let template = [tokenizer.endOfTurnID] + tokenizer.encode(
            "\n<|turn>user\n\(text)<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
            addBOS: false)
        var expected: [Int32] = []
        var expectedEmbedding: [Int32] = []
        var index = 0
        for token in template {
            guard token == MultimodalPromptRenderer.imageTokenID else {
                expected.append(token); expectedEmbedding.append(token); continue
            }
            expected.append(MultimodalPromptRenderer.beginImageTokenID)
            expectedEmbedding.append(MultimodalPromptRenderer.beginImageTokenID)
            expected.append(contentsOf: repeatElement(
                MultimodalPromptRenderer.imageTokenID, count: counts[index]))
            expectedEmbedding.append(contentsOf: repeatElement(
                Int32(0), count: counts[index]))
            expected.append(MultimodalPromptRenderer.endImageTokenID)
            expectedEmbedding.append(MultimodalPromptRenderer.endImageTokenID)
            index += 1
        }
        #expect(bridge.effectiveTokenIDs == expected)
        #expect(bridge.embeddingTokenIDs == expectedEmbedding)
        #expect(bridge.imageTokenRanges.count == counts.count)
        for (range, count) in zip(bridge.imageTokenRanges, counts) {
            #expect(range.count == count)
            #expect(bridge.effectiveTokenIDs[range]
                    .allSatisfy { $0 == MultimodalPromptRenderer.imageTokenID })
            #expect(bridge.effectiveTokenIDs[range.lowerBound - 1]
                    == MultimodalPromptRenderer.beginImageTokenID)
            #expect(bridge.effectiveTokenIDs[range.upperBound]
                    == MultimodalPromptRenderer.endImageTokenID)
        }
    }

    @Test func bridgeStartsWithTheTurnBoundaryTheCacheExpects() async throws {
        let tokenizer = try await GFTokenizer.load()
        let bridge = try tokenizer.encodeMultimodalUserContinuation(
            textAndImages: [.image, .text("what is it")], imageTokenCounts: [42])
        #expect(bridge.effectiveTokenIDs.first == tokenizer.endOfTurnID)
        #expect(bridge.effectiveTokenIDs.count == bridge.embeddingTokenIDs.count)
    }

    @Test func mismatchedImageCountsAreRefused() async throws {
        let tokenizer = try await GFTokenizer.load()
        #expect(throws: MultimodalPromptRendererError.self) {
            _ = try tokenizer.encodeMultimodalUserContinuation(
                textAndImages: [.image, .image], imageTokenCounts: [10])
        }
        #expect(throws: MultimodalPromptRendererError.self) {
            _ = try tokenizer.encodeMultimodalUserContinuation(
                textAndImages: [.text("no images")], imageTokenCounts: [])
        }
    }

    @Test func aReservedMarkerInUserTextIsRefused() async throws {
        let tokenizer = try await GFTokenizer.load()
        #expect(throws: MultimodalPromptRendererError.self) {
            _ = try tokenizer.encodeMultimodalUserContinuation(
                textAndImages: [.text(MultimodalPromptRenderer.placeholder), .image],
                imageTokenCounts: [10])
        }
    }

    /// A reserved marker split across two text parts survives a per-part check
    /// but reassembles on concatenation into a real image token. Left unguarded
    /// it produced more placeholders than counts and trapped while indexing —
    /// reachable from a request body, so a remote crash.
    @Test func aMarkerSplitAcrossTextPartsIsRefusedRatherThanTrapping() async throws {
        let tokenizer = try await GFTokenizer.load()
        let marker = MultimodalPromptRenderer.placeholder
        let cut = marker.index(marker.startIndex, offsetBy: 4)
        #expect(throws: MultimodalPromptRendererError.self) {
            _ = try tokenizer.encodeMultimodalUserContinuation(
                textAndImages: [.text(String(marker[..<cut])),
                                .text(String(marker[cut...])),
                                .image],
                imageTokenCounts: [10])
        }
    }
}
