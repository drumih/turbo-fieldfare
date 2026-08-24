import Metal
import Testing
@testable import TurboFieldfare

@Suite struct MultimodalPromptRendererTests {
    @Test func expandsImagesInContentOrderAndZerosEmbeddingRows() async throws {
        let tokenizer = try await GFTokenizer.load()
        let context = try MetalContext()
        let firstID = UUID()
        let secondID = UUID()
        let first = try features(rows: 2, context: context)
        let second = try features(rows: 3, context: context)

        let input = try MultimodalPromptRenderer.render(
            messages: [MultimodalMessage(
                role: .user,
                content: [
                    .text("before"), .image(id: firstID),
                    .text("between"), .image(id: secondID), .text("after"),
                ])],
            featuresByID: [secondID: second, firstID: first],
            tokenizer: tokenizer)

        #expect(input.imageSpans.map(\.features.tokenCount) == [2, 3])
        for span in input.imageSpans {
            #expect(input.embeddingTokenIDs[span.tokenRange].allSatisfy { $0 == 0 })
            #expect(input.effectiveTokenIDs[span.tokenRange].allSatisfy {
                $0 == MultimodalPromptRenderer.imageTokenID
            })
            #expect(input.effectiveTokenIDs[span.tokenRange.lowerBound - 1]
                == MultimodalPromptRenderer.beginImageTokenID)
            #expect(input.effectiveTokenIDs[span.tokenRange.upperBound]
                == MultimodalPromptRenderer.endImageTokenID)
        }
    }

    @Test func rejectsReservedMarkersAndMissingFeatures() async throws {
        let tokenizer = try await GFTokenizer.load()
        #expect(throws: MultimodalPromptRendererError.reservedImageMarker) {
            try MultimodalPromptRenderer.render(
                messages: [MultimodalMessage(
                    role: .user,
                    content: [.text("typed <|image|> marker")])],
                featuresByID: [:],
                tokenizer: tokenizer)
        }
        let missing = UUID()
        #expect(throws: MultimodalPromptRendererError.missingImage(missing)) {
            try MultimodalPromptRenderer.render(
                messages: [MultimodalMessage(
                    role: .user,
                    content: [.image(id: missing)])],
                featuresByID: [:],
                tokenizer: tokenizer)
        }
    }

    private func features(rows: Int, context: MetalContext) throws -> VisionFeatures {
        let bytes = rows * VisionConfig().textHiddenSize * MemoryLayout<Float16>.stride
        let buffer = try #require(context.device.makeBuffer(
            length: bytes, options: .storageModePrivate))
        return VisionFeatures(
            buffer: buffer,
            tokenCount: rows,
            hiddenSize: VisionConfig().textHiddenSize,
            gpuNanoseconds: 0,
            scratchBytes: bytes,
            attentionVariant: .native72Q16,
            projectorPath: .affineThreadgroupF16,
            expertResidencyTransition: nil,
            preprocessing: nil)
    }
}
