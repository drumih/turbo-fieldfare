import Metal
import Testing
@testable import TurboFieldfare

@Suite struct MultimodalPrefillInputTests {
    @Test func validatesProjectedFeatureShapeAndDualTokenStreams() throws {
        let context = try MetalContext()
        let rows = 4
        let bytes = rows * VisionConfig().textHiddenSize * MemoryLayout<Float16>.stride
        let buffer = try #require(context.device.makeBuffer(
            length: bytes,
            options: .storageModePrivate))
        let features = VisionFeatures(
            buffer: buffer,
            tokenCount: rows,
            hiddenSize: VisionConfig().textHiddenSize,
            gpuNanoseconds: 0,
            scratchBytes: bytes,
            attentionVariant: .native72Q16,
            projectorPath: .affineThreadgroupF16,
            expertResidencyTransition: nil,
            preprocessing: nil)

        let input = try MultimodalPrefillInput(
            effectiveTokenIDs: [10, 20, 20, 20, 20, 30],
            embeddingTokenIDs: [10, 0, 0, 0, 0, 30],
            imageTokenRange: 1..<5,
            imageFeatures: features)

        #expect(input.imageTokenRange == 1..<5)
        #expect(input.effectiveTokenIDs[1] == 20)
        #expect(input.embeddingTokenIDs[1] == 0)
    }

    @Test func rejectsFeatureTokenCountMismatch() throws {
        let context = try MetalContext()
        let buffer = try #require(context.device.makeBuffer(
            length: 3 * VisionConfig().textHiddenSize * MemoryLayout<Float16>.stride,
            options: .storageModePrivate))
        let features = VisionFeatures(
            buffer: buffer,
            tokenCount: 3,
            hiddenSize: VisionConfig().textHiddenSize,
            gpuNanoseconds: 0,
            scratchBytes: buffer.length,
            attentionVariant: .native72Q16,
            projectorPath: .affineThreadgroupF16,
            expertResidencyTransition: nil,
            preprocessing: nil)

        #expect(throws: MultimodalPrefillInputError.featureShapeMismatch) {
            try MultimodalPrefillInput(
                effectiveTokenIDs: [10, 20, 20, 30],
                embeddingTokenIDs: [10, 0, 0, 30],
                imageTokenRange: 1..<3,
                imageFeatures: features)
        }
    }

    @Test func acceptsOrderedNonOverlappingImageSpans() throws {
        let context = try MetalContext()
        func features(_ rows: Int) throws -> VisionFeatures {
            let bytes = rows * VisionConfig().textHiddenSize * MemoryLayout<Float16>.stride
            let buffer = try #require(context.device.makeBuffer(
                length: bytes, options: .storageModePrivate))
            return VisionFeatures(
                buffer: buffer, tokenCount: rows,
                hiddenSize: VisionConfig().textHiddenSize,
                gpuNanoseconds: 0, scratchBytes: bytes,
                attentionVariant: .native72Q16,
                projectorPath: .affineThreadgroupF16,
                expertResidencyTransition: nil, preprocessing: nil)
        }
        let input = try MultimodalPrefillInput(
            effectiveTokenIDs: [1, 2, 2, 3, 4, 4, 4, 5],
            embeddingTokenIDs: [1, 0, 0, 3, 0, 0, 0, 5],
            imageSpans: [
                MultimodalImageSpan(tokenRange: 1..<3, features: try features(2)),
                MultimodalImageSpan(tokenRange: 4..<7, features: try features(3)),
            ])
        #expect(input.imageSpans.map(\.tokenRange) == [1..<3, 4..<7])
    }
}
