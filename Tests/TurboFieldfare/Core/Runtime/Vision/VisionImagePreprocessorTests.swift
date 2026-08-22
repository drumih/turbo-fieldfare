import Foundation
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
@testable import TurboFieldfare

@Suite struct VisionImagePreprocessorTests {
    private struct Corpus: Decodable {
        struct Case: Decodable {
            let file: String
            let orientation: Int
            let processedWidth: Int
            let processedHeight: Int
            let gridWidth: Int
            let gridHeight: Int

            enum CodingKeys: String, CodingKey {
                case file, orientation
                case processedWidth = "processed_width"
                case processedHeight = "processed_height"
                case gridWidth = "grid_width"
                case gridHeight = "grid_height"
            }
        }

        let cases: [Case]
    }

    @Test func geometryMatchesReferenceProcessor() throws {
        let square = try Gemma4ImageGeometry(sourceWidth: 641, sourceHeight: 641)
        #expect(square.processedWidth == 768)
        #expect(square.processedHeight == 768)
        #expect(square.patchGridWidth == 48)
        #expect(square.patchGridHeight == 48)
        #expect(square.softTokenCount == 256)

        let panorama = try Gemma4ImageGeometry(sourceWidth: 1_537, sourceHeight: 257)
        #expect(panorama.processedWidth == 1_920)
        #expect(panorama.processedHeight == 288)
        #expect(panorama.patchCount == 2_160)
        #expect(panorama.softTokenCount == 240)

        let fortyEightMegapixels = try Gemma4ImageGeometry(
            sourceWidth: 8_000, sourceHeight: 6_000)
        #expect(fortyEightMegapixels.processedWidth == 912)
        #expect(fortyEightMegapixels.processedHeight == 672)
        #expect(fortyEightMegapixels.patchGridWidth == 57)
        #expect(fortyEightMegapixels.patchGridHeight == 42)
        #expect(fortyEightMegapixels.patchCount == 2_394)
        #expect(fortyEightMegapixels.softTokenCount == 266)

        let veryWide = try Gemma4ImageGeometry(sourceWidth: 32_768, sourceHeight: 1)
        #expect(veryWide.processedWidth == 13_440)
        #expect(veryWide.processedHeight == 48)
        #expect(veryWide.patchCount == 2_520)

        let veryTall = try Gemma4ImageGeometry(sourceWidth: 1, sourceHeight: 32_768)
        #expect(veryTall.processedWidth == 48)
        #expect(veryTall.processedHeight == 13_440)
        #expect(veryTall.patchCount == 2_520)
    }

    @Test func frozenCorpusProducesExactGeometryAndPositions() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let root = try #require(Self.corpusRoot)
        let corpus = try JSONDecoder().decode(
            Corpus.self,
            from: Data(contentsOf: root.appendingPathComponent("corpus.json")))
        let preprocessor = Gemma4ImagePreprocessor(device: device)

        for fixture in corpus.cases {
            let result = try preprocessor.preprocess(
                fileURL: root.appendingPathComponent(fixture.file))
            #expect(result.metadata.orientation == fixture.orientation)
            #expect(result.geometry.processedWidth == fixture.processedWidth)
            #expect(result.geometry.processedHeight == fixture.processedHeight)
            #expect(result.geometry.patchGridWidth == fixture.gridWidth)
            #expect(result.geometry.patchGridHeight == fixture.gridHeight)
            #expect(result.patchesBF16.length
                    == result.geometry.patchCount * VisionConfig().patchDimension * 2)
            #expect(result.positionsInt32x2.length
                    == result.geometry.patchCount * 2 * MemoryLayout<Int32>.stride)
            #expect(result.allocatedBytes < 12 * 1_024 * 1_024)

            let positions = result.positionsInt32x2.contents().bindMemory(
                to: Int32.self, capacity: result.geometry.patchCount * 2)
            for row in 0..<result.geometry.patchCount {
                #expect(positions[row * 2] == Int32(row % fixture.gridWidth))
                #expect(positions[row * 2 + 1] == Int32(row / fixture.gridWidth))
            }
            let patches = result.patchesBF16.contents().bindMemory(
                to: UInt16.self,
                capacity: result.geometry.patchCount * VisionConfig().patchDimension)
            let count = result.geometry.patchCount * VisionConfig().patchDimension
            for index in stride(from: 0, to: count, by: 251) {
                let value = Quantization.bf16ToFloat(patches[index])
                #expect(value.isFinite && value >= 0 && value <= 1)
            }
        }
    }

    @Test func rejectsMalformedEncodedInput() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-malformed-\(UUID().uuidString).jpg")
        try Data([0xFF, 0xD8, 0xFF]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()))
        #expect(throws: VisionImageError.self) {
            try preprocessor.preprocess(fileURL: url)
        }
    }

    @Test func rejectsOversizedSourceBeforeDecode() throws {
        let image = try #require(Self.corpusRoot).appendingPathComponent("natural-square.png")
        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()),
            limits: VisionImageLimits(maximumEncodedBytes: 1))
        #expect(throws: VisionImageError.self) {
            try preprocessor.preprocess(fileURL: image)
        }
    }

    @Test func rejectsSourceDimensionLimitBeforeDecode() throws {
        let image = try #require(Self.corpusRoot).appendingPathComponent("natural-square.png")
        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()),
            limits: VisionImageLimits(
                maximumSourcePixels: 100,
                maximumSourceDimension: 10))
        #expect(throws: VisionImageError.self) {
            try preprocessor.preprocess(fileURL: image)
        }
    }

    @Test func rejectsSymlinkSource() throws {
        let target = try #require(Self.corpusRoot).appendingPathComponent("natural-square.png")
        let link = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-link-\(UUID().uuidString).png")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        defer { try? FileManager.default.removeItem(at: link) }
        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()))
        #expect(throws: VisionImageError.self) {
            try preprocessor.preprocess(fileURL: link)
        }
    }

    @Test func rejectsTruncatedPayload() throws {
        let source = try #require(Self.corpusRoot).appendingPathComponent("natural-odd-4x3.jpg")
        let truncated = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-truncated-\(UUID().uuidString).jpg")
        let prefix = try Data(contentsOf: source).prefix(1_024)
        try Data(prefix).write(to: truncated)
        defer { try? FileManager.default.removeItem(at: truncated) }
        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()))
        #expect(throws: VisionImageError.self) {
            try preprocessor.preprocess(fileURL: truncated)
        }
    }

    @Test func rejectsMultiframeMedia() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-multiframe-\(UUID().uuidString).gif")
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil, width: 1, height: 1, bitsPerComponent: 8,
                bytesPerRow: 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.gif.identifier as CFString, 2, nil) else {
            Issue.record("could not create multiframe fixture")
            return
        }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        defer { try? FileManager.default.removeItem(at: url) }

        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()))
        #expect(throws: VisionImageError.self) {
            try preprocessor.preprocess(fileURL: url)
        }
    }

    @Test func convertsCMYKToSRGB() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-cmyk-\(UUID().uuidString).jpg")
        try writeImage(
            url: url,
            type: .jpeg,
            colorSpace: CGColorSpaceCreateDeviceCMYK(),
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            bytes: [0, 255, 255, 0])
        defer { try? FileManager.default.removeItem(at: url) }
        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()))
        let result = try preprocessor.preprocess(fileURL: url)
        #expect(result.metadata.colorModel == kCGImagePropertyColorModelCMYK as String)
        assertFiniteUnitPatches(result)
    }

    @Test func convertsSixteenBitToEightBit() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-16bit-\(UUID().uuidString).png")
        try writeImage(
            url: url,
            type: .png,
            colorSpace: CGColorSpaceCreateDeviceGray(),
            bitsPerComponent: 16,
            bitsPerPixel: 16,
            bitmapInfo: .byteOrder16Big,
            bytes: [0x80, 0x00])
        defer { try? FileManager.default.removeItem(at: url) }
        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()))
        let result = try preprocessor.preprocess(fileURL: url)
        #expect(result.metadata.bitsPerComponent == 16)
        assertFiniteUnitPatches(result)
    }

    @Test func acceptsSingleFrameTIFF() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-tiff-\(UUID().uuidString).tiff")
        try writeImage(
            url: url,
            type: .tiff,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bitmapInfo: CGBitmapInfo(rawValue:
                CGImageAlphaInfo.premultipliedLast.rawValue),
            bytes: [0x33, 0x66, 0x99, 0xff])
        defer { try? FileManager.default.removeItem(at: url) }
        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()))

        let result = try preprocessor.preprocess(fileURL: url)

        #expect(result.metadata.typeIdentifier == UTType.tiff.identifier)
        assertFiniteUnitPatches(result)
    }

    private func assertFiniteUnitPatches(_ result: VisionPixelBuffer) {
        let count = result.geometry.patchCount * VisionConfig().patchDimension
        let patches = result.patchesBF16.contents().bindMemory(
            to: UInt16.self, capacity: count)
        for index in stride(from: 0, to: count, by: 251) {
            let value = Quantization.bf16ToFloat(patches[index])
            #expect(value.isFinite && value >= 0 && value <= 1)
        }
    }

    private func writeImage(
        url: URL,
        type: UTType,
        colorSpace: CGColorSpace,
        bitsPerComponent: Int,
        bitsPerPixel: Int,
        bitmapInfo: CGBitmapInfo,
        bytes: [UInt8]
    ) throws {
        let data = Data(bytes)
        let provider = try #require(CGDataProvider(data: data as CFData))
        let image = try #require(CGImage(
            width: 1,
            height: 1,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerPixel,
            bytesPerRow: bytes.count,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent))
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }

    /// The frozen corpus, shipped as a test resource.
    private static var corpusRoot: URL? {
        Bundle.module.url(forResource: "images", withExtension: nil)
    }

}
