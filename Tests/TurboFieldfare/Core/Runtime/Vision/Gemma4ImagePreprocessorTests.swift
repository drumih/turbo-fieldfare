import CoreGraphics
import Foundation
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
@testable import TurboFieldfare

@Suite struct Gemma4ImagePreprocessorTests {
    @Test func referenceGridUsesStrictFortyEightPixelFloor() {
        #expect(Gemma4ImagePreprocessor.outputGrid(
            sourceWidth: 1_000,
            sourceHeight: 1_000,
            maximumSoftTokens: 280) == (16, 16))
        #expect(Gemma4ImagePreprocessor.outputGrid(
            sourceWidth: 1_600,
            sourceHeight: 900,
            maximumSoftTokens: 280) == (22, 12))
        #expect(Gemma4ImagePreprocessor.outputGrid(
            sourceWidth: 900,
            sourceHeight: 1_600,
            maximumSoftTokens: 280) == (12, 22))
        #expect(Gemma4ImagePreprocessor.outputGrid(
            sourceWidth: 10_000,
            sourceHeight: 1,
            maximumSoftTokens: 280) == (280, 1))
    }

    @Test func solidRGBImageIsBoundedPatchifiedAndHasStablePositions() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let imageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-preprocess-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: imageURL) }
        try writeSolidPNG(width: 64, height: 32, rgba: (255, 0, 0, 255), to: imageURL)

        let prepared = try Gemma4ImagePreprocessor.prepare(
            imageAt: imageURL,
            device: device)
        #expect(prepared.sourceWidth == 64)
        #expect(prepared.sourceHeight == 32)
        #expect(prepared.resizedWidth.isMultiple(of: 48))
        #expect(prepared.resizedHeight.isMultiple(of: 48))
        #expect(prepared.softTokenCount <= 280)
        #expect(prepared.patchCount == prepared.softTokenCount * 9)
        #expect(prepared.patchValues.length
            == prepared.patchCount * Gemma4PreparedImage.patchVectorSize * 2)
        #expect(prepared.contentHash.count == 64)

        let values = prepared.patchValues.contents().bindMemory(
            to: Float16.self,
            capacity: Gemma4PreparedImage.patchVectorSize)
        #expect(abs(Float(values[0]) - 1) < 0.001)
        #expect(abs(Float(values[1]) + 1) < 0.001)
        #expect(abs(Float(values[2]) + 1) < 0.001)
        let positions = prepared.patchPositionIDs
        #expect(positions.count == prepared.patchCount)
        #expect(positions.first == SIMD2<Int32>(0, 0))
        #expect(positions.last == SIMD2<Int32>(
            Int32(prepared.patchColumns - 1),
            Int32(prepared.patchRows - 1)))
    }

    private func writeSolidPNG(width: Int,
                               height: Int,
                               rgba: (UInt8, UInt8, UInt8, UInt8),
                               to url: URL) throws {
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        for index in stride(from: 0, to: bytes.count, by: 4) {
            bytes[index] = rgba.0
            bytes[index + 1] = rgba.1
            bytes[index + 2] = rgba.2
            bytes[index + 3] = rgba.3
        }
        let data = Data(bytes) as CFData
        guard let provider = CGDataProvider(data: data),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(
                    rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
                        | CGBitmapInfo.byteOrder32Big.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(
                url as CFURL,
                UTType.png.identifier as CFString,
                1,
                nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw CocoaError(.fileWriteUnknown)
        }
    }
}
