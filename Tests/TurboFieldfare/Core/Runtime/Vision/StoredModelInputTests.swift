import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
@testable import TurboFieldfare

/// The stored copy of an image and the checks that decide whether it may stand
/// in for the file the user attached.
@Suite struct StoredModelInputTests {
    @Test(arguments: [false, true])
    func adversarialStoredImageDisappearsOrCorruptsBeforeReplay(corrupt: Bool) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = Gemma4ImagePreprocessor(device: device)
        let source = try Self.writeSourcePNG(width: 48, height: 48)
        defer { try? FileManager.default.removeItem(at: source) }
        _ = try preprocessor.plan(storedModelInput: source)
        if corrupt {
            try Data("not a PNG".utf8).write(to: source)
        } else {
            try FileManager.default.removeItem(at: source)
        }
        #expect(throws: VisionImageError.self) {
            try preprocessor.plan(storedModelInput: source)
        }
    }

    /// The scaling rule maps a source *towards* the token budget in both
    /// directions, so re-deriving a stored image's geometry through it turns a
    /// 48x48 model input into a 768x768 one and every recorded soft-token count
    /// stops matching its span.
    @Test func storedGeometryIsReadOffTheStoredDimensions() throws {
        let stored = try Gemma4ImageGeometry(processedWidth: 48, processedHeight: 48)
        #expect(stored.processedWidth == 48)
        #expect(stored.processedHeight == 48)
        #expect(stored.patchCount == 9)
        #expect(stored.softTokenCount == 1)

        let rescaled = try Gemma4ImageGeometry(sourceWidth: 48, sourceHeight: 48)
        #expect(rescaled.processedWidth == 768)
    }

    @Test func storedGeometryMatchesThePipelineForAnAlreadyProcessedSize() throws {
        let fromSource = try Gemma4ImageGeometry(sourceWidth: 8_000, sourceHeight: 6_000)
        let fromStored = try Gemma4ImageGeometry(
            processedWidth: fromSource.processedWidth,
            processedHeight: fromSource.processedHeight)
        #expect(fromStored == fromSource)
    }

    @Test func storedGeometryRefusesUnalignedAndOversizedPixels() {
        #expect(throws: VisionImageError.self) {
            try Gemma4ImageGeometry(processedWidth: 47, processedHeight: 48)
        }
        #expect(throws: VisionImageError.self) {
            try Gemma4ImageGeometry(processedWidth: 48, processedHeight: 0)
        }
        // 1,008 x 720 is 48-aligned but 725,760 pixels, past the 645,120 budget.
        #expect(throws: VisionImageError.self) {
            try Gemma4ImageGeometry(processedWidth: 1_008, processedHeight: 720)
        }
    }

    @Test func modelInputPixelsDropAlphaAndRowPadding() {
        // Two rows of two pixels in a surface whose rows are four pixels wide,
        // so both the alpha channel and the row remainder must be skipped.
        let rowBytes = 16
        var surface = [UInt8](repeating: 0xAA, count: rowBytes * 2)
        let pixels: [(Int, Int, [UInt8])] = [
            (0, 0, [1, 2, 3, 255]), (0, 1, [4, 5, 6, 255]),
            (1, 0, [7, 8, 9, 255]), (1, 1, [10, 11, 12, 255]),
        ]
        for (y, x, rgba) in pixels {
            for (offset, value) in rgba.enumerated() {
                surface[y * rowBytes + x * 4 + offset] = value
            }
        }
        let packed = surface.withUnsafeBufferPointer {
            Gemma4ImagePreprocessor.modelInputPixels(
                rgba: $0.baseAddress!, rowBytes: rowBytes, width: 2, height: 2)
        }
        #expect(packed.rgb8 == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        #expect(packed.width == 2)
        #expect(packed.height == 2)
        #expect(packed.digest == SHA256.hash(data: packed.rgb8)
            .map { String(format: "%02x", $0) }.joined())
    }

    /// The round trip the whole feature rests on: preprocess a source image,
    /// store the model-input copy, and get byte-identical patches back from the
    /// stored file. If this ever stops holding, a reopened conversation would
    /// continue on an image the model never saw, so the digest — not this
    /// test's optimism — is what production relies on.
    @Test func aStoredCopyPatchifiesToTheSameBytesAsTheSourceDid() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = Gemma4ImagePreprocessor(device: device)
        let source = try Self.writeSourcePNG(width: 200, height: 100)
        defer { try? FileManager.default.removeItem(at: source) }

        let live = try preprocessor.preprocess(
            try preprocessor.plan(fileURL: source), capturingModelInput: true)
        let modelInput = try #require(live.modelInput)
        #expect(modelInput.width == live.pixels.geometry.processedWidth)
        #expect(modelInput.height == live.pixels.geometry.processedHeight)

        let stored = try Self.writeModelInputPNG(modelInput)
        defer { try? FileManager.default.removeItem(at: stored) }

        let plan = try preprocessor.plan(storedModelInput: stored)
        #expect(plan.geometry == live.pixels.geometry)

        let replayed = try preprocessor.preprocess(
            storedModelInput: plan, expectedDigest: modelInput.digest)
        #expect(Self.bytes(of: replayed.patchesBF16)
            == Self.bytes(of: live.pixels.patchesBF16))
        #expect(Self.bytes(of: replayed.positionsInt32x2)
            == Self.bytes(of: live.pixels.positionsInt32x2))
    }

    @Test func aStoredCopyWhoseDigestChangedIsRefused() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = Gemma4ImagePreprocessor(device: device)
        let source = try Self.writeSourcePNG(width: 200, height: 100)
        defer { try? FileManager.default.removeItem(at: source) }
        let modelInput = try #require(try preprocessor.preprocess(
            try preprocessor.plan(fileURL: source),
            capturingModelInput: true).modelInput)
        let stored = try Self.writeModelInputPNG(modelInput)
        defer { try? FileManager.default.removeItem(at: stored) }

        // Case is not identity: the same digest in upper case still admits.
        _ = try preprocessor.preprocess(
            storedModelInput: try preprocessor.plan(storedModelInput: stored),
            expectedDigest: modelInput.digest.uppercased())

        // Fails closed rather than replaying pixels the model never saw. PNG
        // decode of an sRGB 8-bit file is very likely an identity and nowhere
        // documented as one, which is why the check is not optional.
        #expect(throws: VisionImageError.self) {
            try preprocessor.preprocess(
                storedModelInput: try preprocessor.plan(storedModelInput: stored),
                expectedDigest: String(repeating: "0", count: 64))
        }
    }

    @Test func aStoredImageOffTheAlignmentGridIsRefusedBeforeItIsDecoded() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = Gemma4ImagePreprocessor(device: device)
        let url = try Self.writeSourcePNG(width: 50, height: 48)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: VisionImageError.self) {
            try preprocessor.plan(storedModelInput: url)
        }
    }

    private static func bytes(of buffer: MTLBuffer) -> [UInt8] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: UInt8.self),
            count: buffer.length))
    }

    /// Writes the model-input RGB8 as a lossless PNG, the way the conversation
    /// store does: 24-bit, no alpha, sRGB, no resampling.
    private static func writeModelInputPNG(_ pixels: VisionModelInputPixels) throws -> URL {
        let provider = try #require(CGDataProvider(data: Data(pixels.rgb8) as CFData))
        let image = try #require(CGImage(
            width: pixels.width, height: pixels.height,
            bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: pixels.width * 3,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent))
        return try write(image, name: "model-input")
    }

    private static func writeSourcePNG(width: Int, height: Int) throws -> URL {
        let rowBytes = width * 4
        var bytes = [UInt8](repeating: 255, count: rowBytes * height)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * rowBytes + x * 4
                bytes[offset] = UInt8((x * 5 + y * 3) % 256)
                bytes[offset + 1] = UInt8((x * 7) % 256)
                bytes[offset + 2] = UInt8((y * 11) % 256)
            }
        }
        let image = try bytes.withUnsafeMutableBytes { raw -> CGImage in
            let context = try #require(CGContext(
                data: raw.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: rowBytes,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue))
            return try #require(context.makeImage())
        }
        return try write(image, name: "source")
    }

    private static func write(_ image: CGImage, name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)-\(UUID().uuidString).png")
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return url
    }
}
