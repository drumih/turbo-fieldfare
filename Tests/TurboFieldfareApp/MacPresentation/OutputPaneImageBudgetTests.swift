import AppKit
import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import TurboFieldfareMacPresentation

/// The decode budget the output pane draws attached images under.
///
/// Both the composer's thumbnail row and the transcript's inline images decode
/// through `TranscriptImageLoader.thumbnail`, so these cover both call sites at
/// once — the transcript path is the one that shipped unguarded.
///
/// The budget is injected rather than shipped: proving the refusal at the real
/// 96 MB ceiling would mean materialising about 100 MB of pixels inside the
/// test, which is the memory rule the ceiling exists to keep.
/// Serialized because the decode cache is one process-wide `NSCache`: the two
/// cache tests each clear it on entry and on exit, so run in parallel one wipes
/// the entry the other has just stored and is about to read back. Pre-existing,
/// and it fails on this suite alone.
@Suite(.serialized) struct OutputPaneImageBudgetTests {
    /// One ceiling covers every format, because the reduction that used to
    /// make JPEG cheap now happens in our own filter rather than in ImageIO.
    /// The same pixels are therefore refused whichever container they arrive
    /// in — and refusing them is right, since the runtime refuses them too.
    @Test func aSourceOverTheCeilingIsRefusedInEveryFormat() throws {
        let side = 320
        let decodedBytes = side * side * 4
        let budget = TranscriptImageLoader.Budget(
            maximumSourcePixels: 50_000_000,
            maximumSourceDimension: 32_768,
            maximumDecodedBytes: decodedBytes / 2,
            allowedTypeIdentifiers: Self.supportedTypes)

        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let png = directory.appendingPathComponent("square.png")
        let jpeg = directory.appendingPathComponent("square.jpg")
        try Self.writeSolidImage(width: side, height: side, type: .png, to: png)
        try Self.writeSolidImage(width: side, height: side, type: .jpeg, to: jpeg)

        #expect(TranscriptImageLoader.thumbnail(
            at: png, maximumPixelSize: 64, budget: budget) == nil,
                "a \(side)x\(side) PNG needs \(decodedBytes) decoded bytes against a \(decodedBytes / 2)-byte ceiling, so ImageIO must never be asked for it")
        #expect(TranscriptImageLoader.thumbnail(
            at: jpeg, maximumPixelSize: 64, budget: budget) == nil,
                "the same pixels as a JPEG cost the same to hold, so the container must not change the answer")
    }

    /// The accepted path still has to produce a bounded tile, or the budget
    /// would be buying nothing: a source many times the requested size comes
    /// back at the requested size, not at its own.
    @Test func anAcceptedSourceComesBackBoundedByTheRequestedSize() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("small.png")
        try Self.writeSolidImage(width: 64, height: 64, type: .png, to: url)

        let image = try #require(TranscriptImageLoader.thumbnail(
            at: url, maximumPixelSize: 16, budget: Self.generousBudget),
                                 "a 64x64 PNG is far under every ceiling and has to load")
        // `size` rather than the representation's pixel count: the transcript
        // lays the image out from `size`, and the snapshot representation
        // reports backing-scaled pixels, which differ per display.
        #expect(image.size.width > 0 && image.size.width <= 16,
                "the tile came back \(image.size.width) wide, so the requested 16-pixel bound was not applied")
    }

    /// The dimension and pixel caps are separate refusals from the byte
    /// ceiling; a source can sit under the decoded-byte ceiling and still be
    /// a shape the app will not draw.
    @Test func sourcesOverTheDimensionOrPixelCapAreRefused() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("small.png")
        try Self.writeSolidImage(width: 64, height: 64, type: .png, to: url)

        let overDimension = TranscriptImageLoader.Budget(
            maximumSourcePixels: 50_000_000,
            maximumSourceDimension: 32,
            maximumDecodedBytes: 1 << 30,
            allowedTypeIdentifiers: Self.supportedTypes)
        #expect(TranscriptImageLoader.thumbnail(
            at: url, maximumPixelSize: 16, budget: overDimension) == nil,
                "a 64-pixel edge is over a 32-pixel dimension cap")

        let overPixels = TranscriptImageLoader.Budget(
            maximumSourcePixels: 1_000,
            maximumSourceDimension: 32_768,
            maximumDecodedBytes: 1 << 30,
            allowedTypeIdentifiers: Self.supportedTypes)
        #expect(TranscriptImageLoader.thumbnail(
            at: url, maximumPixelSize: 16, budget: overPixels) == nil,
                "4096 pixels is over a 1000-pixel cap")
    }

    /// Refusal and unreadability have to arrive the same way, because that is
    /// the value both call sites already draw a placeholder for: a missing file
    /// or a file that is not an image returns nil rather than throwing or
    /// handing back an empty image the transcript would draw as a broken tile.
    @Test func unreadableSourcesReturnNilRatherThanAnEmptyImage() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let missing = directory.appendingPathComponent("absent.png")
        #expect(TranscriptImageLoader.thumbnail(
            at: missing, maximumPixelSize: 16, budget: Self.generousBudget) == nil)

        let notAnImage = directory.appendingPathComponent("notes.png")
        try Data("this is not a PNG".utf8).write(to: notAnImage)
        #expect(TranscriptImageLoader.thumbnail(
            at: notAnImage, maximumPixelSize: 16, budget: Self.generousBudget) == nil)
    }

    private static let generousBudget = TranscriptImageLoader.Budget(
        maximumSourcePixels: 50_000_000,
        maximumSourceDimension: 32_768,
        maximumDecodedBytes: 1 << 30,
        allowedTypeIdentifiers: Self.supportedTypes)

    /// Decoding is what costs: `kCGImageSourceCreateThumbnailFromImageAlways`
    /// pays a full decode for most formats, and it ran again on every redraw —
    /// on the main thread, inside the render pass, at the moment Run was
    /// pressed. Deleting the file between calls is how a hit is proved: a miss
    /// has nothing left to read.
    @Test func aDecodedThumbnailIsServedFromTheCacheOnTheSecondCall() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        TranscriptImageLoader.clearCache()
        defer { TranscriptImageLoader.clearCache() }

        let url = directory.appendingPathComponent("cached.png")
        try Self.writeSolidImage(width: 64, height: 64, type: .png, to: url)
        let key = "digest-of-cached-png"

        #expect(TranscriptImageLoader.thumbnail(
            at: url, maximumPixelSize: 32, budget: Self.permissiveBudget,
            cacheKey: key) != nil)

        try FileManager.default.removeItem(at: url)

        #expect(TranscriptImageLoader.thumbnail(
            at: url, maximumPixelSize: 32, budget: Self.permissiveBudget,
            cacheKey: key) != nil,
                "the second call decoded again instead of reusing the first")
        // The size is part of the identity: the transcript asks for 720 and the
        // composer for 96, and neither may be served the other's pixels.
        #expect(TranscriptImageLoader.thumbnail(
            at: url, maximumPixelSize: 720, budget: Self.permissiveBudget,
            cacheKey: key) == nil,
                "a different requested size was served from the wrong entry")
    }

    /// Cached copies are keyed on contents and released on demand, so a cleared
    /// transcript does not leave decoded images held beside a resident model.
    @Test func clearingTheCacheForcesTheNextCallToDecodeAgain() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        TranscriptImageLoader.clearCache()
        defer { TranscriptImageLoader.clearCache() }

        let url = directory.appendingPathComponent("cleared.png")
        try Self.writeSolidImage(width: 64, height: 64, type: .png, to: url)
        let key = "digest-of-cleared-png"

        #expect(TranscriptImageLoader.thumbnail(
            at: url, maximumPixelSize: 32, budget: Self.permissiveBudget,
            cacheKey: key) != nil)
        TranscriptImageLoader.clearCache()
        try FileManager.default.removeItem(at: url)

        #expect(TranscriptImageLoader.thumbnail(
            at: url, maximumPixelSize: 32, budget: Self.permissiveBudget,
            cacheKey: key) == nil,
                "a cleared cache still answered from a decoded copy")
    }

    /// Without a key there is nothing to key on, so the caller must keep
    /// getting a fresh read rather than another caller's image.
    @Test func aCallWithoutAKeyIsNeverCached() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        TranscriptImageLoader.clearCache()
        defer { TranscriptImageLoader.clearCache() }

        let url = directory.appendingPathComponent("uncached.png")
        try Self.writeSolidImage(width: 64, height: 64, type: .png, to: url)

        #expect(TranscriptImageLoader.thumbnail(
            at: url, maximumPixelSize: 32, budget: Self.permissiveBudget) != nil)
        try FileManager.default.removeItem(at: url)
        #expect(TranscriptImageLoader.thumbnail(
            at: url, maximumPixelSize: 32, budget: Self.permissiveBudget) == nil,
                "an unkeyed call was answered from the cache")
    }

    private static let permissiveBudget = TranscriptImageLoader.Budget(
        maximumSourcePixels: 50_000_000,
        maximumSourceDimension: 32_768,
        maximumDecodedBytes: 96 * 1_024 * 1_024,
        allowedTypeIdentifiers: Self.supportedTypes)

    /// The runtime's allowlist, restated rather than imported so that widening
    /// `VisionImageLimits` shows up here as a failing test instead of as a
    /// silently larger decode surface.
    private static let supportedTypes: Set<String> = [
        "public.jpeg", "public.png", "public.heic", "public.heif",
        "public.tiff", "com.compuserve.gif", "public.heics",
    ]

    /// A format the pack was never validated against must not reach ImageIO's
    /// decoder for it. Every other limit here was copied from the runtime; the
    /// type list was not, so a camera RAW or EXR was decoded at full size inside
    /// the process holding the model and only then refused by the run.
    ///
    /// BMP stands in for the whole excluded set: ImageIO can both write and read
    /// it, so the only thing separating it from the PNG below is the allowlist.
    @Test func aFormatOutsideTheAllowlistIsRefusedBeforeDecoding() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let bmp = directory.appendingPathComponent("swatch.bmp")
        try Self.writeSolidImage(width: 64, height: 64, type: .bmp, to: bmp)
        #expect(TranscriptImageLoader.thumbnail(
            at: bmp, maximumPixelSize: 16, budget: Self.generousBudget) == nil,
                "BMP is outside the runtime's allowlist, so it must not be decoded")

        let png = directory.appendingPathComponent("swatch.png")
        try Self.writeSolidImage(width: 64, height: 64, type: .png, to: png)
        #expect(TranscriptImageLoader.thumbnail(
            at: png, maximumPixelSize: 16, budget: Self.generousBudget) != nil,
                "the same pixels as a PNG are inside the allowlist and must load")
    }

    /// The extension is not the format. ImageIO sniffs the bytes, so a file
    /// named `.png` that holds BMP has to be refused on what it is.
    @Test func theAllowlistFollowsTheBytesRatherThanTheExtension() throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let disguised = directory.appendingPathComponent("actually-bmp.png")
        try Self.writeSolidImage(width: 64, height: 64, type: .bmp, to: disguised)
        #expect(TranscriptImageLoader.thumbnail(
            at: disguised, maximumPixelSize: 16, budget: Self.generousBudget) == nil,
                "the .png name must not admit BMP bytes to the decoder")
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("output-pane-budget-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func writeSolidImage(
        width: Int, height: Int, type: UTType, to url: URL
    ) throws {
        let colorSpace = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let bitmap = try #require(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        bitmap.setFillColor(CGColor(gray: 0.5, alpha: 1))
        bitmap.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try #require(bitmap.makeImage())
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }
}
