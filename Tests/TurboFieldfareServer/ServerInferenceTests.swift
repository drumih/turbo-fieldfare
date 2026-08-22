import CoreGraphics
import Foundation
import ImageIO
import Metal
import Testing
import UniformTypeIdentifiers
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Server vision capability")
struct ServerVisionCapabilityTests {
    @Test func unsupportedHardwareDominatesPackAvailability() {
        #expect(ServerModelSession.hardwareVisionCapability(
            supportsVisionRuntime: false) == "unsupported")
        #expect(ServerModelSession.hardwareVisionCapability(
            supportsVisionRuntime: true) == nil)
    }

    @Test func unsupportedHardwareDoesNotInvalidateThePack() {
        #expect(ServerModelSession.unavailableVisionCapability(
            for: VisionRuntimeError.unsupportedKernel("requires M2")) == "unsupported")
        #expect(ServerModelSession.unavailableVisionCapability(
            for: VisionPackError.invalidMetadata("bad manifest")) == "invalid")
    }
}

/// How a request's images are read on the way into a prefill.
///
/// The count that lays out an image's placeholder span and the encode that
/// fills it have to come from one open file. Reading the file twice sniffed,
/// metadata-parsed and marker-walked every image a second time per request,
/// and the two opens could see two files: a staged file rewritten in between
/// gave a count the laid-out span no longer matched.
@Suite("Server request images")
struct ServerRequestImagesTests {
    /// A square source projects to 256 soft tokens and a 2:1 source to 253, so
    /// writing one shape over the other moves the count without decoding a
    /// pixel. The tests assert the relation rather than the numbers.
    private static let plannedWidth = 96
    private static let plannedHeight = 96
    private static let rewrittenWidth = 128
    private static let rewrittenHeight = 64

    /// What an image's span was laid out for, beside what its encode read.
    private struct EncodedImage {
        let planned: Int
        let encoded: Int
    }

    /// The full-prefill path: every image of the request encoded from the plan
    /// its count came from, so a file rewritten after the request was planned
    /// cannot move the count out from under the span already laid out for it.
    @Test func aFullPrefillEncodesEveryImageFromThePlanItsCountCameFrom() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = Gemma4ImagePreprocessor(device: device)
        let directory = try Self.makeStagingDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        var imageFiles: [UUID: URL] = [:]
        for index in 0..<3 {
            let url = directory.appendingPathComponent("staged-\(index).png")
            try Self.writeSolidImage(
                width: Self.plannedWidth, height: Self.plannedHeight, to: url)
            imageFiles[UUID()] = url
        }

        // Rewritten from inside the first encode, which is the only point that
        // sits between the counts and the rest of the encodes. It stands in for
        // a staged file replaced while the request was in flight.
        var rewritten = false
        let encoded = try ServerRequestImages.encodeAll(
            imageFiles,
            with: preprocessor,
            encode: { image -> EncodedImage in
                if !rewritten {
                    rewritten = true
                    for url in imageFiles.values {
                        try Self.writeSolidImage(
                            width: Self.rewrittenWidth,
                            height: Self.rewrittenHeight,
                            to: url)
                    }
                }
                let tokens = try ServerRequestImages.encode(
                    image,
                    fromPlan: { $0.geometry.softTokenCount },
                    byReopening: {
                        try preprocessor.plan(fileURL: $0).geometry.softTokenCount
                    })
                return EncodedImage(planned: image.softTokenCount, encoded: tokens)
            })

        #expect(encoded.count == imageFiles.count)
        let plannedCounts = Set(encoded.values.map(\.planned))
        let reread = try ServerRequestImages.plans(
            for: Array(imageFiles.values), with: preprocessor)
        let rereadCounts = Set(reread.map(\.softTokenCount))
        try #require(
            plannedCounts.isDisjoint(with: rereadCounts),
            "the rewrite has to move the projected count or a second read is invisible")
        for (id, image) in encoded {
            #expect(image.encoded == image.planned,
                    "image \(id) was read again at encode time: its span was laid out for \(image.planned) tokens and the encode read \(image.encoded)")
        }
    }

    /// The bound on open descriptors is the one place a second read is
    /// intended, and it applies from the image past it onwards.
    @Test func onlyTheImagesPastTheOpenPlanBoundAreReadTwice() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = Gemma4ImagePreprocessor(device: device)
        let directory = try Self.makeStagingDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        var urls: [URL] = []
        for index in 0..<2 {
            let url = directory.appendingPathComponent("staged-\(index).png")
            try Self.writeSolidImage(
                width: Self.plannedWidth, height: Self.plannedHeight, to: url)
            urls.append(url)
        }
        let planned = try ServerRequestImages.plans(
            for: urls, with: preprocessor, maximumOpenPlans: 1)
        for url in urls {
            try Self.writeSolidImage(
                width: Self.rewrittenWidth, height: Self.rewrittenHeight, to: url)
        }
        let encoded = try planned.map { image in
            try ServerRequestImages.encode(
                image,
                fromPlan: { $0.geometry.softTokenCount },
                byReopening: {
                    try preprocessor.plan(fileURL: $0).geometry.softTokenCount
                })
        }

        #expect(encoded[0] == planned[0].softTokenCount,
                "the image inside the bound kept its plan, so its encode reads \(planned[0].softTokenCount) tokens rather than \(encoded[0])")
        #expect(encoded[1] != planned[1].softTokenCount,
                "the image past the bound gave its descriptor up, so its encode has to reopen the file rather than report \(encoded[1])")
    }

    /// An unreadable image is the request's problem whichever position it sits
    /// in, and the tower is the expensive part: planning every image first
    /// refuses the request before any of the others is encoded.
    @Test func anUnreadableImageIsRefusedBeforeAnyOtherImageIsEncoded() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = Gemma4ImagePreprocessor(device: device)
        let directory = try Self.makeStagingDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        var imageFiles: [UUID: URL] = [:]
        for index in 0..<2 {
            let url = directory.appendingPathComponent("staged-\(index).png")
            try Self.writeSolidImage(
                width: Self.plannedWidth, height: Self.plannedHeight, to: url)
            imageFiles[UUID()] = url
        }
        let broken = directory.appendingPathComponent("staged-broken.png")
        try Data("this is not an image, whatever the extension claims".utf8)
            .write(to: broken)
        imageFiles[UUID()] = broken

        var encodes = 0
        var failure: (any Error)?
        do {
            _ = try ServerRequestImages.encodeAll(
                imageFiles,
                with: preprocessor,
                encode: { image -> Int in
                    encodes += 1
                    return image.softTokenCount
                })
        } catch {
            failure = error
        }

        #expect(failure != nil, "an image that cannot be read has to fail the request")
        #expect(encodes == 0,
                "the request was refused only after \(encodes) of its images had been encoded")
    }

    private static func makeStagingDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-request-images-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// A solid PNG of the given shape, replacing whatever is at the URL. Only
    /// the dimensions matter here: the projected token count follows from them
    /// alone, and these are small enough that nothing decodes a real surface.
    private static func writeSolidImage(width: Int, height: Int, to url: URL) throws {
        let colorSpace = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let bitmap = try #require(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        bitmap.setFillColor(CGColor(gray: 0.5, alpha: 1))
        bitmap.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try #require(bitmap.makeImage())
        try? FileManager.default.removeItem(at: url)
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }
}
