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

    /// The soft-token count an image's encode would read, taken the way
    /// `encodeTurnImage` takes it.
    private static func encodedCount(
        of image: ServerRequestImages.Planned,
        with preprocessor: Gemma4ImagePreprocessor
    ) throws -> Int {
        switch ServerRequestImages.source(for: image) {
        case .plan(let plan):
            return plan.geometry.softTokenCount
        case .reopened(let url):
            return try preprocessor.plan(fileURL: url).geometry.softTokenCount
        }
    }

    /// The full-prefill path: every image of the request encoded from the plan
    /// its count came from, so a file rewritten after the request was planned
    /// cannot move the count out from under the span already laid out for it.
    ///
    /// The assertion is against a fresh read of the same file, not against
    /// `Planned.softTokenCount` — that field is copied from the plan's own
    /// geometry, so comparing the two would hold however the code behaved.
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
        let planned = try ServerRequestImages.plannedEntries(
            imageFiles, with: preprocessor)

        // Every staged file replaced after the request was planned, standing in
        // for one rewritten while the request was in flight.
        for url in imageFiles.values {
            try Self.writeSolidImage(
                width: Self.rewrittenWidth, height: Self.rewrittenHeight, to: url)
        }

        #expect(planned.count == imageFiles.count)
        for entry in planned {
            let rereadCount = try preprocessor
                .plan(fileURL: entry.image.url).geometry.softTokenCount
            try #require(
                rereadCount != entry.image.softTokenCount,
                "the rewrite has to move the projected count or a second read is invisible")
            let encoded = try Self.encodedCount(of: entry.image, with: preprocessor)
            #expect(encoded != rereadCount,
                    "\(entry.image.url.lastPathComponent) was read again at encode time: the encode saw the rewritten file's \(rereadCount) tokens")
            #expect(encoded == entry.image.softTokenCount)
        }
    }

    /// The features of one image must not be filed under another's id. The
    /// dictionary has no order, so the ids and the plans have to come from one
    /// snapshot; taking them from two walks pairs them wrongly, and when both
    /// images project to the same count nothing downstream can notice.
    @Test func everyPlanIsPairedWithItsOwnImageID() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = Gemma4ImagePreprocessor(device: device)
        let directory = try Self.makeStagingDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        var imageFiles: [UUID: URL] = [:]
        for index in 0..<8 {
            let url = directory.appendingPathComponent("staged-\(index).png")
            try Self.writeSolidImage(
                width: Self.plannedWidth, height: Self.plannedHeight, to: url)
            imageFiles[UUID()] = url
        }

        let planned = try ServerRequestImages.plannedEntries(
            imageFiles, with: preprocessor)
        #expect(planned.count == imageFiles.count)
        #expect(Set(planned.map(\.id)) == Set(imageFiles.keys))
        for entry in planned {
            #expect(entry.image.url == imageFiles[entry.id],
                    "\(entry.id) was paired with \(entry.image.url.lastPathComponent)")
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

        guard case .plan = ServerRequestImages.source(for: planned[0]) else {
            Issue.record("the image inside the bound gave its plan up")
            return
        }
        guard case .reopened = ServerRequestImages.source(for: planned[1]) else {
            Issue.record("the image past the bound kept a plan it had to release")
            return
        }

        let encoded = try planned.map {
            try Self.encodedCount(of: $0, with: preprocessor)
        }
        #expect(encoded[0] == planned[0].softTokenCount,
                "the image inside the bound kept its plan, so its encode reads \(planned[0].softTokenCount) tokens rather than \(encoded[0])")
        #expect(encoded[1] != planned[1].softTokenCount,
                "the image past the bound gave its descriptor up, so its encode has to reopen the file rather than report \(encoded[1])")
    }

    /// An unreadable image is the request's problem whichever position it sits
    /// in, and the tower is the expensive part: planning is one complete pass
    /// that either yields a plan for every image or throws, so the caller has
    /// nothing to encode from until every image has been read. The broken image
    /// is last, which is the position that used to run the tower on the two
    /// ahead of it before failing.
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

        var planned: [(id: UUID, image: ServerRequestImages.Planned)]?
        var failure: (any Error)?
        do {
            planned = try ServerRequestImages.plannedEntries(
                imageFiles, with: preprocessor)
        } catch {
            failure = error
        }
        #expect(failure != nil, "an image that cannot be read has to fail the request")
        #expect(planned == nil,
                "planning handed the caller \(planned?.count ?? 0) images to encode from a request it could not read in full")
    }

    /// An image the client sent that cannot be read is a bad request. The
    /// encode re-plans with stream verification on, so a truncated upload is
    /// admitted and fails there; reaching the generic handler made it a 500,
    /// which official clients retry and 4xx they do not.
    @Test func anUnreadableImageAtEncodeTimeIsAClientError() {
        struct ReadFailure: Error {}

        let mapped = ServerRequestImages.requestError(forUnreadable: ReadFailure())
        guard case .invalid(_, _, let code) = mapped as? ServerRequestError else {
            Issue.record("an unreadable image mapped to \(mapped), not a request error")
            return
        }
        #expect(code == "invalid_image")

        // A refusal the caller already classified, and an abandoned request,
        // both keep their own meaning.
        let alreadyClassified = ServerRequestError.invalid(
            message: "image support is unavailable",
            param: "messages", code: "vision_unavailable")
        let preserved = ServerRequestImages.requestError(forUnreadable: alreadyClassified)
        guard case .invalid(_, _, let preservedCode) = preserved as? ServerRequestError else {
            Issue.record("a classified refusal was rewritten to \(preserved)")
            return
        }
        #expect(preservedCode == "vision_unavailable")
        #expect(ServerRequestImages.requestError(
            forUnreadable: CancellationError()) is CancellationError)
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
