import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// Two image adds can be in flight at once — the file picker and a drop, or a
/// drop and a paste — and `canRun` is gated on none of them being in flight.
///
/// `isAddingImages` was a Bool, so whichever batch finished first cleared it
/// while the other was still copying: Generate reopened, the run snapshotted a
/// partial set, and the remaining images were appended afterwards where
/// `removeImage` and `clearImages` are no-ops. It is a count now, and the risk a
/// count carries is the opposite one — an unbalanced decrement that leaves
/// Generate disabled forever — so both directions are pinned here.
@Suite struct AppImageAddConcurrencyTests {
    @MainActor
    @Test func overlappingAddsAllLandAndReopenGenerateExactlyOnce() async throws {
        let directory = try makeVisionReadyModelInstall("image-add-overlap")
        defer { try? FileManager.default.removeItem(at: directory) }
        let staging = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: staging) }

        let model = AppModel(
            modelDirectory: directory,
            attachmentStore: AppImageAttachmentStore(directoryURL: staging))
        model.maxContextTokens = AppContextLengthOption.sixtyFourK.tokens
        try #require(model.maximumImageAttachments >= 3)

        let first = try Self.writePNG(named: "first.png", in: staging)
        let second = try Self.writePNG(named: "second.png", in: staging)
        let third = try Self.writePNG(named: "third.png", in: staging)

        // Both batches admitted before either can finish: this is the exact
        // interleaving the Bool got wrong.
        model.addImages([first, second])
        model.addImages([third])
        #expect(model.isAddingImages)
        #expect(!model.canRun, "Generate was live while images were still copying")

        try await Self.untilIdle(model)

        #expect(model.imageAttachments.count == 3,
                "only \(model.imageAttachments.count) of 3 images survived two overlapping adds")
        #expect(!model.isAddingImages,
                "the counter never returned to zero, so Generate stays disabled forever")
    }

    /// A failing batch has to give its count back too. The error path decrements
    /// on its own line rather than through the success path's `defer`, which is
    /// exactly the kind of asymmetry that strands the counter above zero.
    @MainActor
    @Test func afailedBatchStillReleasesItsCount() async throws {
        let directory = try makeVisionReadyModelInstall("image-add-failure")
        defer { try? FileManager.default.removeItem(at: directory) }
        let staging = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: staging) }

        let model = AppModel(
            modelDirectory: directory,
            attachmentStore: AppImageAttachmentStore(directoryURL: staging))

        model.addImages([staging.appendingPathComponent("does-not-exist.png")])
        #expect(model.isAddingImages)

        try await Self.untilIdle(model)

        #expect(!model.isAddingImages,
                "a failed staging batch left the counter above zero")
        #expect(model.imageAttachments.isEmpty)
        #expect(model.imageAttachmentError != nil, "the failure was not reported")
    }

    /// Interleaving a data paste with a file add exercises the second entry
    /// point into the counter.
    @MainActor
    @Test func adataPasteAndAFileAddShareTheSameCount() async throws {
        let directory = try makeVisionReadyModelInstall("image-add-mixed")
        defer { try? FileManager.default.removeItem(at: directory) }
        let staging = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: staging) }

        let model = AppModel(
            modelDirectory: directory,
            attachmentStore: AppImageAttachmentStore(directoryURL: staging))
        model.maxContextTokens = AppContextLengthOption.sixtyFourK.tokens
        try #require(model.maximumImageAttachments >= 2)

        let file = try Self.writePNG(named: "from-disk.png", in: staging)
        model.addImages([file])
        model.addImageData(try Self.pngData(), displayName: "pasted.png")
        #expect(model.isAddingImages)
        #expect(!model.canRun)

        try await Self.untilIdle(model)

        #expect(model.imageAttachments.count == 2)
        #expect(!model.isAddingImages)
    }

    /// Polls rather than sleeps: staging runs on a detached task and reports
    /// back on the main actor, so yielding until the counter settles is the
    /// deterministic wait. The bound is a failure, not a timeout to tune.
    @MainActor
    private static func untilIdle(_ model: AppModel) async throws {
        for _ in 0..<2_000 {
            if !model.isAddingImages { return }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        Issue.record("image staging never finished")
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("image-add-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// The smallest thing `ImageMetadataReader` will accept: a real 1x1 PNG.
    private static func pngData() throws -> Data {
        let base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42m"
            + "NkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        guard let data = Data(base64Encoded: base64) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return data
    }

    private static func writePNG(named name: String, in directory: URL) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try pngData().write(to: url)
        return url
    }
}
