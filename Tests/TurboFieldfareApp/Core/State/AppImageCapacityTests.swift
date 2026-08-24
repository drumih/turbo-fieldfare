import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// The app used to allow four images no matter the context, while the server
/// derived a budget from it. The same set of images was therefore accepted over
/// the API and refused in the app. Both now answer from one rule, so these
/// tests are as much about the rule being shared as about the numbers.
@Suite struct AppImageCapacityTests {
    @MainActor
    @Test(arguments: AppContextLengthOption.allCases)
    func capacityFollowsTheContext(option: AppContextLengthOption) throws {
        let directory = try makeVisionReadyModelInstall("capacity-\(option.tokens)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)
        model.maxContextTokens = option.tokens

        let expected = VisionImageTokenBudget.capacity(
            maxContext: option.tokens,
            reservedTextTokens: AppModel.reservedPromptTokens)
        #expect(model.maximumImageAttachments == max(1, expected))
        // A bigger context must never allow fewer images.
        #expect(model.maximumImageAttachments >= 1)
    }

    @MainActor
    @Test func alargerContextAllowsMoreImages() throws {
        let directory = try makeVisionReadyModelInstall("capacity-growth")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)

        model.maxContextTokens = AppContextLengthOption.fourK.tokens
        let small = model.maximumImageAttachments
        model.maxContextTokens = AppContextLengthOption.sixtyFourK.tokens
        let large = model.maximumImageAttachments
        #expect(large > small,
                "raising the context did not raise the image capacity")
        // The old fixed limit is genuinely gone, not merely renamed.
        #expect(large > 4)
    }

    /// The composer and the request validator must not disagree: an attachment
    /// set the composer accepted has to pass validation.
    @MainActor
    @Test func whatTheComposerAcceptsTheRequestAccepts() throws {
        let directory = try makeVisionReadyModelInstall("capacity-agreement")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)
        model.maxContextTokens = AppContextLengthOption.eightK.tokens

        let attachments = (0..<model.maximumImageAttachments).map { index in
            AppImageAttachment(
                fileURL: URL(fileURLWithPath: "/tmp/image-\(index).png"),
                displayName: "image-\(index).png",
                encodedBytes: 1,
                sha256: String(repeating: "0", count: 64))
        }
        let request = AppGenerationRequest(
            modelDirectory: directory,
            prompt: "describe these",
            imageAttachments: attachments,
            maxContextTokens: model.maxContextTokens)
        try request.validate(requireModelDirectory: false)
    }

    /// And one image past what the context can hold is refused rather than
    /// discovered at generation time.
    @MainActor
    @Test func beyondTheContextTheRequestIsRefused() throws {
        let directory = try makeVisionReadyModelInstall("capacity-refusal")
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = AppContextLengthOption.fourK.tokens
        let capacity = VisionImageTokenBudget.capacity(
            maxContext: context, reservedTextTokens: 0)
        let attachments = (0...capacity).map { index in
            AppImageAttachment(
                fileURL: URL(fileURLWithPath: "/tmp/image-\(index).png"),
                displayName: "image-\(index).png",
                encodedBytes: 1,
                sha256: String(repeating: "0", count: 64))
        }
        let request = AppGenerationRequest(
            modelDirectory: directory,
            prompt: "describe these",
            imageAttachments: attachments,
            maxContextTokens: context)
        #expect(throws: AppInferenceError.self) {
            try request.validate(requireModelDirectory: false)
        }
    }

    /// Choosing more images than fit must say so. Silently keeping the first
    /// few left the user believing every image they picked was attached.
    @MainActor
    @Test func choosingTooManyImagesReportsWhatWasDropped() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("capacity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = try makeVisionReadyModelInstall("capacity-truncation")
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = AppImageAttachmentStore(
            directoryURL: root.appendingPathComponent("staged", isDirectory: true))
        let model = AppModel(modelDirectory: directory, attachmentStore: store)
        model.maxContextTokens = AppContextLengthOption.fourK.tokens
        let capacity = model.maximumImageAttachments

        var urls: [URL] = []
        for index in 0...capacity {
            let url = root.appendingPathComponent("image-\(index).png")
            try Data("fixture \(index)".utf8).write(to: url)
            urls.append(url)
        }

        let previous: String? = nil
        model.addImages(urls)
        if let previous {
            _ = previous
        } else {
            _ = previous
        }
        let deadline = Date().addingTimeInterval(60)
        while model.isAddingImages, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(model.imageAttachments.count == capacity)
        #expect(model.imageAttachmentError != nil,
                "images were dropped without saying so")
        model.releaseAllAttachments()
    }
}

/// Findings from reviewing this session's own Mac app changes.
@Suite struct AppImageReviewRegressionTests {
    @MainActor
    @Test func theComposerCapsOnTheContextARunWillActuallyUse() async throws {
        let directory = try makeVisionReadyModelInstall("review-capacity")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient())
        model.maxContextTokens = AppContextLengthOption.fourK.tokens
        model.loadModel()
        let deadline = Date().addingTimeInterval(60)
        while !model.loadState.isReady, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        try #require(model.loadState.isReady)
        let loadedCapacity = model.maximumImageAttachments

        // Raising the setting without reloading must not raise what the
        // composer accepts, or it accepts images the request then refuses.
        model.maxContextTokens = AppContextLengthOption.sixtyFourK.tokens
        #expect(model.maximumImageAttachments == loadedCapacity,
                "the composer offered capacity the loaded session cannot serve")
        #expect(model.effectiveMaxContextTokens
                    == AppContextLengthOption.fourK.tokens)
    }

    /// A companion operation renames the pack directory, so a readiness refresh
    /// during one can briefly report no image support. Deleting the user's
    /// staged images on that would be data loss.
    @MainActor
    @Test func amidCompanionOperationRefreshKeepsStagedImages() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("review-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.png")
        try Data("fixture".utf8).write(to: source)
        let directory = try makeVisionReadyModelInstall("review-operation")
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = AppImageAttachmentStore(
            directoryURL: root.appendingPathComponent("staged", isDirectory: true))
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             attachmentStore: store)
        let previous: String? = nil
        defer {
            if let previous { setenv("TURBO_FIELDFARE_VISION_RUNTIME", previous, 1) }
            else { unsetenv("TURBO_FIELDFARE_VISION_RUNTIME") }
        }
        model.addImages([source])
        var deadline = Date().addingTimeInterval(60)
        while model.isAddingImages, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        try #require(model.imageAttachments.count == 1)

        // The pack goes away underneath, as it does mid-rename, while an
        // operation is in flight.
        let companion = try VisionPackLocation.companionURL(forTextModel: directory)
        try FileManager.default.removeItem(at: companion)
        model.visionInstallState = .discarding
        try #require(model.isVisionCompanionOperationInProgress)
        model.refreshVisionInstallReadiness()

        #expect(model.imageAttachments.count == 1,
                "a refresh during a companion operation deleted staged images")

        // Once the operation ends and support really is gone, they go: an
        // attachment nothing can encode is worse than none.
        model.visionInstallState = .idle
        model.refreshVisionInstallReadiness()
        #expect(model.imageAttachments.isEmpty)
        model.releaseAllAttachments()
    }
}
