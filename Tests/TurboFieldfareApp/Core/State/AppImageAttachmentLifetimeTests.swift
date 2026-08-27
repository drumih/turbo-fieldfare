import Darwin
import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Who owns a staged image file, and when it is deleted. Every case here used
/// to leak a full-size copy or delete one that was still on screen.
@Suite(.serialized) struct AppImageAttachmentLifetimeTests {
    /// `addImages` is gated on the vision runtime, and it is the only way in:
    /// `imageAttachments` is read-only from outside, which is what keeps the
    /// ownership rules in one place.
    private func withVisionEnabled<T>(_ body: () throws -> T) rethrows -> T {
        let previous = ProcessInfo.processInfo.environment[
            "TURBO_FIELDFARE_VISION_RUNTIME"]
        setenv("TURBO_FIELDFARE_VISION_RUNTIME", "1", 1)
        defer {
            if let previous {
                setenv("TURBO_FIELDFARE_VISION_RUNTIME", previous, 1)
            } else {
                unsetenv("TURBO_FIELDFARE_VISION_RUNTIME")
            }
        }
        return try body()
    }

    @MainActor
    private func attach(_ model: AppModel, _ urls: [URL]) async {
        withVisionEnabled { model.addImages(urls) }
        let deadline = Date().addingTimeInterval(5)
        while model.isAddingImages, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    private func makeStore() -> (root: URL, store: AppImageAttachmentStore) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("attachments-\(UUID().uuidString)", isDirectory: true)
        return (root, AppImageAttachmentStore(
            directoryURL: root.appendingPathComponent("staged", isDirectory: true)))
    }

    private func stagedFileCount(_ store: AppImageAttachmentStore) -> Int {
        guard let walker = FileManager.default.enumerator(
            at: store.directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey]) else { return 0 }
        var count = 0
        for case let url as URL in walker {
            if (try? url.resourceValues(forKeys: [.isRegularFileKey]))?
                .isRegularFile == true { count += 1 }
        }
        return count
    }

    private func writeSource(_ root: URL, _ name: String, _ bytes: String) throws -> URL {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent(name)
        try Data(bytes.utf8).write(to: url)
        return url
    }

    /// Adding images is all-or-nothing, so a batch that fails partway used to
    /// leave the copies it had already made referenced by nothing at all.
    @MainActor
    @Test func aFailedBatchLeavesNoStagedFilesBehind() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let good = try writeSource(root, "good.png", "fixture")
        let missing = root.appendingPathComponent("not-there.png")

        let directory = try makeVisionReadyModelInstall("lifetime-failed-batch")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, attachmentStore: store)
        await attach(model, [good, missing])

        #expect(model.imageAttachments.isEmpty)
        #expect(model.imageAttachmentError != nil, "the failure must be reported")
        #expect(stagedFileCount(store) == 0,
                "the images staged before the failure were orphaned")
    }

    /// The transcript renders from the files it was handed. Clearing the
    /// composer used to delete them out from under a finished answer.
    @Test func aRetainedImageSurvivesTheComposerDeletingItsCopy() throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try writeSource(root, "source.png", "fixture")
        let staged = try store.stage(source)

        let retained = try store.retain(staged)
        #expect(retained.fileURL != staged.fileURL)
        #expect(retained.sha256 == staged.sha256)
        #expect(retained.id == staged.id)

        store.remove(staged)
        #expect(!FileManager.default.fileExists(atPath: staged.fileURL.path))
        #expect(try Data(contentsOf: retained.fileURL) == Data("fixture".utf8),
                "the transcript's copy was deleted with the composer's")

        // A link, not a second copy: one inode, and the last link frees it.
        var link = stat()
        #expect(lstat(retained.fileURL.path, &link) == 0)
        #expect(link.st_nlink == 1)
        store.remove(retained)
        #expect(!FileManager.default.fileExists(atPath: retained.fileURL.path))
    }

    /// A cleared prompt used to leave the images attached, so the next Run
    /// silently sent them with a completely different message.
    @MainActor
    @Test func clearingTheSentPromptAlsoDropsItsImages() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try writeSource(root, "source.png", "fixture")

        let directory = try makeVisionReadyModelInstall("lifetime-clear")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = AppModel(modelDirectory: directory, client: client,
                             attachmentStore: store)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        await attach(model, [source])
        #expect(model.imageAttachments.count == 1)
        model.promptText = "describe it"
        model.maxNewTokensOverride = 1
        model.run()

        #expect(model.promptText.isEmpty)
        #expect(model.imageAttachments.isEmpty,
                "images outlived the prompt they were sent with")
        // The transcript still has them, from its own files.
        #expect(model.outputImageAttachments.count == 1)
        let shown = try #require(model.outputImageAttachments.first)
        #expect(try Data(contentsOf: shown.fileURL) == Data("fixture".utf8))

        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        // Clearing the transcript is what finally frees them.
        model.newChat()
        #expect(model.outputImageAttachments.isEmpty)
        #expect(stagedFileCount(store) == 0)
    }

    /// What the test above could not see. It asserted the composer was emptied
    /// and the transcript still rendered, both of which were true while every
    /// image run under this setting failed: the request still pointed at the
    /// composer's paths, and `.clear` unlinked them before the run opened them.
    /// The transcript's copies are hard links at *different* paths, so they
    /// never covered for it. Fails on the old code now that the mock opens what
    /// it is handed.
    @MainActor
    @Test func aClearedPromptLeavesTheRunAbleToOpenItsImages() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try writeSource(root, "source.png", "fixture")

        let directory = try makeVisionReadyModelInstall("lifetime-clear-runs")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = AppModel(modelDirectory: directory, client: client,
                             attachmentStore: store)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        await attach(model, [source])
        model.promptText = "describe it"
        model.maxNewTokensOverride = 1
        model.run()

        for _ in 0..<400 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(model.error == nil,
                "the run could not open the images the composer had just deleted")
        #expect(!model.outputText.isEmpty, "the run produced nothing")
    }

    /// Without its own reference nothing guarantees the files outlive the
    /// composer, so the run is refused instead of started against files that
    /// are about to be removed. It must not fall back to the composer's own
    /// attachment either: that gave one file two owners, and whichever side
    /// cleaned up first deleted the other's image.
    @MainActor
    @Test func aRunIsRefusedWhenTheTranscriptCannotTakeItsOwnReference() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try writeSource(root, "source.png", "fixture")

        let directory = try makeVisionReadyModelInstall("lifetime-retain-fails")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = AppModel(modelDirectory: directory, client: client,
                             attachmentStore: store)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        await attach(model, [source])
        model.promptText = "describe it"

        // A regular file where `retain` needs to create a directory, so the
        // hard link cannot be made and every retain fails the same way.
        try Data().write(to: store.directoryURL
            .appendingPathComponent("retained", isDirectory: false))

        model.run()

        #expect(!model.isRunning, "the run started without stable references")
        #expect(model.error != nil, "the refusal was silent")
        #expect(model.outputImageAttachments.isEmpty,
                "the transcript aliased the composer's own file")
        #expect(model.imageAttachments.count == 1,
                "a refused run must leave the composer's images alone")
        let staged = try #require(model.imageAttachments.first)
        #expect(FileManager.default.fileExists(atPath: staged.fileURL.path),
                "a refused run deleted the composer's staged copy")
    }

    /// Every early return in `addImages` owns the promise directory. The
    /// staging task's `defer` is the only other thing that deletes it, and a
    /// return above that task means the task never exists — while the sweep
    /// covers only the staging root, so nothing in the app ever reclaims it.
    @MainActor
    @Test func aDropRefusedDuringARunIsExplainedAndDiscardsItsPromiseDirectory() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let directory = try makeVisionReadyModelInstall("lifetime-promise-leak")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        client.holdsDuringPrefill = true
        let model = AppModel(modelDirectory: directory, client: client,
                             attachmentStore: store)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        model.promptText = "hello"
        model.run()
        #expect(model.isRunning)

        let promises = root.appendingPathComponent("promises", isDirectory: true)
        let dropped = try writeSource(promises, "dropped.png", "promised")
        withVisionEnabled {
            model.addImages([dropped], discardingSourceDirectory: promises)
        }

        #expect(model.imageAttachmentError != nil,
                "the drop vanished with nothing said about it")
        #expect(!FileManager.default.fileExists(atPath: promises.path),
                "the promise directory outlived the drop that created it")
        model.cancel()
    }

    /// Staging copies the files the request will carry, so starting a run while
    /// it is in flight sent a request without those images and then landed them
    /// on the next message instead.
    @MainActor
    @Test func aRunCannotStartWhileItsImagesAreStillStaging() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try writeSource(root, "source.png", "fixture")

        let directory = try makeVisionReadyModelInstall("lifetime-staging-race")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockInferenceClient(tokenDelayNanos: 1),
                             attachmentStore: store)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        model.promptText = "describe it"
        #expect(model.canRun)

        withVisionEnabled { model.addImages([source]) }
        #expect(model.isAddingImages)
        #expect(!model.canRun,
                "a run could start while its own images were still being copied")

        let deadline = Date().addingTimeInterval(5)
        while model.isAddingImages, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(model.canRun)
        #expect(model.imageAttachments.count == 1)
    }

    /// Two adds in flight at once — the picker and a drop — each sized itself
    /// against the count it saw at admission, so the second to land appended
    /// past the cap. The surplus copies also have to be deleted, or the clamp
    /// just trades an over-full composer for staged files nothing references.
    @MainActor
    @Test func overlappingAddsCannotPushPastTheCap() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }

        let directory = try makeVisionReadyModelInstall("lifetime-overlapping-adds")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockInferenceClient(tokenDelayNanos: 1),
                             attachmentStore: store)
        let capacity = model.maximumImageAttachments
        let first = try (0..<capacity).map {
            try writeSource(root, "first-\($0).png", "fixture-\($0)")
        }
        let second = try (0..<capacity).map {
            try writeSource(root, "second-\($0).png", "other-\($0)")
        }

        // Both batches admit themselves against a composer that is still empty.
        withVisionEnabled { model.addImages(first) }
        withVisionEnabled { model.addImages(second) }

        let deadline = Date().addingTimeInterval(10)
        while model.isAddingImages, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        try? await Task.sleep(nanoseconds: 100_000_000)

        #expect(model.imageAttachments.count == capacity,
                "the second batch appended past the cap")
        #expect(stagedFileCount(store) == capacity,
                "the copies that did not fit were left staged and unreferenced")
    }

    @MainActor
    @Test func sendingMovesImagesFromComposerToTranscriptWithoutSharingFiles() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try writeSource(root, "source.png", "fixture")

        let directory = try makeVisionReadyModelInstall("lifetime-keep")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = AppModel(modelDirectory: directory, client: client,
                             attachmentStore: store)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        await attach(model, [source])
        model.promptText = "describe it"
        model.maxNewTokensOverride = 1
        model.run()
        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(model.imageAttachments.isEmpty)
        #expect(model.outputImageAttachments.count == 1)
        model.clearImages()
        #expect(model.imageAttachments.isEmpty)
        let shown = try #require(model.outputImageAttachments.first)
        #expect(try Data(contentsOf: shown.fileURL) == Data("fixture".utf8),
                "clearing the composer deleted the transcript's image")
    }

    @MainActor
    @Test func newChatReleasesTheImagesOfEveryTurnNotJustTheLast() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = try writeSource(root, "first.png", "one")
        let second = try writeSource(root, "second.png", "two")

        let directory = try makeVisionReadyModelInstall("lifetime-multi-turn")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = AppModel(modelDirectory: directory, client: client,
                             attachmentStore: store)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        model.maxNewTokensOverride = 1

        for source in [first, second] {
            await attach(model, [source])
            model.promptText = "describe it"
            model.run()
            for _ in 0..<200 where model.isRunning {
                try? await Task.sleep(nanoseconds: 5_000_000)
            }
        }
        #expect(model.conversation.turns.compactMap { $0.images.first }.count == 2,
                "the fixture needs two turns each holding an image")

        model.newChat()
        #expect(stagedFileCount(store) == 0,
                "New chat left an earlier turn's staged image behind")
    }

    /// Quitting is the one moment every staged file is certainly unwanted, and
    /// nothing called `removeAll()` at all.
    @MainActor
    @Test func releasingEverythingEmptiesTheStore() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try writeSource(root, "source.png", "fixture")
        let directory = try makeVisionReadyModelInstall("lifetime-release-all")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, attachmentStore: store)
        await attach(model, [source, source])
        #expect(stagedFileCount(store) == 2)

        model.releaseAllAttachments()
        #expect(model.imageAttachments.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: store.directoryURL.path))
    }

    /// A run killed before it could clean up leaves its whole directory. Only a
    /// later run can reclaim it, and only if it can tell live from dead.
    @Test func sweepReclaimsDeadRunsAndSparesLiveOnes() throws {
        let manager = FileManager.default
        let root = manager.temporaryDirectory
            .appendingPathComponent("attachment-sweep-\(UUID().uuidString)",
                                    isDirectory: true)
        defer { try? manager.removeItem(at: root) }
        let dead = root.appendingPathComponent("pid-4194300", isDirectory: true)
        let live = root.appendingPathComponent("pid-\(getpid())", isDirectory: true)
        let stray = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
        for directory in [dead, live, stray] {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data([0x1]).write(to: directory.appendingPathComponent("image"))
        }

        let reclaimed = AppImageAttachmentStore.sweepAbandoned(in: root)
        #expect(reclaimed.contains("pid-4194300"))
        #expect(reclaimed.contains(stray.lastPathComponent))
        #expect(!manager.fileExists(atPath: dead.path))
        #expect(manager.fileExists(atPath: live.appendingPathComponent("image").path),
                "the sweep deleted a live run's images")
    }
}

/// Images can arrive with no file behind them: copied from another app, or
/// dragged from one that only promises to write the bytes later. Both used to
/// be refused with an instruction to go and find a file.
@Suite(.serialized) struct AppPastedImageTests {
    private func withVisionEnabled<T>(_ body: () throws -> T) rethrows -> T {
        let previous = ProcessInfo.processInfo.environment[
            "TURBO_FIELDFARE_VISION_RUNTIME"]
        setenv("TURBO_FIELDFARE_VISION_RUNTIME", "1", 1)
        defer {
            if let previous {
                setenv("TURBO_FIELDFARE_VISION_RUNTIME", previous, 1)
            } else {
                unsetenv("TURBO_FIELDFARE_VISION_RUNTIME")
            }
        }
        return try body()
    }

    private func makeStore() -> (root: URL, store: AppImageAttachmentStore) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("pasted-\(UUID().uuidString)", isDirectory: true)
        return (root, AppImageAttachmentStore(
            directoryURL: root.appendingPathComponent("staged", isDirectory: true)))
    }

    @Test func stagingBytesProducesTheSameSealedCopyAsAFile() throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let bytes = Data("pretend png".utf8)

        let attachment = try store.stage(data: bytes, displayName: "Pasted image.png")
        #expect(attachment.displayName == "Pasted image.png")
        #expect(attachment.encodedBytes == bytes.count)
        #expect(try Data(contentsOf: attachment.fileURL) == bytes)
        let attributes = try FileManager.default.attributesOfItem(
            atPath: attachment.fileURL.path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o400,
                "a pasted image must be sealed like any other")
    }

    @Test func pastedBytesAreBoundedTheSameWayAFileIs() throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: VisionImageError.self) {
            _ = try store.stage(data: Data(), displayName: "empty.png")
        }
        let limit = VisionImageLimits().maximumEncodedBytes
        #expect(throws: VisionImageError.self) {
            _ = try store.stage(
                data: Data(repeating: 0x41, count: limit + 1),
                displayName: "huge.png")
        }
    }

    @MainActor
    @Test func pastedImagesJoinTheComposerAndRespectItsCap() async throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = try makeVisionReadyModelInstall("pasted-cap")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, attachmentStore: store)
        // The smallest context the app offers, so the cap is reached without
        // staging dozens of files.
        model.maxContextTokens = AppContextLengthOption.fourK.tokens
        let capacity = model.maximumImageAttachments

        for index in 0..<(capacity + 1) {
            withVisionEnabled {
                model.addImageData(Data("image \(index)".utf8),
                                   displayName: "Pasted image.png")
            }
            let deadline = Date().addingTimeInterval(5)
            while model.isAddingImages, Date() < deadline {
                try? await Task.sleep(nanoseconds: 5_000_000)
            }
        }

        #expect(model.imageAttachments.count == capacity)
        #expect(model.imageAttachmentError != nil,
                "the one past the cap was accepted silently")
        model.releaseAllAttachments()
    }
}

/// The runtime flag says this build can use images; the companion pack says
/// this model can. Gating the composer on the flag alone offered an
/// Add-images button with no tower behind it, and the failure only appeared
/// when the user pressed Generate.
@Suite(.serialized) struct AppImageAvailabilityTests {
    private func withVisionEnabled<T>(_ body: () throws -> T) rethrows -> T {
        let previous = ProcessInfo.processInfo.environment[
            "TURBO_FIELDFARE_VISION_RUNTIME"]
        setenv("TURBO_FIELDFARE_VISION_RUNTIME", "1", 1)
        defer {
            if let previous {
                setenv("TURBO_FIELDFARE_VISION_RUNTIME", previous, 1)
            } else {
                unsetenv("TURBO_FIELDFARE_VISION_RUNTIME")
            }
        }
        return try body()
    }

    @MainActor
    @Test func imageInputNeedsTheCompanionAndNotJustTheFlag() throws {
        let directory = try makeCompleteModelInstall("image-availability")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)
        #expect(model.isModelInstalled)
        #expect(!model.isVisionPackInstalled)

        withVisionEnabled {
            #expect(model.visionRuntimeEnabled)
            #expect(!model.isImageInputAvailable,
                    "the composer offered images with no companion installed")
        }
        #expect(!model.isImageInputAvailable)
    }

    /// The app always releases the image tower after each image, so nothing in
    /// the UI can ask for the other policy. It used to be a Memory setting; the
    /// figure that would have justified it — about 1 GB of page cache held —
    /// is charged to no process and so could never be shown honestly beside the
    /// memory it was traded against.
    @MainActor
    @Test func theAppAlwaysReleasesTheImageTower() throws {
        let directory = try makeVisionReadyModelInstall("residency")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient())
        #expect(model.runtimeOptions.visionResidencyPolicy == .onDemand)
        model.promptText = "go"
        #expect(try model.makeRequest().runtimeOptions
            .visionResidencyPolicy == .onDemand,
                "a run asked for a residency the app no longer offers")
    }

    /// The switch is only meaningful once the tower exists.
    @MainActor
    @Test func residencyIsOnlyOfferedWithTheCompanionInstalled() throws {
        let withPack = try makeVisionReadyModelInstall("residency-with")
        let withoutPack = try makeCompleteModelInstall("residency-without")
        defer {
            try? FileManager.default.removeItem(at: withPack)
            try? FileManager.default.removeItem(at: withoutPack)
        }
        setenv("TURBO_FIELDFARE_VISION_RUNTIME", "1", 1)
        defer { unsetenv("TURBO_FIELDFARE_VISION_RUNTIME") }

        #expect(AppModel(modelDirectory: withPack).isImageInputAvailable)
        #expect(!AppModel(modelDirectory: withoutPack).isImageInputAvailable)
        #expect(!AppModel(modelDirectory: withPack, visionRuntimeSupported: false)
            .isImageInputAvailable,
                "an installed pack exposed image input on unsupported hardware")
    }
}
