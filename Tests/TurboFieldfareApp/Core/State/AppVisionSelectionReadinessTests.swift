import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AppVisionSelectionReadinessTests {
    @MainActor @Test func selectingInstalledPairClearsMissingTextFailureAndSurvivesRelaunch() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("vision-selection-\(UUID())")
        let text = try makeVisionReadyModelInstall("selection-readiness")
        let vision = try VisionPackLocation.companionURL(forTextModel: text)
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: text)
            try? FileManager.default.removeItem(at: vision)
        }
        let model = AppModel(appDataDirectory: root,
                             environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path],
                             client: MockLifecycleInferenceClient(), settingsPersistenceEnabled: true)
        defer { model.shutdownForTermination() }
        #expect(model.visionInstallReadiness == .failed("Install the text model first"))
        await model.selectModelLocation(text, intent: .existing)
        #expect(model.isModelInstalled)
        #expect(model.installReadiness == .installed)
        #expect(!model.canInstallModel)
        #expect(model.isVisionPackInstalled)
        #expect(model.isImageInputAvailable)
        // Previously the completed-pack early return retained the first-run failure.
        #expect(model.visionInstallReadiness == .installed)
        #expect(!model.canInstallVisionPack)
        model.refreshVisionInstallReadiness()
        #expect(model.visionInstallReadiness == .installed)
        model.shutdownForTermination()
        let reopened = AppModel(appDataDirectory: root,
                                environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path],
                                client: MockLifecycleInferenceClient(), settingsPersistenceEnabled: true)
        defer { reopened.shutdownForTermination() }
        #expect(reopened.isVisionPackInstalled)
        #expect(reopened.visionInstallReadiness == .installed)
    }

    @MainActor @Test func selectingSameModelRefreshesVisionAndClearsOnlyItsSelectionError() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("same-model-\(UUID())")
        let text = try makeCompleteModelInstall("same-model-refresh")
        let vision = try VisionPackLocation.companionURL(forTextModel: text)
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: text)
            try? FileManager.default.removeItem(at: vision)
        }
        let model = AppModel(appDataDirectory: root,
                             environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path],
                             client: MockLifecycleInferenceClient(), settingsPersistenceEnabled: true)
        defer { model.shutdownForTermination() }
        await model.selectModelLocation(text, intent: .existing)
        #expect(!model.isVisionPackInstalled)
        model.loadState = .ready(modelDirectory: text, loadSeconds: 0)
        await model.reviewExistingModelLocation(root.appendingPathComponent("absent"))
        #expect(model.modelLocationDiagnostic != nil)
        #expect(model.error != nil)
        try installVisionCompanion(forTextModel: text)
        // Reselecting the active path previously returned before refreshing its companion.
        await model.reviewExistingModelLocation(text)
        #expect(model.isVisionPackInstalled)
        #expect(model.visionInstallReadiness == .installed)
        #expect(model.loadState.isReady)
        #expect(model.modelLocationDiagnostic == nil)
        #expect(model.error == nil)
        model.error = .connectionLost("independent failure")
        await model.reviewExistingModelLocation(text)
        #expect(model.error == .connectionLost("independent failure"))
    }

    @MainActor @Test func modelSwitchBlocksImagesWhileWaitingForChatPersistence() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("switch-images-\(UUID())")
        let text = try makeVisionReadyModelInstall("switch-images")
        let vision = try VisionPackLocation.companionURL(forTextModel: text)
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: text)
            try? FileManager.default.removeItem(at: vision)
        }
        let model = AppModel(appDataDirectory: root,
                             environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path],
                             client: MockLifecycleInferenceClient(), settingsPersistenceEnabled: true)
        defer { model.shutdownForTermination() }
        await model.selectModelLocation(text, intent: .existing)
        let gate = AsyncStream<Void>.makeStream()
        model.persistenceTail = Task { for await _ in gate.stream {} }
        let destination = root.appendingPathComponent("new-destination")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let change = Task { await model.selectModelLocation(destination, intent: .download) }
        while !model.isChangingModelLocation { await Task.yield() }
        // Promised drops can arrive after the panel has begun switching models.
        let promised = root.appendingPathComponent("promised")
        try FileManager.default.createDirectory(at: promised, withIntermediateDirectories: true)
        let file = promised.appendingPathComponent("image.png")
        try Data([0]).write(to: file)
        model.addImages([file], discardingSourceDirectory: promised)
        #expect(!model.isAddingImages)
        #expect(!FileManager.default.fileExists(atPath: promised.path))
        model.addImageData(Data([0]), displayName: "pasted.png")
        #expect(!model.isAddingImages)
        #expect(model.imageAttachmentError?.contains("model location change") == true)
        gate.continuation.finish()
        await change.value
    }

    @MainActor @Test func installedPairClearsStorageWarningButMissingOrDamagedFilesRemainVisible() throws {
        let text = try makeCompleteModelInstall("vision-readiness-transitions")
        let vision = try VisionPackLocation.companionURL(forTextModel: text)
        defer {
            try? FileManager.default.removeItem(at: text)
            try? FileManager.default.removeItem(at: vision)
        }
        let installer = MockVisionPackInstallerClient(requirement: .init(probePath: text.path, requiredBytes: 10, availableBytes: 0))
        let model = AppModel(modelDirectory: text, client: MockLifecycleInferenceClient(), visionInstaller: installer)
        defer { model.shutdownForTermination() }
        #expect(model.visionInstallReadiness.requirement?.shortfallBytes == 10)
        try installVisionCompanion(forTextModel: text)
        model.recheckModelAtCurrentLocation()
        #expect(model.visionInstallReadiness == .installed)
        #expect(model.isImageInputAvailable)

        let manifest = text.appendingPathComponent("manifest.json")
        let saved = try Data(contentsOf: manifest)
        try FileManager.default.removeItem(at: manifest)
        model.recheckModelAtCurrentLocation()
        #expect(model.visionInstallReadiness == .failed("Install the text model first"))
        #expect(!model.isImageInputAvailable)
        try saved.write(to: manifest)
        model.recheckModelAtCurrentLocation()
        #expect(model.visionInstallReadiness == .installed)

        try Data("broken".utf8).write(to: vision.appendingPathComponent("manifest.json"))
        model.recheckModelAtCurrentLocation()
        if case .partial = model.visionInstallationStatus {} else { Issue.record("Corrupt companion must remain a visible repair failure") }
        #expect(!model.isImageInputAvailable)
        #expect(model.visionInstallReadiness != .installed)
        try FileManager.default.removeItem(at: vision)
        model.recheckModelAtCurrentLocation()
        #expect(model.visionInstallationStatus == .missing)
        #expect(model.visionInstallReadiness.requirement?.shortfallBytes == 10)
    }
}
