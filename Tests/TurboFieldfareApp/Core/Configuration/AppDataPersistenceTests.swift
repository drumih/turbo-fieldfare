import Foundation
import Synchronization
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppDataPersistenceTests {
    @Test func settingsSaveDoesNotRemoveAnotherWritersStagingFile() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = AppDataLocation.settingsURL(in: root)
        let otherWriter = file.appendingPathExtension("tmp")
        let bytes = Data("another writer owns this file".utf8)
        try bytes.write(to: otherWriter)
        // The shared .tmp path used to be unlinked by an unrelated save.
        try MacAppSettingsFileStore.save(.init(temperature: 0.71), at: file)
        #expect(try Data(contentsOf: otherWriter) == bytes)
        #expect(try MacAppSettingsFileStore.load(at: file)?.temperature == 0.71)
    }

    @Test func concurrentSettingsSavesPublishWholeSnapshotsWithoutInterfering() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = AppDataLocation.settingsURL(in: root)
        let failures = Mutex<[String]>([])
        // Writers must not share or remove each other's staging paths, even
        // when they enter the file operation concurrently without a store lock.
        DispatchQueue.concurrentPerform(iterations: 32) { index in
            do {
                let settings = MacAppSettings(selectedModelPath: "/models/\(index).gturbo", topK: index + 1)
                try MacAppSettingsFileStore.save(settings, at: file)
                guard let loaded = try MacAppSettingsFileStore.load(at: file),
                      loaded.selectedModelPath == "/models/\(loaded.topK - 1).gturbo" else {
                    throw CocoaError(.fileReadCorruptFile)
                }
            } catch { failures.withLock { $0.append(String(describing: error)) } }
        }
        #expect(failures.withLock { $0 }.isEmpty)
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path) == [MacAppSettings.fileName])
    }

    private func root() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("app data тест \(UUID())")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @MainActor @Test func savedUnavailableLocationWinsAndSurvivesPreferenceWrites() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let unavailable = "/Volumes/Unavailable-\(UUID())/gemma4.gturbo"
        try MacAppSettingsFileStore.save(.init(selectedModelPath: unavailable, temperature: 0.7),
                                         at: AppDataLocation.settingsURL(in: root))
        let model = AppModel(appDataDirectory: root, environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path], client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true)
        #expect(model.modelPathText == unavailable)
        #expect(!model.isModelLocationAvailable)
        #expect(!model.canInstallModel)
        if case .failed(let message) = model.installReadiness {
            #expect(message.contains("selected volume is unavailable"))
        } else { Issue.record("Missing drive must report unavailable instead of local-volume capacity") }
        model.setTextSize(.largest)
        let saved = try #require(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: root)))
        #expect(saved.selectedModelPath == unavailable)
        #expect(saved.temperature == 0.7)
        #expect(model.conversationBinding?.rootURL.path == root.appendingPathComponent("conversations").path)
        #expect(!FileManager.default.fileExists(atPath: "/Volumes/" + URL(fileURLWithPath: unavailable).pathComponents[2]))
    }

    @MainActor @Test func launchOverrideNeverRewritesSavedSelectionOrHistoryRoot() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let saved = root.appendingPathComponent("selected.gturbo").path
        try MacAppSettingsFileStore.save(.init(selectedModelPath: saved), at: AppDataLocation.settingsURL(in: root))
        let model = AppModel(appDataDirectory: root,
                             environment: ["TURBO_FIELDFARE_MODEL_PATH": root.appendingPathComponent("override.gturbo").path],
                             client: MockLifecycleInferenceClient(), settingsPersistenceEnabled: true)
        #expect(model.isModelLocationOverridden)
        #expect(!model.canChangeModelLocation)
        model.setTextSize(.largest)
        #expect(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: root))?.selectedModelPath == saved)
        #expect(model.conversationBinding?.rootURL.path == root.appendingPathComponent("conversations").path)
    }

    @MainActor @Test(arguments: [Data("broken json".utf8), Data("{\"version\":999,\"future\":true}".utf8)])
    func damagedOrNewerSettingsDisableWritesUntilExplicitBackupRecovery(_ data: Data) async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = AppDataLocation.settingsURL(in: root)
        try data.write(to: file)
        let model = AppModel(appDataDirectory: root, environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path], client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true)
        #expect(model.settingsDiagnostic != nil)
        #expect(!model.canChangeModelLocation)
        model.setTextSize(.largest)
        model.shutdownForTermination()
        #expect(try Data(contentsOf: file) == data)
        model.resetSettingsAfterBackingUp()
        #expect(model.settingsDiagnostic == nil)
        let backup = try #require(FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            .first { $0.lastPathComponent.contains(".backup-") })
        #expect(try Data(contentsOf: backup) == data)
        #expect(try MacAppSettingsFileStore.load(at: file)?.textSize == .largest)
        // Recovery has no saved model selection yet. Explicitly choosing the
        // displayed default must commit it even though the runtime path is equal.
        await model.selectModelLocation(root, intent: .download)
        #expect(try MacAppSettingsFileStore.load(at: file)?.selectedModelPath == model.modelPathText)
    }

    @MainActor @Test func durableSelectionPreservesPreferencesAndStoreAcrossRepeatedSwitches() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(appDataDirectory: root, environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path], client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true)
        let store = try #require(model.conversationStore)
        try await store.activate()
        model.setTextSize(.largest)
        for name in ["one тест", "two with spaces", "one тест"] {
            let parent = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            await model.selectModelLocation(parent, intent: .download)
            #expect(model.modelPathText == parent.appendingPathComponent("gemma4.gturbo").path)
            #expect(model.conversationStore === store)
            #expect(await !store.isReadOnly)
            #expect(model.textSize == .largest)
            #expect(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: root))?.selectedModelPath == model.modelPathText)
        }
        let current = URL(fileURLWithPath: model.modelPathText)
        let alias = root.appendingPathComponent("same-parent-alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: current.deletingLastPathComponent())
        model.loadState = .ready(modelDirectory: current, loadSeconds: 0)
        await model.selectModelLocation(alias, intent: .download)
        #expect(model.loadState.isReady)
        #expect(model.conversationStore === store)
        let restored = AppModel(appDataDirectory: root, environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path], client: MockLifecycleInferenceClient(),
                                settingsPersistenceEnabled: true)
        #expect(restored.modelPathText == model.modelPathText)
        #expect(restored.textSize == .largest)
    }

    @MainActor @Test func failedSelectionWritePreservesOldPathAndLoadedRuntime() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(appDataDirectory: root, environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path], client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true)
        let previous = model.modelPathText
        model.loadState = .ready(modelDirectory: URL(fileURLWithPath: previous), loadSeconds: 0)
        let epoch = model.conversation.epoch
        let settings = AppDataLocation.settingsURL(in: root)
        try FileManager.default.removeItem(at: settings)
        try FileManager.default.createDirectory(at: settings, withIntermediateDirectories: false)
        let other = root.appendingPathComponent("another download")
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: false)
        await model.selectModelLocation(other, intent: .download)
        #expect(model.modelPathText == previous)
        #expect(model.loadState.isReady)
        #expect(model.conversation.epoch == epoch)
        #expect(model.modelLocationDiagnostic?.contains("previous selection retained") == true)
    }

    @MainActor @Test func switchingIsRefusedWhileLoadOrInstallOwnsTheRuntime() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(appDataDirectory: root, environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path], client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true)
        model.installState = .copyingPayload(reusedBytes: 0, downloadedThisRunBytes: 1, totalBytes: 2)
        let previous = model.modelPathText
        #expect(!model.canChangeModelLocation)
        await model.selectModelLocation(root, intent: .download)
        #expect(model.modelPathText == previous)
    }

    @Test func appDataOverrideIsIndependentOfModelOverride() {
        let environment = ["TURBO_FIELDFARE_APP_DATA_PATH": "/tmp/app-data", "TURBO_FIELDFARE_MODEL_PATH": "/tmp/model"]
        #expect(AppDataLocation.resolve(environment: environment).path == URL(fileURLWithPath: "/tmp/app-data").resolvingSymlinksInPath().path)
        #expect(AppDataLocation.resolve(environment: [:], applicationSupportURL: URL(fileURLWithPath: "/tmp/support")).path == "/tmp/support/TurboFieldfare")
    }

    @MainActor @Test(arguments: [AppModelLoadState.loading(.tokenizer), .cancelling, .unloading])
    func locationActionRefusesBusyTransitionAndWorksAfterItSettles(_ state: AppModelLoadState) async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(appDataDirectory: root,
                             environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path],
                             client: MockLifecycleInferenceClient(), settingsPersistenceEnabled: true)
        let destination = root.appendingPathComponent("another destination")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
        let original = model.modelPathText
        model.loadState = state
        #expect(!model.canChangeModelLocation)
        await model.selectModelLocation(destination, intent: .download)
        #expect(model.modelPathText == original)
        model.loadState = .notLoaded
        #expect(model.canChangeModelLocation)
        await model.selectModelLocation(destination, intent: .download)
        #expect(model.modelPathText == destination.appendingPathComponent("gemma4.gturbo").path)
        model.shutdownForTermination()
    }
}
