import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct MacAppSettingsTests {
    @Test func settingsFileLivesBesideModelDirectory() {
        let model = URL(fileURLWithPath: "/tmp/TurboFieldfare/gemma4.gturbo",
                        isDirectory: true)
        #expect(MacAppSettingsFileStore.fileURL(forModelDirectory: model).path
            == "/tmp/TurboFieldfare/mac-app-settings.json")
    }

    @Test func missingFileCreatesReadableDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)

        #expect(settings == MacAppSettings())
        #expect(FileManager.default.fileExists(atPath: fileURL.path))
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: Data(contentsOf: fileURL))
        #expect(decoded == MacAppSettings())
    }

    @Test func malformedFileIsReplacedWithDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        try Data("not json".utf8).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        #expect(settings == MacAppSettings())
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: Data(contentsOf: fileURL))
        #expect(decoded == MacAppSettings())
    }

    @Test func invalidValuesAreReplacedWithDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let invalid = MacAppSettings(contextTokens: 123)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        try JSONEncoder().encode(invalid).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        #expect(settings == MacAppSettings())
    }

    @MainActor
    @Test func appModelLoadsAndSavesPersistedSettings() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory,
            withIntermediateDirectories: true)
        let initial = MacAppSettings(
            contextTokens: 8_192,
            expertCacheSlots: 24,
            temperature: 0.4,
            topKEnabled: false,
            topK: 32,
            topPEnabled: false,
            topP: 0.8,
            prefillEnabled: false)
        try MacAppSettingsFileStore.save(initial, forModelDirectory: modelDirectory)

        let model = AppModel(
            modelDirectory: modelDirectory,
            settingsPersistenceEnabled: true)
        #expect(model.maxContextTokens == 8_192)
        #expect(model.runtimeOptions.expertCacheSlots == 24)
        #expect(model.temperature == 0.4)
        #expect(!model.topKEnabled)
        #expect(model.topK == 32)
        #expect(!model.topPEnabled)
        #expect(model.topP == 0.8)
        #expect(!model.runtimeOptions.prefillEnabled)

        model.temperature = 0.6
        model.runtimeOptions.expertCacheSlots = 32
        model.runtimeOptions.prefillEnabled = true
        let beforeGenerate = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(beforeGenerate == initial)

        model.loadState = .ready(modelDirectory: modelDirectory, loadSeconds: 0)
        model.promptText = "Save these settings"
        model.run()
        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(saved.temperature == 0.6)
        #expect(saved.expertCacheSlots == 32)
        #expect(saved.prefillEnabled)
        model.cancel()
    }

    private func makeTemporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("MacAppSettingsTests-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        return root
    }
}
