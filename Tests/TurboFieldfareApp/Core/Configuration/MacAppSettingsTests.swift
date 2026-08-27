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

    @Test func legacySettingsDefaultToReturnWithoutLosingValues() throws {
        let data = Data("""
        {
          "version": 1,
          "contextTokens": 8192,
          "expertCacheSlots": 24,
          "temperature": 0.4,
          "topKEnabled": false,
          "topK": 32,
          "topPEnabled": false,
          "topP": 0.8,
          "prefillEnabled": false
        }
        """.utf8)

        let settings = try JSONDecoder().decode(MacAppSettings.self, from: data)

        #expect(settings.contextTokens == 8_192)
        #expect(settings.expertCacheSlots == 24)
        #expect(settings.temperature == 0.4)
        #expect(!settings.topKEnabled)
        #expect(settings.topK == 32)
        #expect(!settings.topPEnabled)
        #expect(settings.topP == 0.8)
        #expect(!settings.prefillEnabled)
        #expect(settings.newlineShortcut == .return)
        #expect(settings.showPromptExamples)
        #expect(settings.visionResidencyPolicy == .onDemand)
    }

    @Test(arguments: AppNewlineShortcut.allCases)
    func newlineShortcutRoundTrips(_ shortcut: AppNewlineShortcut) throws {
        let initial = MacAppSettings(newlineShortcut: shortcut)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @Test func sendMessageOptionsUseUserFacingOrderAndLabels() {
        #expect(AppNewlineShortcut.sendMessageOptions == [.shiftReturn, .return])
        #expect(AppNewlineShortcut.shiftReturn.sendMessageLabel == "Return")
        #expect(AppNewlineShortcut.return.sendMessageLabel == "Command-Return")
    }

    @Test(arguments: [true, false])
    func showPromptExamplesRoundTrips(_ show: Bool) throws {
        let initial = MacAppSettings(showPromptExamples: show)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @Test func invalidNewlineShortcutIsReplacedWithDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let invalid = Data("""
        {
          "version": 1,
          "contextTokens": 4096,
          "expertCacheSlots": 16,
          "temperature": 0.2,
          "topKEnabled": true,
          "topK": 64,
          "topPEnabled": true,
          "topP": 0.95,
          "prefillEnabled": true,
          "newlineShortcut": "invalid"
        }
        """.utf8)
        try invalid.write(to: fileURL)

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
            prefillEnabled: false,
            newlineShortcut: .shiftReturn,
            showPromptExamples: false)
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
        #expect(model.newlineShortcut == .shiftReturn)
        #expect(!model.showPromptExamples)

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
        #expect(saved.newlineShortcut == .shiftReturn)
        #expect(!saved.showPromptExamples)
        model.cancel()
    }

    @MainActor
    @Test func newlineShortcutPersistsImmediately() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = AppModel(
            modelDirectory: modelDirectory,
            settingsPersistenceEnabled: true)

        model.setNewlineShortcut(.shiftReturn)

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(saved.newlineShortcut == .shiftReturn)
    }

    @MainActor
    @Test func showPromptExamplesPersistsImmediately() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = AppModel(
            modelDirectory: modelDirectory,
            settingsPersistenceEnabled: true)

        model.setShowPromptExamples(false)

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(!saved.showPromptExamples)
    }

    @MainActor
    @Test func changingModelDirectoryLoadsItsNewlineShortcut() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first/model.gturbo", isDirectory: true)
        let second = root.appendingPathComponent("second/model.gturbo", isDirectory: true)
        try MacAppSettingsFileStore.save(
            MacAppSettings(newlineShortcut: .return, showPromptExamples: true),
            forModelDirectory: first)
        try MacAppSettingsFileStore.save(
            MacAppSettings(newlineShortcut: .shiftReturn, showPromptExamples: false),
            forModelDirectory: second)
        let model = AppModel(modelDirectory: first, settingsPersistenceEnabled: true)
        #expect(model.newlineShortcut == .return)
        #expect(model.showPromptExamples)

        model.setModelURL(second)

        #expect(model.newlineShortcut == .shiftReturn)
        #expect(!model.showPromptExamples)
    }

    /// Phase D item 15. The newer-version branch says "Every key decodes with
    /// `decodeIfPresent`, so it reads cleanly", and that is false: nine keys use
    /// a hard `decode`. So a version-3 file whose schema moved any of those nine
    /// throws inside `JSONDecoder().decode` *before* the version guard is
    /// reached, and lands in the `catch` that deletes the file - destroying a
    /// newer build's settings, which is the exact outcome that branch exists to
    /// prevent. A version bump that cannot change the schema protects nothing.
    @Test func aNewerSettingsFileSurvivesAKeyThisBuildDoesNotKnow() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: modelDirectory)

        // A plausible version 3: `topP` became `topProbability`. Everything else
        // this build knows is still present and still valid.
        let newer = """
        {
          "version": 3,
          "contextTokens": 8192,
          "expertCacheSlots": 16,
          "temperature": 0.2,
          "topKEnabled": true,
          "topK": 64,
          "topPEnabled": true,
          "topProbability": 0.95,
          "prefillEnabled": true
        }
        """
        try Data(newer.utf8).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: modelDirectory)

        #expect(settings == MacAppSettings(), "this build must run on defaults")
        #expect(FileManager.default.fileExists(atPath: fileURL.path),
                "a newer build's settings file was deleted by an older build")
        #expect(try String(contentsOf: fileURL, encoding: .utf8) == newer,
                "a newer build's settings file was rewritten by an older build")
    }

    /// Phase D item 16. RDADVISE is a Picker in `InspectorView` bound to
    /// `runtimeOptions.rdadvisePolicy`, and `RUNTIME_CONTROLS.md` lists it as an
    /// app control - but it had no `CodingKey`, so every relaunch silently
    /// reverted it while the UI kept offering it as a persistent setting.
    @Test func rdadviseSurvivesARelaunch() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)

        try MacAppSettingsFileStore.save(
            MacAppSettings(rdadvisePolicy: .bounded),
            forModelDirectory: modelDirectory)
        let reloaded = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: modelDirectory)

        #expect(reloaded.rdadvisePolicy == .bounded)
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
    @Test func visionResidencyRoundTrips() throws {
        let initial = MacAppSettings(visionResidencyPolicy: .keepReady)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @MainActor
    @Test func changingModelDirectoryStillReleasesTheImageTower() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first/model.gturbo", isDirectory: true)
        let second = root.appendingPathComponent("second/model.gturbo", isDirectory: true)
        try MacAppSettingsFileStore.save(
            MacAppSettings(visionResidencyPolicy: .onDemand),
            forModelDirectory: first)
        try MacAppSettingsFileStore.save(
            MacAppSettings(visionResidencyPolicy: .keepReady),
            forModelDirectory: second)
        let model = AppModel(modelDirectory: first, settingsPersistenceEnabled: true)
        #expect(model.runtimeOptions.visionResidencyPolicy == .onDemand)

        model.setModelURL(second)

        #expect(model.runtimeOptions.visionResidencyPolicy == .onDemand,
                "a persisted keep-ready came back through the model path change")
    }

}
