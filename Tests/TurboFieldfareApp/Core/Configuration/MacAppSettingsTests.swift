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

    /// Additive, like every field around it: a settings file written before the
    /// sidebar existed decodes with the list shown, and a hidden one survives a
    /// round trip. Neither costs a version bump.
    @Test func aSettingsFileWithoutSidebarVisibleDecodesToShown() throws {
        let json = """
        {"version":2,"contextTokens":8192,"expertCacheSlots":16,\
        "temperature":0.2,"topKEnabled":true,"topK":64,"topPEnabled":true,\
        "topP":0.95,"prefillEnabled":true}
        """
        let settings = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(json.utf8))
        #expect(settings.sidebarVisible)
        #expect(settings.version == 2)
    }

    @Test(arguments: [true, false])
    func sidebarVisibleRoundTrips(_ visible: Bool) throws {
        let initial = MacAppSettings(sidebarVisible: visible)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self, from: try JSONEncoder().encode(initial))
        #expect(decoded.sidebarVisible == visible)
        #expect(decoded.version == MacAppSettings.currentVersion)
    }

    @Test func selectedConversationRoundTripsAndIsAdditive() throws {
        let id = UUID()
        let initial = MacAppSettings(selectedConversationID: id)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self, from: try JSONEncoder().encode(initial))
        #expect(decoded.selectedConversationID == id)

        let oldJSON = """
        {"version":2,"contextTokens":8192,"expertCacheSlots":16,
        "temperature":0.2,"topKEnabled":true,"topK":64,"topPEnabled":true,
        "topP":0.95,"prefillEnabled":true}
        """
        let old = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(oldJSON.utf8))
        #expect(old.selectedConversationID == nil)
    }

    @MainActor
    @Test func contextWithoutASelectedRowPersistsImmediately() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("model.gturbo", isDirectory: true)
        let model = AppModel(modelDirectory: directory, settingsPersistenceEnabled: true)

        model.setMaxContextTokens(4_096)

        let saved = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
        #expect(saved.contextTokens == 4_096)
    }

    @MainActor
    @Test func unknownRestoredSelectionIsClearedFromSettings() async throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("model.gturbo", isDirectory: true)
        try MacAppSettingsFileStore.save(
            MacAppSettings(selectedConversationID: UUID()),
            forModelDirectory: directory)

        let model = AppModel(modelDirectory: directory, settingsPersistenceEnabled: true)
        try await waitUntil {
            MacAppSettingsFileStore.loadOrCreate(
                forModelDirectory: directory).selectedConversationID == nil
        }

        #expect(model.history.selection == nil)
    }

    /// The Inspector's visibility is remembered on the same terms as the
    /// sidebar's: additive, no version bump, and a file that predates it opens
    /// with the panel shown rather than with the window missing a side.
    @Test func aSettingsFileWithoutInspectorVisibleDecodesToShown() throws {
        let json = """
        {"version":2,"contextTokens":8192,"expertCacheSlots":16,\
        "temperature":0.2,"topKEnabled":true,"topK":64,"topPEnabled":true,\
        "topP":0.95,"prefillEnabled":true,"sidebarVisible":false}
        """
        let settings = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(json.utf8))
        #expect(settings.inspectorVisible)
        #expect(!settings.sidebarVisible)
        #expect(settings.version == 2)
    }

    @Test(arguments: [true, false])
    func inspectorVisibleRoundTrips(_ visible: Bool) throws {
        let initial = MacAppSettings(inspectorVisible: visible)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self, from: try JSONEncoder().encode(initial))
        #expect(decoded.inspectorVisible == visible)
        #expect(decoded.version == MacAppSettings.currentVersion)
    }

    /// The v1 file is restamped, and nothing else about it changes.
    ///
    /// 4K has never been the default — `AppContextLengthOption.eightK` is
    /// 8K in v1 too — so a stored 4,096 is a choice someone made to fit an 8 GB
    /// machine, not a stale default to repair. Rewriting it here doubled the
    /// full-attention KV of that session and wrote the original off disk, with
    /// 4K still offered in the picker.
    /// The mirror image of the migration test. A file written by a newer build
    /// decodes cleanly here — every key is `decodeIfPresent` — so rejecting it
    /// on the version number alone sent it through the catch that deletes the
    /// file, and running an older build once against the same model directory
    /// silently reset everything its owner had chosen, a deliberate 4K context
    /// included. This build runs on defaults and leaves the file alone.
    @Test func afileFromANewerBuildIsLeftAloneRatherThanDeleted() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let future = MacAppSettings(
            version: MacAppSettings.currentVersion + 1,
            contextTokens: 4_096,
            expertCacheSlots: 24,
            temperature: 0.4)
        try JSONEncoder().encode(future).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        #expect(settings == MacAppSettings(),
                "an unknown version should be run on defaults, not adopted")
        #expect(FileManager.default.fileExists(atPath: fileURL.path),
                "the newer build's settings file was deleted")
        let persisted = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(contentsOf: fileURL))
        #expect(persisted == future,
                "the newer build's settings were overwritten by this one")
    }

    @Test func versionOneMigrationKeepsAChosen4KContext() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let legacy = MacAppSettings(
            version: 1,
            contextTokens: 4_096,
            expertCacheSlots: 24,
            temperature: 0.4)
        try JSONEncoder().encode(legacy).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)
        let persisted = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(contentsOf: fileURL))

        #expect(settings.version == MacAppSettings.currentVersion)
        #expect(settings.contextTokens == 4_096,
                "a deliberately chosen 4K context was rewritten by the migration")
        #expect(settings.expertCacheSlots == 24)
        #expect(settings.temperature == 0.4)
        #expect(persisted == settings)
        #expect(AppContextLengthOption.eightK.tokens != 4_096,
                "if 4K ever becomes the default, this test is asking the wrong question")
    }

    /// A v1 file at any other context is restamped untouched too — the version
    /// bump must not be a licence to edit values.
    @Test func versionOneMigrationOnlyRestampsTheVersion() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let legacy = MacAppSettings(version: 1, contextTokens: 8_192)
        try JSONEncoder().encode(legacy).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        var expected = legacy
        expected.version = MacAppSettings.currentVersion
        #expect(settings == expected)
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
    @Test func appModelLoadsAndSavesPersistedSettings() async throws {
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
        model.send()
        await SendWaiting.generationStarts(model)
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
