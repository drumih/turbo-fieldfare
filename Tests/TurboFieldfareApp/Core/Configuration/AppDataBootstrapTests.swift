import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AppDataBootstrapTests {
    @Test(arguments: [false, true])
    func completedOverrideImportIsNotRepeatedAfterCentralChatChanges(explicitOverride: Bool) throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = root.appendingPathComponent("legacy/gemma4.gturbo")
        let central = root.appendingPathComponent("app-data")
        let source = ConversationStoreLocation.url(forModelDirectory: legacy)
        let id = UUID().uuidString
        let conversation = source.appendingPathComponent(id)
        try FileManager.default.createDirectory(at: conversation, withIntermediateDirectories: true)
        let original = Data("legacy chat".utf8)
        for name in ["conversation.json", "transcript.jsonl"] {
            try original.write(to: conversation.appendingPathComponent(name))
        }
        let first = AppDataBootstrap(root: central, legacyModel: legacy,
            environment: explicitOverride ? [:] : ["TURBO_FIELDFARE_MODEL_PATH": legacy.path],
            modelOverride: explicitOverride, enabled: true)
        #expect(first.failure == nil)
        #expect(first.settings.selectedModelPath == nil)
        #expect(first.settings.migratedHistorySources == [source.path])
        let imported = ConversationStoreLocation.url(forAppDataDirectory: central).appendingPathComponent(id)
        let changed = Data("continued and renamed centrally".utf8)
        for name in ["conversation.json", "transcript.jsonl"] {
            try changed.write(to: imported.appendingPathComponent(name))
        }
        // Reimporting this completed source used to report an ID conflict and
        // disable all settings writes and model selection on normal relaunch.
        let next = AppDataBootstrap(root: central, legacyModel: legacy, environment: [:], enabled: true)
        #expect(next.failure == nil)
        #expect(next.settings.selectedModelPath == legacy.path)
        #expect(next.settings.migratedHistorySources == [source.path])
        #expect(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: central))?.selectedModelPath == legacy.path)
        for name in ["conversation.json", "transcript.jsonl"] {
            #expect(try Data(contentsOf: imported.appendingPathComponent(name)) == changed)
            #expect(try Data(contentsOf: conversation.appendingPathComponent(name)) == original)
        }
    }

    private func root() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("bootstrap-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @Test func firstLaunchImportsLegacyPreferencesAndHistoryWithoutMovingOriginals() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = root.appendingPathComponent("legacy/gemma4.gturbo")
        let central = root.appendingPathComponent("app-data")
        try MacAppSettingsFileStore.save(.init(temperature: 0.73, textSize: .largest),
            at: AppDataLocation.settingsURL(in: legacy.deletingLastPathComponent()))
        let name = UUID().uuidString
        let conversation = ConversationStoreLocation.url(forModelDirectory: legacy).appendingPathComponent(name)
        try FileManager.default.createDirectory(at: conversation, withIntermediateDirectories: true)
        for file in ["conversation.json", "transcript.jsonl"] {
            try Data("fixture".utf8).write(to: conversation.appendingPathComponent(file))
        }
        let bootstrap = AppDataBootstrap(root: central, legacyModel: legacy, environment: [:], enabled: true)
        #expect(bootstrap.failure == nil)
        #expect(bootstrap.settings.temperature == 0.73)
        #expect(bootstrap.settings.textSize == .largest)
        #expect(bootstrap.settings.selectedModelPath == legacy.path)
        let copied = ConversationStoreLocation.url(forAppDataDirectory: central).appendingPathComponent(name)
        #expect(try Data(contentsOf: copied.appendingPathComponent("transcript.jsonl")) == Data("fixture".utf8))
        #expect(FileManager.default.fileExists(atPath: conversation.path))
        let relaunched = AppDataBootstrap(root: central, legacyModel: legacy, environment: [:], enabled: true)
        #expect(relaunched.failure == nil)
        #expect(relaunched.settings.migratedHistorySources.count == 1)
    }

    @Test func centralSettingsWinWithoutReadingMalformedLegacySettings() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let central = root.appendingPathComponent("app-data")
        let legacy = root.appendingPathComponent("legacy/gemma4.gturbo")
        let legacySettings = AppDataLocation.settingsURL(in: legacy.deletingLastPathComponent())
        try FileManager.default.createDirectory(at: legacySettings.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("broken".utf8).write(to: legacySettings)
        let saved = "/Volumes/unavailable/gemma4.gturbo"
        try MacAppSettingsFileStore.save(.init(selectedModelPath: saved, temperature: 0.81),
                                        at: AppDataLocation.settingsURL(in: central))
        let bootstrap = AppDataBootstrap(root: central, legacyModel: legacy, environment: [:], enabled: true)
        #expect(bootstrap.failure == nil)
        #expect(bootstrap.settings.selectedModelPath == saved)
        #expect(bootstrap.settings.temperature == 0.81)
        #expect(try Data(contentsOf: legacySettings) == Data("broken".utf8))
    }

    @Test func isolatedRootDoesNotReadLegacyPreferencesOrHistory() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let central = root.appendingPathComponent("app-data")
        let legacy = root.appendingPathComponent("legacy/gemma4.gturbo")
        let legacySettings = AppDataLocation.settingsURL(in: legacy.deletingLastPathComponent())
        try FileManager.default.createDirectory(at: legacySettings.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("broken".utf8).write(to: legacySettings)
        let bootstrap = AppDataBootstrap(root: central, legacyModel: legacy,
            environment: ["TURBO_FIELDFARE_APP_DATA_PATH": central.path], enabled: true)
        #expect(bootstrap.failure == nil)
        #expect(bootstrap.settings.temperature == MacAppSettings().temperature)
        #expect(bootstrap.settings.migratedHistorySources.isEmpty)
    }

    @Test func multipleKnownInstallsRequireChoiceAndKeepCentralPreferencesOnRelaunch() throws {
        let root = try root()
        let legacy = try makeCompleteModelInstall("bootstrap-ambiguity")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: legacy)
        }
        let second = root.appendingPathComponent("gemma4.gturbo")
        try FileManager.default.copyItem(at: legacy, to: second)
        let receipt = second.appendingPathComponent("verified-install.json")
        var metadata = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: receipt)) as? [String: Any])
        metadata["modelDirectoryPath"] = second.standardizedFileURL.path
        try JSONSerialization.data(withJSONObject: metadata).write(to: receipt)
        #expect(AppModelInstallationProbe.status(at: second) == .complete)
        try MacAppSettingsFileStore.save(.init(temperature: 0.91), at: AppDataLocation.settingsURL(in: root))
        for _ in 0..<2 {
            let bootstrap = AppDataBootstrap(root: root, legacyModel: legacy, environment: [:], enabled: true)
            #expect(bootstrap.failure == nil)
            #expect(bootstrap.modelChoiceRequired)
            #expect(bootstrap.settings.selectedModelPath == nil)
            #expect(bootstrap.settings.temperature == 0.91)
            #expect(bootstrap.settings.migratedHistorySources.isEmpty)
        }
    }

    @Test func failedHistoryRecoveryPreservesReadableSettingsAndExplainsTheCorrectRepair() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let saved = root.appendingPathComponent("selected.gturbo").path
        let file = AppDataLocation.settingsURL(in: root)
        try MacAppSettingsFileStore.save(.init(selectedModelPath: saved, temperature: 0.67), at: file)
        let bytes = try Data(contentsOf: file)
        let store = ConversationStoreLocation.url(forAppDataDirectory: root)
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        try Data("broken journal".utf8).write(to: store.appendingPathComponent(".history-import.json"))
        let bootstrap = AppDataBootstrap(root: root, legacyModel: root.appendingPathComponent("default.gturbo"),
                                         environment: [:], enabled: true)
        #expect(bootstrap.settings.selectedModelPath == saved)
        #expect(bootstrap.settings.temperature == 0.67)
        #expect(bootstrap.failure?.contains("Recovering an interrupted history import") == true)
        #expect(bootstrap.failure?.contains("reset settings") == false)
        #expect(!bootstrap.canResetSettings)
        #expect(try Data(contentsOf: file) == bytes)
    }

    @Test func interruptedImportRecoversOnStartupEvenWhenSavedSourceDriveIsUnavailable() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let saved = "/Volumes/Unavailable-\(UUID())/gemma4.gturbo"
        try MacAppSettingsFileStore.save(.init(selectedModelPath: saved, temperature: 0.67),
                                        at: AppDataLocation.settingsURL(in: root))
        let store = ConversationStoreLocation.url(forAppDataDirectory: root)
        let name = UUID().uuidString
        let published = store.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: published, withIntermediateDirectories: true)
        let data = Data("bounded fixture".utf8)
        var fingerprints: [String: String] = [:]
        for file in ["conversation.json", "transcript.jsonl"] {
            try data.write(to: published.appendingPathComponent(file))
            fingerprints[file] = Sha256Verifier.hashData(data)
        }
        let journal = store.appendingPathComponent(".history-import.json")
        try JSONEncoder().encode(ConversationStoreImport.Journal(source: saved, fingerprints: [name: fingerprints])).write(to: journal)
        let bootstrap = AppDataBootstrap(root: root, legacyModel: URL(fileURLWithPath: saved), environment: [:], enabled: true)
        #expect(bootstrap.failure == nil)
        #expect(bootstrap.settings.selectedModelPath == saved)
        #expect(bootstrap.settings.temperature == 0.67)
        #expect(!FileManager.default.fileExists(atPath: journal.path))
        #expect(!FileManager.default.fileExists(atPath: published.path))
    }
}
