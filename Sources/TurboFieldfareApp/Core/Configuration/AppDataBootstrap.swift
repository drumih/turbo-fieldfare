import Foundation
import TurboFieldfareRepackCore

struct AppDataBootstrap {
    var settings = MacAppSettings()
    var failure: String?
    var modelChoiceRequired = false
    var canResetSettings = true

    init(root: URL, legacyModel: URL, environment: [String: String], modelOverride: Bool = false, enabled: Bool) {
        guard enabled else { return }
        do {
            let settingsURL = AppDataLocation.settingsURL(in: root)
            let central = try MacAppSettingsFileStore.load(at: settingsURL)
            if let central { settings = central }
            let history = ConversationStoreLocation.url(forAppDataDirectory: root)
            do {
                try ConversationStoreImport.recover(at: history)
            } catch {
                canResetSettings = false
                failure = "Recovering an interrupted history import at \(history.path) failed: \(error). Existing settings and history were preserved. Repair the history recovery files and reopen the app; automatic settings writes are disabled."
                return
            }
            if let central {
                if central.selectedModelPath != nil || modelOverride
                    || environment["TURBO_FIELDFARE_MODEL_PATH"] != nil { return }
            }
            let isolated = environment["TURBO_FIELDFARE_APP_DATA_PATH"] != nil
            let legacyRoot = isolated ? root : legacyModel.deletingLastPathComponent()
            let legacySettingsURL = AppDataLocation.settingsURL(in: legacyRoot)
            if central == nil, legacySettingsURL != settingsURL,
               let legacySettings = try MacAppSettingsFileStore.load(at: legacySettingsURL) {
                settings = legacySettings
            }
            var selectedLegacy = legacyModel
            if !modelOverride && environment["TURBO_FIELDFARE_MODEL_PATH"] == nil {
                let candidates = Array(Set([legacyModel.standardizedFileURL,
                    root.appendingPathComponent("gemma4.gturbo").standardizedFileURL]))
                let installed = try candidates.filter { candidate in
                    if AppModelInstallationProbe.status(at: candidate) == .complete { return true }
                    return try RemoteStreamingRepacker.inspectPersistentInstall(
                        outputDirectory: candidate.path,
                        repoID: AppModelInstallDescriptor.default.repoID,
                        requestedRevision: AppModelInstallDescriptor.default.revision) != nil
                }
                if installed.count > 1 {
                    modelChoiceRequired = true
                } else {
                    selectedLegacy = installed.first ?? legacyModel
                    settings.selectedModelPath = selectedLegacy.path
                }
            }
            let source = ConversationStoreLocation.url(forModelDirectory: selectedLegacy)
            let destination = ConversationStoreLocation.url(forAppDataDirectory: root)
            if !isolated, !modelChoiceRequired, source != destination,
               !settings.migratedHistorySources.contains(source.path),
               FileManager.default.fileExists(atPath: source.path) {
                try ConversationStoreImport.run(from: source, to: destination)
                settings.migratedHistorySources.append(source.path)
            }
            try MacAppSettingsFileStore.save(settings, at: settingsURL)
        } catch {
            failure = "Could not initialize app data: \(error). Reveal the settings file and repair it, or explicitly reset settings after keeping a backup. Automatic settings writes are disabled."
        }
    }
}
