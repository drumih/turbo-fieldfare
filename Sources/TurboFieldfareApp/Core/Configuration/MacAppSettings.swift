import Foundation
import TurboFieldfare

struct MacAppSettings: Codable, Equatable, Sendable {
    static let fileName = "mac-app-settings.json"
    static let currentVersion = 2

    var version: Int = currentVersion
    var contextTokens: Int = AppContextLengthOption.eightK.tokens
    var expertCacheSlots: Int = 16
    var temperature: Double = 0.2
    var topKEnabled: Bool = true
    var topK: Int = 64
    var topPEnabled: Bool = true
    var topP: Double = 0.95
    var prefillEnabled: Bool = true
    var newlineShortcut: AppNewlineShortcut = .return
    var showPromptExamples: Bool = true
    /// Whether the list of chats is showing. Remembered because hiding it is a
    /// choice about how the window looks, and a window that forgot it every
    /// launch would be making that choice again for the user each time.
    var sidebarVisible: Bool = true
    /// Whether the Inspector is showing. Remembered for the same reason the
    /// sidebar is: it is a choice about the window, not about one session.
    var inspectorVisible: Bool = true
    var visionResidencyPolicy: VisionResidencyPolicy = .onDemand
    var rdadvisePolicy: AppRDAdvicePolicy = .off
    var loadModelOnLaunch: Bool = false
    var selectedConversationID: UUID?

    private enum CodingKeys: String, CodingKey {
        case version
        case contextTokens
        case expertCacheSlots
        case temperature
        case topKEnabled
        case topK
        case topPEnabled
        case topP
        case prefillEnabled
        case newlineShortcut
        case showPromptExamples
        case sidebarVisible
        case inspectorVisible
        case visionResidencyPolicy
        case rdadvisePolicy
        case loadModelOnLaunch
        case selectedConversationID
    }

    init(version: Int = currentVersion,
         contextTokens: Int = AppContextLengthOption.eightK.tokens,
         expertCacheSlots: Int = 16,
         temperature: Double = 0.2,
         topKEnabled: Bool = true,
         topK: Int = 64,
         topPEnabled: Bool = true,
         topP: Double = 0.95,
         prefillEnabled: Bool = true,
         newlineShortcut: AppNewlineShortcut = .return,
         showPromptExamples: Bool = true,
         sidebarVisible: Bool = true,
         inspectorVisible: Bool = true,
         visionResidencyPolicy: VisionResidencyPolicy = .onDemand,
         rdadvisePolicy: AppRDAdvicePolicy = .off,
         loadModelOnLaunch: Bool = false,
         selectedConversationID: UUID? = nil) {
        self.version = version
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.prefillEnabled = prefillEnabled
        self.newlineShortcut = newlineShortcut
        self.showPromptExamples = showPromptExamples
        self.sidebarVisible = sidebarVisible
        self.inspectorVisible = inspectorVisible
        self.visionResidencyPolicy = visionResidencyPolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.loadModelOnLaunch = loadModelOnLaunch
        self.selectedConversationID = selectedConversationID
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        contextTokens = try container.decode(Int.self, forKey: .contextTokens)
        expertCacheSlots = try container.decode(Int.self, forKey: .expertCacheSlots)
        temperature = try container.decode(Double.self, forKey: .temperature)
        topKEnabled = try container.decode(Bool.self, forKey: .topKEnabled)
        topK = try container.decode(Int.self, forKey: .topK)
        topPEnabled = try container.decode(Bool.self, forKey: .topPEnabled)
        topP = try container.decode(Double.self, forKey: .topP)
        prefillEnabled = try container.decode(Bool.self, forKey: .prefillEnabled)
        newlineShortcut = try container.decodeIfPresent(
            AppNewlineShortcut.self,
            forKey: .newlineShortcut) ?? .return
        showPromptExamples = try container.decodeIfPresent(
            Bool.self,
            forKey: .showPromptExamples) ?? true
        // Additive, like every field around it: absent means the default, so
        // no version bump and no rewrite of a file an older build can still
        // read.
        sidebarVisible = try container.decodeIfPresent(
            Bool.self,
            forKey: .sidebarVisible) ?? true
        inspectorVisible = try container.decodeIfPresent(
            Bool.self,
            forKey: .inspectorVisible) ?? true
        visionResidencyPolicy = try container.decodeIfPresent(
            VisionResidencyPolicy.self,
            forKey: .visionResidencyPolicy) ?? .onDemand
        rdadvisePolicy = try container.decodeIfPresent(
            AppRDAdvicePolicy.self,
            forKey: .rdadvisePolicy) ?? .off
        loadModelOnLaunch = try container.decodeIfPresent(
            Bool.self,
            forKey: .loadModelOnLaunch) ?? false
        selectedConversationID = try container.decodeIfPresent(
            UUID.self,
            forKey: .selectedConversationID)
    }

    func isValid() -> Bool {
        AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
            && AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots)
            && temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
    }
}

/// Just enough of the file to route on, so a newer build's schema cannot throw
/// before its version has been read.
private struct VersionStamp: Decodable {
    let version: Int
}

enum MacAppSettingsFileStore {
    static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(MacAppSettings.fileName, isDirectory: false)
    }

    static func loadOrCreate(forModelDirectory modelDirectory: URL,
                             fileManager: FileManager = .default) -> MacAppSettings {
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                // The version is read on its own, before the full decode, because
                // nine keys decode with a hard `decode` and a newer build is
                // entitled to have moved any of them. Decoding first threw on
                // exactly the files this branch exists to protect, and the throw
                // reached the `catch` below, which deletes the file - so an older
                // build silently replaced a newer one's settings with its own
                // defaults. A version gate that cannot survive a schema change
                // gates nothing.
                let stamp = try JSONDecoder().decode(VersionStamp.self, from: data)
                // This build does not know what a later version means, so it runs
                // on defaults and leaves the file exactly as its owner wrote it.
                guard stamp.version <= MacAppSettings.currentVersion else {
                    return MacAppSettings()
                }
                var settings = try JSONDecoder().decode(MacAppSettings.self, from: data)
                let needsMigration = settings.version < MacAppSettings.currentVersion
                if needsMigration {
                    // Version only: every existing value is still a deliberate
                    // choice, and rewriting one here would silently discard it.
                    settings.version = MacAppSettings.currentVersion
                }
                guard settings.isValid() else { throw InvalidSettings() }
                if needsMigration {
                    try? save(settings, forModelDirectory: modelDirectory,
                              fileManager: fileManager)
                }
                return settings
            } catch {
                try? fileManager.removeItem(at: fileURL)
            }
        }

        let settings = MacAppSettings()
        try? save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
        return settings
    }

    static func save(_ settings: MacAppSettings,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        guard settings.isValid() else { throw InvalidSettings() }
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(settings)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    private struct InvalidSettings: Error {}
}
