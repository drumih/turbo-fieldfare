import Foundation
import TurboFieldfare
import TurboFieldfareRepackCore

struct MacAppSettings: Codable, Equatable, Sendable {
    static let fileName = "mac-app-settings.json"
    static let currentVersion = 4

    var version: Int = currentVersion
    var selectedModelPath: String?
    var migratedHistorySources: [String] = []
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
    var textSize: AppTextSize = .standard
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

    private enum CodingKeys: String, CodingKey {
        case version
        case selectedModelPath
        case migratedHistorySources
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
        case textSize
        case sidebarVisible
        case inspectorVisible
        case visionResidencyPolicy
        case rdadvisePolicy
        case loadModelOnLaunch
    }

    init(version: Int = currentVersion,
         selectedModelPath: String? = nil,
         migratedHistorySources: [String] = [],
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
         textSize: AppTextSize = .standard,
         sidebarVisible: Bool = true,
         inspectorVisible: Bool = true,
         visionResidencyPolicy: VisionResidencyPolicy = .onDemand,
         rdadvisePolicy: AppRDAdvicePolicy = .off,
         loadModelOnLaunch: Bool = false) {
        self.version = version
        self.selectedModelPath = selectedModelPath
        self.migratedHistorySources = migratedHistorySources
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
        self.textSize = textSize
        self.sidebarVisible = sidebarVisible
        self.inspectorVisible = inspectorVisible
        self.visionResidencyPolicy = visionResidencyPolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.loadModelOnLaunch = loadModelOnLaunch
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        selectedModelPath = try container.decodeIfPresent(String.self, forKey: .selectedModelPath)
        migratedHistorySources = try container.decodeIfPresent([String].self, forKey: .migratedHistorySources) ?? []
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
        // A damaged appearance preference must not reach settings-wide recovery,
        // which disables settings writes and requires explicit recovery.
        // Foundation can throw a non-DecodingError for a fractional integer.
        if container.contains(.textSize) {
            do {
                textSize = try container.decode(AppTextSize.self, forKey: .textSize)
            } catch {
                FileHandle.standardError.write(Data(
                    "Invalid Mac app setting textSize: unsupported value or type; using 100%.\n".utf8))
                textSize = .standard
            }
        }
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
    }

    /// Schema routing happens before decoding; value validation never replaces a damaged file.
    func isValid() -> Bool {
        (selectedModelPath.map { $0.hasPrefix("/") && !$0.contains("\0") } ?? true)
            && AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
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

    static func load(at fileURL: URL) throws -> MacAppSettings? {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        let data = try Data(contentsOf: fileURL)
        let stamp = try JSONDecoder().decode(VersionStamp.self, from: data)
        guard stamp.version <= MacAppSettings.currentVersion else {
            throw SettingsError.unsupportedVersion(stamp.version)
        }
        var settings = try JSONDecoder().decode(MacAppSettings.self, from: data)
        guard settings.isValid() else { throw SettingsError.invalidValues }
        settings.version = MacAppSettings.currentVersion
        if let path = settings.selectedModelPath {
            settings.selectedModelPath = URL(fileURLWithPath: path).standardizedFileURL.path
        }
        return settings
    }

    static func loadOrCreate(forModelDirectory modelDirectory: URL,
                             fileManager: FileManager = .default) -> MacAppSettings {
        do {
            if let settings = try load(at: fileURL(forModelDirectory: modelDirectory)) {
                let original = try JSONDecoder().decode(VersionStamp.self, from: Data(contentsOf: fileURL(forModelDirectory: modelDirectory)))
                if original.version < MacAppSettings.currentVersion {
                    try save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
                }
                return settings
            }
            let settings = MacAppSettings()
            try save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
            return settings
        } catch {
            FileHandle.standardError.write(Data("Reading Mac app settings failed: \(error)\n".utf8))
            return MacAppSettings()
        }
    }

    enum SettingsError: Error, CustomStringConvertible {
        case unsupportedVersion(Int)
        case invalidValues
        var description: String {
            switch self {
            case .unsupportedVersion(let version): "Settings version \(version) requires a newer app. The original file was preserved."
            case .invalidValues: "Settings contain invalid values. The original file was preserved."
            }
        }
    }

    static func save(_ settings: MacAppSettings,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        guard settings.isValid() else { throw SettingsError.invalidValues }
        try save(settings, at: fileURL(forModelDirectory: modelDirectory), fileManager: fileManager)
    }

    static func save(_ settings: MacAppSettings, at fileURL: URL,
                     fileManager: FileManager = .default) throws {
        guard settings.isValid() else { throw SettingsError.invalidValues }
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(settings)
        data.append(0x0A)
        try Posix.atomicWrite(data, to: fileURL.path,
                              durableIn: fileURL.deletingLastPathComponent().path,
                              uniqueTemporary: true)
    }

}
