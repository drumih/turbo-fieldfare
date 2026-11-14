import Foundation

public enum AppDataLocation {
    public static func resolve(explicitURL: URL? = nil,
                               environment: [String: String] = ProcessInfo.processInfo.environment,
                               applicationSupportURL: URL? = nil) -> URL {
        if let explicitURL { return explicitURL.standardizedFileURL.resolvingSymlinksInPath() }
        if let path = environment["TURBO_FIELDFARE_APP_DATA_PATH"], !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
        }
        let support = applicationSupportURL ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support", isDirectory: true)
        return support.appendingPathComponent("TurboFieldfare", isDirectory: true).standardizedFileURL
    }

    static func settingsURL(in root: URL) -> URL {
        root.appendingPathComponent(MacAppSettings.fileName)
    }
}
