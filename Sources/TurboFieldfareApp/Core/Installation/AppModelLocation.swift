import Foundation

public enum AppModelLocation {
    public static let storageKey = "TurboFieldfare.modelDirectory"

    public static func defaultURL() -> URL {
        defaultURL(userDefaults: .standard)
    }

    public static func remember(_ url: URL) {
        remember(url, userDefaults: .standard)
    }

    static func defaultURL(userDefaults: UserDefaults) -> URL {
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        return resolve(
            explicitURL: preferredURL(userDefaults: userDefaults),
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:))
    }

    static func remember(_ url: URL, userDefaults: UserDefaults) {
        userDefaults.set(url.standardizedFileURL.path, forKey: storageKey)
    }

    private static func preferredURL(userDefaults: UserDefaults) -> URL? {
        guard let path = userDefaults.string(forKey: storageKey),
              !path.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static func resolve(explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        fileExists: (String) -> Bool) -> URL {
        if let explicitURL {
            return absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
        }
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return root.appendingPathComponent("scratch/gemma4.gturbo", isDirectory: true)
                .standardizedFileURL
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return root.appendingPathComponent("scratch/gemma4.gturbo", isDirectory: true)
                .standardizedFileURL
        }
        return applicationSupportURL
            .appendingPathComponent("TurboFieldfare", isDirectory: true)
            .appendingPathComponent("gemma4.gturbo", isDirectory: true)
            .standardizedFileURL
    }

    private static func absoluteURL(_ url: URL, relativeTo base: URL) -> URL {
        if url.path.hasPrefix("/") {
            return url.standardizedFileURL
        }
        return base.appendingPathComponent(url.path, isDirectory: true).standardizedFileURL
    }

    private static func packageRoot(startingAt start: URL,
                                    fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            let package = candidate.appendingPathComponent("Package.swift").path
            let appSources = candidate.appendingPathComponent(
                "Sources/TurboFieldfareApp/Mac", isDirectory: true).path
            if fileExists(package), fileExists(appSources) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
