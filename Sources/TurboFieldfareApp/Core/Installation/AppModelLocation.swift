import Foundation

enum AppModelLocation {
    static let modelDirectoryName = "model.gturbo"

    static func defaultURL(forRepoID repoID: String) throws -> URL {
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        return try resolve(
            repoID: repoID,
            explicitURL: nil,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:))
    }

    /// Default location for the curated model, for callers that cannot throw.
    ///
    /// The curated repository ID is a compile-time constant that always passes
    /// slug validation, so the throwing path is unreachable; the fallback
    /// exists only to keep this total. Replaced by the user's selection once a
    /// model has been picked.
    static func curatedDefaultURL() -> URL {
        let repoID = ModelCatalog.curated.first?.repoID ?? ""
        if let url = try? defaultURL(forRepoID: repoID) { return url }
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        return supportDirectory(applicationSupportURL: applicationSupport)
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(modelDirectoryName, isDirectory: true)
            .standardizedFileURL
    }

    /// The directory holding `catalog.json`, `conversations.json`, and the
    /// per-model `models/` tree.
    static func supportDirectory(applicationSupportURL: URL) -> URL {
        applicationSupportURL
            .appendingPathComponent("TurboFieldfare", isDirectory: true)
            .standardizedFileURL
    }

    static func resolve(repoID: String,
                        explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        fileExists: (String) -> Bool) throws -> URL {
        if let explicitURL {
            return absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
        }
        // Derived before any path is built, so a hostile repository ID cannot
        // reach `appendingPathComponent`.
        let slug = try ModelSlug.make(repoID: repoID)
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return modelURL(under: root.appendingPathComponent("scratch", isDirectory: true),
                            slug: slug)
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return modelURL(under: root.appendingPathComponent("scratch", isDirectory: true),
                            slug: slug)
        }
        return modelURL(under: supportDirectory(applicationSupportURL: applicationSupportURL),
                        slug: slug)
    }

    private static func modelURL(under base: URL, slug: String) -> URL {
        base.appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(slug, isDirectory: true)
            .appendingPathComponent(modelDirectoryName, isDirectory: true)
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
