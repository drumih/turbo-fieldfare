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

    /// Where the single-model builds installed weights, before per-model
    /// directories existed.
    static let legacyDirectoryName = "gemma4.gturbo"

    static func legacyInstallURL(executableURL: URL?,
                                 currentDirectoryURL: URL,
                                 applicationSupportURL: URL,
                                 fileExists: (String) -> Bool) -> URL {
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return root.appendingPathComponent("scratch", isDirectory: true)
                .appendingPathComponent(legacyDirectoryName, isDirectory: true)
                .standardizedFileURL
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return root.appendingPathComponent("scratch", isDirectory: true)
                .appendingPathComponent(legacyDirectoryName, isDirectory: true)
                .standardizedFileURL
        }
        return supportDirectory(applicationSupportURL: applicationSupportURL)
            .appendingPathComponent(legacyDirectoryName, isDirectory: true)
            .standardizedFileURL
    }

    /// Adopts an install made by a pre-multi-model build.
    ///
    /// A rename, not a copy: both paths sit on the same volume, so this is
    /// instant even for a 14 GB model. The alternative — leaving the old
    /// directory stranded — would silently ask the user to re-download weights
    /// they already have.
    ///
    /// Returns true only when weights were actually moved.
    @discardableResult
    static func migrateLegacyInstall(legacy: URL,
                                     destination: URL,
                                     fileManager: FileManager = .default) -> Bool {
        var legacyIsDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: legacy.path, isDirectory: &legacyIsDirectory),
              legacyIsDirectory.boolValue else { return false }
        // Never clobber a real install; leave the legacy copy for the user to
        // remove deliberately.
        guard !fileManager.fileExists(atPath: destination.path) else { return false }
        do {
            try fileManager.createDirectory(at: destination.deletingLastPathComponent(),
                                            withIntermediateDirectories: true)
            try fileManager.moveItem(at: legacy, to: destination)
            rewriteInstallReceiptPath(inModelDirectory: destination, fileManager: fileManager)
            return true
        } catch {
            return false
        }
    }

    /// Repoints `verified-install.json` at the directory it now lives in.
    ///
    /// The receipt records an absolute `modelDirectoryPath` and
    /// `VerifiedInstallReceipt.validate` rejects a mismatch, so a move that
    /// skipped this would leave weights that pass an on-disk verify but throw
    /// `trustedReceiptInvalid` at load. Only that one field is touched; the
    /// per-file hashes still describe the same bytes.
    private static func rewriteInstallReceiptPath(inModelDirectory directory: URL,
                                                  fileManager: FileManager) {
        let receiptURL = directory
            .appendingPathComponent("verified-install.json", isDirectory: false)
        guard fileManager.fileExists(atPath: receiptURL.path),
              let data = try? Data(contentsOf: receiptURL),
              var receipt = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return }
        receipt["modelDirectoryPath"] = directory.standardizedFileURL.path
        guard let updated = try? JSONSerialization.data(
            withJSONObject: receipt,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]) else { return }
        try? updated.write(to: receiptURL, options: .atomic)
    }

    /// Migrates the curated model's legacy install, if there is one.
    @discardableResult
    static func migrateLegacyInstallIfNeeded(repoID: String) -> Bool {
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        let currentDirectory = URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                   isDirectory: true)
        guard let destination = try? resolve(
            repoID: repoID,
            explicitURL: nil,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: currentDirectory,
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:)) else { return false }
        let legacy = legacyInstallURL(
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: currentDirectory,
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:))
        return migrateLegacyInstall(legacy: legacy, destination: destination)
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
