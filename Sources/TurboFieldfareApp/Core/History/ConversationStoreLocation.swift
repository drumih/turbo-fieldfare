import Foundation

public enum ConversationStoreLocation {
    public static let directoryName = "conversations"

    public static func url(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(directoryName, isDirectory: true)
    }

    public static func url(forAppDataDirectory root: URL) -> URL {
        root.standardizedFileURL.appendingPathComponent(directoryName, isDirectory: true)
    }

    public static func contains(_ fileURL: URL, forAppDataDirectory appDataDirectory: URL) -> Bool {
        contains(fileURL, root: url(forAppDataDirectory: appDataDirectory))
    }

    /// Whether `fileURL` names something inside the store for this model.
    ///
    /// The decode service is handed image paths over a socket and opens
    /// whatever they name, so this is a trust boundary, not a tidiness check:
    /// without it a peer could use the service to read any file the user can.
    /// Same shape as `AppImageAttachmentStore.contains` — resolve, then require
    /// a strict path-component prefix, so `.../conversations-elsewhere` cannot
    /// pass as `.../conversations`.
    public static func contains(_ fileURL: URL, forModelDirectory modelDirectory: URL) -> Bool {
        contains(fileURL, root: url(forModelDirectory: modelDirectory))
    }

    private static func contains(_ fileURL: URL, root: URL) -> Bool {
        guard fileURL.isFileURL else { return false }
        let resolved = fileURL.standardizedFileURL.resolvingSymlinksInPath()
        let root = root.standardizedFileURL.resolvingSymlinksInPath()
        let fileComponents = resolved.pathComponents
        let rootComponents = root.pathComponents
        guard fileComponents.count > rootComponents.count else { return false }
        return Array(fileComponents.prefix(rootComponents.count)) == rootComponents
    }
}
