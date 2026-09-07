import Foundation

/// Where conversations live: beside the settings file and the model, one
/// directory per conversation.
///
/// The same derivation as `mac-app-settings.json`, deliberately. One location
/// rule is the whole point: an installed app puts both under Application
/// Support, a dev checkout puts both under `scratch/`, and moving the model with
/// `TURBO_FIELDFARE_MODEL_PATH` moves both together.
public enum ConversationStoreLocation {
    public static let directoryName = "conversations"

    public static func url(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(directoryName, isDirectory: true)
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
        guard fileURL.isFileURL else { return false }
        let resolved = fileURL.standardizedFileURL.resolvingSymlinksInPath()
        let root = url(forModelDirectory: modelDirectory)
            .standardizedFileURL.resolvingSymlinksInPath()
        let fileComponents = resolved.pathComponents
        let rootComponents = root.pathComponents
        guard fileComponents.count > rootComponents.count else { return false }
        return Array(fileComponents.prefix(rootComponents.count)) == rootComponents
    }
}
