import Foundation

/// Snapshot fingerprints pinned by the project. Adding a new entry means the
/// importer has been validated against a fresh upload of the source.
public enum SourceFingerprint {
    public static let knownFingerprints: [String: String] = [
        SupportedModelSource.repoID: SupportedModelSource.sourceIndexSHA256,
    ]

    /// Returns the recognised model ID for a given index.json SHA-256, or nil.
    public static func modelID(forIndexSha256 sha256Hex: String) -> String? {
        for (id, sha) in knownFingerprints where sha == sha256Hex { return id }
        return nil
    }
}
