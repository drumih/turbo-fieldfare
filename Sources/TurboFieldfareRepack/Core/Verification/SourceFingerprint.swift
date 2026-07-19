import Foundation

/// Snapshot fingerprints pinned by the project. Adding a new entry means the
/// importer has been validated against a fresh upload of the source.
enum SourceFingerprint {
    static let knownFingerprints: [String: String] = [
        "majentik/gemma-4-26B-A4B-TurboQuant-MLX-4bit":
            "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80"
    ]

    /// Returns the recognised model ID for a given index.json SHA-256, or nil.
    static func modelID(forIndexSha256 sha256Hex: String) -> String? {
        for (id, sha) in knownFingerprints where sha == sha256Hex { return id }
        return nil
    }
}
