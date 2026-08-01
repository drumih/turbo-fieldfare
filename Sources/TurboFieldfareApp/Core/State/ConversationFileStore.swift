import Foundation

/// Persists every conversation for one model directory in a single file, using
/// the same discipline as `MacAppSettingsFileStore`: written beside the model
/// directory, atomic, and discarded outright when it fails to decode.
///
/// One file rather than one per conversation: writes happen only at commit
/// points (a few per generation), so a whole-file rewrite is cheap, and a
/// single atomic write keeps the store internally consistent without a manifest.
public enum ConversationFileStore {
    /// ISO-8601 *with* fractional seconds. The plain `.iso8601` strategy
    /// truncates to whole seconds, which would let two conversations touched in
    /// the same second reorder in the sidebar across a reload.
    public static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(ConversationStoreFile.fileName, isDirectory: false)
    }

    public static func load(forModelDirectory modelDirectory: URL,
                            fileManager: FileManager = .default) -> ConversationStoreFile {
        let url = fileURL(forModelDirectory: modelDirectory)
        guard fileManager.fileExists(atPath: url.path) else {
            return ConversationStoreFile()
        }
        do {
            // Dates use the default numeric strategy on purpose:
            // ISO8601DateFormatter round-trips with ~1e-7 s of float error, so a
            // reloaded conversation would compare unequal to the identical one
            // in memory. Sorting the sidebar depends on that comparison holding.
            let decoder = JSONDecoder()
            let store = try decoder.decode(
                ConversationStoreFile.self,
                from: try Data(contentsOf: url))
            guard store.isValid() else { throw InvalidStore() }
            return store
        } catch {
            // A corrupt or future-version store is discarded rather than
            // partially migrated; chat history is recreatable, a wedged app is
            // not recoverable by the user.
            try? fileManager.removeItem(at: url)
            return ConversationStoreFile()
        }
    }

    public static func save(_ store: ConversationStoreFile,
                            forModelDirectory modelDirectory: URL,
                            fileManager: FileManager = .default) throws {
        guard store.isValid() else { throw InvalidStore() }
        let url = fileURL(forModelDirectory: modelDirectory)
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(store)
        data.append(0x0A)
        try data.write(to: url, options: .atomic)
    }

    struct InvalidStore: Error {}
}
