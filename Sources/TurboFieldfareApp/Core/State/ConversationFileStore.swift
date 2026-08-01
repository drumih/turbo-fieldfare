import Foundation

/// Persists every conversation in a single file, using the same discipline as
/// `MacAppSettingsFileStore`: atomic, and discarded outright when it fails to
/// decode.
///
/// One file rather than one per conversation: writes happen only at commit
/// points (a few per generation), so a whole-file rewrite is cheap, and a
/// single atomic write keeps the store internally consistent without a manifest.
///
/// The live store is the *global* one in the support directory. The per-model
/// API survives only so `migrate` can read stores written beside a model
/// directory back when each model kept its own history: once models can be
/// installed and deleted individually, history stored beside a model would
/// disappear with it and would not follow the user across a switch.
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

    public static let migratedFileName = "conversations.migrated.json"

    public static func globalFileURL(inSupportDirectory supportDirectory: URL) -> URL {
        supportDirectory.standardizedFileURL
            .appendingPathComponent(ConversationStoreFile.fileName, isDirectory: false)
    }

    public static func loadGlobal(inSupportDirectory supportDirectory: URL,
                                  fileManager: FileManager = .default) -> ConversationStoreFile {
        let url = globalFileURL(inSupportDirectory: supportDirectory)
        guard fileManager.fileExists(atPath: url.path) else {
            return ConversationStoreFile()
        }
        do {
            let store = try JSONDecoder().decode(
                ConversationStoreFile.self,
                from: try Data(contentsOf: url))
            guard store.isValid() else { throw InvalidStore() }
            return store
        } catch {
            try? fileManager.removeItem(at: url)
            return ConversationStoreFile()
        }
    }

    public static func saveGlobal(_ store: ConversationStoreFile,
                                  inSupportDirectory supportDirectory: URL,
                                  fileManager: FileManager = .default) throws {
        guard store.isValid() else { throw InvalidStore() }
        let url = globalFileURL(inSupportDirectory: supportDirectory)
        try fileManager.createDirectory(at: supportDirectory,
                                        withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(store)
        data.append(0x0A)
        try data.write(to: url, options: .atomic)
    }

    /// Folds per-model stores into the global one, tagging every imported turn.
    ///
    /// Renaming the source file is what makes this idempotent: a second run
    /// finds nothing to read. Renaming rather than deleting keeps the original
    /// recoverable if the merge is ever wrong.
    @discardableResult
    public static func migrate(fromModelDirectories directories: [URL],
                               modelIDsByDirectory: [URL: String],
                               inSupportDirectory supportDirectory: URL,
                               fileManager: FileManager = .default) throws -> Int {
        var global = loadGlobal(inSupportDirectory: supportDirectory,
                                fileManager: fileManager)
        var imported = 0
        for directory in directories {
            let legacyURL = fileURL(forModelDirectory: directory)
            guard fileManager.fileExists(atPath: legacyURL.path) else { continue }
            let legacy = load(forModelDirectory: directory, fileManager: fileManager)
            let modelID = modelIDsByDirectory[directory]
            for var conversation in legacy.conversations where !conversation.isEmpty {
                for index in conversation.turns.indices
                where conversation.turns[index].modelID == nil {
                    conversation.turns[index].modelID = modelID
                }
                global.conversations.append(conversation)
                imported += 1
            }
            let renamed = legacyURL.deletingLastPathComponent()
                .appendingPathComponent(migratedFileName, isDirectory: false)
            try? fileManager.removeItem(at: renamed)
            try fileManager.moveItem(at: legacyURL, to: renamed)
        }
        if imported > 0 {
            try saveGlobal(global, inSupportDirectory: supportDirectory,
                           fileManager: fileManager)
        }
        return imported
    }

    struct InvalidStore: Error {}
}
