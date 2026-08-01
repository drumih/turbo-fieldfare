import Foundation

/// Persists user-added catalog entries.
///
/// A corrupt file is renamed rather than deleted: unlike chat history, the
/// catalog can hold entries the user typed by hand and would want back, and a
/// silent delete would look like the app forgot them.
public enum ModelCatalogStore {
    public static let corruptFileName = "catalog.corrupt.json"

    public static func fileURL(inSupportDirectory supportDirectory: URL) -> URL {
        supportDirectory.standardizedFileURL
            .appendingPathComponent(ModelCatalogFile.fileName, isDirectory: false)
    }

    public static func load(inSupportDirectory supportDirectory: URL,
                            fileManager: FileManager = .default) -> ModelCatalogFile {
        let url = fileURL(inSupportDirectory: supportDirectory)
        guard fileManager.fileExists(atPath: url.path) else {
            return ModelCatalogFile()
        }
        do {
            let file = try JSONDecoder().decode(
                ModelCatalogFile.self,
                from: try Data(contentsOf: url))
            guard file.isValid() else { throw InvalidCatalog() }
            return file
        } catch {
            quarantine(url, in: supportDirectory, fileManager: fileManager)
            return ModelCatalogFile()
        }
    }

    public static func save(_ file: ModelCatalogFile,
                            inSupportDirectory supportDirectory: URL,
                            fileManager: FileManager = .default) throws {
        guard file.isValid() else { throw InvalidCatalog() }
        let url = fileURL(inSupportDirectory: supportDirectory)
        try fileManager.createDirectory(at: supportDirectory,
                                        withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(file)
        data.append(0x0A)
        try data.write(to: url, options: .atomic)
    }

    private static func quarantine(_ url: URL,
                                   in supportDirectory: URL,
                                   fileManager: FileManager) {
        let backup = supportDirectory
            .appendingPathComponent(corruptFileName, isDirectory: false)
        try? fileManager.removeItem(at: backup)
        try? fileManager.moveItem(at: url, to: backup)
        try? fileManager.removeItem(at: url)
    }

    struct InvalidCatalog: Error, Sendable {}
}
