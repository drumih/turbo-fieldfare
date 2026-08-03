import Foundation
import OSLog

/// Persistent library of extracted documents, keyed by filename. Lets the
/// same file be reused across conversations without re-extracting, and
/// backs the retrieval path for large documents.
@MainActor
@Observable
public final class DocumentLibrary {
    private static let logger = Logger(
        subsystem: "app.turbofieldfare",
        category: "document-library")

    public private(set) var documents: [DocumentAttachment] = []

    private let storageURL: URL
    private let fileManager: FileManager

    public init(storageURL: URL? = nil,
                fileManager: FileManager = .default) {
        let base = storageURL ?? Self.defaultStorageURL()
        self.storageURL = base
        self.fileManager = fileManager
        load()
    }

    public static func defaultStorageURL() -> URL {
        let folder = ConversationStore.defaultStorageURL().deletingLastPathComponent()
        return folder.appendingPathComponent("documents.json")
    }

    /// Adds or replaces the entry with the same filename.
    public func upsert(_ attachment: DocumentAttachment) {
        if let index = documents.firstIndex(where: { $0.filename == attachment.filename }) {
            documents[index] = attachment
        } else {
            documents.append(attachment)
        }
        documents.sort { $0.filename.localizedCaseInsensitiveCompare($1.filename) == .orderedAscending }
        save()
    }

    /// Removes an entry by id.
    public func remove(_ attachment: DocumentAttachment) {
        documents.removeAll { $0.id == attachment.id }
        save()
    }

    public func clear() {
        documents.removeAll()
        save()
    }

    /// Looks up a stored document by filename.
    public func document(named filename: String) -> DocumentAttachment? {
        documents.first { $0.filename == filename }
    }

    private func load() {
        guard let data = try? Data(contentsOf: storageURL), !data.isEmpty else { return }
        do {
            documents = try JSONDecoderValue.decode([DocumentAttachment].self, from: data)
        } catch {
            Self.logger.error("Failed to decode document library at \(self.storageURL.path): \(error)")
            documents = []
        }
    }

    private func save() {
        do {
            let folder = storageURL.deletingLastPathComponent()
            try fileManager.createDirectory(at: folder, withIntermediateDirectories: true)
            let data = try JSONEncoderValue.encode(documents)
            try data.write(to: storageURL, options: .atomic)
        } catch {
            Self.logger.error("Failed to save document library at \(self.storageURL.path): \(error)")
        }
    }
}

// JSON helpers with ISO8601 dates (documents carry no dates today, but the
// encoder/decoder pair is shared with the conversation store for symmetry).
private enum JSONDecoderValue {
    static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(T.self, from: data)
    }
}

private enum JSONEncoderValue {
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        return try encoder.encode(value)
    }
}
