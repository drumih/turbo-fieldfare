import Foundation

/// A past conversation between the user and the model.
public struct Conversation: Identifiable, Codable, Equatable, Sendable {
    public let id: UUID
    public var title: String
    public var prompt: String
    public var response: String
    public var createdAt: Date
    public var updatedAt: Date
    public var promptTokenCount: Int?
    public var generatedTokenCount: Int?
    public var stopReason: String?

    public init(id: UUID = UUID(),
                title: String,
                prompt: String,
                response: String,
                createdAt: Date = Date(),
                updatedAt: Date = Date(),
                promptTokenCount: Int? = nil,
                generatedTokenCount: Int? = nil,
                stopReason: String? = nil) {
        self.id = id
        self.title = title
        self.prompt = prompt
        self.response = response
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.stopReason = stopReason
    }

    /// Automatic title from the prompt when none is provided.
    public static func title(from prompt: String, maxLength: Int = 60) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "New conversation" }
        if trimmed.count <= maxLength { return trimmed }
        let end = trimmed.index(trimmed.startIndex, offsetBy: maxLength)
        return String(trimmed[..<end]) + "…"
    }

    /// Short response preview for list display.
    public var responsePreview: String {
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count <= 100 { return trimmed }
        let end = trimmed.index(trimmed.startIndex, offsetBy: 100)
        return String(trimmed[..<end]) + "…"
    }
}

/// Persistent conversation store, saved as JSON.
@MainActor
@Observable
public final class ConversationStore {
    public private(set) var conversations: [Conversation] = []

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
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory,
                                                  in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory())
        let folder = appSupport.appendingPathComponent("TurboFieldfare", isDirectory: true)
        return folder.appendingPathComponent("conversations.json")
    }

    /// Adds or updates a conversation.
    public func upsert(_ conversation: Conversation) {
        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            conversations[index] = conversation
        } else {
            conversations.insert(conversation, at: 0)
        }
        // Keep most recently updated first
        conversations.sort { $0.updatedAt > $1.updatedAt }
        save()
    }

    /// Deletes a conversation.
    public func delete(_ conversation: Conversation) {
        conversations.removeAll { $0.id == conversation.id }
        save()
    }

    /// Deletes all conversations.
    public func clearAll() {
        conversations.removeAll()
        save()
    }

    /// Reloads from disk.
    public func reload() {
        load()
    }

    private func load() {
        guard let data = try? Data(contentsOf: storageURL), !data.isEmpty else { return }
        do {
            conversations = try JSONDecoderValue.decode([Conversation].self, from: data)
            conversations.sort { $0.updatedAt > $1.updatedAt }
        } catch {
            conversations = []
        }
    }

    private func save() {
        do {
            let folder = storageURL.deletingLastPathComponent()
            try fileManager.createDirectory(at: folder, withIntermediateDirectories: true)
            let data = try JSONEncoderValue.encode(conversations)
            try data.write(to: storageURL, options: .atomic)
        } catch {
            // Silent: history is not critical
        }
    }
}

// JSON helpers with ISO8601 dates
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