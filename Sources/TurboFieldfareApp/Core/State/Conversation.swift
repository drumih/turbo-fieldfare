import Foundation
import OSLog

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
    public var attachments: [DocumentAttachment]
    public var isPinned: Bool

    public init(id: UUID = UUID(),
                title: String,
                prompt: String,
                response: String,
                createdAt: Date = Date(),
                updatedAt: Date = Date(),
                promptTokenCount: Int? = nil,
                generatedTokenCount: Int? = nil,
                stopReason: String? = nil,
                attachments: [DocumentAttachment] = [],
                isPinned: Bool = false) {
        self.id = id
        self.title = title
        self.prompt = prompt
        self.response = response
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.stopReason = stopReason
        self.attachments = attachments
        self.isPinned = isPinned
    }

    /// Returns a copy with the given mutable fields replaced.
    public func replacing(title: String? = nil,
                          response: String? = nil,
                          attachments: [DocumentAttachment]? = nil,
                          isPinned: Bool? = nil) -> Conversation {
        Conversation(id: id,
                     title: title ?? self.title,
                     prompt: prompt,
                     response: response ?? self.response,
                     createdAt: createdAt,
                     updatedAt: updatedAt,
                     promptTokenCount: promptTokenCount,
                     generatedTokenCount: generatedTokenCount,
                     stopReason: stopReason,
                     attachments: attachments ?? self.attachments,
                     isPinned: isPinned ?? self.isPinned)
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

    private enum CodingKeys: String, CodingKey {
        case id, title, prompt, response, createdAt, updatedAt
        case promptTokenCount, generatedTokenCount, stopReason
        case attachments, isPinned
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(UUID.self, forKey: .id)
        self.title = try container.decode(String.self, forKey: .title)
        self.prompt = try container.decode(String.self, forKey: .prompt)
        self.response = try container.decode(String.self, forKey: .response)
        self.createdAt = try container.decode(Date.self, forKey: .createdAt)
        self.updatedAt = try container.decode(Date.self, forKey: .updatedAt)
        self.promptTokenCount = try container.decodeIfPresent(Int.self, forKey: .promptTokenCount)
        self.generatedTokenCount = try container.decodeIfPresent(Int.self, forKey: .generatedTokenCount)
        self.stopReason = try container.decodeIfPresent(String.self, forKey: .stopReason)
        // Older history files predate document attachments and pinning.
        self.attachments = try container.decodeIfPresent([DocumentAttachment].self, forKey: .attachments) ?? []
        self.isPinned = try container.decodeIfPresent(Bool.self, forKey: .isPinned) ?? false
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(title, forKey: .title)
        try container.encode(prompt, forKey: .prompt)
        try container.encode(response, forKey: .response)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(updatedAt, forKey: .updatedAt)
        try container.encodeIfPresent(promptTokenCount, forKey: .promptTokenCount)
        try container.encodeIfPresent(generatedTokenCount, forKey: .generatedTokenCount)
        try container.encodeIfPresent(stopReason, forKey: .stopReason)
        try container.encode(attachments, forKey: .attachments)
        try container.encode(isPinned, forKey: .isPinned)
    }
}

/// Persistent conversation store, saved as JSON.
@MainActor
@Observable
public final class ConversationStore {
    private static let logger = Logger(
        subsystem: "app.turbofieldfare",
        category: "conversation-store")

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
        sortConversations()
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

    /// Exports all conversations in the same JSON format the store uses.
    public func exportJSON() throws -> Data {
        try JSONEncoderValue.encode(conversations)
    }

    /// Imports conversations from exported JSON. Existing entries with the
    /// same id are replaced; new ones are inserted. Returns the count.
    public func importJSON(_ data: Data) throws -> Int {
        let imported = try JSONDecoderValue.decode([Conversation].self, from: data)
        for conversation in imported {
            if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
                conversations[index] = conversation
            } else {
                conversations.append(conversation)
            }
        }
        sortConversations()
        save()
        return imported.count
    }

    /// Pinned conversations first, then most recently updated.
    private func sortConversations() {
        conversations.sort {
            if $0.isPinned != $1.isPinned { return $0.isPinned }
            return $0.updatedAt > $1.updatedAt
        }
    }

    private func load() {
        guard let data = try? Data(contentsOf: storageURL), !data.isEmpty else { return }
        do {
            conversations = try JSONDecoderValue.decode([Conversation].self, from: data)
            sortConversations()
        } catch {
            Self.logger.error("Failed to decode conversation history at \(self.storageURL.path): \(error)")
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
            Self.logger.error("Failed to save conversation history at \(self.storageURL.path): \(error)")
        }
    }
}

// JSON helpers with ISO8601 dates. Decoding accepts both whole-second and
// fractional-second timestamps so files written by other tools still load.
private enum JSONDecoderValue {
    static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let string = try container.decode(String.self)
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let date = formatter.date(from: string) {
                return date
            }
            formatter.formatOptions = [.withInternetDateTime]
            if let date = formatter.date(from: string) {
                return date
            }
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Invalid ISO8601 date: \(string)")
        }
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
