import Foundation
import OSLog

/// One turn of a multi-turn conversation.
public struct Turn: Identifiable, Codable, Equatable, Sendable {
    public enum Role: String, Codable, Sendable {
        case user
        case model
    }

    public let id: UUID
    public var role: Role
    public var text: String
    public var createdAt: Date
    /// Token count attributed to this turn, if known.
    public var tokenCount: Int?

    public init(id: UUID = UUID(),
                role: Role,
                text: String,
                createdAt: Date = Date(),
                tokenCount: Int? = nil) {
        self.id = id
        self.role = role
        self.text = text
        self.createdAt = createdAt
        self.tokenCount = tokenCount
    }
}

/// A past conversation between the user and the model.
public struct Conversation: Identifiable, Codable, Equatable, Sendable {
    public let id: UUID
    public var title: String
    /// Ordered turns of the conversation. Non-empty for loaded history.
    public var turns: [Turn]
    public var createdAt: Date
    public var updatedAt: Date
    /// Prompt token count of the last generated turn.
    public var promptTokenCount: Int?
    /// Generated token count of the last generated turn.
    public var generatedTokenCount: Int?
    /// Stop reason of the last generated turn.
    public var stopReason: String?
    public var attachments: [DocumentAttachment]
    public var isPinned: Bool
    /// Per-conversation max response length; nil means "use the global default".
    public var maxNewTokens: Int?
    /// Id of the conversation this one was forked from, if any.
    public var parentConversationID: UUID?
    /// Whether this conversation is a reusable template.
    public var isTemplate: Bool
    /// User-defined tags used for filtering (e.g. "#work").
    public var tags: [String]

    /// The first user turn, or an empty string when there is none.
    public var firstPrompt: String {
        turns.first { $0.role == .user }?.text ?? ""
    }

    /// The last model turn, or an empty string when there is none.
    public var lastResponse: String {
        turns.last { $0.role == .model }?.text ?? ""
    }

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
                isPinned: Bool = false,
                maxNewTokens: Int? = nil,
                parentConversationID: UUID? = nil,
                isTemplate: Bool = false,
                tags: [String] = []) {
        self.id = id
        self.title = title
        self.turns = [
            Turn(role: .user, text: prompt, createdAt: createdAt,
                 tokenCount: promptTokenCount),
            Turn(role: .model, text: response, createdAt: updatedAt,
                 tokenCount: generatedTokenCount),
        ].filter { !$0.text.isEmpty }
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.stopReason = stopReason
        self.attachments = attachments
        self.isPinned = isPinned
        self.maxNewTokens = maxNewTokens
        self.parentConversationID = parentConversationID
        self.isTemplate = isTemplate
        self.tags = tags
    }

    /// Creates a conversation with an explicit turn list (for multi-turn history).
    public init(id: UUID = UUID(),
                title: String,
                turns: [Turn],
                createdAt: Date = Date(),
                updatedAt: Date = Date(),
                promptTokenCount: Int? = nil,
                generatedTokenCount: Int? = nil,
                stopReason: String? = nil,
                attachments: [DocumentAttachment] = [],
                isPinned: Bool = false,
                maxNewTokens: Int? = nil,
                parentConversationID: UUID? = nil,
                isTemplate: Bool = false,
                tags: [String] = []) {
        self.id = id
        self.title = title
        self.turns = turns
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.stopReason = stopReason
        self.attachments = attachments
        self.isPinned = isPinned
        self.maxNewTokens = maxNewTokens
        self.parentConversationID = parentConversationID
        self.isTemplate = isTemplate
        self.tags = tags
    }

    /// Returns a copy with the given mutable fields replaced.
    public func replacing(title: String? = nil,
                          turns: [Turn]? = nil,
                          isPinned: Bool? = nil,
                          isTemplate: Bool? = nil,
                          tags: [String]? = nil) -> Conversation {
        Conversation(id: id,
                     title: title ?? self.title,
                     turns: turns ?? self.turns,
                     createdAt: createdAt,
                     updatedAt: updatedAt,
                     promptTokenCount: promptTokenCount,
                     generatedTokenCount: generatedTokenCount,
                     stopReason: stopReason,
                     attachments: attachments,
                     isPinned: isPinned ?? self.isPinned,
                     maxNewTokens: maxNewTokens,
                     parentConversationID: parentConversationID,
                     isTemplate: isTemplate ?? self.isTemplate,
                     tags: tags ?? self.tags)
    }

    /// Formats the full multi-turn history as a single prompt for generation.
    /// Each turn is labelled; the model is expected to continue as model.
    public func asPrompt() -> String {
        turns.map { turn in
            switch turn.role {
            case .user: return "User:\n\(turn.text)"
            case .model: return "Model:\n\(turn.text)"
            }
        }.joined(separator: "\n\n") + "\n\nModel:"
    }

    /// Human-readable transcript for display and copy operations.
    public var displayTranscript: String {
        turns.map { turn in
            switch turn.role {
            case .user: return "You:\n\(turn.text)"
            case .model: return "Answer:\n\(turn.text)"
            }
        }.joined(separator: "\n\n")
    }

    /// Short response preview for list display (last model turn).
    public var responsePreview: String {
        let trimmed = lastResponse.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count <= 100 { return trimmed }
        let end = trimmed.index(trimmed.startIndex, offsetBy: 100)
        return String(trimmed[..<end]) + "…"
    }

    /// Total character count across all turns (for context estimates).
    public var characterCount: Int {
        turns.reduce(0) { $0 + $1.text.count }
    }

    private enum CodingKeys: String, CodingKey {
        case id, title, createdAt, updatedAt
        case promptTokenCount, generatedTokenCount, stopReason
        case attachments, isPinned, maxNewTokens, turns
        case parentConversationID, isTemplate, tags
        // Legacy single-exchange keys, kept for backward decoding.
        case prompt, response
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(UUID.self, forKey: .id)
        self.title = try container.decode(String.self, forKey: .title)
        self.createdAt = try container.decode(Date.self, forKey: .createdAt)
        self.updatedAt = try container.decode(Date.self, forKey: .updatedAt)
        self.promptTokenCount = try container.decodeIfPresent(Int.self, forKey: .promptTokenCount)
        self.generatedTokenCount = try container.decodeIfPresent(Int.self, forKey: .generatedTokenCount)
        self.stopReason = try container.decodeIfPresent(String.self, forKey: .stopReason)
        self.attachments = try container.decodeIfPresent([DocumentAttachment].self, forKey: .attachments) ?? []
        self.isPinned = try container.decodeIfPresent(Bool.self, forKey: .isPinned) ?? false
        self.maxNewTokens = try container.decodeIfPresent(Int.self, forKey: .maxNewTokens)
        self.parentConversationID = try container.decodeIfPresent(UUID.self, forKey: .parentConversationID)
        self.isTemplate = try container.decodeIfPresent(Bool.self, forKey: .isTemplate) ?? false
        self.tags = try container.decodeIfPresent([String].self, forKey: .tags) ?? []
        // Newer files carry turns; legacy files used prompt/response fields.
        if let turns = try container.decodeIfPresent([Turn].self, forKey: .turns),
           !turns.isEmpty {
            self.turns = turns
        } else {
            let prompt = try container.decode(String.self, forKey: .prompt)
            let response = try container.decode(String.self, forKey: .response)
            self.turns = [
                Turn(role: .user, text: prompt, createdAt: createdAt),
                Turn(role: .model, text: response, createdAt: updatedAt),
            ].filter { !$0.text.isEmpty }
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(title, forKey: .title)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(updatedAt, forKey: .updatedAt)
        try container.encodeIfPresent(promptTokenCount, forKey: .promptTokenCount)
        try container.encodeIfPresent(generatedTokenCount, forKey: .generatedTokenCount)
        try container.encodeIfPresent(stopReason, forKey: .stopReason)
        try container.encode(attachments, forKey: .attachments)
        try container.encode(isPinned, forKey: .isPinned)
        try container.encodeIfPresent(maxNewTokens, forKey: .maxNewTokens)
        try container.encodeIfPresent(parentConversationID, forKey: .parentConversationID)
        try container.encode(isTemplate, forKey: .isTemplate)
        try container.encode(tags, forKey: .tags)
        try container.encode(turns, forKey: .turns)
    }

    /// Placeholder title from the first user turn when no title exists.
    public static func title(from prompt: String, maxLength: Int = 60) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "New conversation" }
        if trimmed.count <= maxLength { return trimmed }
        let end = trimmed.index(trimmed.startIndex, offsetBy: maxLength)
        return String(trimmed[..<end]) + "…"
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

    /// Set to true when the stored history file was unreadable and had to be
    /// reset. The existing file is preserved with a `.corrupted` suffix, so no
    /// data is lost; the flag tells the UI a re-review may be worthwhile.
    public private(set) var didLoadFromCorrupted: Bool = false

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
            didLoadFromCorrupted = true
            // Preserve the unreadable file for inspection, then start fresh.
            let backupURL = storageURL.appendingPathExtension("corrupted")
            try? fileManager.moveItem(at: storageURL, to: backupURL)
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
