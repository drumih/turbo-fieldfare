import Foundation
import TurboFieldfareDecodeProtocol

public enum ChatRole: String, Codable, Sendable {
    case user
    case assistant
}

/// One committed exchange half. Committed turns are immutable: the in-flight
/// turn lives in `AppModel`'s live fields until a generation reaches a terminal
/// event, so a turn in a conversation is never partially written.
public struct ChatTurn: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var role: ChatRole
    public var content: String
    public var timestamp: Date

    public init(id: UUID = UUID(),
                role: ChatRole,
                content: String,
                timestamp: Date = Date()) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
    }

    var decodeMessage: DecodeChatMessage {
        DecodeChatMessage(role: role == .user ? .user : .assistant,
                          content: content)
    }
}

public struct Conversation: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var title: String
    /// A manual rename sticks; otherwise the title tracks the first user turn.
    public var titleIsCustom: Bool
    public var createdAt: Date
    public var updatedAt: Date
    public var turns: [ChatTurn]
    /// Exact prompt token count reported by the last generation in this
    /// conversation. Anchors the context estimate so only the turns added since
    /// then have to be guessed at.
    public var lastPromptTokenCount: Int?

    public init(id: UUID = UUID(),
                title: String = Conversation.untitled,
                titleIsCustom: Bool = false,
                createdAt: Date = Date(),
                updatedAt: Date = Date(),
                turns: [ChatTurn] = [],
                lastPromptTokenCount: Int? = nil) {
        self.id = id
        self.title = title
        self.titleIsCustom = titleIsCustom
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.turns = turns
        self.lastPromptTokenCount = lastPromptTokenCount
    }

    public static let untitled = "New Chat"

    public var isEmpty: Bool { turns.isEmpty }

    /// Derives a title from the first user turn. Cheap and deterministic on
    /// purpose: titling with the model would cost a full prefill on the 8 GB
    /// machines this app targets.
    public static func derivedTitle(from content: String) -> String {
        let collapsed = content
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
        guard !collapsed.isEmpty else { return untitled }
        let limit = 40
        guard collapsed.count > limit else { return collapsed }
        let clipped = collapsed.prefix(limit)
        // Prefer a word boundary, but only when it keeps most of the budget.
        if let space = clipped.lastIndex(of: " "),
           clipped.distance(from: clipped.startIndex, to: space) >= limit / 2 {
            return String(clipped[..<space]) + "…"
        }
        return String(clipped) + "…"
    }

    mutating func retitleIfNeeded() {
        guard !titleIsCustom,
              let first = turns.first(where: { $0.role == .user }) else { return }
        title = Self.derivedTitle(from: first.content)
    }
}

/// Versioned envelope for the whole store, mirroring `MacAppSettings`: a
/// version mismatch or decode failure discards the file rather than trying to
/// migrate partial state.
public struct ConversationStoreFile: Codable, Equatable, Sendable {
    public static let currentVersion = 1
    public static let fileName = "conversations.json"

    public var version: Int
    public var conversations: [Conversation]

    public init(version: Int = Self.currentVersion,
                conversations: [Conversation] = []) {
        self.version = version
        self.conversations = conversations
    }

    public func isValid() -> Bool {
        guard version == Self.currentVersion else { return false }
        let ids = conversations.map(\.id)
        return Set(ids).count == ids.count
    }
}
