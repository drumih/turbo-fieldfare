import Foundation
import TurboFieldfare

/// Settings that were in force when a conversation was recorded.
///
/// Informational, not the continuability test: a 3,000-token chat recorded at
/// 8K continues perfectly well at 4K. What it is for is telling the user why
/// reopening will reload the model.
public struct ConversationSessionSettings: Codable, Equatable, Sendable {
    public var contextTokens: Int
    public var expertCacheSlots: Int
    public var visionResidencyPolicy: String

    public init(contextTokens: Int, expertCacheSlots: Int,
                visionResidencyPolicy: String) {
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.visionResidencyPolicy = visionResidencyPolicy
    }
}

/// Sampling as it was when the chat was last used. Restored into the composer
/// on reopen; it never affects the KV, so it is not part of replay identity.
public struct ConversationSampling: Codable, Equatable, Sendable {
    public var temperature: Double
    public var topKEnabled: Bool
    public var topK: Int
    public var topPEnabled: Bool
    public var topP: Double
    public var maxNewTokens: Int

    public init(temperature: Double, topKEnabled: Bool, topK: Int,
                topPEnabled: Bool, topP: Double, maxNewTokens: Int) {
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.maxNewTokens = maxNewTokens
    }
}

extension ConversationSampling {
    /// What a conversation reads as when nothing recorded its sampling: a
    /// legacy store whose `conversation.json` is gone and whose transcript
    /// predates the `origin` record. Sampling never enters replay identity, so
    /// the window's own defaults are the right answer here — refusing to open
    /// a readable chat over a slider position would not be.
    public static let windowDefaults = ConversationSampling(
        temperature: MacAppSettings().temperature,
        topKEnabled: MacAppSettings().topKEnabled,
        topK: MacAppSettings().topK,
        topPEnabled: MacAppSettings().topPEnabled,
        topP: MacAppSettings().topP,
        // The window's default reply length, which the settings file does not
        // carry.
        maxNewTokens: 4_096)
}

/// The token a run left outside the KV, kept so a reopened conversation replays
/// it exactly as a live one would.
public struct ConversationBoundary: Codable, Equatable, Sendable {
    public var tokens: [Int32]
    public var needsReplay: Bool

    public init(tokens: [Int32] = [], needsReplay: Bool = false) {
        self.tokens = tokens
        self.needsReplay = needsReplay
    }
}

public enum ConversationTitleSource: String, Codable, Sendable {
    case firstMessage
    case user
}

/// The small file the sidebar reads: a cache of the transcript, not a second
/// copy of it.
///
/// Every field here is `ConversationMetaProjection.project` over the records in
/// `transcript.jsonl`, and nothing writes it by hand. It exists because listing
/// a hundred conversations must never mean parsing a hundred transcripts — and
/// because it is derived rather than maintained, it cannot drift from what the
/// conversation actually holds. A copy that disagrees is replaced on the next
/// open; a hand edit to this file does not survive one.
public struct ConversationMeta: Codable, Equatable, Sendable {
    /// A newer major is opened read-only and its bytes are left exactly as its
    /// owner wrote them.
    public static let currentVersion = 1
    public static let fileName = "conversation.json"

    public var version: Int
    public var id: UUID
    public var title: String
    public var titleSource: ConversationTitleSource
    public var createdAt: Date
    public var updatedAt: Date
    public var turnCount: Int
    /// Tokens the KV held when this conversation was last written. Optional and
    /// meant to stay that way: a conversation whose count was never reported
    /// cannot be replayed, and guessing one would put the user in front of a
    /// chat that fails on its first turn.
    public var kvTokens: Int?
    public var imageCount: Int
    public var identity: ConversationIdentity
    public var session: ConversationSessionSettings
    public var sampling: ConversationSampling
    public var boundary: ConversationBoundary
    /// A turn of this conversation reported a picture whose stored copy could
    /// not be written, so its recorded IDs cite placeholders no file backs.
    /// Projected from the turn records; `nil` means none of them reported one.
    public var imageWriteFailed: Bool?
    public var pinned: Bool

    public init(version: Int = ConversationMeta.currentVersion,
                id: UUID,
                title: String,
                titleSource: ConversationTitleSource = .firstMessage,
                createdAt: Date,
                updatedAt: Date,
                turnCount: Int = 0,
                kvTokens: Int? = nil,
                imageCount: Int = 0,
                identity: ConversationIdentity,
                session: ConversationSessionSettings,
                sampling: ConversationSampling,
                boundary: ConversationBoundary = ConversationBoundary(),
                imageWriteFailed: Bool? = nil,
                pinned: Bool = false) {
        self.version = version
        self.id = id
        self.title = title
        self.titleSource = titleSource
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.turnCount = turnCount
        self.kvTokens = kvTokens
        self.imageCount = imageCount
        self.identity = identity
        self.session = session
        self.sampling = sampling
        self.boundary = boundary
        self.imageWriteFailed = imageWriteFailed
        self.pinned = pinned
    }
}

/// One image as the transcript records it.
///
/// Two digests, because they answer different questions: `sourceDigest` is the
/// file the user attached and names the stored copies, `modelInputDigest` is the
/// pixels the tower consumed and is what replay verifies. A file whose bytes
/// hash the same can still decode differently on a later macOS, which is why
/// the second one exists at all.
public struct ConversationImageRecord: Codable, Equatable, Sendable {
    public var id: UUID
    public var displayName: String
    /// Store-relative, so moving the conversations folder does not break it.
    public var pixelsFile: String
    public var thumbnailFile: String
    public var sourceDigest: String
    public var modelInputDigest: String
    public var width: Int
    public var height: Int
    public var softTokens: Int

    public init(id: UUID, displayName: String, pixelsFile: String,
                thumbnailFile: String, sourceDigest: String,
                modelInputDigest: String, width: Int, height: Int,
                softTokens: Int) {
        self.id = id
        self.displayName = displayName
        self.pixelsFile = pixelsFile
        self.thumbnailFile = thumbnailFile
        self.sourceDigest = sourceDigest
        self.modelInputDigest = modelInputDigest
        self.width = width
        self.height = height
        self.softTokens = softTokens
    }
}

public struct ConversationTurnRecord: Codable, Equatable, Sendable {
    public enum Role: String, Codable, Sendable { case user, assistant }

    public var id: UUID
    public var role: Role
    public var at: Date
    public var text: String
    public var images: [ConversationImageRecord]
    /// Exactly what this turn put into the KV. The replay record; `text` is the
    /// reading copy and cannot substitute for it.
    public var tokens: [Int32]?
    /// Kept for display and structurally excluded from replay: the tokens above
    /// already contain whatever the model emitted, thought spans included.
    public var thinking: String?
    public var promptTokens: Int?
    public var cachedTokens: Int?
    public var generatedTokens: Int?
    public var stopReason: String?
    /// Sampling as it was when this turn was sent. On the user half, because
    /// that is the half the settings applied to. Informational: it never enters
    /// replay identity.
    public var sampling: ConversationSampling?
    /// The token this turn left outside the KV. On the assistant half, because
    /// that is the half that produced it, and it is a fact about the end of the
    /// conversation rather than about the turn.
    public var boundary: ConversationBoundary?
    /// This turn carried pictures whose stored copies could not be written.
    ///
    /// `tokens` above still holds their placeholder runs, so a replay would
    /// hand the model a span of image tokens with nothing behind them and the
    /// context would silently differ from the one the answer was produced in.
    /// The write used to be attempted with `try?`: the picture vanished from
    /// the record, the placeholders stayed in the IDs, and nothing said so.
    /// Optional because a transcript written before this build has no answer to
    /// give; absent reads as "no failure was reported".
    public var imageWriteFailed: Bool?

    public init(id: UUID = UUID(), role: Role, at: Date, text: String,
                images: [ConversationImageRecord] = [],
                tokens: [Int32]? = nil, thinking: String? = nil,
                promptTokens: Int? = nil, cachedTokens: Int? = nil,
                generatedTokens: Int? = nil, stopReason: String? = nil,
                sampling: ConversationSampling? = nil,
                boundary: ConversationBoundary? = nil,
                imageWriteFailed: Bool? = nil) {
        self.id = id
        self.role = role
        self.at = at
        self.text = text
        self.images = images
        self.tokens = tokens
        self.thinking = thinking
        self.promptTokens = promptTokens
        self.cachedTokens = cachedTokens
        self.generatedTokens = generatedTokens
        self.stopReason = stopReason
        self.sampling = sampling
        self.boundary = boundary
        self.imageWriteFailed = imageWriteFailed
    }
}

/// The first line of every transcript: what the conversation is and what it was
/// made under.
///
/// The three creation facts are optional because a transcript written before
/// they moved here does not carry them; for those, `open` hoists them out of
/// the legacy `conversation.json` into an `origin` record once. An older build
/// reading this one's header simply does not ask for the extra keys.
public struct ConversationHeaderRecord: Codable, Equatable, Sendable {
    public var version: Int
    public var id: UUID
    public var createdAt: Date
    public var identity: ConversationIdentity?
    public var session: ConversationSessionSettings?
    /// Sampling as the window had it when the chat was started. Superseded by
    /// any turn that records its own, which is every turn this build writes.
    public var sampling: ConversationSampling?

    public init(version: Int = ConversationMeta.currentVersion,
                id: UUID, createdAt: Date,
                identity: ConversationIdentity? = nil,
                session: ConversationSessionSettings? = nil,
                sampling: ConversationSampling? = nil) {
        self.version = version
        self.id = id
        self.createdAt = createdAt
        self.identity = identity
        self.session = session
        self.sampling = sampling
    }
}

/// The creation facts of a conversation whose header predates them, lifted out
/// of its `conversation.json` once and appended so the projection has a record
/// to read them from.
///
/// Written by `open` under the writer lock and never again: a header carrying
/// identity, or an `origin` already in the file, means there is nothing to do.
/// A build older than this one reads the line as `.unknown("origin")` and
/// ignores it, which is what the record enum's default case is for.
public struct ConversationOriginRecord: Codable, Equatable, Sendable {
    public var identity: ConversationIdentity
    public var session: ConversationSessionSettings
    public var sampling: ConversationSampling
    public var boundary: ConversationBoundary
    public var at: Date

    public init(identity: ConversationIdentity,
                session: ConversationSessionSettings,
                sampling: ConversationSampling,
                boundary: ConversationBoundary,
                at: Date) {
        self.identity = identity
        self.session = session
        self.sampling = sampling
        self.boundary = boundary
        self.at = at
    }
}

public struct ConversationTitleRecord: Codable, Equatable, Sendable {
    public var title: String
    public var source: ConversationTitleSource
    public var at: Date

    public init(title: String, source: ConversationTitleSource, at: Date) {
        self.title = title
        self.source = source
        self.at = at
    }
}

/// One line of `transcript.jsonl`.
///
/// The `type` is read before anything else and an unrecognised one becomes
/// `.unknown` rather than an error, so a file written by a later build stays
/// readable to this one instead of taking the whole conversation down. The same
/// rule every surviving design in this space arrived at, and the reason
/// `MacAppSettings` already reads its version through a stamp.
public enum TranscriptRecord: Equatable, Sendable {
    case header(ConversationHeaderRecord)
    case turn(ConversationTurnRecord)
    /// A checkpoint of a reply still being generated. A later `turn` with the
    /// same id supersedes it.
    case partial(ConversationTurnRecord)
    case title(ConversationTitleRecord)
    /// A legacy conversation's creation facts, hoisted out of its meta once.
    case origin(ConversationOriginRecord)
    case unknown(type: String)
}

extension TranscriptRecord: Codable {
    private enum CodingKeys: String, CodingKey {
        case type, header, turn, partial, title, origin
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "header":
            self = .header(try container.decode(
                ConversationHeaderRecord.self, forKey: .header))
        case "turn":
            self = .turn(try container.decode(
                ConversationTurnRecord.self, forKey: .turn))
        case "partial":
            self = .partial(try container.decode(
                ConversationTurnRecord.self, forKey: .partial))
        case "title":
            self = .title(try container.decode(
                ConversationTitleRecord.self, forKey: .title))
        case "origin":
            self = .origin(try container.decode(
                ConversationOriginRecord.self, forKey: .origin))
        default:
            self = .unknown(type: type)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .header(let value):
            try container.encode("header", forKey: .type)
            try container.encode(value, forKey: .header)
        case .turn(let value):
            try container.encode("turn", forKey: .type)
            try container.encode(value, forKey: .turn)
        case .partial(let value):
            try container.encode("partial", forKey: .type)
            try container.encode(value, forKey: .partial)
        case .title(let value):
            try container.encode("title", forKey: .type)
            try container.encode(value, forKey: .title)
        case .origin(let value):
            try container.encode("origin", forKey: .type)
            try container.encode(value, forKey: .origin)
        case .unknown(let type):
            try container.encode(type, forKey: .type)
        }
    }
}
