import Foundation

public struct AppChatTurn: Identifiable, Equatable, Sendable {
    public enum Role: Equatable, Sendable { case user, assistant }

    public let id: UUID
    public let role: Role
    public var text: String
    public var images: [AppImageAttachment]
    public var promptTokens: Int?
    public var cachedTokens: Int?
    public var generatedTokens: Int?
    public var stopReason: AppStopReason?

    public init(id: UUID = UUID(), role: Role, text: String,
                images: [AppImageAttachment] = [],
                promptTokens: Int? = nil, cachedTokens: Int? = nil,
                generatedTokens: Int? = nil, stopReason: AppStopReason? = nil) {
        self.id = id
        self.role = role
        self.text = text
        self.images = images
        self.promptTokens = promptTokens
        self.cachedTokens = cachedTokens
        self.generatedTokens = generatedTokens
        self.stopReason = stopReason
    }
}
/// The app's half of one conversation: what the transcript shows, and the turn
/// order the decode service's gate checks against.
///
/// The invariant this type exists to hold is that **the transcript equals the
/// model's context**. A turn shown here that is not in the KV is a lie the user
/// cannot see, and one in the KV but not here is context they cannot account
/// for. Both counts move together or not at all, which is why a turn is only
/// committed once its generation stream completed: a turn that threw was
/// rewound by the runtime, so it is in neither place.
public struct AppConversation: Equatable, Sendable {
    /// What a turn must present to the service to be admitted.
    public struct Ticket: Equatable, Sendable {
        public let epoch: UUID
        public let index: Int
    }

    public private(set) var epoch: UUID
    public private(set) var turns: [AppChatTurn]
    /// Turns whose tokens are in the KV. Must equal the service gate's count.
    public private(set) var committedTurns: Int
    /// Tokens the KV holds, for the context gauge and the image budget. `nil`
    /// means the client committed a turn without reporting its exact position.
    public private(set) var kvTokens: Int?
    /// Set when the runtime reports the KV no longer matches this conversation.
    /// Nothing can continue it; only a new chat clears it.
    public private(set) var isLineageLost: Bool
    private var pendingUserTurnID: UUID?

    public init(epoch: UUID = UUID()) {
        self.epoch = epoch
        self.turns = []
        self.committedTurns = 0
        self.kvTokens = 0
        self.isLineageLost = false
    }

    public var isEmpty: Bool { turns.isEmpty }

    /// The turns that are finished, as the pairs the transcript draws. A user
    /// turn still decoding is not here: it is the live turn, drawn separately
    /// so its answer can be appended token by token.
    public var completedPairs: [(user: AppChatTurn, assistant: AppChatTurn)] {
        var pairs: [(user: AppChatTurn, assistant: AppChatTurn)] = []
        var index = 0
        while index + 1 < turns.count {
            let user = turns[index]
            let assistant = turns[index + 1]
            guard user.role == .user, assistant.role == .assistant else { break }
            pairs.append((user, assistant))
            index += 2
        }
        return pairs
    }
    public var hasTurnInFlight: Bool { pendingUserTurnID != nil }
    public var canSend: Bool { !isLineageLost && pendingUserTurnID == nil }

    /// Starts a fresh lineage. The caller is responsible for telling the
    /// inference side about `epoch` before the next turn is sent.
    public mutating func startNew(epoch: UUID = UUID()) {
        self = AppConversation(epoch: epoch)
    }

    /// Shows the user's turn and reserves its position, or refuses.
    ///
    /// Refusing rather than trapping: a second send while one is decoding is a
    /// double-tap on Send, and a send after the lineage broke is a user who has
    /// not read the banner yet. Neither should crash, and neither may be
    /// admitted.
    public mutating func beginTurn(text: String,
                                   images: [AppImageAttachment] = []) -> Ticket? {
        guard canSend else { return nil }
        let turn = AppChatTurn(role: .user, text: text, images: images)
        turns.append(turn)
        pendingUserTurnID = turn.id
        return Ticket(epoch: epoch, index: committedTurns)
    }

    /// Attaches the retained image links to the turn that is in flight. They
    /// are staged after the turn is reserved, because retaining them can fail
    /// and a failure gives the position straight back.
    public mutating func attachImagesToPendingTurn(_ images: [AppImageAttachment]) {
        guard let pendingUserTurnID,
              let index = turns.firstIndex(where: { $0.id == pendingUserTurnID }) else {
            return
        }
        turns[index].images = images
    }

    /// Records the reply. Called only when the generation stream completed,
    /// including a turn the user stopped: that one ends at a token boundary
    /// with its partial reply committed to the KV.
    public mutating func completeTurn(text: String, diagnostics: AppDiagnostics?) {
        guard let pendingUserTurnID,
              let userIndex = turns.firstIndex(where: { $0.id == pendingUserTurnID }) else {
            return
        }
        turns[userIndex].promptTokens = diagnostics?.promptTokenCount
        turns[userIndex].cachedTokens = diagnostics?.cachedPromptTokens
        turns.append(AppChatTurn(
            role: .assistant, text: text,
            generatedTokens: diagnostics?.generatedTokens,
            stopReason: diagnostics?.stopReason))
        committedTurns += 1
        self.pendingUserTurnID = nil
        // Only the runtime knows the committed position. Sampled-token counts
        // include a boundary token that may not be in KV, and stop-string
        // cleanup can rewind more than one token. Missing data stays unknown so
        // capacity checks fail closed instead of trusting a stale lower bound.
        kvTokens = diagnostics?.conversationTokens
    }

    /// Drops the in-flight user turn and hands it back, because the runtime
    /// rewound it: it is not in the KV, so it must not stay in the transcript.
    /// The caller restores its text and images to the composer rather than
    /// making the user retype them.
    @discardableResult
    public mutating func abandonTurn() -> AppChatTurn? {
        guard let pendingUserTurnID,
              let index = turns.firstIndex(where: { $0.id == pendingUserTurnID }) else {
            return nil
        }
        let turn = turns.remove(at: index)
        self.pendingUserTurnID = nil
        return turn
    }

    /// The KV no longer matches this conversation. The transcript stays
    /// readable — losing what the user already read helps nobody — but nothing
    /// further can be sent.
    @discardableResult
    public mutating func markLineageLost() -> AppChatTurn? {
        let abandoned = abandonTurn()
        isLineageLost = true
        return abandoned
    }
}
