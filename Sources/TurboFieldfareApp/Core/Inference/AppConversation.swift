import Foundation

public struct AppChatTurn: Identifiable, Equatable, Sendable {
    public enum Role: Equatable, Sendable { case user, assistant }

    public let id: UUID
    public let role: Role
    public var text: String
    public var images: [ChatImage]
    public var promptTokens: Int?
    public var cachedTokens: Int?
    public var generatedTokens: Int?
    public var stopReason: AppStopReason?
    /// What this turn put into the KV, in the model's own IDs.
    ///
    /// A user turn carries the continuation-encoded prompt it contributed; an
    /// assistant turn carries the reply the cache kept. `text` is what the
    /// reader sees and these are what the model saw, and the two are not
    /// interchangeable: the pinned template strips historical thought spans, so
    /// re-rendering `text` produces a different sequence. Nil on the
    /// single-prompt path, and on any turn the runtime rewound.
    public var tokenIDs: [Int32]?

    public init(id: UUID = UUID(), role: Role, text: String,
                images: [ChatImage] = [],
                promptTokens: Int? = nil, cachedTokens: Int? = nil,
                generatedTokens: Int? = nil, stopReason: AppStopReason? = nil,
                tokenIDs: [Int32]? = nil) {
        self.id = id
        self.role = role
        self.text = text
        self.images = images
        self.promptTokens = promptTokens
        self.cachedTokens = cachedTokens
        self.generatedTokens = generatedTokens
        self.stopReason = stopReason
        self.tokenIDs = tokenIDs
    }
}

/// One image of a stored conversation as the app hands it back for replay.
///
/// The span is a lower bound and a count rather than a `Range`, matching the
/// wire type it becomes: this travels to another process, and a range that
/// arrives inverted has to be refusable rather than fatal.
public struct AppConversationReplayImage: Sendable, Equatable {
    public let tokenLowerBound: Int
    public let tokenCount: Int
    public let fileURL: URL
    public let expectedDigest: String

    public init(tokenLowerBound: Int, tokenCount: Int,
                fileURL: URL, expectedDigest: String) {
        self.tokenLowerBound = tokenLowerBound
        self.tokenCount = tokenCount
        self.fileURL = fileURL
        self.expectedDigest = expectedDigest
    }
}

/// A stored conversation as the inference side needs it: the token IDs the KV
/// held, the images those IDs stand in for, and the turn count the reopened
/// lineage continues from.
public struct AppConversationLineage: Sendable, Equatable {
    public let tokenIDs: [Int32]
    public let images: [AppConversationReplayImage]
    public let boundaryTokenIDs: [Int32]
    public let boundaryNeedsReplay: Bool
    public let committedTurns: Int

    public init(tokenIDs: [Int32],
                images: [AppConversationReplayImage] = [],
                boundaryTokenIDs: [Int32] = [],
                boundaryNeedsReplay: Bool = false,
                committedTurns: Int) {
        self.tokenIDs = tokenIDs
        self.images = images
        self.boundaryTokenIDs = boundaryTokenIDs
        self.boundaryNeedsReplay = boundaryNeedsReplay
        self.committedTurns = committedTurns
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
    /// A token the model emitted that the KV never took, because the last turn
    /// stopped on max tokens or was cancelled. The next turn replays it, so a
    /// stored conversation has to carry it or the reopened context is missing a
    /// token the model already produced.
    public private(set) var boundaryTokenIDs: [Int32] = []
    public private(set) var boundaryNeedsReplay: Bool = false
    /// The turns of this window's conversation whose KV a load or an unload
    /// took.
    ///
    /// They stay on screen because the app deliberately keeps a transcript
    /// across lifecycle actions, and the transcript draws a break above them to
    /// say the model can no longer see them. They belong to the conversation
    /// rather than to what is being drawn: held in the same array as the
    /// read-only copy of a chat being browsed, a reload while browsing drew
    /// that chat as this conversation's own out-of-context turns.
    ///
    /// Kept as turns and paired on demand so `Equatable` stays synthesized: an
    /// array of tuples has no conformance to derive from.
    private var outOfContextTurns: [AppChatTurn] = []
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
        Self.pairs(of: turns)
    }

    /// The exchanges the transcript draws above the context break: on screen,
    /// and out of the model's context.
    public var outOfContextPairs: [(user: AppChatTurn, assistant: AppChatTurn)] {
        Self.pairs(of: outOfContextTurns)
    }

    private static func pairs(of turns: [AppChatTurn])
        -> [(user: AppChatTurn, assistant: AppChatTurn)] {
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

    /// Starts a fresh lineage, optionally carrying the exchanges a released KV
    /// left on screen. The caller is responsible for telling the inference side
    /// about `epoch` before the next turn is sent.
    public mutating func startNew(
        epoch: UUID = UUID(),
        carryingOutOfContext pairs: [(user: AppChatTurn, assistant: AppChatTurn)] = []
    ) {
        self = AppConversation(epoch: epoch)
        outOfContextTurns = pairs.flatMap { [$0.user, $0.assistant] }
    }

    /// Forgets the exchanges a released KV left on screen, because the window
    /// is drawing something else now.
    public mutating func dropOutOfContextPairs() {
        outOfContextTurns = []
    }

    /// Adopts a stored conversation whose token IDs are back in the KV.
    ///
    /// `committedTurns` is derived from the turns rather than passed in, so the
    /// count the service's gate was opened at and the count this type reports
    /// come from the same transcript. Two independently supplied numbers is
    /// exactly the drift that makes the gate refuse every later turn.
    public mutating func adoptRestored(epoch: UUID,
                                       turns: [AppChatTurn],
                                       kvTokens: Int,
                                       boundaryTokenIDs: [Int32],
                                       boundaryNeedsReplay: Bool) {
        self = AppConversation(epoch: epoch)
        self.turns = turns
        self.committedTurns = turns.filter { $0.role == .user }.count
        self.kvTokens = kvTokens
        self.boundaryTokenIDs = boundaryTokenIDs
        self.boundaryNeedsReplay = boundaryNeedsReplay
    }

    /// Shows the user's turn and reserves its position, or refuses.
    ///
    /// Refusing rather than trapping: a second send while one is decoding is a
    /// double-tap on Send, and a send after the lineage broke is a user who has
    /// not read the banner yet. Neither should crash, and neither may be
    /// admitted.
    public mutating func beginTurn(text: String,
                                   images: [StagedImage] = []) -> Ticket? {
        guard canSend else { return nil }
        let turn = AppChatTurn(role: .user, text: text,
                               images: images.map(ChatImage.staged))
        turns.append(turn)
        pendingUserTurnID = turn.id
        return Ticket(epoch: epoch, index: committedTurns)
    }

    /// Attaches the retained image links to the turn that is in flight. They
    /// are staged after the turn is reserved, because retaining them can fail
    /// and a failure gives the position straight back.
    /// Only staged files reach a live turn: a stored conversation's pictures
    /// arrive through `adoptRestored`, already owned by the conversation store.
    public mutating func attachImagesToPendingTurn(_ images: [StagedImage]) {
        guard let pendingUserTurnID,
              let index = turns.firstIndex(where: { $0.id == pendingUserTurnID }) else {
            return
        }
        turns[index].images = images.map(ChatImage.staged)
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
        turns[userIndex].tokenIDs = diagnostics?.promptTokenIDs
        turns.append(AppChatTurn(
            role: .assistant, text: text,
            generatedTokens: diagnostics?.generatedTokens,
            stopReason: diagnostics?.stopReason,
            tokenIDs: diagnostics?.generatedTokenIDs))
        committedTurns += 1
        boundaryTokenIDs = diagnostics?.boundaryTokenIDs ?? []
        boundaryNeedsReplay = diagnostics?.boundaryNeedsReplay ?? false
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
