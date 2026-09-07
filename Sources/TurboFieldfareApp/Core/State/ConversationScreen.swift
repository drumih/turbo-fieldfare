import Foundation

/// What the transcript pane is showing.
///
/// Exactly one case at a time. The six stored fields this replaces —
/// `pendingReplayConversationID`, `storedDisplayPairs`, `archivedPairs`,
/// `storedCopyRenderID`, `isRestoringConversation`, `openedConversationState` —
/// admitted dozens of combinations for the four states that actually exist, and
/// twelve of the feature's defects were two of those fields disagreeing: the
/// chat that was read left on screen under the row returned to, a conversation
/// drawn twice, a replay landing on top of a New Chat.
public enum ConversationScreen: Equatable, Sendable {
    /// The conversation the KV holds, empty or not. Always continuable.
    case live
    /// A stored conversation drawn from disk while the KV keeps `live`.
    ///
    /// `document` is nil until the load lands: reading a conversation is a file
    /// read and continuing it is a full prefill, so the row is drawn at once
    /// and filled when the file has been walked. `renderID` is new on every
    /// entry, including a re-entry into the same conversation, because the
    /// transcript renderer appends and cannot take pairs back — it has to be
    /// told that what is on screen is a different thing to draw.
    case reading(id: UUID, document: ConversationDocument?,
                 state: ConversationContinuability, renderID: UUID)
    /// A stored conversation being prefilled into the KV ahead of the turn that
    /// asked for it.
    ///
    /// `pending` is that turn, held out of the composer for the length of the
    /// replay and handed straight back if it fails. `renderID` is carried over
    /// from `reading` rather than derived from the conversation, so the
    /// transcript is not rebuilt at the moment a send starts.
    case replaying(id: UUID, document: ConversationDocument?,
                   pending: PreparedTurn, renderID: UUID)
    /// A stored conversation whose transcript could not be read at all.
    ///
    /// The window draws the live conversation, exactly as it did before the row
    /// was clicked; this case only carries why the row that was clicked is not
    /// on screen, so the notice can say so instead of showing the empty state a
    /// chat that was never written also shows.
    case unreadable(id: UUID, error: AppInferenceError)

    /// Which case a screen is, without its payload.
    ///
    /// What the machine's matrix test enumerates. A new screen case forces a
    /// new `Shape` case through the exhaustive switch below, which grows
    /// `allCases`, which fails the completeness test until every event has a
    /// row for it.
    public enum Shape: String, CaseIterable, Sendable {
        case live, reading, replaying, unreadable
    }

    public var shape: Shape {
        switch self {
        case .live: return .live
        case .reading: return .reading
        case .replaying: return .replaying
        case .unreadable: return .unreadable
        }
    }

    public static var allShapes: [Shape] { Shape.allCases }

    /// The stored conversation this screen names, if any. Nil only for `live`.
    public var conversationID: UUID? {
        switch self {
        case .live: return nil
        case .reading(let id, _, _, _): return id
        case .replaying(let id, _, _, _): return id
        case .unreadable(let id, _): return id
        }
    }

    /// The stored conversation as it was read off disk, once it has landed.
    public var document: ConversationDocument? {
        switch self {
        case .live, .unreadable: return nil
        case .reading(_, let document, _, _): return document
        case .replaying(_, let document, _, _): return document
        }
    }

    /// Distinguishes repeated reads of the same conversation across actor hops.
    var documentReadID: UUID? {
        switch self {
        case .reading(_, _, _, let id), .replaying(_, _, _, let id): return id
        case .live, .unreadable: return nil
        }
    }

    /// A reopened conversation is being prefilled into the KV right now.
    ///
    /// Not `isRunning`: no turn is generating, so every guard written against
    /// that let New Chat through mid-replay — and the replay then landed on top
    /// of the empty chat the user had just asked for.
    public var isReplaying: Bool {
        if case .replaying = self { return true }
        return false
    }

    /// The chat on screen is not the one the KV is holding.
    ///
    /// Browsing does not disturb the model: the live conversation stays exactly
    /// where it is, and what is drawn is a copy read off disk. So the live
    /// fields belong to a different chat for as long as this is true, and
    /// nothing may draw them under the one being read.
    public var isShowingStoredCopy: Bool {
        switch self {
        case .reading, .replaying: return true
        case .live, .unreadable: return false
        }
    }

    /// Whether a message typed now has a conversation to go to.
    ///
    /// The composer used to ask only the live conversation, which is always
    /// willing. A row the notice had just called unable to continue could
    /// therefore be sent: its tokens were replayed under a checkpoint they were
    /// not recorded for, or the held chat's KV was dropped for a restore that
    /// was always going to be refused. What is on screen decides, and the
    /// answer is the same one the notice gives.
    public var allowsSend: Bool {
        switch self {
        case .live: return true
        case .reading(_, _, let state, _): return state == .continuable
        // A replay is already carrying a message; the composer is closed
        // until it lands. An unreadable row has nothing to replay, and a send
        // would silently land on the held conversation under the wrong row.
        case .replaying, .unreadable: return false
        }
    }
}
