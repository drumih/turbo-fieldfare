import Foundation

/// Every transition of what the window is showing, in one pure value.
///
/// The shape is the one `DecodeConversationGate` already proved in this tree: a
/// value, a typed event, a new value plus the effects the caller performs. The
/// gate has produced no defects while the flags around it produced twelve, and
/// the reason is that a transition nobody thought of is a cell in a table here
/// rather than an interaction between six fields nobody enumerated.
///
/// Two rules hold for every event:
///
/// 1. An event naming a conversation the screen no longer names is a no-op with
///    no effects. Two `guard` lines used to do this by hand for the async load;
///    a replay landing after a delete had no such guard at all.
/// 2. The machine performs nothing. It cannot read a file, start a task, or
///    touch the composer, so a transition cannot half-happen.
public struct ConversationScreenMachine: Equatable, Sendable {
    public private(set) var screen: ConversationScreen = .live

    public init() {}

    /// Starts at a given screen.
    ///
    /// Internal, and used only by the matrix test: it has to put the machine
    /// into every cell without replaying the events that reach one, and a
    /// settable `screen` would let production code do the same.
    init(screen: ConversationScreen) {
        self.screen = screen
    }

    public enum Event: Equatable, Sendable {
        /// A row in the sidebar was clicked. `heldID` and `kvMatchesHeld`
        /// describe the conversation the KV is holding, because clicking the
        /// row it is already holding costs nothing and must not rebuild it.
        case rowClicked(id: UUID, heldID: UUID?, kvMatchesHeld: Bool,
                        state: ConversationContinuability, renderID: UUID)
        /// The context in force changed, so the row on screen is worth a
        /// different answer than the one taken when it was clicked. `state` is
        /// that answer, passed as data rather than as a closure so the event
        /// stays `Equatable` and the matrix test can name the cell.
        case contextChanged(state: ConversationContinuability, heldID: UUID?,
                            kvMatchesHeld: Bool, renderID: UUID)
        case documentLoaded(id: UUID, renderID: UUID, document: ConversationDocument,
                            state: ConversationContinuability)
        case documentFailed(id: UUID, renderID: UUID, AppInferenceError)
        /// A send while a stored conversation is on screen. Its tokens have to
        /// be back in the KV before the turn can be numbered against them.
        case sendRequested(PreparedTurn)
        case replaySucceeded(id: UUID, document: ConversationDocument,
                             epoch: UUID, kvTokens: Int)
        /// Putting the conversation back failed for a reason outside it: the
        /// service refused the digest, the service died, a load is in flight.
        /// The record is intact, so the row stays continuable and the send can
        /// be retried once the cause is dealt with. The restore was attempted,
        /// so the held conversation's KV is gone with it.
        case replayFailed(id: UUID, error: AppInferenceError)
        /// The replay never reached the model: a check on this side refused it
        /// first, such as the companion pack being absent for a conversation
        /// with pictures. Distinct from `replayFailed` because nothing was
        /// asked of the service, so the held conversation is exactly where it
        /// was. Reported as a failed replay instead, its intact KV was
        /// abandoned and its staged images deleted for a restore that had not
        /// happened.
        case replayNotStarted(id: UUID, error: AppInferenceError)
        /// What is on disk cannot be replayed: a turn with no recorded tokens,
        /// an image whose placeholders are not where its turn says, a
        /// transcript that will not read. A permanent property of the record,
        /// so the row says so rather than offering the same failure again.
        ///
        /// `state` is what the row shows afterwards: a record that will not
        /// replay by default, or the answer a fresh read of the record gave
        /// when it disagreed with the cached one the click saw.
        case replayRefused(id: UUID, error: AppInferenceError,
                           state: ConversationContinuability
                               = .cannotReplay(reason: .tokenCountUnknown))
        case newChat
        case deleted(id: UUID, heldID: UUID?)
        /// A load, an unload, a reload or a gate refusal took the KV. Whatever
        /// was on screen stops being a conversation the model can be asked
        /// about.
        case lineageEnded

        /// Which case an event is, without its payload. See
        /// `ConversationScreen.Shape`.
        public enum Shape: String, CaseIterable, Sendable {
            case rowClicked, contextChanged, documentLoaded, documentFailed
            case sendRequested, replaySucceeded, replayFailed, replayNotStarted
            case replayRefused
            case newChat, deleted, lineageEnded
        }

        public var shape: Shape {
            switch self {
            case .rowClicked: return .rowClicked
            case .contextChanged: return .contextChanged
            case .documentLoaded: return .documentLoaded
            case .documentFailed: return .documentFailed
            case .sendRequested: return .sendRequested
            case .replaySucceeded: return .replaySucceeded
            case .replayFailed: return .replayFailed
            case .replayNotStarted: return .replayNotStarted
            case .replayRefused: return .replayRefused
            case .newChat: return .newChat
            case .deleted: return .deleted
            case .lineageEnded: return .lineageEnded
            }
        }

        public static var allShapes: [Shape] { Shape.allCases }
    }

    /// The work a transition asks for. Performed by `AppModel.perform`, which
    /// is the only place any of it happens.
    public enum Effect: Equatable, Sendable {
        /// Which row the sidebar highlights. Always the conversation on screen,
        /// including when that is the live one and including nil.
        case select(UUID?)
        /// The turns of a released KV that were still on screen. A click draws
        /// one conversation and only that one, so they go with everything else
        /// the window was drawing.
        case dropOutOfContextTurns
        case loadDocument(UUID, renderID: UUID)
        /// The window's sliders follow the conversation it is showing.
        case applySampling(ConversationSampling)
        case reportError(AppInferenceError)
        case replay(id: UUID, pending: PreparedTurn)
        /// The staged copies of the conversation the KV is about to stop
        /// holding. Emitted only where it is actually given up: after a replay
        /// succeeds, and on New Chat. Released before a replay instead, a
        /// restore that then failed left the held chat on screen pointing at
        /// files that had already been deleted.
        ///
        /// `includingLiveTurn` covers the pictures the live fields are still
        /// drawing. False after a replay, because by then those belong to the
        /// message that started it and the send is about to hard-link them.
        case releaseImagesOfHeldConversation(includingLiveTurn: Bool)
        /// A restore replaces the service KV before it can fail. The previously
        /// held lineage is therefore gone even though the requested row remains
        /// readable and retryable.
        case endFailedReplayLineage
        case adoptRestored(ConversationDocument, epoch: UUID, kvTokens: Int)
        /// The message the replay was holding goes back to the composer.
        ///
        /// The error is optional because a failure that has already recorded
        /// itself must not be recorded twice: a rewound generation sets
        /// `error` as it ends, and a second copy of the same cause would
        /// replace the one the window is already showing.
        case restoreComposer(PreparedTurn, AppInferenceError?)
        /// The message the replay was holding may start its turn.
        ///
        /// Read by the send pipeline, which is the thing holding the message.
        /// Performing it anywhere else is what made `run()` call itself.
        case startTurn(PreparedTurn)
    }

    public mutating func apply(_ event: Event) -> [Effect] {
        switch event {
        case .rowClicked(let id, let heldID, let kvMatchesHeld, let state,
                         let renderID):
            // Refused while a replay runs. The conversation being put into the
            // KV is the one the next turn is numbered against, so drawing
            // another one over it would leave the window naming a chat the send
            // is not going to.
            if screen.isReplaying { return [] }
            return show(id: id, heldID: heldID, kvMatchesHeld: kvMatchesHeld,
                        state: state, renderID: renderID)

        case .contextChanged(let state, let heldID, let kvMatchesHeld,
                             let renderID):
            // Only the row being read is redrawn: the live conversation's own
            // continuability is not in question, and a replay owns the screen
            // until it lands.
            guard case .reading(let id, _, _, _) = screen else { return [] }
            return show(id: id, heldID: heldID, kvMatchesHeld: kvMatchesHeld,
                        state: state, renderID: renderID)

        case .documentLoaded(let id, let requestID, let document, let refreshedState):
            switch screen {
            case .reading(let readingID, _, _, let renderID)
                where readingID == id && renderID == requestID:
                screen = .reading(id: id, document: document, state: refreshedState,
                                  renderID: renderID)
                // A row that cannot be continued does not move the window's
                // sliders: nothing is going to generate under them.
                return refreshedState == .continuable
                    ? [.applySampling(document.meta.sampling)] : []
            case .replaying(let replayingID, _, let pending, let renderID)
                where replayingID == id && renderID == requestID:
                // The click's read can land after the send that started the
                // replay. The transcript still wants it: what is drawn under a
                // replay is the conversation being put back.
                screen = .replaying(id: id, document: document, pending: pending,
                                    renderID: renderID)
                // Send owns the sampling now; only the transcript may catch up.
                return []
            default:
                return []
            }

        case .documentFailed(let id, let requestID, let error):
            // Not while replaying: that path reads the same file itself and
            // reports its own failure, and taking the screen from it would
            // leave the turn it is holding with nowhere to go back to.
            guard case .reading(let readingID, _, _, let renderID) = screen,
                  readingID == id, renderID == requestID else { return [] }
            screen = .unreadable(id: id, error: error)
            return [.reportError(error)]

        case .sendRequested(let pending):
            // A send on the live conversation changes nothing about what is on
            // screen, and a second one during a replay is refused before it
            // gets here. A row that cannot be continued is refused here as
            // well as at the composer: replaying it would put tokens recorded
            // under another checkpoint into the KV, or drop the held chat for
            // a restore the context cannot hold.
            guard case .reading(let id, let document, let state, let renderID)
                    = screen, state == .continuable
            else { return [] }
            screen = .replaying(id: id, document: document, pending: pending,
                                renderID: renderID)
            return [.replay(id: id, pending: pending)]

        case .replaySucceeded(let id, let document, let epoch, let kvTokens):
            guard case .replaying(let replayingID, _, let pending, _) = screen,
                  replayingID == id else { return [] }
            screen = .live
            return [.releaseImagesOfHeldConversation(includingLiveTurn: false),
                    .adoptRestored(document, epoch: epoch, kvTokens: kvTokens),
                    .select(id),
                    .startTurn(pending)]

        case .replayFailed(let id, let error):
            // Nothing about the conversation changed, so the row is what it
            // was. A replay is only ever started from a continuable row — the
            // composer and the `.sendRequested` guard both refuse the others —
            // so that is what it goes back to.
            var effects = endReplay(id: id, state: .continuable, error: error)
            if !effects.isEmpty { effects.insert(.endFailedReplayLineage, at: 0) }
            return effects

        case .replayNotStarted(let id, let error):
            // The same hand-back without the lineage end: the service was
            // never asked, so the held conversation and its pictures stay.
            return endReplay(id: id, state: .continuable, error: error)

        case .replayRefused(let id, let error, let state):
            return endReplay(id: id, state: state, error: error)

        case .newChat:
            screen = .live
            return [.releaseImagesOfHeldConversation(includingLiveTurn: true),
                    .select(nil)]

        case .deleted(let id, let heldID):
            // The chat being read is gone, so there is nothing left to draw.
            // What comes back is the chat the KV is still holding, and that is
            // the row the sidebar highlights: nil left the window showing a
            // conversation whose row was not selected.
            guard screen.conversationID == id else { return [] }
            // A replay of the deleted chat cannot land any more, and the send
            // waiting on it stops when it finds the screen gone. The message
            // goes back here, as it does when the KV goes under a replay.
            if case .replaying(_, _, let pending, _) = screen {
                screen = .live
                return [.restoreComposer(pending, nil), .select(heldID)]
            }
            screen = .live
            return [.select(heldID)]

        case .lineageEnded:
            // A replay that loses the KV under it can no longer land: the
            // send waiting on it will find the screen gone and stop, so the
            // message it was holding goes back to the composer here. Dropped
            // with the screen instead, the prompt and its staged pictures were
            // never handed back to anything.
            if case .replaying(_, _, let pending, _) = screen {
                screen = .live
                return [.restoreComposer(pending, nil), .select(nil)]
            }
            screen = .live
            return [.select(nil)]
        }
    }

    /// The one rule both a click and a context change follow.
    private mutating func show(id: UUID, heldID: UUID?, kvMatchesHeld: Bool,
                               state: ConversationContinuability,
                               renderID: UUID) -> [Effect] {
        // The conversation the model is already holding. Looking at another
        // chat and coming back used to cost a full replay of a conversation the
        // KV had never let go of, because opening anything started a new
        // lineage on this side. Nothing here has to be rebuilt: the transcript,
        // the epoch and the service's cache all still agree.
        //
        // Whatever its record says. `.cannotReplay` is about putting the
        // record back, and the held chat is not being put back: a turn whose
        // stored picture failed to write is still in the KV. Requiring
        // `.continuable` here read that chat as a copy the composer refuses,
        // with New Chat as the only way out of a conversation the model was
        // still holding. `.needsContext` is different: the context in force
        // no longer holds the chat, so the copy and its notice are the truth.
        if id == heldID, kvMatchesHeld {
            if case .needsContext = state {} else {
                screen = .live
                return [.select(id), .dropOutOfContextTurns]
            }
        }
        screen = .reading(id: id, document: nil, state: state,
                          renderID: renderID)
        return [.select(id), .dropOutOfContextTurns, .loadDocument(id, renderID: renderID)]
    }

    private mutating func endReplay(id: UUID,
                                    state: ConversationContinuability,
                                    error: AppInferenceError) -> [Effect] {
        guard case .replaying(let replayingID, let document, let pending,
                              let renderID) = screen,
              replayingID == id else { return [] }
        screen = .reading(id: id, document: document, state: state,
                          renderID: renderID)
        return [.restoreComposer(pending, error)]
    }
}
