import Foundation

/// Decides whether an incoming generate may join the open conversation.
///
/// Extracted from the service's command loop so the decision is testable
/// without a socket, a model, or a Metal device. The loop cannot express the
/// boundary cases this holds — a turn from a replaced conversation, a turn that
/// skips or repeats a position, a one-shot arriving while a lineage is open —
/// and every one of them ends with a message the user never sent being appended
/// to a KV, which no later check can detect.
public struct DecodeConversationGate: Equatable, Sendable {
    /// Why a turn cannot run. Typed rather than a string so a test pins the
    /// case, and the loop owns the wording.
    public enum Rejection: Error, Equatable, Sendable {
        /// The turn names a conversation that is no longer the open one.
        case staleConversation(requested: UUID, open: UUID?)
        /// The turn's position does not follow the committed turns.
        case outOfOrderTurn(requested: Int?, committed: Int)
        /// A one-shot generate arrived while a conversation was open. Running
        /// it would reset the KV under a lineage the app still believes in.
        case oneShotDuringConversation(open: UUID)
    }

    public enum Admission: Equatable, Sendable {
        /// No conversation is open: reset the KV and prefill the whole prompt.
        case oneShot
        /// Append to the open lineage at `index`.
        case turn(epoch: UUID, index: Int)
    }

    public private(set) var openEpoch: UUID?
    public private(set) var committedTurns = 0

    public init() {}

    /// Starts a new lineage. The caller drops the KV; this only records that it
    /// did.
    public mutating func reset(to epoch: UUID) {
        openEpoch = epoch
        committedTurns = 0
    }

    /// Opens a lineage that already has turns behind it: a stored conversation
    /// whose token IDs are back in the KV.
    ///
    /// Opening it at zero, as a reset does, would reject the reopened
    /// conversation's very next turn as out of order and keep rejecting every
    /// turn after it. A negative count is clamped rather than trusted: the
    /// figure arrives over a socket.
    public mutating func restore(to epoch: UUID, committedTurns: Int) {
        openEpoch = epoch
        self.committedTurns = max(0, committedTurns)
    }

    /// Ends any lineage. Both unload and load reach here: each releases or
    /// rebuilds the runner and the KV, so the tokens the epoch named are gone,
    /// and a turn resuming onto them would resume onto nothing.
    public mutating func endLineage() {
        openEpoch = nil
        committedTurns = 0
    }

    /// A restore was asked for and did not complete.
    ///
    /// The session drops the KV it held before it prefills the requested
    /// record, so by the time a restore can fail — a digest that does not
    /// match, a token outside the vocabulary — the previous lineage's tokens
    /// are already gone. Leaving that epoch open admitted its next turn onto an
    /// empty cache as an opening turn, with the caller's transcript showing
    /// every exchange the model could no longer see. Fails closed for a
    /// restore refused before the session was touched too: the app treats
    /// every failed replay as the end of what it held, and the two sides must
    /// not disagree about which epoch is open.
    public mutating func restoreFailed() {
        endLineage()
    }

    public func admit(_ request: DecodeGenerationRequest) -> Result<Admission, Rejection> {
        guard let requested = request.conversationEpoch else {
            if let openEpoch { return .failure(.oneShotDuringConversation(open: openEpoch)) }
            return .success(.oneShot)
        }
        guard requested == openEpoch else {
            return .failure(.staleConversation(requested: requested, open: openEpoch))
        }
        guard request.turnIndex == committedTurns else {
            return .failure(.outOfOrderTurn(requested: request.turnIndex,
                                            committed: committedTurns))
        }
        return .success(.turn(epoch: requested, index: committedTurns))
    }

    /// Records that a turn actually ran. Called after the generation, not at
    /// admission: a turn rejected downstream, or one that failed before
    /// touching the KV, must not advance the position the next turn has to
    /// match.
    public mutating func commit(_ admission: Admission) {
        guard case .turn(let epoch, _) = admission, epoch == openEpoch else { return }
        committedTurns += 1
    }
}
