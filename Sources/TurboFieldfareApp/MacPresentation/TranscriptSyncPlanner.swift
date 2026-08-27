import Foundation

/// Decides what the transcript must do to catch up with the conversation.
///
/// Extracted from the view coordinator because that is where three defects hid
/// and none of them were reachable by a test: the context break was drawn
/// before the history it separates, its "already drawn" flag latched even when
/// the append was refused, and a seal that no-ops still advanced the rendered
/// count, dropping a turn from the transcript for good. The coordinator now
/// executes these steps rather than deciding them.
public enum TranscriptSyncStep: Equatable, Sendable {
    /// A different conversation: everything drawn belongs to the old one.
    case reset
    /// Freeze the turn already on screen into the history above it.
    case sealDrawnTurn
    /// Draw history pair `index`, then freeze it.
    case drawPair(index: Int)
    /// Mark where the model's context begins.
    case appendContextBreak
}

public struct TranscriptSyncPlanner: Equatable, Sendable {
    /// Pairs already frozen into the document.
    public private(set) var renderedHistory = 0
    /// The lineage the drawn document belongs to.
    public private(set) var renderedEpoch: UUID?
    public private(set) var renderedContextBreak = false

    public init() {}

    public struct Input: Equatable, Sendable {
        public let epoch: UUID
        public let historyCount: Int
        /// Pairs above this index are on screen but not in the model's context.
        public let contextBreak: Int?
        public let startedNewRun: Bool
        /// Whether the coordinator has ever drawn anything for this lineage.
        public let firstSynchronize: Bool

        public init(epoch: UUID, historyCount: Int, contextBreak: Int?,
                    startedNewRun: Bool, firstSynchronize: Bool) {
            self.epoch = epoch
            self.historyCount = historyCount
            self.contextBreak = contextBreak
            self.startedNewRun = startedNewRun
            self.firstSynchronize = firstSynchronize
        }
    }

    public mutating func plan(_ input: Input) -> [TranscriptSyncStep] {
        var steps: [TranscriptSyncStep] = []
        if renderedEpoch != input.epoch {
            steps.append(.reset)
            renderedEpoch = input.epoch
            renderedHistory = 0
            renderedContextBreak = false
        }

        if renderedHistory < input.historyCount {
            if input.startedNewRun,
               renderedHistory == input.historyCount - 1,
               !input.firstSynchronize {
                steps.append(.sealDrawnTurn)
                renderedHistory = input.historyCount
            } else {
                while renderedHistory < input.historyCount {
                    steps.append(.drawPair(index: renderedHistory))
                    renderedHistory += 1
                }
            }
        }

        // After the history, not before it. Drawing the break first put it above
        // the very turns it is supposed to sit under, and on the pass that
        // resets for a new epoch it was skipped entirely — leaving archived
        // turns on screen unmarked until some unrelated change forced another
        // pass.
        if let contextBreak = input.contextBreak,
           !renderedContextBreak,
           renderedHistory >= contextBreak {
            steps.append(.appendContextBreak)
        }
        return steps
    }

    /// Records that the break was actually drawn.
    ///
    /// Separate from planning because the append can refuse — it only writes at
    /// the frozen boundary. Latching the flag regardless meant one refusal
    /// suppressed the break for the rest of the session.
    public mutating func markContextBreakDrawn() {
        renderedContextBreak = true
    }

    /// Records that a seal did nothing, so the pair it was meant to freeze is
    /// still owed. Advancing past it left a turn that is in the KV permanently
    /// absent from the transcript.
    public mutating func sealFoundNothingToFreeze(historyCount: Int) {
        renderedHistory = max(0, min(renderedHistory, historyCount) - 1)
    }
}
