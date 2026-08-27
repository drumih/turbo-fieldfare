import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

/// These decisions lived inside a private view coordinator, where no test could
/// reach them — which is why three defects survived two reviews. Each test here
/// pins one of them.
@Suite struct TranscriptSyncPlannerTests {
    private func input(epoch: UUID, history: Int, contextBreak: Int? = nil,
                       startedNewRun: Bool = false,
                       firstSynchronize: Bool = false) -> TranscriptSyncPlanner.Input {
        .init(epoch: epoch, historyCount: history, contextBreak: contextBreak,
              startedNewRun: startedNewRun, firstSynchronize: firstSynchronize)
    }

    @Test func afreshCoordinatorDrawsTheHistoryItFinds() {
        var planner = TranscriptSyncPlanner()
        let epoch = UUID()
        #expect(planner.plan(input(epoch: epoch, history: 3, firstSynchronize: true))
                == [.reset, .drawPair(index: 0), .drawPair(index: 1), .drawPair(index: 2)])
        // Nothing owed on the next pass.
        #expect(planner.plan(input(epoch: epoch, history: 3)).isEmpty)
    }

    @Test func anewRunSealsTheTurnAlreadyOnScreenRatherThanRedrawingIt() {
        var planner = TranscriptSyncPlanner()
        let epoch = UUID()
        _ = planner.plan(input(epoch: epoch, history: 0, firstSynchronize: true))
        // Turn one finished and a second run started: history grew by one and
        // that pair is already drawn as the live turn.
        #expect(planner.plan(input(epoch: epoch, history: 1, startedNewRun: true))
                == [.sealDrawnTurn])
    }

    @Test func anewEpochResetsBeforeAnythingElse() {
        var planner = TranscriptSyncPlanner()
        let first = UUID()
        _ = planner.plan(input(epoch: first, history: 2, firstSynchronize: true))
        let steps = planner.plan(input(epoch: UUID(), history: 0))
        #expect(steps == [.reset])
        #expect(planner.renderedHistory == 0)
        #expect(!planner.renderedContextBreak)
    }

    /// The break separates the turns above it from the model's context, so it
    /// has to be written after they are drawn. Planning it first put it above
    /// the very turns it sits under.
    @Test func thecontextBreakComesAfterTheHistoryItSeparates() {
        var planner = TranscriptSyncPlanner()
        let epoch = UUID()
        let steps = planner.plan(input(epoch: epoch, history: 2, contextBreak: 2,
                                       firstSynchronize: true))
        #expect(steps == [.reset, .drawPair(index: 0), .drawPair(index: 1),
                          .appendContextBreak])
        let breakIndex = try? #require(steps.firstIndex(of: .appendContextBreak))
        let lastPair = try? #require(steps.lastIndex(of: .drawPair(index: 1)))
        if let breakIndex, let lastPair { #expect(lastPair < breakIndex) }
    }

    /// A reload archives its turns and changes the epoch in the same pass. The
    /// break used to be skipped on exactly that pass, and nothing scheduled
    /// another — archived turns sat on screen unmarked until an unrelated
    /// change forced one.
    @Test func areloadDrawsItsBreakOnTheSamePassAsTheReset() {
        var planner = TranscriptSyncPlanner()
        let before = UUID()
        _ = planner.plan(input(epoch: before, history: 1, firstSynchronize: true))
        let steps = planner.plan(input(epoch: UUID(), history: 1, contextBreak: 1))
        #expect(steps.contains(.appendContextBreak),
                "the archived turns were left with no break until some later pass")
        #expect(steps.first == .reset)
    }

    /// The append only writes at the frozen boundary and can refuse. Latching
    /// the flag anyway meant one refusal suppressed the break for good.
    @Test func arefusedBreakIsPlannedAgainInsteadOfBeingLatchedOff() {
        var planner = TranscriptSyncPlanner()
        let epoch = UUID()
        let first = planner.plan(input(epoch: epoch, history: 1, contextBreak: 1,
                                       firstSynchronize: true))
        #expect(first.contains(.appendContextBreak))
        // The coordinator does not call `markContextBreakDrawn` — the append
        // refused because a live turn was drawn.
        #expect(planner.plan(input(epoch: epoch, history: 1, contextBreak: 1))
                == [.appendContextBreak])

        planner.markContextBreakDrawn()
        #expect(planner.plan(input(epoch: epoch, history: 1, contextBreak: 1)).isEmpty,
                "the break was drawn twice")
    }

    /// A seal that finds nothing to freeze must not consume the pair it was
    /// meant to freeze, or that turn is in the KV and permanently absent from
    /// the transcript.
    @Test func asealThatFrozeNothingLeavesThePairStillOwed() {
        var planner = TranscriptSyncPlanner()
        let epoch = UUID()
        _ = planner.plan(input(epoch: epoch, history: 0, firstSynchronize: true))
        #expect(planner.plan(input(epoch: epoch, history: 1, startedNewRun: true))
                == [.sealDrawnTurn])
        #expect(planner.renderedHistory == 1)

        planner.sealFoundNothingToFreeze(historyCount: 1)
        #expect(planner.renderedHistory == 0)
        // So the next pass draws it properly instead of skipping it forever.
        #expect(planner.plan(input(epoch: epoch, history: 1))
                == [.drawPair(index: 0)])
    }

    @Test func nothingIsPlannedWhenTheTranscriptIsAlreadyInStep() {
        var planner = TranscriptSyncPlanner()
        let epoch = UUID()
        _ = planner.plan(input(epoch: epoch, history: 2, firstSynchronize: true))
        planner.markContextBreakDrawn()
        #expect(planner.plan(input(epoch: epoch, history: 2, contextBreak: 2)).isEmpty)
    }
}
