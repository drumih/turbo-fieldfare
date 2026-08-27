import Testing
@testable import TurboFieldfare

@Suite struct MultimodalConversationKVRecoveryTests {
    private enum FixtureError: Error, CustomStringConvertible {
        case generation
        case rewind

        var description: String {
            switch self {
            case .generation: "generation fixture"
            case .rewind: "rewind fixture"
            }
        }
    }

    @Test func failedTurnAtAnEmptyLineageResetsWithoutRewinding() throws {
        var reset = false
        try MultimodalConversationKVRecovery.restoreAfterFailure(
            positionBefore: 0,
            generationError: FixtureError.generation,
            rewind: { _ in Issue.record("empty lineage tried to rewind") },
            reset: { reset = true })

        #expect(reset)
    }

    @Test func failedRewindReportsLineageLossWithBothCauses() {
        do {
            try MultimodalConversationKVRecovery.restoreAfterFailure(
                positionBefore: 280,
                generationError: FixtureError.generation,
                rewind: { _ in throw FixtureError.rewind },
                reset: { Issue.record("nonempty lineage reset") })
            Issue.record("failed rewind was treated as recovered")
        } catch let error as MultimodalConversationError {
            guard case .lineageRecoveryFailed = error else {
                Issue.record("expected lineageRecoveryFailed, got \(error)")
                return
            }
            #expect(error.description.contains("generation fixture"))
            #expect(error.description.contains("rewind fixture"))
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    @Test func stopStringTrimRewindsToTheVisiblePrefix() throws {
        var target: Int?
        let kept = try MultimodalConversationKVRecovery.trimHiddenStopTokens(
            [1, 2, 3, 4],
            withheld: 2,
            rewind: { target = $0 },
            reset: { Issue.record("nonempty prefix reset") })

        #expect(target == 2)
        #expect(kept == [1, 2])
    }

    @Test func stopStringTrimToZeroResetsTheRunner() throws {
        var reset = false
        let kept = try MultimodalConversationKVRecovery.trimHiddenStopTokens(
            [1, 2],
            withheld: 2,
            rewind: { _ in Issue.record("zero target tried to rewind") },
            reset: { reset = true })

        #expect(reset)
        #expect(kept.isEmpty)
    }

    @Test func failedStopStringTrimReportsLineageLoss() {
        #expect(throws: MultimodalConversationError.self) {
            _ = try MultimodalConversationKVRecovery.trimHiddenStopTokens(
                [1, 2, 3],
                withheld: 1,
                rewind: { _ in throw FixtureError.rewind },
                reset: {})
        }
    }
}
