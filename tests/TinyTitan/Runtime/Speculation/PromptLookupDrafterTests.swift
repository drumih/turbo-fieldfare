import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

@Suite struct PromptLookupDrafterTests {
    /// The definition the replay tool uses, restated naively: the most recent
    /// earlier occurrence of the last n tokens whose continuation exists,
    /// longest n first.
    static func naivePropose(
        _ history: [Int32], limit: Int, minN: Int, maxN: Int
    ) -> ([Int32], Int) {
        guard limit > 0 else { return ([], 0) }
        for n in stride(from: maxN, through: minN, by: -1) where history.count >= n {
            let suffix = Array(history.suffix(n))
            // `end` is where the occurrence's continuation starts.
            var end = history.count - 1
            while end - n >= 0 {
                if Array(history[(end - n)..<end]) == suffix {
                    return (Array(history[end..<min(history.count, end + limit)]), n)
                }
                end -= 1
            }
        }
        return ([], 0)
    }

    @Test func copiesTheContinuationOfAnEarlierOccurrence() {
        var drafter = PromptLookupDrafter()
        drafter.append(contentsOf: [1, 2, 3, 4, 5, 9, 1, 2, 3] as [Int32])
        let (tokens, n) = drafter.propose(limit: 3)
        #expect(Array(tokens) == [4, 5, 9])
        #expect(n == 3)
    }

    @Test func prefersTheLongestMatchThenTheMostRecent() {
        var drafter = PromptLookupDrafter()
        // [7, 8] occurs twice; the later one wins at n = 2. [6, 7, 8] occurs
        // once, earlier, and wins because it is longer.
        drafter.append(contentsOf: [6, 7, 8, 1, 7, 8, 2, 6, 7, 8] as [Int32])
        let (tokens, n) = drafter.propose(limit: 2)
        #expect(n == 3)
        #expect(Array(tokens) == [1, 7])
    }

    @Test func neverMatchesTheSuffixAgainstItself() {
        var drafter = PromptLookupDrafter()
        drafter.append(contentsOf: [1, 2, 3] as [Int32])
        #expect(drafter.propose(limit: 4).tokens.isEmpty)
        #expect(drafter.propose(limit: 4).matchedN == 0)
    }

    @Test func aRepeatingPatternDraftsItsNextPeriod() {
        var drafter = PromptLookupDrafter()
        drafter.append(contentsOf: [5, 5, 5] as [Int32])
        // The proposal may run into the suffix: [5, 5] followed by 5.
        #expect(Array(drafter.propose(limit: 4).tokens) == [5])
    }

    @Test func respectsTheLimitAndResets() {
        var drafter = PromptLookupDrafter()
        drafter.append(contentsOf: [1, 2, 3, 4, 5, 6, 1, 2] as [Int32])
        #expect(Array(drafter.propose(limit: 2).tokens) == [3, 4])
        #expect(drafter.propose(limit: 0).tokens.isEmpty)
        drafter.reset()
        #expect(drafter.history.isEmpty)
        drafter.append(contentsOf: [1, 2] as [Int32])
        #expect(drafter.propose(limit: 2).tokens.isEmpty)
    }

    @Test func matchesTheNaiveDefinitionOnRandomStreams() {
        var generator = SplitMix64(seed: 0x5EED)
        for alphabet in [2, 3, 5, 16] {
            for (minN, maxN) in [(2, 8), (1, 4), (3, 3)] {
                var drafter = PromptLookupDrafter(minN: minN, maxN: maxN)
                var history: [Int32] = []
                for _ in 0..<400 {
                    let token = Int32(generator.next() % UInt64(alphabet))
                    drafter.append(token)
                    history.append(token)
                    let limit = Int(generator.next() % 9)
                    let (tokens, n) = drafter.propose(limit: limit)
                    let (expected, expectedN) = Self.naivePropose(
                        history, limit: limit, minN: minN, maxN: maxN)
                    #expect(Array(tokens) == expected)
                    #expect(n == expectedN)
                }
            }
        }
    }
}

@Suite struct WideVerifyControllerTests {
    @Test func expectedEmittedIsTheGeometricSum() {
        #expect(WideVerifyController.expectedEmitted(drafted: 0, acceptance: 0.5) == 1)
        #expect(WideVerifyController.expectedEmitted(drafted: 3, acceptance: 1) == 4)
        let value = WideVerifyController.expectedEmitted(drafted: 2, acceptance: 0.5)
        #expect(abs(value - 1.75) < 1e-12)
    }

    @Test func noMatchMeansAScalarStep() {
        let controller = WideVerifyController()
        #expect(controller.draftLength(available: 7, matchedN: 0) == 0)
        #expect(controller.draftLength(available: 0, matchedN: 8) == 0)
    }

    @Test func draftsOnlyWhenThePassPaysForItself() {
        // Plan projection: a W=8 pass costs 810 ms = 3.84 scalar tokens.
        var controller = WideVerifyController(costModel: .planProjection, maxRows: 8)
        // Near-certain acceptance: the widest pass wins.
        for _ in 0..<200 { controller.record(matchedN: 8, drafted: 7, acceptedCount: 7) }
        #expect(controller.draftLength(available: 7, matchedN: 8) == 7)
        // Near-zero acceptance on short matches: never worth a pass.
        for _ in 0..<200 { controller.record(matchedN: 2, drafted: 7, acceptedCount: 0) }
        #expect(controller.draftLength(available: 7, matchedN: 2) == 0)
    }

    @Test func aCheapPassDraftsAtModerateAcceptance() {
        let cheap = WideVerifyCostModel(
            scalarMilliseconds: 200, passBaseMilliseconds: 150, passRowMilliseconds: 20)
        let controller = WideVerifyController(costModel: cheap, maxRows: 16, priorAcceptance: 0.6)
        #expect(controller.draftLength(available: 15, matchedN: 4) > 0)
        let dear = WideVerifyCostModel(
            scalarMilliseconds: 200, passBaseMilliseconds: 900, passRowMilliseconds: 100)
        let refused = WideVerifyController(costModel: dear, maxRows: 16, priorAcceptance: 0.6)
        #expect(refused.draftLength(available: 15, matchedN: 4) == 0)
    }

    /// Gate 2's verdict, held in the default: on the measured chunked-prefill
    /// pass, 90% per-token acceptance loses to scalar decode at every width,
    /// and only near-certain acceptance at W=16 pays, by at most ~1.25x.
    @Test func theMeasuredDefaultDraftsOnlyNearCertainCopies() {
        let model = WideVerifyCostModel.measuredChunkedPrefill
        let scalarRate = 1 / model.scalarMilliseconds
        for drafted in 1...15 {
            let rate =
                WideVerifyController.expectedEmitted(drafted: drafted, acceptance: 0.9)
                / model.passMilliseconds(rows: drafted + 1)
            #expect(rate < scalarRate, "W=\(drafted + 1) at p=0.9")
        }
        let best = 16 / model.passMilliseconds(rows: 16) / scalarRate
        #expect(best > 1 && best < 1.3)
        var controller = WideVerifyController(maxRows: 16, priorAcceptance: 0.9)
        #expect(controller.draftLength(available: 15, matchedN: 8) == 0)
        for _ in 0..<200 { controller.record(matchedN: 8, drafted: 15, acceptedCount: 15) }
        #expect(controller.draftLength(available: 15, matchedN: 8) == 15)
    }

    @Test func estimatesForgetSoARegimeChangeIsFollowed() {
        var controller = WideVerifyController(decay: 0.9)
        for _ in 0..<100 { controller.record(matchedN: 5, drafted: 7, acceptedCount: 7) }
        let high = controller.acceptance(matchedN: 5)
        for _ in 0..<60 { controller.record(matchedN: 5, drafted: 7, acceptedCount: 0) }
        #expect(high > 0.95)
        #expect(controller.acceptance(matchedN: 5) < 0.2)
    }
}
