import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct ConversationContextPresentationTests {
    @Test func gaugeIsCompactAtBothScales() {
        #expect(ConversationContextPresentation.gauge(kvTokens: 58, maxContext: 2_048)
                == "58/2.0K")
        #expect(ConversationContextPresentation.gauge(kvTokens: 1_500, maxContext: 16_384)
                == "1.5K/16K")
        #expect(ConversationContextPresentation.gauge(kvTokens: 0, maxContext: 0)
                == "\u{2014}", "an unknown window must not read as a full one")
    }

    /// The committed count only moves when a turn finishes, so a gauge reading
    /// it sits still through the whole run and then jumps. These pin the live
    /// figure through each phase of a turn.
    @Test func theLiveCountTracksTheTurnThroughEveryPhase() {
        // Nothing reported yet — an image encoding, or the first chunk not in.
        // The last thing that was true is the committed figure.
        #expect(ConversationContextPresentation.liveTokens(
            prefillDone: 0, prefillTotal: 0, generated: 0, committed: 56) == 56)
        // Prefill reports absolute position, so it is the KV position directly.
        #expect(ConversationContextPresentation.liveTokens(
            prefillDone: 70, prefillTotal: 120, generated: 0, committed: 56) == 70)
        // The newest sampled token is the next producer boundary and is not in
        // KV yet. The first sample therefore leaves the committed position at
        // the prompt boundary.
        #expect(ConversationContextPresentation.liveTokens(
            prefillDone: 120, prefillTotal: 120, generated: 1, committed: 56) == 120)
        #expect(ConversationContextPresentation.liveTokens(
            prefillDone: 120, prefillTotal: 120, generated: 9, committed: 56) == 128)
    }

    @Test func theLiveCountNeverGoesBackwardsWithinATurn() {
        var previous = 0
        for done in stride(from: 0, through: 120, by: 20) {
            let value = ConversationContextPresentation.liveTokens(
                prefillDone: done, prefillTotal: 120, generated: 0, committed: 0)
            #expect(value >= previous, "the gauge went backwards during prefill")
            previous = value
        }
        for generated in 1...10 {
            let value = ConversationContextPresentation.liveTokens(
                prefillDone: 120, prefillTotal: 120, generated: generated, committed: 0)
            #expect(value >= previous, "the gauge went backwards during decode")
            previous = value
        }
    }

    @Test func fractionIsClampedBothWays() {
        #expect(ConversationContextPresentation.fraction(kvTokens: 0, maxContext: 100) == 0)
        #expect(ConversationContextPresentation.fraction(kvTokens: 50, maxContext: 100) == 0.5)
        // A KV count larger than the window is a bug elsewhere, not a reason to
        // draw a gauge past its end.
        #expect(ConversationContextPresentation.fraction(kvTokens: 500, maxContext: 100) == 1)
        #expect(ConversationContextPresentation.fraction(kvTokens: 5, maxContext: 0) == 0)
    }

    @Test func theWarningArrivesBeforeTheWallNotAtIt() {
        #expect(!ConversationContextPresentation.isNearingLimit(
            kvTokens: 79, maxContext: 100))
        #expect(ConversationContextPresentation.isNearingLimit(
            kvTokens: 80, maxContext: 100), "the boundary is inclusive")
        #expect(ConversationContextPresentation.isNearingLimit(
            kvTokens: 100, maxContext: 100))
    }

    @Test func theExplanationNamesTheReusedCount() {
        let text = ConversationContextPresentation.explanation(
            kvTokens: 1_234, maxContext: 8_192, cachedTokens: 1_000)
        #expect(text.contains("1,234"))
        #expect(text.contains("8,192"))
        #expect(text.contains("1,000"))
        #expect(text.contains("reused"))
    }

    @Test func afirstTurnSaysItHadNothingToReuse() {
        let text = ConversationContextPresentation.explanation(
            kvTokens: 20, maxContext: 8_192, cachedTokens: 0)
        #expect(text.contains("nothing to reuse"))
    }

    @Test func aoneShotRunSaysNothingAboutReuseAtAll() {
        let text = ConversationContextPresentation.explanation(
            kvTokens: 20, maxContext: 8_192, cachedTokens: nil)
        #expect(!text.contains("reuse"))
    }

    @Test func nearingTheLimitExplainsTheRefusalInAdvance() {
        let text = ConversationContextPresentation.explanation(
            kvTokens: 7_500, maxContext: 8_192, cachedTokens: 7_000)
        #expect(text.contains("refused"))
        #expect(text.contains("New chat") || text.contains("new chat"))
    }
}
