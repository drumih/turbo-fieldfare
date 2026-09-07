import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Admission has to promise what it checks.
///
/// Before this, `generate` admitted any turn whose prompt alone fit — room for
/// exactly one generated token — while the refusal the user saw named their
/// whole max response. A chat near the ceiling was accepted and then answered
/// with a single word, and the same arithmetic now decides whether a stored
/// conversation can be continued, so an over-generous guard would put a row
/// marked "continue" in front of a composer that refuses everything.
@Suite struct ConversationGenerationReserveTests {
    /// The old guard was `prompt + 1 <= maxContext`. Anything that admits a
    /// prompt with only one token of headroom fails here.
    @Test func aTurnNeedsRoomForARealReplyNotOneToken() {
        let maxContext = 4_096
        let reserve = ConversationGenerationReserve.tokens
        #expect(reserve > 1, "a one-token reserve is what this test exists to reject")

        #expect(ConversationGenerationReserve.fits(
            tokens: maxContext - reserve, maxContext: maxContext))
        #expect(!ConversationGenerationReserve.fits(
            tokens: maxContext - reserve + 1, maxContext: maxContext))
        // The old guard was `prompt + 1 <= maxContext`, which admitted this.
        #expect(!ConversationGenerationReserve.fits(
            tokens: maxContext - 1, maxContext: maxContext))
    }

    /// The message names the figure the guard used. Reporting the user's max
    /// response instead is how someone lowers their reply length, sees no
    /// change, and has no way to find out why.
    @Test func theRefusalNamesTheReserveItActuallyChecked() {
        let error = AppInferenceError.conversationContextExhausted(
            prompt: 3_900, reserve: ConversationGenerationReserve.tokens,
            maxContext: 4_096)
        let message = error.userMessage
        #expect(message.contains("3900") || message.contains("3,900"))
        #expect(message.contains("\(ConversationGenerationReserve.tokens)"))
        #expect(message.contains("4096") || message.contains("4,096"))
        // The way out is a longer context or a new chat, not a shorter prompt.
        #expect(message.contains("context length"))
        #expect(message.contains("new chat"))
    }

    /// A conversation that could not take a turn must not be replayed at all:
    /// restoring it would spend minutes of prefill to arrive at a composer that
    /// refuses everything typed into it.
    ///
    /// The lineage rule leaves room for the smallest turn, not just for the
    /// reply. Asking `fits` with the lineage alone admitted a chat whose
    /// next message, once wrapped in the turn template, no longer fit — a
    /// row that said "continue" and refused every message.
    @Test func restoreAndSendAgreeOnTheSameCeiling() {
        let maxContext = 8_192
        let envelope = ConversationGenerationReserve.turnEnvelope
        let restorable = maxContext - ConversationGenerationReserve.tokens - envelope
        #expect(ConversationGenerationReserve.fitsLineage(
            tokens: restorable, maxContext: maxContext))
        #expect(!ConversationGenerationReserve.fitsLineage(
            tokens: restorable + 1, maxContext: maxContext))
        // The lineage the row admits still fits once the smallest turn is on
        // it, which is what `generate` will ask.
        #expect(ConversationGenerationReserve.fits(
            tokens: restorable + envelope, maxContext: maxContext))
        // The context `needsContext` suggests is one the same rule accepts.
        let required = ConversationGenerationReserve.contextRequired(
            forLineage: restorable + 1)
        #expect(ConversationGenerationReserve.fitsLineage(
            tokens: restorable + 1, maxContext: required))
        #expect(!ConversationGenerationReserve.fitsLineage(
            tokens: restorable + 1, maxContext: required - 1))
    }

    /// The envelope is what a one-word turn costs in template tokens, and a
    /// zero would put the rule back where it was.
    @Test func theTurnEnvelopeIsARealAllowance() {
        #expect(ConversationGenerationReserve.turnEnvelope > 1)
        #expect(ConversationGenerationReserve.turnEnvelope
            < ConversationGenerationReserve.tokens)
    }
}
