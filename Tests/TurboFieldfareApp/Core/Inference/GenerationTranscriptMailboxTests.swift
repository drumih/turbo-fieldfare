import Testing
@testable import TurboFieldfareAppCore

@Suite struct GenerationTranscriptMailboxTests {
    @Test func drainIsLosslessAndExactlyOnce() {
        let mailbox = GenerationTranscriptMailbox()
        mailbox.append("alpha")
        mailbox.append(" beta")

        let first = mailbox.drain()
        #expect(first.pendingText == "alpha beta")
        #expect(first.completeText == "alpha beta")

        let second = mailbox.drain()
        #expect(second.pendingText.isEmpty)
        #expect(second.completeText == "alpha beta")
    }

    @Test func UnicodeSuffixesRemainLossless() {
        let mailbox = GenerationTranscriptMailbox()
        mailbox.append("Fieldfare ")
        mailbox.append("\u{1F426}\u{200D}\u{2B1B}")
        mailbox.append(" caf\u{00E9}")

        #expect(mailbox.drain().completeText == "Fieldfare \u{1F426}\u{200D}\u{2B1B} caf\u{00E9}")
    }

    @Test func resetClearsPendingAndCanonicalText() {
        let mailbox = GenerationTranscriptMailbox()
        mailbox.append("old")
        mailbox.reset()

        let snapshot = mailbox.drain()
        #expect(snapshot.pendingText.isEmpty)
        #expect(snapshot.completeText.isEmpty)
    }
}
