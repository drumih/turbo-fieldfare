import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct StreamingResponseBufferTests {
    @Test func revealsUnicodeByWholeCharacters() {
        var buffer = StreamingResponseBuffer()
        buffer.begin(with: "A \u{1F426}\u{200D}\u{2B1B} cafe\u{301}")

        #expect(buffer.revealNext(maximumCharacterCount: 3))
        #expect(buffer.displayedText == "A \u{1F426}\u{200D}\u{2B1B}")
        #expect(buffer.pendingCharacterCount == 5)

        #expect(buffer.revealNext(maximumCharacterCount: 10))
        #expect(buffer.displayedText == "A \u{1F426}\u{200D}\u{2B1B} cafe\u{301}")
        #expect(!buffer.hasPendingText)
    }

    @Test func appendedSnapshotsKeepTheCurrentRevealPosition() {
        var buffer = StreamingResponseBuffer()
        buffer.begin(with: "Hello")
        _ = buffer.revealNext(maximumCharacterCount: 2)
        buffer.receive("Hello, world")

        #expect(buffer.displayedText == "He")
        #expect(buffer.targetText == "Hello, world")

        buffer.finish()
        #expect(buffer.displayedText == "Hello, world")
    }

    @Test func replacementSnapshotsNeverMixDifferentResponses() {
        var buffer = StreamingResponseBuffer()
        buffer.begin(with: "First answer")
        _ = buffer.revealNext(maximumCharacterCount: 5)
        buffer.receive("Replacement answer")

        #expect(buffer.targetText == "Replacement answer")
        #expect(buffer.displayedText == "Replacement answer")
        #expect(!buffer.hasPendingText)
    }

    @Test func recommendedBatchIsBoundedAndMakesProgress() {
        #expect(StreamingResponseBuffer.recommendedBatchSize(for: 0) == 0)
        #expect(StreamingResponseBuffer.recommendedBatchSize(for: 1) == 1)
        #expect(StreamingResponseBuffer.recommendedBatchSize(for: 10_000) == 72)
    }
}
