import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

/// A chat renders every completed turn above the live one. The rule that makes
/// that affordable is that history is drawn once: the controller replaces only
/// the live turn, so TextKit never lays out the conversation again per token.
/// Every test here pins one way that used to break — `rebuild` called
/// `setAttributedString`, which took the history with it.
@MainActor
@Suite struct TranscriptMultiTurnTests {
    private func controller() -> InstructionTranscriptDocumentController {
        InstructionTranscriptDocumentController()
    }

    private func firstTurn(
        _ controller: InstructionTranscriptDocumentController,
        _ storage: NSMutableAttributedString,
        prompt: String = "first question",
        response: String = "first answer"
    ) {
        _ = controller.synchronize(
            storage: storage, prompt: prompt, response: response, isTerminal: true)
        controller.sealTurn(storage: storage)
    }

    @Test func asealedTurnBecomesHistoryAndTheLiveTurnStartsAfterIt() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        firstTurn(controller, storage)

        #expect(controller.frozenLength == storage.length)
        #expect(storage.string.contains("first question"))
        #expect(storage.string.contains("first answer"))
        #expect(controller.assistantRange.location == controller.frozenLength)
        #expect(controller.assistantRange.length == 0)
    }

    @Test func asecondTurnLeavesTheFirstBytesUntouched() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        firstTurn(controller, storage)
        let history = storage.attributedSubstring(
            from: NSRange(location: 0, length: controller.frozenLength))

        _ = controller.synchronize(
            storage: storage, prompt: "second question", response: "", isTerminal: false)
        _ = controller.synchronize(
            storage: storage, prompt: "second question", response: "second", isTerminal: false)
        _ = controller.synchronize(
            storage: storage, prompt: "second question", response: "second answer",
            isTerminal: true)

        let after = storage.attributedSubstring(
            from: NSRange(location: 0, length: controller.frozenLength))
        #expect(after.isEqual(to: history), "the first turn was redrawn")
        #expect(storage.string.contains("first answer"))
        #expect(storage.string.contains("second answer"))
    }

    /// Streaming into turn two must still be an append, not a rebuild. If it
    /// rebuilds per token the conversation is laid out again on every tick, and
    /// the cost grows with the whole transcript rather than the answer.
    @Test func streamingTheSecondTurnAppendsRatherThanRebuilds() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        firstTurn(controller, storage)

        _ = controller.synchronize(
            storage: storage, prompt: "second", response: "a", isTerminal: false)
        var mutations: [InstructionTranscriptDocumentController.Mutation] = []
        for text in ["ab", "abc", "abcd"] {
            let update = controller.synchronize(
                storage: storage, prompt: "second", response: text, isTerminal: false)
            mutations.append(update.mutation)
        }
        #expect(!mutations.contains(.rebuilt), "a token tick rebuilt the document")
        #expect(controller.assistantRange.location >= controller.frozenLength)
    }

    /// A rebuild inside the live turn is ordinary — a prompt edit, a response
    /// that does not extend the last one. It must still not reach above the
    /// frozen mark.
    @Test func arebuildInTheLiveTurnDoesNotReachTheHistory() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        firstTurn(controller, storage)
        let frozen = controller.frozenLength
        let history = storage.attributedSubstring(
            from: NSRange(location: 0, length: frozen))

        _ = controller.synchronize(
            storage: storage, prompt: "second", response: "aaa", isTerminal: false)
        // Not an extension of "aaa", so this is the rebuild path.
        _ = controller.synchronize(
            storage: storage, prompt: "second", response: "bbb", isTerminal: false)

        #expect(controller.frozenLength == frozen)
        #expect(storage.attributedSubstring(
            from: NSRange(location: 0, length: frozen)).isEqual(to: history))
        #expect(storage.string.contains("bbb"))
        #expect(!storage.string.contains("aaa"))
    }

    @Test func threeTurnsAccumulateInOrder() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        for index in 1...3 {
            _ = controller.synchronize(
                storage: storage, prompt: "question \(index)",
                response: "answer \(index)", isTerminal: true)
            controller.sealTurn(storage: storage)
        }
        let text = storage.string
        let one = try? #require(text.range(of: "answer 1"))
        let two = try? #require(text.range(of: "answer 2"))
        let three = try? #require(text.range(of: "answer 3"))
        #expect(one != nil && two != nil && three != nil)
        if let one, let two, let three {
            #expect(one.lowerBound < two.lowerBound)
            #expect(two.lowerBound < three.lowerBound)
        }
    }

    @Test func sealingAnEmptyLiveTurnIsANoOp() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        firstTurn(controller, storage)
        let frozen = controller.frozenLength
        // A refused turn leaves nothing drawn; sealing it would insert a blank
        // gap the user never earned.
        controller.sealTurn(storage: storage)
        #expect(controller.frozenLength == frozen)
        #expect(storage.length == frozen)
    }

    /// A reload takes the KV but not the transcript, so the turns above the
    /// break are on screen and not in context. Saying so is the difference
    /// between a transcript the user can trust and one that quietly implies the
    /// model remembers what it cannot.
    @Test func acontextBreakFreezesAndSeparatesTheArchivedTurns() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        firstTurn(controller, storage)
        let beforeBreak = controller.frozenLength

        controller.appendContextBreak(
            storage: storage, text: "Earlier turns are no longer in the model's context")
        #expect(controller.frozenLength > beforeBreak)
        #expect(storage.string.contains("no longer in the model's context"))
        #expect(controller.assistantRange.location == controller.frozenLength)

        // The turn after the break draws below it and leaves it alone.
        _ = controller.synchronize(
            storage: storage, prompt: "after", response: "reply", isTerminal: true)
        let text = storage.string
        let marker = try? #require(text.range(of: "no longer in the model's context"))
        let after = try? #require(text.range(of: "after"))
        if let marker, let after { #expect(marker.lowerBound < after.lowerBound) }
        #expect(text.contains("first answer"))
    }

    @Test func acontextBreakIsRefusedWhileALiveTurnIsDrawn() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        firstTurn(controller, storage)
        _ = controller.synchronize(
            storage: storage, prompt: "live", response: "partial", isTerminal: false)
        let frozen = controller.frozenLength
        // Inserting it here would put the marker above the live turn's text
        // instead of between the turns, and strand `frozenLength` mid-turn.
        controller.appendContextBreak(storage: storage, text: "break")
        #expect(controller.frozenLength == frozen)
        #expect(!storage.string.contains("break"))
    }

    /// Right-clicking inside a turn has to name *that* turn's answer. Falling
    /// back to the newest one would copy the wrong text while looking correct,
    /// which is worse than offering nothing.
    @Test func eachTurnReportsItsOwnAnswerByCharacterIndex() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        firstTurn(controller, storage, prompt: "q1", response: "answer one")
        let firstEnd = controller.frozenLength
        firstTurn(controller, storage, prompt: "q2", response: "answer two")
        let secondEnd = controller.frozenLength
        _ = controller.synchronize(
            storage: storage, prompt: "q3", response: "live answer", isTerminal: false)

        #expect(controller.answer(at: 1) == "answer one")
        #expect(controller.answer(at: firstEnd - 1) == "answer one")
        #expect(controller.answer(at: firstEnd + 1) == "answer two")
        #expect(controller.answer(at: secondEnd - 1) == "answer two")
        // The live turn is included: it is the one a reader most wants, and it
        // is not sealed until the next run starts.
        #expect(controller.answer(at: secondEnd + 1) == "live answer")
    }

    @Test func aturnWithNoAnswerOffersNothingToCopy() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        _ = controller.synchronize(
            storage: storage, prompt: "asked", response: "", isTerminal: false)
        #expect(controller.answer(at: 0) == nil)
        controller.sealTurn(storage: storage)
        #expect(controller.answer(at: 0) == nil, "an empty answer is not copyable")
    }

    @Test func resetTranscriptForgetsTheTurnAnswers() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        firstTurn(controller, storage, prompt: "q", response: "old answer")
        controller.resetTranscript(storage: storage)
        #expect(controller.answer(at: 0) == nil)
    }

    @Test func resetTranscriptClearsHistoryAndTheFrozenMark() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        firstTurn(controller, storage)
        _ = controller.synchronize(
            storage: storage, prompt: "second", response: "live", isTerminal: false)

        controller.resetTranscript(storage: storage)
        #expect(storage.length == 0)
        #expect(controller.frozenLength == 0)
        #expect(controller.assistantRange == NSRange(location: 0, length: 0))

        // And the next turn behaves like a first turn again.
        _ = controller.synchronize(
            storage: storage, prompt: "fresh", response: "answer", isTerminal: true)
        #expect(storage.string.contains("fresh"))
        #expect(!storage.string.contains("first question"))
    }

    /// The single-turn path is what every existing test covers, so its
    /// behaviour must be identical while nothing has been sealed.
    @Test func withNothingSealedTheControllerStillReplacesTheWholeDocument() {
        let controller = controller()
        let storage = NSMutableAttributedString()
        _ = controller.synchronize(
            storage: storage, prompt: "one", response: "first", isTerminal: false)
        _ = controller.synchronize(
            storage: storage, prompt: "two", response: "second", isTerminal: true)
        #expect(controller.frozenLength == 0)
        #expect(!storage.string.contains("one"))
        #expect(storage.string.contains("two"))
    }
}
