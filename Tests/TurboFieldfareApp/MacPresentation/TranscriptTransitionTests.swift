import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@MainActor
@Suite struct TranscriptTransitionTests {
    @Test(arguments: [true, false])
    func replayDrawsTheSavedPairBeforeThePendingImageTurn(progressive: Bool) {
        let controller = InstructionTranscriptDocumentController(
            environment: ["TURBO_FIELDFARE_PROGRESSIVE_RENDER": progressive ? "1" : "0"])
        let storage = NSMutableAttributedString()
        var planner = TranscriptSyncPlanner()
        let prefix = NSAttributedString(string: "[saved image]\n")
        func update(_ epoch: UUID, first: Bool, newRun: Bool) {
            controller.synchronizeHistory(storage: storage, planner: &planner,
                input: .init(epoch: epoch, historyCount: 1, contextBreak: nil,
                             startedNewRun: newRun, firstSynchronize: first)) { _ in
                _ = controller.synchronize(storage: storage, prompt: "code word",
                    response: "OK", isTerminal: true,
                    promptPrefix: prefix, promptPrefixIdentifier: "saved-image")
            }
        }
        update(UUID(), first: true, newRun: false)
        let restored = UUID()
        update(restored, first: false, newRun: true)
        _ = controller.synchronize(storage: storage, prompt: "explain it", response: "",
            isTerminal: false, showsPrefillPlaceholder: true,
            promptPrefix: NSAttributedString(string: "[pending image]\n"),
            promptPrefixIdentifier: "pending-image")
        // Inspect the first update, with no later timer/model event to repair it.
        #expect(storage.string.contains("code word"))
        #expect(storage.string.contains("OK"))
        #expect(storage.string.contains("[saved image]"))
        #expect(storage.string.contains("explain it"))
        #expect(storage.string.contains("[pending image]"))
        let frozen = controller.frozenLength
        let history = storage.attributedSubstring(from: NSRange(location: 0, length: frozen))
        update(restored, first: false, newRun: false)
        #expect(controller.frozenLength == frozen)
        #expect(storage.attributedSubstring(from: NSRange(location: 0, length: frozen))
            .isEqual(to: history))
        #expect(storage.string.components(separatedBy: "code word").count == 2)
    }

    @Test(arguments: [true, false], ["$$x = 1$$", "$$x = 1$$\n\n",
        "Final paragraph $x^2$", "$$\\begin{bmatrix}1 & 2 \\\\ 3 & 4\\end{bmatrix}$$"])
    func completedMathSurvivesSealingAndLaterTokens(progressive: Bool, answer: String) {
        let typesetter = FakeMathTypesetter()
        let controller = InstructionTranscriptDocumentController(
            renderer: ResponseMarkdownRenderer(typesetter: typesetter),
            environment: ["TURBO_FIELDFARE_PROGRESSIVE_RENDER": progressive ? "1" : "0"])
        let storage = NSMutableAttributedString()
        var planner = TranscriptSyncPlanner()
        let epoch = UUID()
        controller.synchronizeHistory(storage: storage, planner: &planner,
            input: .init(epoch: epoch, historyCount: 0, contextBreak: nil,
                         startedNewRun: false, firstSynchronize: true)) { _ in
            Issue.record("No history should be drawn")
        }
        _ = controller.synchronize(storage: storage, prompt: "formula", response: answer,
                                   isTerminal: true)
        let completed = NSAttributedString(attributedString: storage)
        let calls = typesetter.calls.count
        #expect(calls > 0)
        #expect(!storage.string.contains("$$"))
        // The queued-send model regression separately pins that this stays terminal.
        _ = controller.synchronize(storage: storage, prompt: "formula", response: answer,
                                   isTerminal: true)
        controller.synchronizeHistory(storage: storage, planner: &planner,
            input: .init(epoch: epoch, historyCount: 1, contextBreak: nil,
                         startedNewRun: true, firstSynchronize: false)) { _ in
            Issue.record("Same-epoch history must be sealed without rerendering")
        }
        for response in ["", "next", "next answer"] {
            _ = controller.synchronize(storage: storage, prompt: "follow up", response: response,
                                       isTerminal: false)
            #expect(storage.attributedSubstring(from: NSRange(location: 0,
                length: completed.length)).isEqual(to: completed))
        }
        #expect(controller.answer(at: 0) == answer, "Copy must retain the original LaTeX")
        #expect(typesetter.calls.count == calls)
        #expect(storage.string.components(separatedBy: "follow up").count == 2)
    }
}
