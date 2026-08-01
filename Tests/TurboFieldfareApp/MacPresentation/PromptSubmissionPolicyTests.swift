import SwiftUI
import Testing
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

@Suite struct PromptSubmissionPolicyTests {
    @Test(arguments: [true, false])
    func returnNewlineModeAlwaysDefersToEditor(canRun: Bool) {
        #expect(decision(.return, [], canRun: canRun) == .deferToEditor)
        #expect(decision(.return, .shift, canRun: canRun) == .deferToEditor)
        #expect(decision(.return, .command, canRun: canRun) == .deferToEditor)
        #expect(decision(.return, .option, canRun: canRun) == .deferToEditor)
        #expect(decision(.return, .control, canRun: canRun) == .deferToEditor)
    }

    @Test func plainReturnSubmitsOnlyWhenGenerationIsAvailable() {
        #expect(decision(.shiftReturn, [], canRun: true) == .submit)
        #expect(decision(.shiftReturn, [], canRun: false) == .consume)
    }

    @Test(arguments: [
        EventModifiers.shift,
        EventModifiers.command,
        EventModifiers.option,
        EventModifiers.control,
        [EventModifiers.command, .shift],
    ])
    func modifiedReturnDefersToEditor(_ modifiers: EventModifiers) {
        #expect(decision(.shiftReturn, modifiers, canRun: true) == .deferToEditor)
    }

    @Test func markedTextAlwaysDefersToEditor() {
        #expect(decision(.shiftReturn, [], canRun: true, hasMarkedText: true)
            == .deferToEditor)
        #expect(decision(.shiftReturn, [], canRun: false, hasMarkedText: true)
            == .deferToEditor)
    }

    @Test func plainReturnRepeatsAreConsumedInSendMode() {
        #expect(decision(.shiftReturn, [], canRun: true, isRepeat: true) == .consume)
        #expect(decision(.shiftReturn, [], canRun: false, isRepeat: true) == .consume)
        #expect(decision(
            .shiftReturn, [], canRun: true, hasMarkedText: true, isRepeat: true)
            == .consume)
    }

    @Test func newlineAndModifiedReturnRepeatsRemainNative() {
        #expect(decision(.return, [], canRun: true, isRepeat: true) == .deferToEditor)
        #expect(decision(
            .shiftReturn, .shift, canRun: true, isRepeat: true) == .deferToEditor)
    }

    private func decision(
        _ shortcut: AppNewlineShortcut,
        _ modifiers: EventModifiers,
        canRun: Bool,
        hasMarkedText: Bool = false,
        isRepeat: Bool = false
    ) -> PromptSubmissionDecision {
        PromptSubmissionPolicy.decision(
            newlineShortcut: shortcut,
            modifiers: modifiers,
            canRun: canRun,
            hasMarkedText: hasMarkedText,
            isRepeat: isRepeat)
    }
}
