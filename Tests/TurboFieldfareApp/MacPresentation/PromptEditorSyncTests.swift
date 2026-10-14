import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct PromptEditorSyncTests {
    @Test func markedHypothesisKeepsStaleBindingOffTheView() {
        let decision = decision(
            bindingText: "",
            viewText: "hel",
            hasMarkedText: true,
            focusRequest: 1,
            appliedFocusRequest: 1)
        #expect(decision == PromptEditorSyncDecision(
            replacementText: nil,
            makeFirstResponder: false,
            appliedFocusRequest: 1))
    }

    @Test func markedFocusRequestStaysPendingUntilTextIsUnmarked() {
        let marked = decision(
            bindingText: "",
            viewText: "hel",
            hasMarkedText: true,
            focusRequest: 1,
            appliedFocusRequest: 0,
            isInWindow: true,
            isFirstResponder: false)
        #expect(marked == PromptEditorSyncDecision(
            replacementText: nil,
            makeFirstResponder: false,
            appliedFocusRequest: 0))

        let unmarked = decision(
            bindingText: "",
            viewText: "hel",
            hasMarkedText: false,
            focusRequest: 1,
            appliedFocusRequest: marked.appliedFocusRequest,
            isInWindow: true,
            isFirstResponder: false)
        #expect(unmarked == PromptEditorSyncDecision(
            replacementText: "",
            makeFirstResponder: true,
            appliedFocusRequest: 1))

        let again = decision(
            bindingText: "",
            viewText: "",
            hasMarkedText: false,
            focusRequest: 1,
            appliedFocusRequest: unmarked.appliedFocusRequest,
            isInWindow: true,
            isFirstResponder: false)
        #expect(again == PromptEditorSyncDecision(
            replacementText: nil,
            makeFirstResponder: false,
            appliedFocusRequest: 1))
    }

    @Test func emptyBindingReplacesDraftWithoutMovingFocus() {
        #expect(decision(
            bindingText: "",
            viewText: "draft",
            focusRequest: 1,
            appliedFocusRequest: 1,
            isFirstResponder: false) == PromptEditorSyncDecision(
                replacementText: "",
                makeFirstResponder: false,
                appliedFocusRequest: 1))
    }

    @Test func presetBindingReplacesEmptyViewWithoutMovingFocus() {
        #expect(decision(
            bindingText: "preset",
            viewText: "",
            focusRequest: 2,
            appliedFocusRequest: 2) == PromptEditorSyncDecision(
                replacementText: "preset",
                makeFirstResponder: false,
                appliedFocusRequest: 2))
    }

    @Test func equalStringsAreNotWrittenBack() {
        #expect(decision(
            bindingText: "draft",
            viewText: "draft",
            focusRequest: 1,
            appliedFocusRequest: 1,
            isInWindow: true,
            isFirstResponder: false) == PromptEditorSyncDecision(
                replacementText: nil,
                makeFirstResponder: false,
                appliedFocusRequest: 1))
    }

    @Test func advancedFocusRequestMakesFirstResponderOnce() {
        let focused = decision(
            bindingText: "draft",
            viewText: "draft",
            focusRequest: 1,
            appliedFocusRequest: 0,
            isInWindow: true,
            isFirstResponder: false)
        #expect(focused == PromptEditorSyncDecision(
            replacementText: nil,
            makeFirstResponder: true,
            appliedFocusRequest: 1))

        // Dictation took first responder. The same request must not take it back.
        let dictationTookFocus = decision(
            bindingText: "draft",
            viewText: "draft",
            focusRequest: 1,
            appliedFocusRequest: focused.appliedFocusRequest,
            isInWindow: true,
            isFirstResponder: false)
        #expect(dictationTookFocus == PromptEditorSyncDecision(
            replacementText: nil,
            makeFirstResponder: false,
            appliedFocusRequest: 1))
    }

    @Test func advancedFocusRequestOnFirstResponderIsConsumedWithoutRefocusing() {
        #expect(decision(
            bindingText: "draft",
            viewText: "draft",
            focusRequest: 1,
            appliedFocusRequest: 0,
            isInWindow: true,
            isFirstResponder: true) == PromptEditorSyncDecision(
                replacementText: nil,
                makeFirstResponder: false,
                appliedFocusRequest: 1))
    }

    @Test func focusRequestBeforeTheViewIsInAWindowStaysPending() {
        let waiting = decision(
            bindingText: "",
            viewText: "draft",
            focusRequest: 1,
            appliedFocusRequest: 0,
            isInWindow: false,
            isFirstResponder: false)
        #expect(waiting == PromptEditorSyncDecision(
            replacementText: "",
            makeFirstResponder: false,
            appliedFocusRequest: 0))

        let attached = decision(
            bindingText: "",
            viewText: "",
            focusRequest: 1,
            appliedFocusRequest: waiting.appliedFocusRequest,
            isInWindow: true,
            isFirstResponder: false)
        #expect(attached == PromptEditorSyncDecision(
            replacementText: nil,
            makeFirstResponder: true,
            appliedFocusRequest: 1))
    }

    @Test func imageOnlyClearFocusesWhenTheStringIsAlreadyEmpty() {
        #expect(decision(
            bindingText: "",
            viewText: "",
            focusRequest: 1,
            appliedFocusRequest: 0,
            isInWindow: true,
            isFirstResponder: false) == PromptEditorSyncDecision(
                replacementText: nil,
                makeFirstResponder: true,
                appliedFocusRequest: 1))
    }

    private func decision(
        bindingText: String,
        viewText: String,
        hasMarkedText: Bool = false,
        focusRequest: Int = 0,
        appliedFocusRequest: Int = 0,
        isInWindow: Bool = true,
        isFirstResponder: Bool = false
    ) -> PromptEditorSyncDecision {
        PromptEditorSync.decision(
            bindingText: bindingText,
            viewText: viewText,
            hasMarkedText: hasMarkedText,
            focusRequest: focusRequest,
            appliedFocusRequest: appliedFocusRequest,
            isInWindow: isInWindow,
            isFirstResponder: isFirstResponder)
    }
}
