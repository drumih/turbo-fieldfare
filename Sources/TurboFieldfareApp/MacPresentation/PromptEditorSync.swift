/// What `PromptTextEditor.updateNSView` should do to catch the AppKit field up
/// with the composer binding.
public struct PromptEditorSyncDecision: Equatable, Sendable {
    /// Binding text to assign, or nil when the view string stays as it is.
    public let replacementText: String?
    public let makeFirstResponder: Bool
    /// Value to remember. Unchanged means the focus request is still pending.
    public let appliedFocusRequest: Int

    public init(replacementText: String?, makeFirstResponder: Bool, appliedFocusRequest: Int) {
        self.replacementText = replacementText
        self.makeFirstResponder = makeFirstResponder
        self.appliedFocusRequest = appliedFocusRequest
    }
}

/// Decides how the prompt field catches up with its binding.
///
/// System dictation inserts a marked hypothesis and moves first responder to
/// its own UI until recognition ends. Writing `string` over that hypothesis
/// cancels the session, and so does `makeFirstResponder` on every refresh
/// where the field is focused but not first responder — that is the dictation
/// UI for the whole recognition. Clear is the only programmatic focus, so it
/// is a counted request applied once. The count stays pending while text is
/// marked or the view has no window, and a later move of first responder is
/// left alone once the count has been applied.
public enum PromptEditorSync {
    public static func decision(
        bindingText: String,
        viewText: String,
        hasMarkedText: Bool,
        focusRequest: Int,
        appliedFocusRequest: Int,
        isInWindow: Bool,
        isFirstResponder: Bool
    ) -> PromptEditorSyncDecision {
        // The marked range is the live hypothesis. The binding catches up on
        // the next unmarked pass, including a clear that arrived mid-dictation.
        guard !hasMarkedText else {
            return PromptEditorSyncDecision(
                replacementText: nil,
                makeFirstResponder: false,
                appliedFocusRequest: appliedFocusRequest)
        }

        let replacementText: String? = bindingText == viewText ? nil : bindingText
        // The text view cannot become first responder until it is in a window.
        // Consuming the count here would drop the clear button's focus.
        let canApplyFocus = focusRequest != appliedFocusRequest && isInWindow
        return PromptEditorSyncDecision(
            replacementText: replacementText,
            makeFirstResponder: canApplyFocus && !isFirstResponder,
            appliedFocusRequest: canApplyFocus ? focusRequest : appliedFocusRequest)
    }
}
