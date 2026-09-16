import AppKit

@MainActor
open class PromptEditorTextView: NSTextView {
    public var typography = ConversationTypography() {
        didSet { applyTypography() }
    }

    open override func unmarkText() {
        super.unmarkText()
        applyTypography()
    }

    open override func insertText(_ insertString: Any, replacementRange: NSRange) {
        super.insertText(insertString, replacementRange: replacementRange)
        applyTypography()
    }

    /// Changing the backing string to restyle it would lose undo and marked
    /// input. AppKit owns those; typography only changes its font attributes.
    @discardableResult
    func applyTypography() -> Bool {
        guard !hasMarkedText() else { return false }
        let font = typography.bodyFont
        guard self.font != font else { return false }
        let ranges = selectedRanges
        self.font = font
        typingAttributes[.font] = font
        selectedRanges = ranges
        return true
    }
}
