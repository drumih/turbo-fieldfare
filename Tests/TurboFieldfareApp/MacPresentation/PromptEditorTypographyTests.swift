import AppKit
import Testing
import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

@MainActor @Suite struct PromptEditorTypographyTests {
    @Test func changingSizePreservesStringSelectionAndUndo() throws {
        let editor = PromptEditorTextView()
        editor.isRichText = false
        editor.allowsUndo = true
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 400, height: 200),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = editor
        window.makeFirstResponder(editor)
        editor.typography = .init()
        editor.insertText("A cafe\u{301} draft", replacementRange: NSRange(location: 0, length: 0))
        editor.breakUndoCoalescing()
        let text = editor.string
        let range = NSRange(location: 2, length: 5)
        editor.setSelectedRange(range)
        let undo = try #require(editor.undoManager)
        #expect(undo.canUndo)
        for size in AppTextSize.allCases + [.standard] {
            editor.typography = .init(size)
            #expect(editor.string == text)
            #expect(editor.selectedRange() == range)
            #expect(editor.font?.pointSize == NSFont.systemFontSize * CGFloat(size.scale))
        }
        undo.undo()
        #expect(editor.string.isEmpty)
    }

    @Test func unmarkWithoutEditingAppliesDeferredSize() {
        let editor = PromptEditorTextView()
        editor.isRichText = false
        editor.typography = .init()
        editor.setMarkedText("draft", selectedRange: NSRange(location: 5, length: 0),
                             replacementRange: NSRange(location: 0, length: 0))
        #expect(editor.hasMarkedText())
        editor.typography = .init(.largest)
        #expect(editor.hasMarkedText())
        #expect(editor.font?.pointSize == NSFont.systemFontSize)
        editor.unmarkText()
        #expect(editor.string == "draft")
        #expect(!editor.hasMarkedText())
        #expect(editor.font?.pointSize == NSFont.systemFontSize * 2)
    }

    @Test func emptyEditorTypesAtNewSizeAndSameSizeIsNoOp() {
        let editor = PromptEditorTextView()
        editor.typography = .init(.larger)
        #expect(!editor.applyTypography())
        editor.insertText("next", replacementRange: NSRange(location: 0, length: 0))
        #expect(editor.font?.pointSize == NSFont.systemFontSize * 1.5)
        #expect((editor.typingAttributes[.font] as? NSFont)?.pointSize == NSFont.systemFontSize * 1.5)
    }
}
