import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@MainActor
@Suite struct TranscriptTextViewTests {
    /// Never the general pasteboard: these tests must not touch what the
    /// person running them has copied.
    static func pasteboard(_ name: String) -> NSPasteboard {
        let board = NSPasteboard(name: NSPasteboard.Name("test-transcript-\(name)"))
        board.clearContents()
        return board
    }

    static func view(_ attributed: NSAttributedString) -> TranscriptTextView {
        let view = TranscriptTextView()
        view.textStorage?.setAttributedString(attributed)
        return view
    }

    static let answer = "The relation is $E = mc^2$ and it holds."

    // MARK: - The shared projection

    @Test func theProjectionPutsTheLatexBackWhereTheAttachmentIs() {
        let renderer = ResponseMarkdownRenderer()
        let rendered = renderer.render(Self.answer).attributedString
        let projected = TranscriptPlainText.string(of: rendered)

        #expect(rendered.string.contains("\u{FFFC}"))
        #expect(projected == "The relation is $E = mc^2$ and it holds.")
        // Copy actions and `plainText` are one code path, so they cannot drift.
        #expect(projected == renderer.plainText(Self.answer))
    }

    // MARK: - Selection copy

    @Test func copyingASelectionWritesTheLatexRatherThanThePlaceholder() {
        let rendered = ResponseMarkdownRenderer().render(Self.answer).attributedString
        let view = Self.view(rendered)
        view.setSelectedRange(NSRange(location: 0, length: rendered.length))

        let board = Self.pasteboard("whole")
        view.copySelection(to: board)

        let copied = board.string(forType: .string)
        #expect(copied == "The relation is $E = mc^2$ and it holds.")
        #expect(copied?.contains("\u{FFFC}") == false)
        // The rich flavour is still AppKit's, so styling and images survive.
        #expect(board.data(forType: .rtf) != nil)
    }

    @Test func copyingPartOfALineKeepsOnlyThatPart() {
        let rendered = ResponseMarkdownRenderer().render(Self.answer).attributedString
        let view = Self.view(rendered)
        let equation = (rendered.string as NSString).range(of: "\u{FFFC}")
        view.setSelectedRange(NSRange(location: equation.location, length: equation.length + 4))

        let board = Self.pasteboard("partial")
        view.copySelection(to: board)

        #expect(board.string(forType: .string) == "$E = mc^2$ and")
    }

    @Test func anEmptySelectionWritesNothing() {
        let rendered = ResponseMarkdownRenderer().render(Self.answer).attributedString
        let view = Self.view(rendered)
        view.setSelectedRange(NSRange(location: 3, length: 0))

        let board = Self.pasteboard("empty")
        board.declareTypes([.string], owner: nil)
        board.setString("untouched", forType: .string)
        view.copySelection(to: board)

        #expect(board.string(forType: .string) == "untouched")
    }

    /// Text with no maths in it must come back exactly as it reads, so the
    /// override cannot become a second, divergent way of producing plain text.
    @Test func copyingPlainProseIsUnchangedByTheOverride() {
        let rendered = ResponseMarkdownRenderer()
            .render("First line.\n\nSecond line.").attributedString
        let view = Self.view(rendered)
        view.setSelectedRange(NSRange(location: 0, length: rendered.length))

        let board = Self.pasteboard("prose")
        view.copySelection(to: board)

        #expect(board.string(forType: .string) == rendered.string)
    }
}
