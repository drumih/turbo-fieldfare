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

    @Test func copyingASelectionWritesTheLatexRatherThanThePlaceholder() throws {
        let rendered = ResponseMarkdownRenderer().render(Self.answer).attributedString
        let view = Self.view(rendered)
        view.setSelectedRange(NSRange(location: 0, length: rendered.length))

        let board = Self.pasteboard("whole")
        view.copySelection(to: board)

        let copied = board.string(forType: .string)
        #expect(copied == "The relation is $E = mc^2$ and it holds.")
        #expect(copied?.contains("\u{FFFC}") == false)
        // The rich flavours carry the LaTeX too. RTF cannot hold an attachment
        // at all, so it used to lose every equation; RTFD held one rasterised
        // TIFF each.
        for type in [NSPasteboard.PasteboardType.rtf, .rtfd] {
            let data = try #require(board.data(forType: type))
            let decoded = try #require(NSAttributedString(
                rtfd: data, documentAttributes: nil)
                ?? NSAttributedString(rtf: data, documentAttributes: nil))
            #expect(decoded.string.contains("$E = mc^2$"), "\(type.rawValue)")
            #expect(!decoded.string.contains("\u{FFFC}"), "\(type.rawValue)")
        }
    }

    /// Cmd-C on the fifty-equation answer wrote a 12.3 MB RTFD of fifty TIFFs
    /// and logged a hundred `CGImageDestinationFinalize` lines, because AppKit
    /// rasterises an attachment that has no file wrapper.
    @Test func copyingAnAnswerFullOfEquationsWritesTextNotImages() throws {
        let source = try TranscriptCorpus.source("fifty-equations")
        let rendered = ResponseMarkdownRenderer().render(source).attributedString
        let attachments = TranscriptRenderCorpusTests.attachments(in: rendered)
        #expect(attachments.count == 50)

        let view = Self.view(rendered)
        view.setSelectedRange(NSRange(location: 0, length: rendered.length))
        let board = Self.pasteboard("fifty")
        view.copySelection(to: board)

        let rtfd = try #require(board.data(forType: .rtfd))
        #expect(rtfd.count < 200_000, "RTFD is \(rtfd.count) bytes")
        let text = try #require(String(data: rtfd, encoding: .isoLatin1))
        #expect(!text.contains("NeXTGraphic"))
        // Nothing was serialised as an attachment, which is what stopped AppKit
        // asking each one for a file wrapper and rasterising it. Reading
        // `fileWrapper` back here would create the wrapper this is about.
        let decoded = try #require(NSAttributedString(rtfd: rtfd, documentAttributes: nil))
        var kept = 0
        decoded.enumerateAttribute(
            .attachment,
            in: NSRange(location: 0, length: decoded.length),
            options: []) { value, _, _ in
            if value != nil { kept += 1 }
        }
        #expect(kept == 0)
        #expect(decoded.string.contains("\\frac"))
    }

    /// A drag and a Services provider go out through `writeSelection` too, so
    /// they cannot drift from what Cmd-C writes.
    @Test func dragAndServicesWriteTheSameBytesAsCopy() throws {
        let rendered = ResponseMarkdownRenderer().render(Self.answer).attributedString
        let view = Self.view(rendered)
        view.setSelectedRange(NSRange(location: 0, length: rendered.length))

        let copied = Self.pasteboard("copy-route")
        view.copySelection(to: copied)

        let dragged = Self.pasteboard("drag-route")
        let types = view.writablePasteboardTypes
        #expect(types == [.rtfd, .rtf, .string])
        dragged.declareTypes(types, owner: nil)
        #expect(view.writeSelection(to: dragged, types: types))

        let service = Self.pasteboard("service-route")
        service.declareTypes([.string], owner: nil)
        #expect(view.writeSelection(to: service, type: .string))

        for type in types {
            #expect(copied.data(forType: type) == dragged.data(forType: type),
                    "\(type.rawValue)")
        }
        #expect(service.string(forType: .string) == copied.string(forType: .string))
    }

    /// An image in the prompt strip is not a math attachment and still travels
    /// in the rich flavour the way it did before.
    @Test func aPromptImageStillTravelsInTheRichFlavour() throws {
        let image = NSImage(size: NSSize(width: 8, height: 8))
        image.lockFocus()
        NSColor.red.setFill()
        NSBezierPath.fill(NSRect(x: 0, y: 0, width: 8, height: 8))
        image.unlockFocus()
        let attachment = NSTextAttachment()
        attachment.image = image
        let document = NSMutableAttributedString(attachment: attachment)
        document.append(ResponseMarkdownRenderer().render(Self.answer).attributedString)

        let view = Self.view(document)
        view.setSelectedRange(NSRange(location: 0, length: document.length))
        let board = Self.pasteboard("prompt-image")
        view.copySelection(to: board)

        let rtfd = try #require(board.data(forType: .rtfd))
        let decoded = try #require(NSAttributedString(rtfd: rtfd, documentAttributes: nil))
        var kept = 0
        decoded.enumerateAttribute(
            .attachment,
            in: NSRange(location: 0, length: decoded.length),
            options: []) { value, _, _ in
            if value != nil { kept += 1 }
        }
        #expect(kept == 1)
        #expect(decoded.string.contains("$E = mc^2$"))
    }

    /// Nothing archives the transcript, so the attachment refuses to be
    /// decoded rather than coming back without the source it exists to carry.
    @Test func aMathAttachmentRefusesToBeDecoded() throws {
        let attachment = MathAttachment(latexSource: "E = mc^2")
        let archived = try NSKeyedArchiver.archivedData(
            withRootObject: attachment, requiringSecureCoding: false)
        let decoded = try? NSKeyedUnarchiver.unarchivedObject(
            ofClass: MathAttachment.self, from: archived)
        #expect(decoded == nil)
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

    @Test func contextMenuKeepsOnlyTheTranscriptCommands() throws {
        let view = Self.view(NSAttributedString(string: "Earlier answer"))
        view.answerAtCharacterIndex = { _ in "Earlier answer" }
        view.lastAnswerText = { "Last answer" }
        view.conversationText = { "Whole conversation" }
        view.startNewChat = {}
        view.setSelectedRange(NSRange(location: 0, length: 7))
        let event = try #require(NSEvent.mouseEvent(
            with: .rightMouseDown,
            location: .zero,
            modifierFlags: [],
            timestamp: 0,
            windowNumber: 0,
            context: nil,
            eventNumber: 0,
            clickCount: 1,
            pressure: 1))

        let menu = try #require(view.menu(for: event))

        #expect(!menu.allowsContextMenuPlugIns,
                "AppKit appends Services when contextual-menu plug-ins are enabled")
        #expect(menu.items.filter { !$0.isSeparatorItem }.map(\.title) == [
            "Copy This Answer",
            "Copy Selection",
            "Copy Last Answer",
            "Copy Conversation",
            "New Chat",
        ])
    }
}
