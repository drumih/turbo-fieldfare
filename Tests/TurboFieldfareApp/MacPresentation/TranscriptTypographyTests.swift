import AppKit
import Testing
import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

@MainActor @Suite struct TranscriptTypographyTests {
    private func input(_ epoch: UUID, history: Int, boundary: Int? = nil) -> TranscriptSyncPlanner.Input {
        .init(epoch: epoch, historyCount: history, contextBreak: boundary,
              startedNewRun: false, firstSynchronize: true)
    }

    // A renderer-only size change leaves frozen turns and cached progressive
    // blocks at their old size, even though subsequent tokens look correct.
    @Test(arguments: AppTextSize.allCases)
    func sealedHistoryAndOpenFenceResizeThenContinue(_ size: AppTextSize) throws {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()
        var planner = TranscriptSyncPlanner()
        let state = input(UUID(), history: 2, boundary: 1)
        func draw(_ index: Int) {
            controller.synchronize(storage: storage, prompt: "question \(index)",
                                   response: "historic \(index)", isTerminal: true)
        }
        controller.synchronizeHistory(storage: storage, planner: &planner, input: state, drawPair: draw)
        let open = "Finished paragraph.\n\n```swift\nlet a = 1"
        controller.synchronize(storage: storage, prompt: "live", response: open, isTerminal: false)
        let baseline = storage.copy() as! NSAttributedString
        let spy = CountingTypographyRenderer(.init(size))
        let update = controller.refreshTypography(
            .init(size), storage: storage, planner: &planner, input: state,
            promptPrefix: NSAttributedString(), renderer: spy, drawPair: draw)
        #expect(update.replaced == nil)
        #expect(storage.string == baseline.string)
        #expect(!controller.isFinalized)
        #expect(planner.renderedHistory == 2)
        #expect(planner.renderedContextBreak)
        baseline.enumerateAttribute(.font, in: NSRange(location: 0, length: baseline.length)) { value, range, _ in
            guard let font = value as? NSFont else { return }
            let scaled = storage.attribute(.font, at: range.location, effectiveRange: nil) as? NSFont
            #expect(scaled?.pointSize == font.pointSize * CGFloat(size.scale))
        }
        let calls = spy.sources.count
        let noChange = controller.refreshTypography(
            .init(size), storage: storage, planner: &planner, input: state,
            promptPrefix: NSAttributedString(), drawPair: { _ in Issue.record("same size replayed history") })
        #expect(noChange.mutation == .none)
        #expect(spy.sources.count == calls)
        let historicCalls = spy.sources.filter { $0.contains("historic") }.count
        controller.synchronize(storage: storage, prompt: "live", response: open + "\nlet b = 2",
                               isTerminal: false)
        controller.synchronize(storage: storage, prompt: "live", response: open + "\nlet b = 2\n```",
                               isTerminal: true)
        #expect(storage.string.contains("let b = 2"))
        #expect(controller.isFinalized)
        #expect(spy.sources.filter { $0.contains("historic") }.count == historicCalls)
        #expect(controller.answer(at: 0) == "historic 0")
        #expect(controller.answer(at: (storage.string as NSString).range(of: "historic 1").location)
                == "historic 1")
        _ = controller.refreshTypography(
            .init(), storage: storage, planner: &planner, input: state,
            promptPrefix: NSAttributedString(), drawPair: draw)
        let body = (storage.string as NSString).range(of: "historic 0")
        #expect((storage.attribute(.font, at: body.location, effectiveRange: nil) as? NSFont)?.pointSize
                == NSFont.systemFontSize)
    }

    @Test(arguments: [0, 1, 3])
    func replayInsertsContextBreakAtExactBoundary(_ boundary: Int) {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()
        var planner = TranscriptSyncPlanner()
        let state = input(UUID(), history: 3, boundary: boundary)
        func draw(_ i: Int) {
            controller.synchronize(storage: storage, prompt: "Q\(i)", response: "A\(i)", isTerminal: true)
        }
        controller.synchronizeHistory(storage: storage, planner: &planner, input: state, drawPair: draw)
        for size in [AppTextSize.largest, .standard] {
            controller.refreshTypography(.init(size), storage: storage, planner: &planner, input: state,
                                         promptPrefix: NSAttributedString(), drawPair: draw)
            let string = storage.string as NSString
            let marker = string.range(of: "Earlier turns").location
            if boundary > 0 { #expect(marker > string.range(of: "A\(boundary - 1)").location) }
            if boundary < 3 { #expect(marker < string.range(of: "Q\(boundary)").location) }
            #expect(storage.string.components(separatedBy: "Earlier turns").count == 2)
        }
    }

    @Test(arguments: [true, false])
    func finalizedMathSelectionAndCopySurviveReflow(progressive: Bool) throws {
        let environment = ["TURBO_FIELDFARE_PROGRESSIVE_RENDER": progressive ? "1" : "0"]
        let controller = InstructionTranscriptDocumentController(environment: environment)
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 400, height: 100))
        let view = TranscriptTextView(frame: scroll.bounds)
        scroll.documentView = view
        let storage = try #require(view.textStorage)
        var planner = TranscriptSyncPlanner()
        let state = input(UUID(), history: 0)
        controller.synchronizeHistory(storage: storage, planner: &planner, input: state) { _ in }
        let answer = "Before $x^2$ after.\n\nTail cafe\u{301}."
        controller.synchronize(storage: storage, prompt: "math", response: answer, isTerminal: true)
        let baseline = storage.string
        let oldAttachment = try #require(attachments(storage).first)
        let selection = (baseline as NSString).range(of: "Before \u{fffc} after.")
        #expect(selection.location != NSNotFound)
        view.setSelectedRange(selection)
        let position = TranscriptReadingPosition(textView: view, scrollView: scroll, followsPrefill: false)
        let copied = TranscriptPlainText.string(of: storage.attributedSubstring(from: selection))
        let update = controller.refreshTypography(
            .init(.largest), storage: storage, planner: &planner, input: state,
            promptPrefix: NSAttributedString(), drawPair: { _ in })
        #expect(update.replaced == nil)
        position.restore(textView: view, scrollView: scroll, replacing: update.replaced)
        #expect(view.selectedRange() == selection)
        #expect(TranscriptPlainText.string(of: storage.attributedSubstring(from: view.selectedRange())) == copied)
        #expect(storage.string == baseline)
        #expect(controller.isFinalized)
        #expect(controller.answer(at: selection.location) == answer)
        let newAttachment = try #require(attachments(storage).first)
        #expect(newAttachment.bounds.height > oldAttachment.bounds.height)
        #expect(newAttachment.latexSource == oldAttachment.latexSource)
    }

    @Test func prefillDotsAndLatePhotoPrefixKeepChosenSize() throws {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()
        var planner = TranscriptSyncPlanner()
        let state = input(UUID(), history: 0)
        controller.synchronizeHistory(storage: storage, planner: &planner, input: state) { _ in }
        controller.synchronize(storage: storage, prompt: "look", response: "", isTerminal: false,
                               showsPrefillPlaceholder: true)
        controller.advancePrefillAnimation(storage: storage)
        controller.advancePrefillAnimation(storage: storage)
        let old = storage.string
        controller.refreshTypography(.init(.largest), storage: storage, planner: &planner, input: state,
                                     promptPrefix: NSAttributedString(), drawPair: { _ in })
        #expect(storage.string == old)
        #expect(controller.showsPrefillPlaceholder)
        let photo = NSTextAttachment()
        photo.bounds = NSRect(x: 0, y: 0, width: 100, height: 60)
        controller.synchronize(storage: storage, prompt: "look", response: "", isTerminal: false,
                               showsPrefillPlaceholder: true,
                               promptPrefix: NSAttributedString(attachment: photo),
                               promptPrefixIdentifier: "photo")
        #expect(photo.bounds.size == NSSize(width: 100, height: 60))
        let word = (storage.string as NSString).range(of: "look")
        #expect((storage.attribute(.font, at: word.location, effectiveRange: nil) as? NSFont)?.pointSize
                == NSFont.systemFontSize * 2)
    }

    @Test(arguments: [false, true])
    func realTextViewPreservesReadingAnchorAndSelection(atBottom: Bool) throws {
        let scroll = TranscriptScrollView(frame: NSRect(x: 0, y: 0, width: 360, height: 150))
        let text = NSTextView(frame: scroll.bounds)
        text.isVerticallyResizable = true
        text.isHorizontallyResizable = false
        text.autoresizingMask = [.width]
        text.textContainer?.widthTracksTextView = true
        scroll.documentView = text
        let controller = InstructionTranscriptDocumentController()
        let storage = try #require(text.textStorage)
        var planner = TranscriptSyncPlanner()
        let state = input(UUID(), history: 0)
        controller.synchronizeHistory(storage: storage, planner: &planner, input: state) { _ in }
        let response = (0..<50).map { "Paragraph \($0): the quick brown fox jumps over the dog." }
            .joined(separator: "\n\n")
        controller.synchronize(storage: storage, prompt: "read", response: response, isTerminal: true)
        let manager = try #require(text.layoutManager)
        let container = try #require(text.textContainer)
        manager.ensureLayout(for: container)
        let selected = (storage.string as NSString).range(of: "Paragraph 8")
        text.setSelectedRange(selected)
        if atBottom {
            text.scrollToEndOfDocument(nil)
        } else {
            let glyph = manager.glyphIndexForCharacter(at: selected.location)
            let y = manager.lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil).minY
                + text.textContainerOrigin.y + 3
            scroll.contentView.scroll(to: NSPoint(x: 0, y: y))
        }
        for size in [AppTextSize.largest, .standard] {
            let position = TranscriptReadingPosition(textView: text, scrollView: scroll, followsPrefill: false)
            let update = controller.refreshTypography(
                .init(size), storage: storage, planner: &planner, input: state,
                promptPrefix: NSAttributedString(), drawPair: { _ in })
            position.restore(textView: text, scrollView: scroll, replacing: update.replaced)
            var follow = TranscriptScrollFollow()
            if atBottom { follow.beginRun() }
            follow.recordScroll(origin: scroll.contentView.bounds.minY, documentHeight: text.bounds.height)
            scroll.didRestoreReadingPosition = {
                follow.recordScroll(origin: scroll.contentView.bounds.minY, documentHeight: text.bounds.height)
            }
            // The composer preference resizes the viewport after typography has
            // restored the anchor. Without resize restoration, the newest text
            // falls below the viewport and subsequent streaming stops following.
            scroll.setFrameSize(NSSize(width: 360, height: size == .largest ? 74 : 150))
            let follows = follow.shouldScrollToBottom(
                origin: scroll.contentView.bounds.minY, documentHeight: text.bounds.height)
            #expect(follows == atBottom)
            scroll.didRestoreReadingPosition = nil
            #expect(text.selectedRange() == selected)
            if atBottom {
                #expect(scroll.contentView.bounds.maxY >= text.bounds.maxY - 24)
            } else {
                let glyph = manager.glyphIndexForCharacter(at: selected.location)
                let expected = manager.lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil).minY
                    + text.textContainerOrigin.y + 3
                #expect(abs(scroll.contentView.bounds.minY - expected) < 1)
            }
        }
    }

    @Test(arguments: ["Equation $x^", "Equation $$x^"])
    func unfinishedEquationStaysRawUntilFinalized(_ prefix: String) throws {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()
        var planner = TranscriptSyncPlanner()
        let state = input(UUID(), history: 0)
        controller.synchronizeHistory(storage: storage, planner: &planner, input: state) { _ in }
        controller.synchronize(storage: storage, prompt: "math", response: prefix, isTerminal: false)
        let original = storage.string
        controller.refreshTypography(.init(.largest), storage: storage, planner: &planner, input: state,
                                     promptPrefix: NSAttributedString(), drawPair: { _ in })
        #expect(storage.string == original)
        #expect(attachments(storage).isEmpty)
        #expect(!controller.isFinalized)
        controller.synchronize(storage: storage, prompt: "math",
                               response: prefix + (prefix.contains("$$") ? "2$$" : "2$"),
                               isTerminal: true)
        let equation = try #require(attachments(storage).first)
        #expect(equation.bounds.height > 0)
        storage.enumerateAttribute(.attachment, in: NSRange(location: 0, length: storage.length)) { value, range, _ in
            guard value is MathAttachment else { return }
            #expect((storage.attribute(.font, at: range.location, effectiveRange: nil) as? NSFont)?.pointSize
                    == NSFont.systemFontSize * 2)
        }
    }

    @Test(arguments: [false, true], [AppTextSize.standard, .largest])
    func readableMixedContentFrames(dark: Bool, size: AppTextSize) throws {
        let source = """
        ## Readable at a larger size

        A response with **emphasis**, a [link](https://example.com), and $x^2 + y^2$.

        - First item
        - Second item

        | Setting | Choice |
        | --- | --- |
        | Text size | \(size.label) |

        ```swift
        let readable = true
        ```
        """
        let result = ResponseMarkdownRenderer(typography: .init(size)).render(source)
        let image = try TranscriptFrameRenderer.image(result.attributedString, width: 530, dark: dark)
        #expect(image.size.height < TranscriptFrameRenderer.maximumHeight)
        try TranscriptFrameRenderer.record(image, named: "text-size-\(size.rawValue)-\(dark ? "dark" : "light")")
    }

    private func attachments(_ string: NSAttributedString) -> [MathAttachment] {
        var result: [MathAttachment] = []
        string.enumerateAttribute(.attachment, in: NSRange(location: 0, length: string.length)) { value, _, _ in
            if let value = value as? MathAttachment { result.append(value) }
        }
        return result
    }
}

@MainActor private final class CountingTypographyRenderer: TranscriptBlockRendering {
    private let renderer: ResponseMarkdownRenderer
    var sources: [String] = []
    init(_ typography: ConversationTypography) {
        renderer = ResponseMarkdownRenderer(typography: typography)
    }
    func render(_ source: String, typesetsMath: Bool) -> ResponseMarkdownRenderer.Result {
        sources.append(source)
        return renderer.render(source, typesetsMath: typesetsMath)
    }
    func blockSeparator(trailingNewlines: Int) -> NSAttributedString {
        renderer.blockSeparator(trailingNewlines: trailingNewlines)
    }
    func streamingCodeAttributes() -> [NSAttributedString.Key: Any] {
        renderer.streamingCodeAttributes()
    }
}
