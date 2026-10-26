import AppKit
import Testing
import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

@MainActor
@Suite struct TranscriptCodeCopyTests {
    static func blocks(_ text: NSAttributedString) -> [(TranscriptCodeBlock, Int)] {
        var result: [(TranscriptCodeBlock, Int)] = []
        text.enumerateAttribute(.transcriptCodeBlock, in: NSRange(location: 0, length: text.length)) { value, range, _ in
            if let block = value as? TranscriptCodeBlock { result.append((block, range.location)) }
        }
        return result
    }

    static func bodies(_ text: NSAttributedString) -> [String] {
        blocks(text).compactMap { TranscriptCodeBlock.body(at: $0.1, in: text, matching: $0.0) }
    }

    @Test(arguments: [
        ("```swift\nlet a = 1\n```", "let a = 1\n"),
        ("~~~text\n\tλ = e\u{301}\n\n  next\n~~~", "\tλ = e\u{301}\n\n  next\n"),
        ("    let a = 1", "let a = 1"),
        ("    let a = 1\n", "let a = 1\n"),
        ("> ```swift\n> let a = 1\n> ```", "let a = 1\n"),
        ("- item\n\n  ```\n  first\n    second\n  ```", "first\n  second\n"),
        ("```\r\nfirst\r\nsecond\r\n```", "first\nsecond\n"),
    ])
    func semanticCodePreservesExactBody(_ source: String, _ expected: String) {
        let rendered = ResponseMarkdownRenderer().render(source)
        #expect(!rendered.usedFallback)
        #expect(Self.bodies(rendered.attributedString) == [expected])
        let finalized = ProgressiveTranscriptRenderTests.finalized(source)
        #expect(Self.bodies(finalized) == [expected])
    }

    @Test func syntheticNewlineTracksStreamingAndClosingFenceRetraction() throws {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        var identity: TranscriptCodeBlock?
        for (source, expected) in [
            ("```\na", "a"), ("```\na\r", "a\n"), ("```\na\r\nb", "a\nb"),
            ("```\na\r\nb\n`", "a\nb\n`"), ("```\na\r\nb\n```", "a\nb\n"),
            ("```\na\r\nb\n```\n\nProse", "a\nb\n"),
        ] {
            controller.synchronize(storage: storage, prompt: "prompt", response: source, isTerminal: false)
            let block = try #require(Self.blocks(storage).first?.0)
            if let identity { #expect(identity === block) } else { identity = block }
            #expect(Self.bodies(storage) == [expected])
        }
        controller.synchronize(storage: storage, prompt: "prompt", response: controller.response, isTerminal: true)
        controller.sealTurn(storage: storage)
        controller.synchronize(storage: storage, prompt: "next", response: "```\nx", isTerminal: false)
        #expect(Self.bodies(storage) == ["a\nb\n", "x"])
        #expect(Self.blocks(storage).first?.0 === identity)
        controller.synchronize(storage: storage, prompt: "next", response: "```\nx", isTerminal: true)
        #expect(Self.bodies(storage) == ["a\nb\n", "x"])
    }

    @Test func nestedCodeKeepsIdentityAcrossStreamingAndPromotion() throws {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        var first: TranscriptCodeBlock?
        for source in ["> ```\n> first", "> ```\n> first\n> second", "> ```\n> first\n> second\n> ```", "> ```\n> first\n> second\n> ```\n\nDone"] {
            controller.synchronize(storage: storage, prompt: "p", response: source, isTerminal: false)
            let block = try #require(Self.blocks(storage).first?.0)
            if let first { #expect(first === block) } else { first = block }
        }
    }

    @Test func nestedAutoClosureDoesNotInventClipboardNewline() {
        let (_, storage) = ProgressiveTranscriptRenderTests.stream("> ```\n> x")
        #expect(Self.bodies(storage) == ["x"])
    }

    @Test func emptyInlineQuotedAndRawFallbackTextHaveNoTargets() {
        for source in ["`inline`", "> quote", "```\n```", "```\nunclosed", "<div>raw</div>", "- ```\n  x\n  ```"] {
            #expect(Self.blocks(ResponseMarkdownRenderer().render(source).attributedString).isEmpty)
        }
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController(environment: ["TURBO_FIELDFARE_PROGRESSIVE_RENDER": "0"])
        controller.synchronize(storage: storage, prompt: "p", response: "```\nx\n```", isTerminal: false)
        #expect(Self.blocks(storage).isEmpty)
        controller.synchronize(storage: storage, prompt: "p", response: "```\nx\n```", isTerminal: true)
        #expect(Self.bodies(storage) == ["x\n"])
    }

    static func host(_ text: NSAttributedString, width: CGFloat = 560) throws -> (NSWindow, TranscriptTextView) {
        let view = TranscriptTextView(frame: NSRect(x: 0, y: 0, width: width, height: 500))
        view.isEditable = false
        view.textContainerInset = NSSize(width: 0, height: 4)
        view.textContainer?.lineFragmentPadding = 0
        let window = NSWindow(contentRect: view.bounds, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        view.textStorage?.setAttributedString(text)
        view.layoutManager?.ensureLayout(for: try #require(view.textContainer))
        view.refreshCodeCopyButtons()
        return (window, view)
    }

    @Test(arguments: [5, 10, 20, 200])
    func tallCodeKeepsItsVisibleHeader(_ lines: Int) throws {
        let source = "```\n" + Array(repeating: "line", count: lines).joined(separator: "\n") + "\n```"
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 400, height: 180))
        let view = TranscriptTextView(frame: scroll.bounds)
        view.isVerticallyResizable = true
        view.textContainer?.widthTracksTextView = true
        scroll.documentView = view
        let window = NSWindow(contentRect: scroll.bounds, styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = scroll
        defer { window.contentView = nil }
        view.textStorage?.setAttributedString(ProgressiveTranscriptRenderTests.finalized(source))
        let layout = try #require(view.layoutManager)
        let container = try #require(view.textContainer)
        layout.ensureLayout(for: container)
        view.setFrameSize(NSSize(width: 400, height: max(180, layout.usedRect(for: container).height)))
        view.refreshCodeCopyButtons()
        // A viewport-truncated attribute range cannot resolve TextKit's cell bounds.
        #expect(view.codeCopyButtons.count == 1)
        if lines >= 20 {
            scroll.contentView.scroll(to: NSPoint(x: 0, y: 150))
            view.refreshCodeCopyButtons()
            #expect(view.codeCopyButtons.isEmpty)
        }
    }

    @Test func cachedCodeHasDistinctIdentitiesInSeparateTurnsAndAfterReset() throws {
        let (window, view) = try Self.host(NSAttributedString())
        defer { window.contentView = nil }
        let storage = try #require(view.textStorage)
        let controller = InstructionTranscriptDocumentController()
        let source = "> ```\n> same\n> ```"
        controller.synchronize(storage: storage, prompt: "first", response: source, isTerminal: true)
        view.refreshCodeCopyButtons()
        let obsolete = try #require(view.codeCopyButtons.values.first)
        controller.sealTurn(storage: storage)
        controller.synchronize(storage: storage, prompt: "second", response: source, isTerminal: true)
        view.refreshCodeCopyButtons()
        #expect(Self.bodies(storage) == ["same\n", "same\n"])
        #expect(Set(Self.blocks(storage).map { $0.0.id }).count == 2)
        #expect(view.codeCopyButtons.count == 2)
        controller.resetTranscript(storage: storage)
        controller.synchronize(storage: storage, prompt: "first", response: source, isTerminal: true)
        view.refreshCodeCopyButtons()
        #expect(!view.codeCopyButtons.keys.contains(obsolete.block.id))
        view.writeCodeToPasteboard = { _ in Issue.record("retired occurrence copied new transcript"); return true }
        obsolete.performClick(nil)
    }

    @Test(arguments: [
        ("    abc<partial", "abc<partial"),
        ("    x\n    ---", "x\n---"),
        ("    [label](https://exa", "[label](https://exa"),
        (">     abc<partial", "abc<partial"),
        ("> ```\n> x", "x"),
        ("> ```\r\n> x\r\n", "x\n"),
        ("> ```\r\n> x", "x"),
        ("> ```\r\n> x\r\n> ```", "x\n"),
        ("- item\n\n  ```\n  x\n  ```", "x\n"),
        ("    ```", "```"),
        ("    ```\n    literal\n    ```", "```\nliteral\n```"),
        ("> paragraph\n>\n>     abc<partial", "abc<partial"),
        ("- item\n\n      abc<partial", "abc<partial"),
        (">     $x$", "$x$"),
        (">     $x$\n>", "$x$\n"),
        ("> - list\n> >     abc<partial", "abc<partial"),
        ("> - list\n> >     [label](url", "[label](url"),
        ("> - item\n>\n> ```\n> x\n> ```", "x\n"),
        ("> - item\n>\n> ~~~\n> x\n> ~~~", "x\n"),
        (">     x\n>", "x\n"),
        (">     x\n> ", "x\n"),
        (">     x\r\n> \t", "x\n"),
        ("> ```\n> x\u{2028}> ```", "x\u{2028}> ```"),
        ("> ```\n> x\u{2028}> ```\n> abc<partial", "x\u{2028}> ```\nabc<partial"),
        ("> ```\n> x\u{2029}> ```", "x\u{2029}> ```"),
        ("> ```\n> x\u{0085}> ```", "x\u{0085}> ```"),
        ("> ```\n> > ```\n> abc<partial", "> ```\nabc<partial"),
        ("> ```\n> > ```\n> abc\n> ```", "> ```\nabc\n"),
        ("> - item\n> > ```\n> > x\n> > ```", "x\n"),
        ("> ```\n> x\n>", "x\n"),
        ("> ```\n> x\n> \n>", "x\n\n"),
        (">     x\n>\n> ", "x\n"),
        ("    x\n\t>", "x\n>"),
        ("    x\n    >", "x\n>"),
    ])
    func sourceCodeSurvivesTailRepairsFinalizationAndHistory(_ source: String, _ expected: String) throws {
        let (window, view) = try Self.host(NSAttributedString())
        defer { window.contentView = nil }
        let storage = try #require(view.textStorage)
        let controller = InstructionTranscriptDocumentController()
        func assertNativeCopy() throws {
            view.refreshCodeCopyButtons()
            #expect(Self.bodies(storage) == [expected])
            let button = try #require(view.codeCopyButtons.values.first)
            var copied: String?
            view.writeCodeToPasteboard = { copied = $0; return true }
            button.performClick(nil)
            #expect(copied == expected)
        }
        for terminal in [false, true] {
            controller.synchronize(storage: storage, prompt: "p", response: source, isTerminal: terminal)
            try assertNativeCopy()
        }
        controller.resetTranscript(storage: storage)
        controller.synchronize(storage: storage, prompt: "p", response: source, isTerminal: true)
        try assertNativeCopy()
    }

    @Test(arguments: [AppTextSize.standard, .largest])
    func visibleHeaderDoesNotRequireVisibleCodeGlyphs(_ size: AppTextSize) throws {
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 400, height: 100))
        let view = TranscriptTextView(frame: scroll.bounds)
        view.isVerticallyResizable = true
        view.textContainer?.widthTracksTextView = true
        view.textContainer?.lineFragmentPadding = 0
        scroll.documentView = view
        let window = NSWindow(contentRect: scroll.bounds, styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = scroll
        defer { window.contentView = nil }
        let source = String(repeating: "Paragraph.\n\n", count: 10) + "```\nline\nline\n```"
        let text = ResponseMarkdownRenderer(typography: .init(size)).render(source).attributedString
        let storage = try #require(view.textStorage)
        storage.setAttributedString(text)
        let layout = try #require(view.layoutManager)
        let container = try #require(view.textContainer)
        layout.ensureLayout(for: container)
        view.setFrameSize(NSSize(width: 400, height: layout.usedRect(for: container).height + 8))
        let block = try #require(Self.blocks(storage).first)
        let style = try #require(storage.attribute(.paragraphStyle, at: block.1, effectiveRange: nil) as? NSParagraphStyle)
        var range = NSRange()
        _ = storage.attribute(.transcriptCodeBlock, at: block.1, longestEffectiveRange: &range, in: NSRange(location: 0, length: storage.length))
        let cell = try #require(style.textBlocks.first)
        let rect = layout.boundsRect(for: cell, glyphRange: layout.glyphRange(forCharacterRange: range, actualCharacterRange: nil))
            .offsetBy(dx: view.textContainerOrigin.x, dy: view.textContainerOrigin.y)
        for points: CGFloat in [0, 3, 6, 10, 20, 30] {
            scroll.contentView.scroll(to: NSPoint(x: 0, y: rect.minY - 100 + points * CGFloat(size.scale)))
            view.refreshCodeCopyButtons()
            #expect(view.codeCopyButtons.count == (points > 4 ? 1 : 0), "\(points) points of the cell are visible at \(size)")
        }
    }

    @Test func emptyQuotedCodeHasNoControlUntilContentArrives() throws {
        let (window, view) = try Self.host(NSAttributedString())
        defer { window.contentView = nil }
        let storage = try #require(view.textStorage)
        let controller = InstructionTranscriptDocumentController()
        for terminal in [false, true] {
            controller.resetTranscript(storage: storage)
            controller.synchronize(storage: storage, prompt: "p", response: "> ```\n> ", isTerminal: terminal)
            view.refreshCodeCopyButtons()
            #expect(view.codeCopyButtons.isEmpty)
            #expect(Self.bodies(storage).isEmpty)
        }
        controller.resetTranscript(storage: storage)
        controller.synchronize(storage: storage, prompt: "p", response: "> ```\n> ", isTerminal: false)
        controller.synchronize(storage: storage, prompt: "p", response: "> ```\n> x", isTerminal: false)
        view.refreshCodeCopyButtons()
        let button = try #require(view.codeCopyButtons.values.first)
        var copied: String?
        view.writeCodeToPasteboard = { copied = $0; return true }
        button.performClick(nil)
        #expect(copied == "x")
    }

    @Test func nativeButtonsCopyCurrentBlockAndReportOnlySuccessfulWrites() throws {
        let source = "```\nidentical\n```\n\n```\nidentical\n```"
        let rendered = ProgressiveTranscriptRenderTests.finalized(source)
        let (window, view) = try Self.host(rendered)
        defer { window.contentView = nil }
        #expect(view.codeCopyButtons.count == 2)
        let buttons = view.codeCopyButtons.values.sorted { $0.characterIndex < $1.characterIndex }
        let first = try #require(buttons.first)
        let second = try #require(buttons.last)
        #expect(first.block.id != second.block.id)
        let copyImage = try #require(first.image)
        #expect(first.title.isEmpty)
        #expect(first.toolTip == "Copy code")
        let board = TranscriptTextViewTests.pasteboard("code-block")
        defer { board.releaseGlobally() }
        view.writeCodeToPasteboard = { board.setString($0, forType: .string) }
        view.setSelectedRange(NSRange(location: 0, length: 2))
        let selection = view.selectedRanges
        first.performClick(nil)
        #expect(board.string(forType: .string) == "identical\n")
        let copiedImage = try #require(first.image)
        #expect(copiedImage !== copyImage)
        #expect(first.title.isEmpty)
        #expect(first.accessibilityLabel() == "Code copied")
        #expect(first.toolTip == "Code copied")
        #expect(view.selectedRanges == selection)
        #expect(view.string == rendered.string)
        view.writeCodeToPasteboard = { _ in false }
        first.performClick(nil)
        #expect(first.image != nil)
        #expect(first.image !== copiedImage)
        #expect(first.title.isEmpty)
        #expect(first.accessibilityLabel()?.contains("failed") == true)
        view.textStorage?.setAttributedString(NSAttributedString(string: "new chat"))
        view.writeCodeToPasteboard = { _ in Issue.record("stale control wrote clipboard"); return true }
        second.performClick(nil)
        #expect(view.codeCopyButtons.isEmpty)
    }

    nonisolated static let layoutCases: [(AppTextSize, CGFloat, Bool)] = [AppTextSize.standard, .largest].flatMap { size in
        [CGFloat(260), CGFloat(700)].flatMap { width in [false, true].map { (size, width, $0) } }
    }

    @Test(arguments: layoutCases)
    func controlsFitCodeHeaderAndFrames(_ size: AppTextSize, _ width: CGFloat, _ dark: Bool) throws {
        let text = ResponseMarkdownRenderer(typography: .init(size)).render("```swift\nlet value = 42\n```\n\nText with $x^2$.").attributedString
        let (window, view) = try Self.host(text, width: width)
        defer { window.contentView = nil }
        let button = try #require(view.codeCopyButtons.values.first)
        let layout = try #require(view.layoutManager)
        let glyphs = layout.glyphRange(forCharacterRange: NSRange(location: button.characterIndex, length: 1), actualCharacterRange: nil)
        let codeRect = layout.boundingRect(forGlyphRange: glyphs, in: try #require(view.textContainer)).offsetBy(dx: view.textContainerOrigin.x, dy: view.textContainerOrigin.y)
        #expect(button.frame.maxY <= codeRect.minY)
        #expect(button.frame.minX >= 0)
        #expect(button.frame.maxX <= width)
        let style = try #require(view.textStorage?.attribute(.paragraphStyle, at: button.characterIndex, effectiveRange: nil) as? NSParagraphStyle)
        let cell = try #require(style.textBlocks.first)
        var codeRange = NSRange()
        _ = try #require(view.textStorage).attribute(.transcriptCodeBlock, at: button.characterIndex, longestEffectiveRange: &codeRange, in: NSRange(location: 0, length: text.length))
        let cellRect = layout.boundsRect(for: cell, glyphRange: layout.glyphRange(forCharacterRange: codeRange, actualCharacterRange: nil))
        #expect(cellRect.width > 0)
        #expect(cellRect.maxX <= width)
        #expect(button.frame.width == button.frame.height)
        #expect(button.frame.width == 24 * CGFloat(size.scale))
        #expect(button.title.isEmpty)
        #expect(button.imagePosition == .imageOnly)
        #expect(!button.isBordered)
        try TranscriptFrameRenderer.record(text, named: "code-copy-\(size)-\(Int(width))-\(dark).png", width: width, dark: dark)
    }

    @Test func visibleControlsStayBoundedWhileReadingAboveALiveAnswer() throws {
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 560, height: 240))
        let view = TranscriptTextView(frame: scroll.bounds)
        view.isVerticallyResizable = true
        view.textContainer?.widthTracksTextView = true
        view.textContainer?.lineFragmentPadding = 0
        scroll.documentView = view
        let window = NSWindow(contentRect: scroll.bounds, styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = scroll
        defer { window.contentView = nil }
        let storage = try #require(view.textStorage)
        let controller = InstructionTranscriptDocumentController()
        let history = (0..<200).map { "```\nblock \($0)\n```" }.joined(separator: "\n\n")
        controller.synchronize(storage: storage, prompt: "history", response: history, isTerminal: true)
        controller.sealTurn(storage: storage)
        let layout = try #require(view.layoutManager)
        let container = try #require(view.textContainer)
        layout.ensureLayout(for: container)
        view.setFrameSize(NSSize(width: 560, height: layout.usedRect(for: container).height + 8))
        view.refreshCodeCopyButtons()
        let firstIDs = Set(view.codeCopyButtons.keys)
        #expect(!firstIDs.isEmpty)
        #expect(firstIDs.count < 8)
        let clock = ContinuousClock()
        var timings: [Duration] = []
        let source = try TranscriptCorpus.source("large-code-block")
        for prefix in TranscriptStreamingTests.ticks(source) {
            timings.append(clock.measure {
                controller.synchronize(storage: storage, prompt: "live", response: prefix, isTerminal: false)
                view.refreshCodeCopyButtons()
            })
            #expect(Set(view.codeCopyButtons.keys) == firstIDs)
            #expect(scroll.contentView.bounds.origin == .zero)
        }
        let timing = try #require(TranscriptStreamingTests.timing(timings))
        print("CODE_COPY_VIEW head_us=\(timing.headDecileMean * 1e6) tail_us=\(timing.tailDecileMean * 1e6) ratio=\(timing.ratio)")
        #expect(timing.ratio < 3)
        layout.ensureLayout(for: container)
        view.setFrameSize(NSSize(width: 560, height: layout.usedRect(for: container).height + 8))
        scroll.contentView.scroll(to: NSPoint(x: 0, y: view.frame.height - 400))
        view.refreshCodeCopyButtons()
        #expect(Set(view.codeCopyButtons.keys).isDisjoint(with: firstIDs))
        #expect(view.codeCopyButtons.count < 8)
    }

    @Test func accessibilityIncludesCodeButtonsWithoutReplacingTextSemantics() throws {
        let (window, view) = try Self.host(ResponseMarkdownRenderer().render("```\nx\n```\n\n```\ny\n```").attributedString)
        defer { window.contentView = nil }
        // NSTextView omits native subviews from its default accessibility
        // children, making visible buttons unreachable to VoiceOver and AXPress.
        let children = (view.accessibilityChildren() ?? []).compactMap { $0 as? NSAccessibilityProtocol }
        #expect(children.count == 2)
        #expect(children.allSatisfy { $0.accessibilityRole() == .button })
        #expect(children.map { $0.accessibilityLabel() } == ["Copy code", "Copy code"])
        #expect(view.accessibilityRole() == .textArea)
    }

    @Test func selectionAcrossCodeAndMathKeepsItsTextAndRichFlavors() throws {
        let rendered = ResponseMarkdownRenderer().render("```\nlet x = 1\n```\n\nValue $x^2$.").attributedString
        let view = TranscriptTextViewTests.view(rendered)
        view.setSelectedRange(NSRange(location: 0, length: rendered.length))
        let board = TranscriptTextViewTests.pasteboard("code-and-math")
        defer { board.releaseGlobally() }
        view.copySelection(to: board)
        #expect(board.string(forType: .string) == TranscriptPlainText.string(of: rendered))
        #expect(board.string(forType: .string)?.contains("Copy") == false)
        #expect(board.data(forType: .rtf) != nil)
        #expect(board.data(forType: .rtfd) != nil)
    }

    @Test func detachedTranscriptRemovesControlsAndStopsRefreshing() async throws {
        let parent = NSView(frame: NSRect(x: 0, y: 0, width: 560, height: 500))
        let view = TranscriptTextView(frame: parent.bounds)
        parent.addSubview(view)
        let storage = try #require(view.textStorage)
        storage.setAttributedString(ResponseMarkdownRenderer().render("```\nx\n```").attributedString)
        view.refreshCodeCopyButtons()
        #expect(view.codeCopyButtons.count == 1)
        view.removeFromSuperview()
        #expect(view.codeCopyButtons.isEmpty)
        storage.append(ResponseMarkdownRenderer().render("```\ny\n```").attributedString)
        try await Task.sleep(for: .milliseconds(50))
        #expect(view.codeCopyButtons.isEmpty)
        parent.addSubview(view)
        try await Task.sleep(for: .milliseconds(50))
        #expect(view.codeCopyButtons.count == 2)
    }
}
