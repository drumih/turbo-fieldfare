import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@MainActor
@Suite struct ResponseMarkdownRendererTests {
    @Test func rendersSupportedMarkdownWithNativeAttributes() throws {
        let source = """
        # Heading

        A **bold** and *italic* sentence with ~~obsolete~~ text, `inlineCode`, and a [link](https://example.com).

        - first
        - second

        > quoted text

        ```swift
        let answer = 42
        ```

        ---
        """

        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text.contains("Heading"))
        #expect(text.contains("bold"))
        #expect(text.contains("italic"))
        #expect(text.contains("•\tfirst\n•\tsecond"))
        #expect(text.contains("│\tquoted text"))
        #expect(text.contains("let answer = 42"))
        #expect(text.contains("────────────────"))
        #expect(!text.contains("**"))
        #expect(!text.contains("```"))

        let linkRange = (text as NSString).range(of: "link")
        #expect(result.attributedString.attribute(.link, at: linkRange.location,
                                                  effectiveRange: nil) == nil)
        let linkColor = result.attributedString.attribute(
            .foregroundColor, at: linkRange.location, effectiveRange: nil) as? NSColor
        #expect(linkColor?.isEqual(NSColor.linkColor) == true)
        #expect(result.attributedString.attribute(.underlineStyle,
                                                  at: linkRange.location,
                                                  effectiveRange: nil) as? Int
            == NSUnderlineStyle.single.rawValue)

        let codeRange = (text as NSString).range(of: "inlineCode")
        let codeFont = try #require(result.attributedString.attribute(
            .font, at: codeRange.location, effectiveRange: nil) as? NSFont)
        #expect(codeFont.fontDescriptor.symbolicTraits.contains(.monoSpace))
        #expect(result.attributedString.attribute(
            .backgroundColor, at: codeRange.location, effectiveRange: nil) != nil)

        let strikeRange = (text as NSString).range(of: "obsolete")
        #expect(result.attributedString.attribute(
            .strikethroughStyle, at: strikeRange.location, effectiveRange: nil) != nil)
    }

    @Test func unfinishedFenceFallsBackToExactRawText() {
        let source = "Before\n\n```python\nprint('unfinished')"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(result.usedFallback)
        #expect(result.attributedString.string == source)
    }

    @Test func unsupportedHTMLAndImagesStayReadableAsRawText() {
        let renderer = ResponseMarkdownRenderer()
        let samples = [
            "<div>Never execute this</div>",
            "![remote](https://example.com/image.png)",
        ]

        for source in samples {
            let result = renderer.render(source)
            #expect(result.usedFallback)
            #expect(result.attributedString.string == source)
        }
    }

    @Test func markdownTableRendersAsTableNotRawFallback() throws {
        let source = """
        | Action | Code |
        | :--- | :--- |
        | **Create** | `x = 1` |
        | Remove | `del x` |
        """
        let result = ResponseMarkdownRenderer().render(source)
        let attr = result.attributedString
        let text = attr.string

        #expect(!result.usedFallback)
        #expect(text.contains("Action"))
        #expect(text.contains("Create"))
        #expect(text.contains("x = 1"))
        #expect(!text.contains("|"))   // pipe delimiters consumed
        #expect(!text.contains("**"))  // bold delimiters consumed

        // Cells are laid out with NSTextTable blocks.
        let cell = (text as NSString).range(of: "Create")
        let paragraph = try #require(
            attr.attribute(.paragraphStyle, at: cell.location, effectiveRange: nil) as? NSParagraphStyle)
        #expect(paragraph.textBlocks.first is NSTextTableBlock)
    }

    @Test func streamingClosesOpenCodeFenceSoTheBlockRendersWhileArriving() {
        let renderer = ResponseMarkdownRenderer()
        let partial = "Here is code:\n\n```swift\nlet answer = 42"

        let strict = renderer.render(partial)
        #expect(strict.usedFallback)
        #expect(strict.attributedString.string == partial)

        let streamed = renderer.render(partial, streaming: true)
        #expect(!streamed.usedFallback)
        #expect(streamed.attributedString.string.contains("let answer = 42"))
        #expect(!streamed.attributedString.string.contains("```"))
    }

    @Test func streamingLeavesBalancedMarkdownIdenticalToStrictRender() {
        let renderer = ResponseMarkdownRenderer()
        let source = "A **bold** point with `code` and a closed block:\n\n```\ndone\n```"

        let strict = renderer.render(source)
        let streamed = renderer.render(source, streaming: true)

        #expect(!streamed.usedFallback)
        #expect(streamed.attributedString.string == strict.attributedString.string)
    }

    @Test func codeWithAngleBracketsRendersInsteadOfRawFallback() {
        let renderer = ResponseMarkdownRenderer()
        let samples = [
            "Use `Array<Int>` and check x > 0.",
            "```swift\nfor i in 2..<n {}\n```\n\n> note",
            "```cpp\n#include <stdio.h>\nint main() { return 0; }\n```",
            "```ts\nfunction id<T>(x: T): T { return x }\n```",
        ]
        for source in samples {
            #expect(!renderer.render(source).usedFallback, "\(source)")
        }
    }

    @Test func codeBlockCarriesContainerMarkerMonoFontAndKeywordColor() throws {
        let result = ResponseMarkdownRenderer().render("```swift\nfunc f() {}\n```")
        let attr = result.attributedString
        let text = attr.string as NSString
        #expect(!result.usedFallback)

        let codeStart = text.range(of: "func f()").location
        #expect(attr.attribute(
            TranscriptCodeStyle.codeBlockAttribute,
            at: codeStart, effectiveRange: nil) != nil)

        let font = try #require(attr.attribute(.font, at: codeStart, effectiveRange: nil) as? NSFont)
        #expect(font.fontDescriptor.symbolicTraits.contains(.monoSpace))

        let keywordColor = attr.attribute(
            .foregroundColor, at: text.range(of: "func").location, effectiveRange: nil) as? NSColor
        #expect(keywordColor?.isEqual(TranscriptCodeStyle.keyword) == true)
    }

    @Test func codeBlockLinesAreSingleSpaced() {
        let result = ResponseMarkdownRenderer().render("```swift\nlet a = 1\nlet b = 2\n```")
        #expect(result.attributedString.string.contains("let a = 1\nlet b = 2"))
        #expect(!result.attributedString.string.contains("let a = 1\n\nlet b = 2"))
    }

    @Test func pythonHashCommentsAreColoredAsComments() {
        let result = ResponseMarkdownRenderer().render("```python\nx = 1  # note\n```")
        let text = result.attributedString.string as NSString
        let color = result.attributedString.attribute(
            .foregroundColor, at: text.range(of: "# note").location, effectiveRange: nil) as? NSColor
        #expect(color?.isEqual(TranscriptCodeStyle.comment) == true)
    }

    @Test func latexRemainsReadableText() {
        let source = "Cosine is $\\frac{u \\cdot v}{||u|| ||v||}$."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string.contains("\\frac"))
        #expect(result.attributedString.string.contains("\\cdot"))
    }

    @Test func boldOnlyModelHeadingStaysOnItsOwnLine() {
        let source = "**Origins**\nFieldfares arrive from northern Europe."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string
            == "Origins\n\nFieldfares arrive from northern Europe.")
    }
}

@MainActor
@Suite struct InstructionTranscriptDocumentControllerTests {
    @Test func appAccentMatchesProductRGB() {
        let color = TurboFieldfareMacTheme.accentNSColor
            .usingColorSpace(.sRGB)
        #expect(color != nil)
        #expect(abs((color?.redComponent ?? 0) - 106.0 / 255.0) < 0.000_001)
        #expect(abs((color?.greenComponent ?? 0) - 186.0 / 255.0) < 0.000_001)
        #expect(abs((color?.blueComponent ?? 0) - 113.0 / 255.0) < 0.000_001)
    }

    @Test func rebuildsThenAppendsOnlyNewResponseSuffix() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()

        let first = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "Hel",
            isTerminal: false)
        let second = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "Hello",
            isTerminal: false)

        #expect(first.mutation == .rebuilt)
        #expect(second.mutation == .appended)
        #expect(storage.string == "You\nExplain this\n\nAnswer\nHello")
        #expect(storage.string.components(separatedBy: "Answer").count == 2)
        let answerRange = (storage.string as NSString).range(of: "Answer")
        let answerColor = storage.attribute(
            .foregroundColor,
            at: answerRange.location,
            effectiveRange: nil) as? NSColor
        #expect(answerColor?.isEqual(TurboFieldfareMacTheme.accentNSColor) == true)
        #expect(controller.response == "Hello")
    }

    @Test func streamingRendersMarkdownLiveInsteadOfRawText() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()

        let first = controller.synchronize(
            storage: storage,
            prompt: "Explain",
            response: "# Title",
            isTerminal: false)
        #expect(first.mutation == .rebuilt)
        #expect(!controller.isFinalized)
        #expect(storage.string == "You\nExplain\n\nAnswer\nTitle")

        let second = controller.synchronize(
            storage: storage,
            prompt: "Explain",
            response: "# Title\n\nA **bold** line",
            isTerminal: false)
        #expect(second.mutation == .appended)
        #expect(!controller.isFinalized)
        #expect(storage.string.contains("bold"))
        #expect(!storage.string.contains("**"))
        #expect(!storage.string.contains("# Title"))
    }

    @Test func streamingCodeBlockRendersBeforeClosingFenceArrives() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()

        let open = controller.synchronize(
            storage: storage,
            prompt: "Write code",
            response: "```swift\nlet x = 1",
            isTerminal: false)

        #expect(open.mutation == .rebuilt)
        #expect(!controller.isFinalized)
        #expect(storage.string.contains("let x = 1"))
        #expect(!storage.string.contains("```"))
    }

    @Test func animatedPrefillPlaceholderIsPresentationOnlyAndFirstResponseRemovesIt() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()

        let prefilling = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "",
            isTerminal: false,
            showsPrefillPlaceholder: true)

        #expect(prefilling.mutation == .rebuilt)
        #expect(controller.showsPrefillPlaceholder)
        #expect(storage.string == "You\nExplain this\n\nAnswer\nProcessing your prompt")
        #expect(controller.response.isEmpty)
        #expect(controller.assistantRange.length == 0)

        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt.."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt..."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt"))

        let responding = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "Hello",
            isTerminal: false,
            showsPrefillPlaceholder: true)

        #expect(responding.mutation == .rebuilt)
        #expect(!controller.showsPrefillPlaceholder)
        #expect(storage.string == "You\nExplain this\n\nAnswer\nHello")
        #expect(!storage.string.contains("Processing your prompt"))
        #expect((storage.string as NSString).substring(with: responding.assistantRange)
            == "Hello")
    }

    @Test func processingAnimationPolicyStopsForTextAndTerminalStates() {
        #expect(InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: false, requested: true))
        #expect(!InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "First token", isTerminal: false, requested: true))
        #expect(!InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: true, requested: true))
        #expect(!InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: false, requested: false))
    }

    @Test func promptChangeOrResponseResetRebuildsWithoutStaleBytes() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        _ = controller.synchronize(
            storage: storage, prompt: "Old", response: "Long response", isTerminal: false)

        let result = controller.synchronize(
            storage: storage, prompt: "New", response: "Short", isTerminal: false)

        #expect(result.mutation == .rebuilt)
        #expect(storage.string == "You\nNew\n\nAnswer\nShort")
        #expect(!storage.string.contains("Old"))
        #expect(!storage.string.contains("Long response"))
    }

    @Test func terminalUpdateFormatsOnlyAssistantRangeAndKeepsRawResponse() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        _ = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "**Bold answer**",
            isTerminal: false)

        let result = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "**Bold answer**",
            isTerminal: true)

        #expect(result.mutation == .finalized)
        #expect(controller.isFinalized)
        #expect(controller.response == "**Bold answer**")
        #expect(storage.string == "You\nQuestion\n\nAnswer\nBold answer")
        #expect((storage.string as NSString).substring(with: result.assistantRange)
            == "Bold answer")

        let unchanged = storage.copy() as! NSAttributedString
        let repeated = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "**Bold answer**",
            isTerminal: true)
        #expect(repeated.mutation == .none)
        #expect(storage.isEqual(to: unchanged))
    }

    @Test func returningToStreamingAfterFinalizeStillRendersMarkdown() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        _ = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "Partial **answer**",
            isTerminal: true)
        #expect(storage.string.hasSuffix("Partial answer"))

        let result = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "Partial **answer**",
            isTerminal: false)
        #expect(result.mutation == .rebuilt)
        #expect(!controller.isFinalized)
        // Streaming renders Markdown live, so a reopened stream shows the
        // formatted text rather than the raw ** delimiters.
        #expect(storage.string.hasSuffix("Partial answer"))
        #expect(!storage.string.contains("**"))
    }

    @Test func truncatedCodeBlockRendersAtTerminalInsteadOfRevertingToRaw() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        let partial = "```cpp\nkernel void matmul() {}"  // ends inside an open fence
        let complete = partial + "\n```"

        let first = controller.synchronize(
            storage: storage,
            prompt: "Write a Metal kernel",
            response: partial,
            isTerminal: true)
        #expect(first.mutation == .finalized)
        // A response that ends inside a fence stays a rendered code block at end of
        // turn rather than reverting to raw ``` text.
        #expect(!storage.string.contains("```"))
        #expect(storage.string.contains("kernel void matmul() {}"))

        let updated = controller.synchronize(
            storage: storage,
            prompt: "Write a Metal kernel",
            response: complete,
            isTerminal: true)

        #expect(updated.mutation == .finalized)
        #expect(controller.response == complete)
        #expect((storage.string as NSString).substring(with: updated.assistantRange)
            == "kernel void matmul() {}\n")
        #expect(!storage.string.contains("```"))
    }

    @Test func streamingThenTerminalKeepsUnterminatedCodeBlockFormatted() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        let text = "See:\n\n```swift\nlet x = 1"  // no closing fence

        _ = controller.synchronize(
            storage: storage, prompt: "Q", response: text, isTerminal: false)
        #expect(!storage.string.contains("```"))  // rendered while streaming

        _ = controller.synchronize(
            storage: storage, prompt: "Q", response: text, isTerminal: true)
        #expect(!storage.string.contains("```"))  // still rendered at terminal — no revert
        let ns = storage.string as NSString
        #expect(storage.attribute(
            TranscriptCodeStyle.codeBlockAttribute,
            at: ns.range(of: "let x").location, effectiveRange: nil) != nil)
    }

    @Test func selectionRangesClampToCurrentStorage() {
        let ranges = InstructionTranscriptDocumentController.clampedRanges([
            NSRange(location: 3, length: 20),
            NSRange(location: 50, length: 2),
        ], toLength: 10)

        #expect(ranges == [
            NSRange(location: 3, length: 7),
            NSRange(location: 10, length: 0),
        ])
    }

    @Test func commonPrefixStopsAtFirstCharacterOrAttributeDifference() {
        let mono = [NSAttributedString.Key.font: NSFont.monospacedSystemFont(
            ofSize: NSFont.systemFontSize, weight: .regular)]
        let plain = [NSAttributedString.Key.font: NSFont.systemFont(ofSize: NSFont.systemFontSize)]

        // Same text, attributes diverge at index 5 → common prefix is 5.
        let a = NSMutableAttributedString(string: "abcde", attributes: plain)
        a.append(NSAttributedString(string: "fgh", attributes: plain))
        let b = NSMutableAttributedString(string: "abcde", attributes: plain)
        b.append(NSAttributedString(string: "fgh", attributes: mono))
        #expect(InstructionTranscriptDocumentController.commonPrefixLength(a, b) == 5)

        // Pure append shares the whole shorter string.
        let short = NSAttributedString(string: "let x = 1", attributes: mono)
        let long = NSAttributedString(string: "let x = 1\nlet y = 2", attributes: mono)
        #expect(InstructionTranscriptDocumentController.commonPrefixLength(short, long) == 9)
    }
}
