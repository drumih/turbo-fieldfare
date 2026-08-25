import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@MainActor
@Suite struct ProgressiveTranscriptRenderTests {
    static func controller(
        renderer: any TranscriptBlockRendering = ResponseMarkdownRenderer(),
        progressive: Bool = true
    ) -> InstructionTranscriptDocumentController {
        InstructionTranscriptDocumentController(
            renderer: renderer,
            environment: ["TURBO_FIELDFARE_PROGRESSIVE_RENDER": progressive ? "1" : "0"])
    }

    /// What the transcript shows when an answer finishes: a fresh controller
    /// handed the whole answer as terminal. Finalize closes the open block in
    /// place and never re-renders the message, so this is the reference every
    /// other path is measured against.
    static func finalized(_ source: String) -> NSAttributedString {
        let storage = NSMutableAttributedString()
        let controller = controller()
        _ = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: source,
            isTerminal: true)
        return assistant(storage, controller)
    }

    static func stream(
        _ source: String,
        renderer: any TranscriptBlockRendering = ResponseMarkdownRenderer(),
        step: Int = 1
    ) -> (controller: InstructionTranscriptDocumentController, storage: NSMutableAttributedString) {
        let storage = NSMutableAttributedString()
        let controller = controller(renderer: renderer)
        var seen = ""
        var index = source.startIndex
        while index < source.endIndex {
            let next = source.index(index, offsetBy: step, limitedBy: source.endIndex)
                ?? source.endIndex
            seen.append(contentsOf: source[index..<next])
            index = next
            controller.synchronize(
                storage: storage,
                prompt: "Explain this",
                response: seen,
                isTerminal: false)
        }
        return (controller, storage)
    }

    static func assistant(
        _ storage: NSMutableAttributedString,
        _ controller: InstructionTranscriptDocumentController
    ) -> NSAttributedString {
        storage.attributedSubstring(from: controller.assistantRange)
    }

    // MARK: - What the reader sees

    @Test func completedBlocksAreStyledWhileTheAnswerIsStillArriving() {
        let (controller, storage) = Self.stream("## Title\n\n**Bold** text.\n\nStill writ")
        let text = Self.assistant(storage, controller).string
        #expect(text == "Title\n\nBold text.\n\nStill writ")
        #expect(!controller.isFinalized)
        // The completed blocks are above the open one and are not rewritten.
        #expect(controller.tailRange.length == "Still writ".utf16.count)
        #expect((storage.string as NSString).substring(with: controller.tailRange)
            == "Still writ")
    }

    @Test func openBlockIsAutoClosedForDisplayWithoutTouchingTheResponse() {
        let (controller, storage) = Self.stream("Intro.\n\nA **bold sta")
        #expect(Self.assistant(storage, controller).string == "Intro.\n\nA bold sta")
        #expect(controller.response == "Intro.\n\nA **bold sta")
    }

    /// The cancel shape from the corpus: the reader stops a code answer in the
    /// middle of the fence and has to be left with readable code, not with the
    /// fence markers and an unstyled listing.
    @Test func cancellingInsideAFenceLeavesReadableCode() throws {
        let source = try TranscriptCorpus.source("unclosed-fence")
        let (controller, storage) = Self.stream(source, step: 8)
        let text = Self.assistant(storage, controller).string
        #expect(text.hasPrefix("Here is the kernel so far:\n\n"))
        #expect(!text.contains("```"))
        #expect(text.contains("kernel void reduce"))
        // The listing is inside a code container, which is what makes it read
        // as code rather than as a paragraph of prose.
        let code = (text as NSString).range(of: "kernel void reduce")
        let style = storage.attribute(
            .paragraphStyle,
            at: controller.assistantRange.location + code.location,
            effectiveRange: nil) as? NSParagraphStyle
        #expect(style?.textBlocks.count == 1)

        // Finishing there is what a cancel is. The whole-message render falls
        // back to raw source for an unclosed fence, so the listing used to
        // lose its box and show its markers the moment the reader stopped it.
        controller.synchronize(
            storage: storage, prompt: "Explain this", response: source, isTerminal: true)
        let finalized = Self.assistant(storage, controller).string
        #expect(!finalized.contains("```"))
        #expect(finalized.contains("kernel void reduce"))
        let finalStyle = storage.attribute(
            .paragraphStyle,
            at: controller.assistantRange.location
                + (finalized as NSString).range(of: "kernel void reduce").location,
            effectiveRange: nil) as? NSParagraphStyle
        #expect(finalStyle?.textBlocks.count == 1)
    }

    @Test func mathInTheOpenBlockStaysAsSourceAndTypesetsWhenTheBlockCloses() {
        let typesetter = FakeMathTypesetter()
        let renderer = ResponseMarkdownRenderer(typesetter: typesetter)
        let storage = NSMutableAttributedString()
        let controller = Self.controller(renderer: renderer)
        let answer = "Step 1.\n$$x = 1$$"

        var seen = ""
        for character in answer {
            seen.append(character)
            controller.synchronize(
                storage: storage, prompt: "Solve", response: seen, isTerminal: false)
        }
        #expect(typesetter.calls.isEmpty)
        #expect(storage.string.hasSuffix("$$x = 1$$"))

        // The blank line that closes the block is what typesets it.
        controller.synchronize(
            storage: storage, prompt: "Solve", response: answer + "\n\n", isTerminal: false)
        #expect(typesetter.calls.map(\.latex) == ["x = 1"])
        #expect(storage.string.hasSuffix("Step 1.\n\n\u{FFFC}"))
    }

    /// The progressive counterpart of the raw path's multi-byte delta test: a
    /// CJK or combining-mark delta has to land in the drawn document exactly
    /// once, and the tail keeps math as source rather than mangling it.
    @Test func multiByteDeltasReachTheDrawnDocumentExactly() {
        let answer = "\u{4E2D}\u{6587}: caf\u{E9} \u{30C6}\u{30B9}\u{30C8} $x^2$ done"
        let (controller, storage) = Self.stream(answer)
        #expect(storage.string == "You\nExplain this\n\nAnswer\n" + answer)
        #expect((storage.string as NSString).substring(with: controller.assistantRange)
            == answer)
        #expect(controller.response == answer)
    }

    // MARK: - Cost

    @Test func eachCompletedBlockIsRenderedExactlyOnce() {
        let renderer = RecordingBlockRenderer()
        let source = "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n\nThird."
        _ = Self.stream(source, renderer: renderer)

        #expect(renderer.completedSources == ["# Title", "First paragraph.", "Second paragraph."])
        // The last block is still open at the end of the stream, so it is only
        // ever rendered as a tail.
        #expect(renderer.tailSources.last == "Third.")
    }

    @Test func theOpenBlockIsRenderedOnEveryTickAndNothingElseIs() {
        let renderer = RecordingBlockRenderer()
        let source = "Done.\n\nabcd"
        _ = Self.stream(source, renderer: renderer)

        #expect(renderer.completedSources == ["Done."])
        // One tail render per tick from the first character of the open block.
        #expect(renderer.tailSources.suffix(4) == ["a", "ab", "abc", "abcd"])
    }

    /// A fenced block never reaches the markdown pass while it streams: it is
    /// the one block with no size bound, and re-parsing it per tick is the
    /// quadratic cost the streaming timing gate exists to catch.
    @Test func fencedBlocksDoNotGoThroughTheMarkdownPass() {
        let renderer = RecordingBlockRenderer()
        let source = "Intro.\n\n```swift\nlet a = 1\nlet b = 2\n```\n\nOutro."
        let (_, storage) = Self.stream(source, renderer: renderer)

        #expect(renderer.calls.allSatisfy { !$0.source.contains("```") })
        #expect(storage.string.contains("let a = 1\nlet b = 2"))
        #expect(!storage.string.contains("```"))
    }

    @Test func aClosedFenceKeepsTheBytesItWasStreamedInto() {
        let renderer = RecordingBlockRenderer()
        let source = "```swift\nlet a = 1\n```\n\nAfter."
        let (controller, storage) = Self.stream(source, renderer: renderer)
        #expect(Self.assistant(storage, controller).string == "let a = 1\n\nAfter.")
        #expect(renderer.completedSources == [])
    }

    /// The closer the tail adds has to carry the container's prefix. Written
    /// flush left it lands outside the quote, so the quote's own fence is
    /// still open when the renderer looks: the whole tail falls back to raw
    /// source and the listing keeps its markers on screen.
    @Test(arguments: [
        "> Run this:\n> ```bash\n> brew install x\n> ```\n\nDone.",
        "1. Install:\n\n   ```bash\n   brew install x\n   ```\n\nDone.",
    ])
    func aFenceInsideAContainerNeverShowsItsMarkersWhileStreaming(_ source: String) {
        let storage = NSMutableAttributedString()
        let controller = Self.controller()
        var seen = ""
        for character in source {
            seen.append(character)
            controller.synchronize(
                storage: storage, prompt: "Ask", response: seen, isTerminal: false)
            #expect(!storage.string.contains("```"), "markers shown at \(seen.debugDescription)")
        }
        #expect(Self.assistant(storage, controller).string.contains("brew install x"))
    }

    /// Block renders are memoised so a rebuild mid-answer does not re-render
    /// the whole transcript. The memo is keyed by position as well as text:
    /// handing two identical tables the same render would hand them the same
    /// `NSTextTable`, and TextKit draws that as one table with eight rows.
    @Test func twoIdenticalTablesGetTheirOwnContainers() {
        let table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        let (controller, storage) = Self.stream("\(table)\n\n\(table)\n\nEnd.", step: 4)
        let digest = AttributedStringDigest(Self.assistant(storage, controller))
        var containers: Set<String> = []
        for run in digest.runs {
            guard let marker = run.paragraph.range(of: #"block#\d+"#, options: .regularExpression)
            else {
                continue
            }
            containers.insert(String(run.paragraph[marker]))
        }
        #expect(containers.count == 8, "two 2x2 tables are eight cells, not four")
    }

    // MARK: - Equality with the finalize render

    /// The whole-document render stays the oracle: the per-block finalize has
    /// to reach the same document for every answer whose blocks the message
    /// pass agrees about. The exceptions are pinned rather than tolerated —
    /// they are the difference this pass exists to make.
    @Test(arguments: TranscriptCorpus.fixtures)
    func perBlockFinalizeMatchesTheWholeDocumentRender(_ fixture: String) throws {
        let source = try TranscriptCorpus.source(fixture)
        let result = ResponseMarkdownRenderer().render(source)
        let drawn = Self.finalized(source)

        if TranscriptCorpus.wholeRenderFallbacks.contains(fixture) {
            // One block trips a message-wide gate and takes the whole answer
            // to raw source with it. Per block, only that block is raw.
            #expect(result.usedFallback)
            #expect(result.attributedString.string == source)
            #expect(drawn.string != source, "\(fixture) finalized wholly raw")
            return
        }
        if TranscriptCorpus.perBlockLimitations.contains(fixture) { return }

        let finalDigest = AttributedStringDigest(result.attributedString)
        let drawnDigest = AttributedStringDigest(drawn)
        #expect(finalDigest.text == drawnDigest.text, "\(fixture) finalized different text")
        guard finalDigest != drawnDigest else { return }

        let expectedRuns = finalDigest.perCharacter
        let actualRuns = drawnDigest.perCharacter
        try #require(expectedRuns.count == actualRuns.count)
        for (left, right) in zip(expectedRuns, actualRuns) where left != right {
            #expect(
                left.text == "\n",
                "\(fixture) differs on \(left.text.debugDescription), not on a separator")
            var probe = left
            probe.paragraph = right.paragraph
            #expect(probe == right, "\(fixture) differs in more than the paragraph style")
            #expect(
                left.paragraph.replacingOccurrences(of: "align=1", with: "align=4")
                    == right.paragraph,
                "\(fixture) differs in more than the alignment of a separator")
        }
        for dark in [false, true] {
            let expected = try TranscriptFrameRenderer.image(result.attributedString, dark: dark)
            let actual = try TranscriptFrameRenderer.image(drawn, dark: dark)
            #expect(
                Self.png(expected) == Self.png(actual),
                "\(fixture) draws differently in \(dark ? "dark" : "light")")
        }
    }

    static func png(_ image: NSImage) -> Data? {
        (image.representations.first as? NSBitmapImageRep)?
            .representation(using: .png, properties: [:])
    }

    /// What the controller actually draws while streaming, finished the way a
    /// real answer finishes. A reader who watched it arrive and a reader who
    /// opened it afterwards have to be looking at the same document, with no
    /// exceptions at all.
    @Test(arguments: TranscriptCorpus.fixtures)
    func streamedThenFinalizedMatchesAFreshTerminalRebuild(_ fixture: String) throws {
        let source = try TranscriptCorpus.source(fixture)
        let (controller, storage) = Self.stream(source, step: 8)
        controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: source,
            isTerminal: true)

        #expect(AttributedStringDigest(Self.assistant(storage, controller))
            == AttributedStringDigest(Self.finalized(source)),
            "\(fixture) streamed into a different document")
    }

    /// An image thumbnail landing mid-answer rebuilds the whole document. What
    /// it finalizes to must not depend on that having happened.
    @Test func aRebuildMidStreamStillFinalizesToTheSameDocument() throws {
        let source = try TranscriptCorpus.source("gemma-derivation-table")
        let storage = NSMutableAttributedString()
        let controller = Self.controller()
        var seen = ""
        var index = source.startIndex
        while index < source.endIndex {
            let next = source.index(index, offsetBy: 8, limitedBy: source.endIndex)
                ?? source.endIndex
            seen.append(contentsOf: source[index..<next])
            index = next
            let halfway = seen.count > source.count / 2
            controller.synchronize(
                storage: storage,
                prompt: "Explain this",
                response: seen,
                isTerminal: false,
                promptPrefix: halfway ? NSAttributedString(string: "[image]") : NSAttributedString(),
                promptPrefixIdentifier: halfway ? "image-1" : "")
        }
        controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: source,
            isTerminal: true,
            promptPrefix: NSAttributedString(string: "[image]"),
            promptPrefixIdentifier: "image-1")

        #expect(AttributedStringDigest(Self.assistant(storage, controller))
            == AttributedStringDigest(Self.finalized(source)))
    }

    /// The streamed render draws a fenced body from the raw bytes, so a
    /// finalize pass that promotes a `**Note**` line inside the fence changed
    /// the listing under the reader the moment the answer ended.
    @Test func aBoldLineInsideAFenceStreamsAndFinalizesToTheSameBytes() {
        let source = "Intro.\n\n```text\n**Note**\nkeep this line attached\n```\n\nAfter."
        let (controller, storage) = Self.stream(source, step: 4)
        let streamed = Self.assistant(storage, controller).string
        controller.synchronize(
            storage: storage, prompt: "Explain this", response: source, isTerminal: true)
        let finalized = Self.assistant(storage, controller).string

        #expect(streamed.contains("**Note**\nkeep this line attached"))
        #expect(finalized == streamed)
    }

    /// An empty fence renders nothing. The fence fast path used to require a
    /// non-empty body, so a rebuild sent "```\n```" through the markdown pass,
    /// which produces an empty document and falls back to raw source — the
    /// fence markers reappeared on screen where the stream had drawn nothing.
    @Test func anEmptyFenceDrawsNothingOnEveryPath() {
        let source = "Intro.\n\n```\n```\n\nAfter."
        let (streamedController, streamedStorage) = Self.stream(source, step: 3)
        let streamed = Self.assistant(streamedStorage, streamedController).string

        let rebuiltStorage = NSMutableAttributedString()
        let rebuiltController = Self.controller()
        _ = rebuiltController.synchronize(
            storage: rebuiltStorage, prompt: "Ask", response: source, isTerminal: false)
        let rebuilt = Self.assistant(rebuiltStorage, rebuiltController).string

        let finalStorage = NSMutableAttributedString()
        let finalController = Self.controller()
        _ = finalController.synchronize(
            storage: finalStorage, prompt: "Ask", response: source, isTerminal: true)
        let finalized = Self.assistant(finalStorage, finalController).string

        #expect(!streamed.contains("```"))
        #expect(rebuilt == streamed)
        #expect(finalized == streamed)
    }

    /// The separator count carried the previous block's trailing newlines only
    /// when the new block was newlines all the way down, so a fence whose body
    /// is one blank line left an extra paragraph gap behind it that the whole
    /// render does not have.
    @Test func aFenceWhoseBodyIsOneBlankLineKeepsOneGapOnEveryPath() {
        let source = "Intro.\n\n```\n\n```\n\nAfter."
        let whole = ResponseMarkdownRenderer().render(source).attributedString.string
        #expect(whole == "Intro.\n\n\nAfter.")

        let (streamedController, streamedStorage) = Self.stream(source, step: 3)
        #expect(Self.assistant(streamedStorage, streamedController).string == whole)

        let rebuiltStorage = NSMutableAttributedString()
        let rebuiltController = Self.controller()
        _ = rebuiltController.synchronize(
            storage: rebuiltStorage, prompt: "Ask", response: source, isTerminal: false)
        #expect(Self.assistant(rebuiltStorage, rebuiltController).string == whole)

        let finalStorage = NSMutableAttributedString()
        let finalController = Self.controller()
        _ = finalController.synchronize(
            storage: finalStorage, prompt: "Ask", response: source, isTerminal: true)
        #expect(Self.assistant(finalStorage, finalController).string == whole)
    }

    /// A tab is four columns wide, so stripping a one-space fence indent off it
    /// leaves three. The filter dropped the whole tab instead and the streamed
    /// listing lost its indentation until the answer finalized.
    @Test func aTabWiderThanTheFenceIndentKeepsTheColumnsItOwns() {
        let source = " ```\n\tfoo\n ```"
        let whole = ResponseMarkdownRenderer().render(source).attributedString.string
        #expect(whole == "   foo\n")

        let (controller, storage) = Self.stream(source, step: 2)
        #expect(Self.assistant(storage, controller).string.contains("   foo"))
    }

    /// A rebuild mid-answer draws the completed blocks with the separators the
    /// incremental path writes. Without them the prefix arrived glued —
    /// "One.Two." — and stayed that way until the answer finalized.
    @Test func aRebuildMidStreamKeepsTheSeparatorsBetweenCompletedBlocks() {
        let source = "One.\n\nTwo.\n\nThree is still writ"
        let (reference, referenceStorage) = Self.stream(source)
        let expected = Self.assistant(referenceStorage, reference).string
        #expect(expected == "One.\n\nTwo.\n\nThree is still writ")

        let storage = NSMutableAttributedString()
        let controller = Self.controller()
        var seen = ""
        for character in source {
            seen.append(character)
            controller.synchronize(
                storage: storage, prompt: "Ask", response: seen, isTerminal: false)
        }
        // What an image thumbnail landing mid-answer does: the prompt prefix
        // identifier changes and the whole document is rebuilt.
        let rebuilt = controller.synchronize(
            storage: storage,
            prompt: "Ask",
            response: source,
            isTerminal: false,
            promptPrefix: NSAttributedString(string: "[image]"),
            promptPrefixIdentifier: "image-1")
        #expect(rebuilt.mutation == .rebuilt)
        #expect(Self.assistant(storage, controller).string == expected)

        let finalized = controller.synchronize(
            storage: storage,
            prompt: "Ask",
            response: source,
            isTerminal: true,
            promptPrefix: NSAttributedString(string: "[image]"),
            promptPrefixIdentifier: "image-1")
        #expect(finalized.mutation == .finalized)
        #expect(Self.assistant(storage, controller).string == expected)
    }

    /// `hasPrefix` compares by canonical equivalence, so a prefix that arrives
    /// recomposed passed the gate while the byte offsets the delta is cut at
    /// no longer matched: the append wrote the wrong characters onto a
    /// document drawn from different ones.
    @Test func aCanonicallyEqualButRewrittenPrefixForcesARebuild() {
        let storage = NSMutableAttributedString()
        let controller = Self.controller()
        _ = controller.synchronize(
            storage: storage, prompt: "Ask", response: "cafe\u{301}", isTerminal: false)

        let update = controller.synchronize(
            storage: storage, prompt: "Ask", response: "caf\u{E9} au lait", isTerminal: false)
        #expect(update.mutation == .rebuilt)
        #expect(Self.assistant(storage, controller).string == "caf\u{E9} au lait")

        // A byte-identical extension is still an append, not a rebuild.
        let appended = controller.synchronize(
            storage: storage, prompt: "Ask", response: "caf\u{E9} au lait now", isTerminal: false)
        #expect(appended.mutation != .rebuilt)
    }

    // MARK: - Finalize and resume

    /// Finalize closes the block that was still being written and leaves the
    /// blocks above it exactly as they were drawn.
    @Test func finalizeClosesTheOpenBlockAndLeavesTheRestInPlace() {
        let renderer = RecordingBlockRenderer()
        let storage = NSMutableAttributedString()
        let controller = Self.controller(renderer: renderer)
        let answer = "# Title\n\nA **bold** claim."
        _ = controller.synchronize(
            storage: storage, prompt: "Ask", response: answer, isTerminal: false)
        let drawnPrefix = controller.tailRange.location
        let final = controller.synchronize(
            storage: storage, prompt: "Ask", response: answer, isTerminal: true)

        #expect(final.mutation == .finalized)
        #expect(controller.isFinalized)
        #expect(storage.string == "You\nAsk\n\nAnswer\nTitle\n\nA bold claim.")
        // Only the open block was rewritten; the heading above it did not move.
        #expect(final.replaced?.previous.location ?? 0 >= drawnPrefix)
        // The heading is rendered once, as a completed block, and never again.
        #expect(renderer.completedSources.filter { $0 == "# Title" }.count == 1)
    }

    /// A terminal rebuild used to render every block once as a block and then
    /// the whole answer once more, so a finished answer cost N+2 passes.
    @Test func aTerminalRebuildRendersEachBlockOnceAndNeverTheTail() {
        let renderer = RecordingBlockRenderer()
        let storage = NSMutableAttributedString()
        let controller = Self.controller(renderer: renderer)
        _ = controller.synchronize(
            storage: storage, prompt: "Ask", response: "A.\n\nB.\n\nC.", isTerminal: true)

        #expect(renderer.completedSources == ["A.", "B.", "C."])
        #expect(renderer.tailSources.isEmpty)
    }

    /// An image thumbnail landing after the answer is finished rebuilds the
    /// document from blocks that are all in the memo already.
    @Test func aLatePromptPrefixOnAFinishedTurnRendersNothingNew() {
        let renderer = RecordingBlockRenderer()
        let storage = NSMutableAttributedString()
        let controller = Self.controller(renderer: renderer)
        let answer = "A.\n\nB.\n\nC."
        _ = controller.synchronize(
            storage: storage, prompt: "Ask", response: answer, isTerminal: true)
        let before = renderer.calls.count

        let rebuilt = controller.synchronize(
            storage: storage,
            prompt: "Ask",
            response: answer,
            isTerminal: true,
            promptPrefix: NSAttributedString(string: "[image]"),
            promptPrefixIdentifier: "image-1")

        #expect(rebuilt.mutation == .finalized)
        #expect(renderer.calls.count == before)
        #expect(storage.string.hasSuffix("A.\n\nB.\n\nC."))
    }

    /// A terminal response that extends a finalized one is extended, not
    /// rebuilt: the blocks already drawn are not rendered again.
    @Test func aFinalizedAnswerThatGrowsIsExtendedNotRebuilt() {
        let renderer = RecordingBlockRenderer()
        let storage = NSMutableAttributedString()
        let controller = Self.controller(renderer: renderer)
        _ = controller.synchronize(
            storage: storage, prompt: "Ask", response: "One.\n\nTwo is open", isTerminal: true)
        let grown = controller.synchronize(
            storage: storage, prompt: "Ask", response: "One.\n\nTwo is open now", isTerminal: true)

        #expect(grown.mutation == .finalized)
        #expect(renderer.completedSources.filter { $0 == "One." }.count == 1)
        #expect(Self.assistant(storage, controller).string == "One.\n\nTwo is open now")
    }

    @Test func resumingAfterFinalizeGoesBackToTheProgressiveDocument() {
        let storage = NSMutableAttributedString()
        let controller = Self.controller()
        let answer = "Partial **answer**"
        _ = controller.synchronize(
            storage: storage, prompt: "Ask", response: answer, isTerminal: true)
        #expect(storage.string.hasSuffix("Partial answer"))

        let resumed = controller.synchronize(
            storage: storage, prompt: "Ask", response: answer, isTerminal: false)
        #expect(resumed.mutation == .rebuilt)
        #expect(!controller.isFinalized)
        #expect(storage.string.hasSuffix("Partial answer"))
        #expect(controller.response == answer)

        let grown = controller.synchronize(
            storage: storage, prompt: "Ask", response: answer + " more", isTerminal: false)
        #expect(grown.mutation == .tailReplaced)
        #expect(storage.string.hasSuffix("Partial answer more"))
    }

    // MARK: - The flag

    @Test func theFlagOffRestoresRawStreamingAppends() {
        let storage = NSMutableAttributedString()
        let controller = Self.controller(progressive: false)
        let first = controller.synchronize(
            storage: storage, prompt: "Ask", response: "# Title\n\nBod", isTerminal: false)
        let second = controller.synchronize(
            storage: storage, prompt: "Ask", response: "# Title\n\nBody", isTerminal: false)

        #expect(first.mutation == .rebuilt)
        #expect(second.mutation == .appended)
        #expect(storage.string == "You\nAsk\n\nAnswer\n# Title\n\nBody")
        #expect(controller.tailRange == controller.assistantRange)
    }

    @Test func theFlagDefaultsToProgressiveWhenUnset() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController(environment: [:])
        _ = controller.synchronize(
            storage: storage, prompt: "Ask", response: "# Title\n\nBody", isTerminal: false)
        #expect(storage.string == "You\nAsk\n\nAnswer\nTitle\n\nBody")
    }

    // MARK: - Scroll follow

    /// A tail re-render must not pull a reader who has scrolled up back to the
    /// bottom; only the finalize restyle does that.
    @Test func tailReplacementFollowsTheSameScrollRuleAsAnAppend() {
        #expect(!InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: false, mutation: .tailReplaced))
        #expect(InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: true, mutation: .tailReplaced))
        #expect(InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: false, mutation: .finalized))
    }
}
