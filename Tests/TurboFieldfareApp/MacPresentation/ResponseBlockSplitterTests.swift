import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct ResponseBlockSplitterTests {
    static func texts(_ source: String) -> [String] {
        let split = ResponseBlockSplitter.split(source)
        return split.blocks.map { ResponseBlockSplitter.text($0.utf8Range, in: source) }
    }

    static func kinds(_ source: String) -> [ResponseBlockSplitter.Kind] {
        ResponseBlockSplitter.split(source).blocks.map(\.kind)
    }

    // MARK: - One block per shape

    @Test func fencedBlockKeepsItsBlankLinesAndInfoString() {
        let source = """
            Intro paragraph.

            ```swift
            struct A {}

            struct B {}
            ```

            Outro.
            """
        #expect(Self.texts(source) == [
            "Intro paragraph.",
            "```swift\nstruct A {}\n\nstruct B {}\n```",
            "Outro.",
        ])
        #expect(Self.kinds(source) == [.paragraph, .fencedCode, .paragraph])
    }

    @Test func tildeFenceClosesOnlyOnItsOwnMarker() {
        let source = "~~~text\n```\nstill inside\n~~~\n\nAfter."
        #expect(Self.texts(source) == ["~~~text\n```\nstill inside\n~~~", "After."])
    }

    /// The shape a model writes constantly: a lead-in sentence, then the fence
    /// on the very next line. Absorbing the fence into the paragraph makes the
    /// blank line inside the code commit that paragraph, turns the indented
    /// lines below into an indented-code block, and boxes the sentence that
    /// follows the closing fence along with them.
    @Test func aFenceLineInterruptsTheParagraphAboveIt() {
        let source = "Here is the fix:\n```swift\nfunc apply() {\n\n    return 1\n}\n```\nDone."
        #expect(Self.texts(source) == [
            "Here is the fix:",
            "```swift\nfunc apply() {\n\n    return 1\n}\n```",
            "Done.",
        ])
        #expect(Self.kinds(source) == [.paragraph, .fencedCode, .paragraph])
    }

    @Test func aTildeFenceInterruptsAParagraphTheSameWay() {
        let source = "Here is the fix:\n~~~swift\nlet a = 1\n~~~\n\nDone."
        #expect(Self.texts(source) == [
            "Here is the fix:",
            "~~~swift\nlet a = 1\n~~~",
            "Done.",
        ])
    }

    /// A blank line is a boundary everywhere except inside an open fence,
    /// where it is code.
    @Test func aBlankLineInsideAnOpenFenceDoesNotCommitTheBlock() {
        let source = "```swift\nfunc apply() {\n\n    return 1\n"
        let split = ResponseBlockSplitter.split(source)
        #expect(split.completed.isEmpty)
        #expect(split.open?.fence?.isClosed == false)
        #expect(ResponseBlockSplitter.text(
            (split.open?.fence?.bodyStart ?? 0)..<(split.open?.fence?.bodyEnd ?? 0),
            in: source) == "func apply() {\n\n    return 1\n")
    }

    /// A list owns the fences inside its items, so the interrupt rule stops at
    /// the list boundary.
    @Test func aFenceIndentedInsideAListStaysWithTheList() {
        let source = "- item one\n  ```swift\n  let a = 1\n  ```\n\nAfter.\n"
        #expect(Self.texts(source) == [
            "- item one\n  ```swift\n  let a = 1\n  ```",
            "After.",
        ])
        #expect(Self.kinds(source) == [.list, .paragraph])
    }

    /// CommonMark forbids a backtick inside a backtick fence's info string,
    /// so "```bash```" in a sentence is a code span. Reading it as an opener
    /// put the rest of the answer in a code box while it streamed.
    @Test func aBacktickInTheInfoStringIsNotAFenceLine() {
        let source = "```bash``` is the fence syntax.\n\nDone."
        #expect(Self.texts(source) == ["```bash``` is the fence syntax.", "Done."])
        #expect(Self.kinds(source) == [.paragraph, .paragraph])
        // The rule is the backtick fence's alone; a tilde fence takes any info.
        #expect(Self.kinds("~~~bash`\nbody\n~~~") == [.fencedCode])
    }

    /// The interrupt rule exempted every list without knowing where the item's
    /// content starts, so a fence written flush left under a bullet was
    /// absorbed into the list: the blank line inside the listing then committed
    /// it and the closing fence and the sentence after it drew as code.
    @Test func aFenceBeforeTheItemContentColumnInterruptsTheList() {
        let source = "- Install:\n```bash\nbrew install x\n\nbrew link x\n```\n\nDone.\n"
        #expect(Self.kinds(source) == [.list, .fencedCode, .paragraph])
        #expect(Self.texts(source) == [
            "- Install:",
            "```bash\nbrew install x\n\nbrew link x\n```",
            "Done.",
        ])
    }

    /// The same listing indented to the item's content column belongs to the
    /// item, blank line included, and the list stays one block.
    @Test func aFenceAtTheItemContentColumnStaysInsideTheItem() {
        let source = "1. Install:\n   ```bash\n   brew install x\n\n   brew link x\n   ```\n\nDone.\n"
        #expect(Self.kinds(source) == [.list, .paragraph])
        #expect(Self.texts(source).first?.hasSuffix("   ```") == true)
    }

    @Test func theItemContentColumnFollowsTheMarkerWidth() {
        // "10. " is four columns wide, so its listing is indented four.
        let wide = "10. Install:\n    ```bash\n    brew install x\n    ```\n\nDone.\n"
        #expect(Self.kinds(wide) == [.list, .paragraph])
        // Three is inside the marker, so the fence is a block of its own.
        let narrow = "10. Install:\n   ```bash\n   brew install x\n   ```\n\nDone.\n"
        #expect(Self.kinds(narrow) == [.list, .fencedCode, .paragraph])
    }

    /// A nested item does not move the column the outer item's listing is
    /// measured against; a sibling or an outer one does.
    @Test func onlyASiblingOrOuterItemReplacesTheContentColumn() {
        let nested = "- Outer\n  - Inner\n  ```bash\n  x\n  ```\n\nDone.\n"
        #expect(Self.kinds(nested) == [.list, .paragraph])
        let sibling = "  - Indented\n- Flush\n```bash\nx\n```\n\nDone.\n"
        #expect(Self.kinds(sibling) == [.list, .fencedCode, .paragraph])
    }

    /// An unclosed fence inside an item cannot hold the rest of the answer:
    /// a non-blank line indented below the column ends the item and the fence.
    @Test func aLessIndentedLineClosesAnInnerFence() {
        let source = "1. Install:\n   ```bash\n   brew install x\nDone.\n\nAfter.\n"
        #expect(Self.kinds(source) == [.list, .paragraph])
        #expect(Self.texts(source).last == "After.")
    }

    @Test func tableRowsAreOneBlock() {
        let source = "| A | B |\n| :--- | ---: |\n| 1 | 2 |\n\nAfter the table."
        #expect(Self.texts(source) == ["| A | B |\n| :--- | ---: |\n| 1 | 2 |", "After the table."])
        #expect(Self.kinds(source) == [.table, .paragraph])
    }

    /// Ordinals come from the parser, so a loose list has to reach it whole:
    /// splitting on the blank line between items would restart the numbering
    /// and separate the items by a paragraph gap.
    @Test func listAbsorbsBlankLinesNestingAndContinuationText() {
        let source = """
            1. First item
               continued on the next line

               A second paragraph inside the item

            2. Second item
                - nested bullet
                    - deeper bullet

            Ordinary paragraph.

            """
        let blocks = Self.texts(source)
        #expect(blocks.count == 2)
        #expect(blocks[0].hasPrefix("1. First item"))
        #expect(blocks[0].hasSuffix("    - deeper bullet"))
        #expect(blocks[0].contains("2. Second item"))
        #expect(blocks[1] == "Ordinary paragraph.")
        #expect(Self.kinds(source) == [.list, .paragraph])
    }

    /// A line that has not ended yet may still turn into a list item, so the
    /// list it would continue cannot be closed around it. The open tail then
    /// covers both, which costs nothing: it is one markdown pass over the same
    /// bytes the whole-document pass would have seen.
    @Test func openTailSpansAListAndTheUnfinishedLineAfterIt() {
        let source = "- one\n\nOrdinary par"
        let split = ResponseBlockSplitter.split(source)
        #expect(split.completed.isEmpty)
        #expect(split.open?.kind == .list)
        #expect(ResponseBlockSplitter.text(
            split.open?.utf8Range ?? 0..<0,
            in: source) == source)

        let terminated = ResponseBlockSplitter.split("- one\n\nOrdinary paragraph.\n")
        #expect(terminated.completed.count == 1)
        #expect(terminated.open?.kind == .paragraph)
    }

    @Test func listEndsWhenAnUnindentedParagraphFollowsABlankLine() {
        let source = "- one\n- two\n\nNot a list line.\n"
        #expect(Self.texts(source) == ["- one\n- two", "Not a list line."])
    }

    @Test func quoteAndHeadingAndDisplayMathAreTheirOwnBlocks() {
        let source = """
            ### Heading

            > quoted line one
            > quoted line two

            $$
            x = 1
            $$

            Tail paragraph.
            """
        #expect(Self.kinds(source) == [.heading, .quote, .displayMath, .paragraph])
        #expect(Self.texts(source)[2] == "$$\nx = 1\n$$")
    }

    /// The parser reads a `#` line and the text under it as a heading plus a
    /// paragraph whether or not a blank line separates them. Splitting there
    /// would be safe, but the blank-line rule is the one that also keeps
    /// `Title` above `---` in the same block, where splitting is not.
    @Test func blankLinesAreTheOnlyBoundary() {
        #expect(Self.texts("#### Checklist\n- [x] done\n- [ ] open")
            == ["#### Checklist\n- [x] done\n- [ ] open"])
        #expect(Self.texts("Title\n---\n\nBody") == ["Title\n---", "Body"])
        #expect(Self.kinds("Title\n---\n\nBody") == [.paragraph, .paragraph])
        #expect(Self.kinds("\n---\n\nBody") == [.thematicBreak, .paragraph])
    }

    @Test func carriageReturnsAreTolerated() {
        let source = "# CRLF\r\n\r\n```swift\r\nlet a = 1\r\n```\r\n\r\nEnd.\r\n"
        #expect(Self.texts(source) == ["# CRLF", "```swift\r\nlet a = 1\r\n```", "End."])
    }

    // MARK: - The open tail

    @Test func trailingBlockIsOpenUntilABlankLineFollowsIt() {
        let closed = ResponseBlockSplitter.split("One paragraph.\n\n")
        #expect(closed.open == nil)
        #expect(closed.completed.count == 1)

        let open = ResponseBlockSplitter.split("One paragraph.\n")
        #expect(open.completed.isEmpty)
        #expect(ResponseBlockSplitter.text(
            open.open?.utf8Range ?? 0..<0,
            in: "One paragraph.\n") == "One paragraph.")
    }

    /// A list can still grow across the blank line that already follows it, so
    /// it stays open where a paragraph would have been completed.
    @Test func trailingListStaysOpenAcrossItsBlankLine() {
        let split = ResponseBlockSplitter.split("- one\n\n")
        #expect(split.completed.isEmpty)
        #expect(split.open?.kind == .list)
    }

    @Test func unclosedFenceIsReportedWithItsBody() {
        let source = "```metal\nkernel void reduce() {\n"
        let split = ResponseBlockSplitter.split(source)
        #expect(split.completed.isEmpty)
        let fence = split.open?.fence
        #expect(fence?.isClosed == false)
        #expect(fence?.marker == UInt8(ascii: "`"))
        #expect(ResponseBlockSplitter.text(
            (fence?.bodyStart ?? 0)..<(fence?.bodyEnd ?? 0),
            in: source) == "kernel void reduce() {\n")
    }

    @Test func closedFenceReportsItsBodyWithoutTheClosingLine() {
        let source = "```swift\nlet a = 1\n```\n"
        let split = ResponseBlockSplitter.split(source)
        #expect(split.open == nil)
        let fence = split.completed.first?.fence
        #expect(fence?.isClosed == true)
        #expect(ResponseBlockSplitter.text(
            (fence?.bodyStart ?? 0)..<(fence?.bodyEnd ?? 0),
            in: source) == "let a = 1\n")
    }

    /// A closing fence that has not reached its newline yet could still turn
    /// into ```` ```json ````, so it closes the block for display without
    /// committing it.
    @Test func partialClosingFenceLineDoesNotCommitTheBlock() {
        let split = ResponseBlockSplitter.split("```swift\nlet a = 1\n```")
        #expect(split.completed.isEmpty)
        #expect(split.open?.fence?.isClosed == true)
    }

    @Test func aDisplayMathLineIsItsOwnBlockOpenOrClosed() {
        #expect(ResponseBlockSplitter.split("$$\\frac{a}{b}").open?.kind == .displayMath)
        #expect(ResponseBlockSplitter.split("$$\\frac{a}{b}$$").open?.kind == .displayMath)
        // A dollar pair inside a sentence is prose, not a display block.
        #expect(ResponseBlockSplitter.split("Costs \\$$5 today").open?.kind == .paragraph)
    }

    // MARK: - Incremental scanning

    /// The controller re-splits on every streaming tick and may not re-read the
    /// whole answer to do it. Resuming has to land on exactly the state a fresh
    /// scan would have produced, including the scanner position, for every
    /// prefix of every corpus fixture.
    @Test(arguments: TranscriptCorpus.fixtures)
    func resumedSplitMatchesAFullSplitForEveryPrefix(_ fixture: String) throws {
        let source = try TranscriptCorpus.source(fixture)
        var incremental = ResponseBlockSplitter.Split()
        var prefix = ""
        for character in source {
            prefix.append(character)
            incremental = ResponseBlockSplitter.split(prefix, resuming: incremental)
            let full = ResponseBlockSplitter.split(prefix)
            #expect(incremental == full, "\(fixture) diverged at \(prefix.count) characters")
            if incremental != full { return }
        }
        // Every byte of the answer is covered exactly once by the blocks.
        let blocks = incremental.blocks
        var cursor = 0
        for block in blocks {
            #expect(block.start >= cursor)
            #expect(block.end > block.start)
            cursor = block.end
        }
        #expect(cursor <= source.utf8.count)
    }
}
