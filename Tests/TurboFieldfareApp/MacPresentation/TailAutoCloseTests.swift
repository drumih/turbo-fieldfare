import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct TailAutoCloseTests {
    static let guardCharacter = String(TailAutoClose.setextGuard)

    // MARK: - Fences

    @Test func unclosedFenceGetsItsClosingLine() {
        #expect(TailAutoClose.close("```swift\nlet a = 1") == "```swift\nlet a = 1\n```")
        #expect(TailAutoClose.close("```swift\nlet a = 1\n") == "```swift\nlet a = 1\n```")
        #expect(TailAutoClose.close("~~~~text\nbody") == "~~~~text\nbody\n~~~~")
    }

    @Test func aLoneBacktickInProseIsNotAnOpener() {
        // "Press the ` key" is prose, not the start of a code span; closing it
        // briefly styles the rest of the sentence as code while it is the tail.
        let prose = "Press the ` key to open the console."
        #expect(TailAutoClose.close(prose) == prose)
        let pair = "The ` and ` keys are both literal here."
        #expect(TailAutoClose.close(pair) == pair)
    }

    @Test func aCodeLikeBacktickOpenerIsStillClosed() {
        #expect(TailAutoClose.close("Use `swift bui") == "Use `swift bui`")
        let closed = "Wrap it in `code ` with a padded closer."
        #expect(TailAutoClose.close(closed) == closed)
    }

    /// Counting the marker run alone made "```bash```" an unclosed fence, so
    /// the tail grew a closing line and the sentence turned into a code box.
    @Test func aBacktickInfoStringIsNotAnOpenFence() {
        let tail = "```bash``` is the fence syn"
        #expect(TailAutoClose.close(tail) == tail)
    }

    @Test func closedFenceIsLeftAlone() {
        let closed = "```swift\nlet a = 1\n```"
        #expect(TailAutoClose.close(closed) == closed)
    }

    /// Nothing inside a fence is markdown. A half-written `**` or a stray
    /// backtick in code must survive exactly as the model wrote it.
    @Test func markersInsideAFenceAreNeverBalanced() {
        let tail = "```bash\necho \"**\" `date` $$\ngrep -o 'a"
        #expect(TailAutoClose.close(tail) == tail + "\n```")
    }

    /// The closer was written flush left whatever container the fence was
    /// opened in, so the parser read it as prose after the item and the
    /// listing kept its markers on screen while it streamed.
    @Test(arguments: [
        ("1. Install:\n    ```bash\n    brew install x", "\n    ```"),
        ("> ```bash\n> brew install x", "\n> ```"),
        ("> 1. Install:\n>    ```bash\n>    brew install x", "\n>    ```"),
        ("- Install:\n  ~~~text\n  body", "\n  ~~~"),
    ])
    func aFenceInsideAContainerIsClosedInsideIt(_ probe: (String, String)) {
        #expect(TailAutoClose.close(probe.0) == probe.0 + probe.1,
                "\(probe.0.debugDescription)")
    }

    @Test func aClosedFenceInsideAnItemIsLeftAlone() {
        let closed = "1. Install:\n    ```bash\n    brew install x\n    ```"
        #expect(TailAutoClose.close(closed) == closed)
    }

    // MARK: - Inline code

    @Test func unterminatedInlineCodeIsClosedWithItsOwnRun() {
        #expect(TailAutoClose.close("Run `swift bui") == "Run `swift bui`")
        #expect(TailAutoClose.close("Use ``a ` b") == "Use ``a ` b``")
    }

    @Test func balancedInlineCodeIsLeftAlone() {
        #expect(TailAutoClose.close("Run `swift build` now") == "Run `swift build` now")
    }

    /// Emphasis that opened before an unterminated code span is still open, and
    /// its closer has to land outside the code, not inside it.
    @Test func closersNestInTheOrderTheyWereOpened() {
        #expect(TailAutoClose.close("**bold `cod") == "**bold `cod`**")
    }

    // MARK: - Emphasis

    @Test func unbalancedStrongAndEmphasisAreClosed() {
        #expect(TailAutoClose.close("A **bold sta") == "A **bold sta**")
        #expect(TailAutoClose.close("An *italic sta") == "An *italic sta*")
        #expect(TailAutoClose.close("An _italic sta") == "An _italic sta_")
        #expect(TailAutoClose.close("***both sta") == "***both sta***")
    }

    @Test func balancedEmphasisIsLeftAlone() {
        #expect(TailAutoClose.close("A **bold** word") == "A **bold** word")
        #expect(TailAutoClose.close("A *thin* word") == "A *thin* word")
    }

    /// The reader should never see the marker itself. Closing it instead would
    /// write `****`, which is a new run of literal asterisks rather than the
    /// emphasis the model is one keystroke away from opening.
    @Test func aMarkerWithNothingAfterItIsRemovedRatherThanClosed() {
        #expect(TailAutoClose.close("A sentence and **") == "A sentence and ")
        #expect(TailAutoClose.close("A sentence and *") == "A sentence and ")
        #expect(TailAutoClose.close("A sentence and `") == "A sentence and ")
        #expect(TailAutoClose.close("Half **bold** and *") == "Half **bold** and ")
    }

    /// Closing these would create structure the source does not have: a bullet
    /// would turn into bold text and a rule would turn into emphasis.
    @Test func listBulletsAndThematicBreaksAreNotEmphasisOpeners() {
        #expect(TailAutoClose.close("* first item") == "* first item")
        #expect(TailAutoClose.close("- one\n* two") == "- one\n* two")
        #expect(TailAutoClose.close("Above\n\n***") == "Above\n\n***")
    }

    @Test func intrawordUnderscoresAreNotEmphasis() {
        #expect(TailAutoClose.close("Call snake_case_name now") == "Call snake_case_name now")
        #expect(TailAutoClose.close("Call snake_case_na") == "Call snake_case_na")
    }

    @Test func escapedMarkersAreNotOpeners() {
        #expect(TailAutoClose.close("Literal \\*star and more") == "Literal \\*star and more")
    }

    // MARK: - Math

    @Test func standaloneDisplayMathIsClosed() {
        #expect(TailAutoClose.close("$$x = \\frac{1}{2") == "$$x = \\frac{1}{2$$")
        #expect(TailAutoClose.close("Step 1.\n$$a + b") == "Step 1.\n$$a + b$$")
    }

    /// The closer arrives one character at a time. Adding a whole `$$` to the
    /// half of it that is already there would show a third dollar for a tick.
    @Test func aHalfTypedDisplayCloserIsCompletedNotDoubled() {
        #expect(TailAutoClose.close("$$x = 1$") == "$$x = 1$$")
        #expect(TailAutoClose.close("Step 1.\n$$a + b$") == "Step 1.\n$$a + b$$")
    }

    /// The transcript keeps the open block's math as source rather than
    /// typesetting a half-typed equation, and a closer nothing will typeset is
    /// two dollars on screen that the model never wrote.
    @Test func displayMathIsNotClosedForACallerThatKeepsTheSource() {
        #expect(TailAutoClose.close("$$x = \\frac{1}{2", typesetsMath: false)
            == "$$x = \\frac{1}{2")
        #expect(TailAutoClose.close("A **bold sta", typesetsMath: false) == "A **bold sta**")
    }

    @Test func closedDisplayMathIsLeftAlone() {
        #expect(TailAutoClose.close("$$x = 1$$") == "$$x = 1$$")
    }

    /// Currency is the reason single `$` is never closed by anyone: `$20` and
    /// `$30` in one sentence would become an equation.
    @Test func singleDollarsAreNeverClosed() {
        #expect(TailAutoClose.close("The plan costs $20 and the pro plan $3")
            == "The plan costs $20 and the pro plan $3")
        #expect(TailAutoClose.close("A mid-sentence $$ pair is not a block")
            == "A mid-sentence $$ pair is not a block")
    }

    @Test func mathContentIsNotTreatedAsEmphasis() {
        #expect(TailAutoClose.close("$$a * b * c") == "$$a * b * c$$")
    }

    /// The tail keeps its math as source, so nothing masked it and every `_`
    /// in a subscript and `*` in an equation read as an emphasis marker.
    @Test(arguments: [
        ("The value $x_{1}$ is small", "The value $x_{1}$ is small"),
        ("The value $x_{1", "The value $x_{1"),
        (#"where $\alpha*\beta"#, #"where $\alpha*\beta"#),
        // Currency is not an equation, so the emphasis after it is still live.
        ("costs $20 and *very goo", "costs $20 and *very goo*"),
    ])
    func mathOnTheTailIsNotEmphasis(_ probe: (String, String)) {
        #expect(TailAutoClose.close(probe.0) == probe.1, "\(probe.0.debugDescription)")
    }

    @Test func displayMathOnTheTailIsNotEmphasisForASourceKeepingCaller() {
        let tail = "$$\n\\sum_{i=1}^n x_i"
        #expect(TailAutoClose.close(tail, typesetsMath: false) == tail)
    }

    /// CommonMark flanking, with the deviations arithmetic prose needs: a `*`
    /// between word characters is multiplication, and `*=` is an operator.
    @Test(arguments: [
        ("Multiply 2*3 to get", "Multiply 2*3 to get"),
        ("Multiply 2*3 to get *nine", "Multiply 2*3 to get *nine*"),
        ("The exponent x**2 stays", "The exponent x**2 stays"),
        ("a *= 2", "a *= 2"),
        ("**Note:** value_1 and *bo", "**Note:** value_1 and *bo*"),
        ("(*paren", "(*paren*"),
    ])
    func emphasisFollowsTheFlankingRule(_ probe: (String, String)) {
        #expect(TailAutoClose.close(probe.0) == probe.1, "\(probe.0.debugDescription)")
    }

    /// Every marker on a line re-read the line from its start, so an 8 KB
    /// paragraph cost 164 ms a tick. Per-line state is carried instead.
    @Test func closingCostIsLinearInTheLine() {
        func line(_ bytes: Int) -> String {
            let unit = "word *emph* and more text. "
            return String(repeating: unit, count: bytes / unit.utf8.count) + "trailing *ope"
        }
        let small = line(2_048)
        let large = line(8_192)
        let clock = ContinuousClock()
        func cost(_ text: String) -> Double {
            let elapsed = clock.measure {
                for _ in 0..<5 { _ = TailAutoClose.close(text) }
            }
            return Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) * 1e-18
        }
        _ = cost(small)
        let smallCost = cost(small)
        let largeCost = cost(large)
        let ratio = largeCost / smallCost
        print(String(
            format: "TAIL-CLOSE 2KB=%.2fms 8KB=%.2fms ratio=%.2f",
            smallCost * 1e3 / 5,
            largeCost * 1e3 / 5,
            ratio))
        #expect(ratio < 8, "four times the line cost \(ratio) times as much")
    }

    // MARK: - Trailing fragments

    @Test func incompleteTagIsStripped() {
        #expect(TailAutoClose.close("Line one<br") == "Line one")
        #expect(TailAutoClose.close("Closing </su") == "Closing ")
    }

    @Test func comparisonsAndCompleteTagsSurvive() {
        #expect(TailAutoClose.close("If a < b then") == "If a < b then")
        #expect(TailAutoClose.close("Line one<br>Line two") == "Line one<br>Line two")
    }

    @Test func incompleteLinkRendersItsLabel() {
        #expect(TailAutoClose.close("See [the docs](https://exa") == "See the docs")
        #expect(TailAutoClose.close("![alt text](https://exa") == "alt text")
    }

    @Test func completeLinkIsLeftAlone() {
        let link = "See [the docs](https://example.com) now"
        #expect(TailAutoClose.close(link) == link)
    }

    // MARK: - Setext

    /// `Title` plus a half-typed rule turns into a large heading for one tick
    /// and back into a paragraph on the next. The guard keeps it a paragraph.
    @Test func trailingRuleUnderAParagraphIsNeutralized() {
        #expect(TailAutoClose.close("Title\n---") == "Title\n---" + Self.guardCharacter)
        #expect(TailAutoClose.close("Title\n===") == "Title\n===" + Self.guardCharacter)
    }

    @Test func ruleAfterABlankLineStaysAThematicBreak() {
        #expect(TailAutoClose.close("Paragraph.\n\n---") == "Paragraph.\n\n---")
    }

    @Test func emptyTailIsUnchanged() {
        #expect(TailAutoClose.close("") == "")
    }

    /// A finished answer must survive the pass untouched: whatever the model
    /// closed itself is the text the finalize render will see. The
    /// source-keeping call is the one the tail render makes and holds for every
    /// fixture; the typesetting call completes an unclosed `$$`, which is what
    /// it is for, so a fixture whose block ends inside one is pinned through
    /// the source-keeping call alone.
    @Test(arguments: TranscriptCorpus.fixtures)
    func closedCorpusBlocksAreUnchanged(_ fixture: String) throws {
        let source = try TranscriptCorpus.source(fixture)
        let split = ResponseBlockSplitter.split(source)
        for block in split.completed {
            let text = ResponseBlockSplitter.text(block.utf8Range, in: source)
            #expect(TailAutoClose.close(text, typesetsMath: false) == text,
                    "\(fixture) rewrote a completed block")
            guard !TranscriptCorpus.unclosedDisplayBlocks.contains(fixture) else { continue }
            #expect(TailAutoClose.close(text) == text, "\(fixture) rewrote a completed block")
        }
    }
}
