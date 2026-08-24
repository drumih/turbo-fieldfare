import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct MathSpanDetectorTests {
    static func sources(_ text: String) -> [String] {
        MathSpanDetector.spans(in: text).map(\.source)
    }

    static func modes(_ text: String) -> [MathSpan.Mode] {
        MathSpanDetector.spans(in: text).map(\.mode)
    }

    // MARK: - Inline basics

    @Test func findsOneSpanWithItsSurroundingTextIntact() {
        let spans = MathSpanDetector.spans(in: "Solve $x^2 + 1$ now.")
        #expect(spans.count == 1)
        #expect(spans[0].source == "$x^2 + 1$")
        #expect(spans[0].latex == "x^2 + 1")
        #expect(spans[0].mode == .inline)
        #expect(spans[0].range == 6..<15)
    }

    @Test(arguments: [
        ("$a$ and $b$", ["$a$", "$b$"]),
        ("$a$ start of line", ["$a$"]),
        ("ends with $b$", ["$b$"]),
        ("$$x = 1$$", ["$$x = 1$$"]),
        ("no math here", []),
    ])
    func findsEverySpanInALine(_ probe: (String, [String])) {
        #expect(Self.sources(probe.0) == probe.1, "\(probe.0)")
    }

    // MARK: - Pandoc conditions

    @Test(arguments: [
        ("$x$", ["$x$"]),
        ("$ x$", []),
        ("$x $", []),
        ("$20,000 and $30,000", []),
        (#"\$5 is cheap"#, []),
        // Two backslashes are an escaped backslash, so the dollar after them
        // still opens a span.
        (#"\\$x$"#, ["$x$"]),
        ("$a\nb$", []),
        ("$$", []),
        ("$", []),
    ])
    func appliesThePandocDollarConditions(_ probe: (String, [String])) {
        #expect(Self.sources(probe.0) == probe.1, "\(probe.0.debugDescription)")
    }

    /// Gemma 4 writes currency inside math as `$\$0.60$`; the escaped dollar
    /// must not be mistaken for the closer.
    @Test func escapedDollarInsideASpanDoesNotCloseIt() {
        let spans = MathSpanDetector.spans(in: #"Unit cost $\$0.60$ each."#)
        #expect(spans.map(\.source) == [#"$\$0.60$"#])
        #expect(spans[0].latex == #"\$0.60"#)
    }

    // MARK: - Adjacency extensions

    @Test(arguments: [
        ("x=$a$b", []),
        ("$FOO/bar$BAZ", []),
        ("https://ex.com/$a/$b", []),
        ("($x$)", ["$x$"]),
        ("word $x$.", ["$x$"]),
        ("cost $x$2 apples", []),
        ("path/$x$", []),
    ])
    func appliesTheAdjacencyExtensions(_ probe: (String, [String])) {
        #expect(Self.sources(probe.0) == probe.1, "\(probe.0.debugDescription)")
    }

    // MARK: - Display

    @Test func findsSingleLineAndMultilineDisplaySpans() {
        #expect(Self.sources("$$x = 1$$") == ["$$x = 1$$"])
        #expect(Self.sources("$$\nx = 1\n$$") == ["$$\nx = 1\n$$"])
        #expect(Self.sources("$$ x = 1 $$") == ["$$ x = 1 $$"])
    }

    /// A rejected `$$` opener resumes at the character after itself, never
    /// after the blank line, so the rest of the answer is untouched.
    @Test func aBlankLineRejectsTheCandidateAndLeavesTheRestOfTheMessage() {
        let source = "$$broken\n\nstill here $x$ and $$y$$"
        #expect(Self.sources(source) == ["$x$", "$$y$$"])
    }

    @Test func displayTakesPrecedenceOverInline() {
        #expect(Self.modes("$$x$$") == [.display])
        #expect(Self.sources("$$$$") == [])
    }

    /// A `$$` region that was rejected is never re-offered to the single
    /// dollar pass, so the two dollars it consumed cannot pair with later
    /// text.
    @Test func adjacentDollarsResolveWithoutReofferingConsumedText() {
        #expect(Self.sources("$a$$b$") == ["$a$", "$b$"])
    }

    @Test func twoDisplayBlocksSeparatedByOneNewlineAreTwoSpans() {
        let source = "$$a = 1$$\n$$b = 2$$"
        #expect(Self.sources(source) == ["$$a = 1$$", "$$b = 2$$"])
        #expect(MathSpanDetector.substitute(source).working
            == "\n\n\u{E000}0\u{E001}\n\n\n\n\n\u{E000}1\u{E001}\n\n")
    }

    // MARK: - Backslash forms

    @Test(arguments: [
        (#"\[x = 1\]"#, [#"\[x = 1\]"#]),
        (#"\(x^2\)"#, [#"\(x^2\)"#]),
        // No command and no operator: prose, not math.
        (#"\(count\)"#, []),
        (#"Regex groups like \(abc\) read as prose."#, []),
        (#"\(\alpha\)"#, [#"\(\alpha\)"#]),
    ])
    func findsBackslashDelimiters(_ probe: (String, [String])) {
        #expect(Self.sources(probe.0) == probe.1, "\(probe.0.debugDescription)")
    }

    @Test func findsKnownEnvironmentsAtLineStartOnly() {
        let known = "\\begin{align}\na &= b\n\\end{align}"
        #expect(Self.sources(known) == [known])
        #expect(Self.modes(known) == [.display])
        #expect(Self.sources("\\begin{unknown}\na\n\\end{unknown}").isEmpty)
        #expect(Self.sources("text \\begin{align}\na\n\\end{align}").isEmpty)
        // `aligned` is an inline environment, not a block one: Gemma emits it
        // inside a single-dollar span and KaTeX does not auto-render it.
        #expect(Self.sources("\\begin{aligned}\na\n\\end{aligned}").isEmpty)
    }

    // MARK: - Code protection

    @Test(arguments: [
        "`$x$`",
        "```\n$$x$$\n```",
        "```bash\necho $HOME\nexport T=\"$HOME/$PATH\"\n```",
        "Before\n\n    $a$ indented code\n\nAfter",
        "~~~\n$a$\n~~~",
    ])
    func codeRegionsHideTheirDollars(_ source: String) {
        #expect(Self.sources(source).isEmpty, "\(source.debugDescription)")
    }

    /// Two lone backticks in different paragraphs are prose, not a code span.
    /// Pairing them masked everything between, so an equation in the middle
    /// typeset while it streamed and reverted to raw source at finalize.
    @Test func backtickSpansDoNotPairAcrossABlankLine() {
        let source = """
            Press the ` key to open the console.

            The relation is $E = mc^2$ exactly.

            Use the ` character to quote a word.
            """
        #expect(Self.sources(source) == ["$E = mc^2$"])
    }

    /// The bound is the blank line, not the line end: a genuine code span
    /// still hides its dollars, wrapped or not.
    @Test func aCodeSpanWithinOneParagraphStillMasksItsDollars() {
        #expect(Self.sources("Run `echo $HOME now` please").isEmpty)
        #expect(Self.sources("Run `echo\n$HOME now` please").isEmpty)
    }

    @Test func fourSpaceListContinuationIsNotCode() {
        let source = "- item\n\n    continues with $a$ inside\n"
        #expect(Self.sources(source) == ["$a$"])
    }

    @Test func mathImmediatelyAfterACodeRegionIsStillFound() {
        #expect(Self.sources("`code` then $x$") == ["$x$"])
        #expect(Self.sources("```\ncode\n```\n\n$$y$$") == ["$$y$$"])
    }

    /// An unmatched backtick opens nothing, but the two that do match hide the
    /// dollars between them, so this string yields no math at all.
    @Test func unbalancedBackticksStillPairByCommonMarkRules() {
        #expect(Self.sources("$a`$ and $b`$").isEmpty)
    }

    // MARK: - Streaming shapes

    @Test func unclosedDisplayOpenersBecomeLiteralProtectSpans() {
        for source in ["Text\n\n$$\\frac{a}{b} = c", "Text\n\n\\[\\frac{a}{b} = c"] {
            let spans = MathSpanDetector.spans(in: source)
            #expect(spans.count == 1, "\(source.debugDescription)")
            #expect(spans.first?.isLiteralProtect == true)
            #expect(spans.first?.mode == .display)
            #expect(spans.first?.source.hasSuffix("= c") == true)
            #expect(spans.first?.latex == spans.first?.source)
        }
    }

    @Test func unclosedSingleDollarStaysPlainText() {
        #expect(Self.sources("The price is $5 and rising").isEmpty)
        #expect(Self.sources("Half an equation $x = ").isEmpty)
    }

    /// Two dollars in a sentence are not a cut-off equation. Protecting from
    /// there to the end of the answer showed every heading, table, and list
    /// below it as raw source.
    @Test(arguments: [
        "Rated $$ on the price scale.\n\n# Heading\n\n| A | B |\n| --- | --- |\n| 1 | 2 |",
        "See \\[ in the note.\n\n# Heading\n\nBody text.",
    ])
    func anOpenerWithABlankLineAfterItIsPlainText(_ source: String) {
        #expect(Self.sources(source).isEmpty, "\(source.debugDescription)")
    }

    /// The interrupted-generation case the protection exists for: nothing
    /// follows the opener but the half-typed equation.
    @Test func aTrailingOpenerWithNoBlankLineAfterItStaysProtected() {
        for source in ["Text\n\n$$\\frac{", "Text\n\n\\[\\frac{"] {
            let spans = MathSpanDetector.spans(in: source)
            #expect(spans.count == 1, "\(source.debugDescription)")
            #expect(spans.first?.isLiteralProtect == true)
        }
    }

    @Test func aTrailingDisplayOpenerWithNoContentIsNotASpan() {
        #expect(Self.sources("Text\n\n$$").isEmpty)
        #expect(Self.sources("Text\n\n$$\n").isEmpty)
    }

    // MARK: - Robustness

    @Test func handlesCRLFAndNonASCIINeighbours() {
        #expect(Self.sources("Line\r\n$$x = 1$$\r\nAfter\r\n") == ["$$x = 1$$"])
        #expect(Self.sources("\u{4E2D}\u{6587} $E = mc^2$ \u{4E2D}\u{6587}") == ["$E = mc^2$"])
        #expect(Self.sources("caf\u{E9} $x$ caf\u{E9}") == ["$x$"])
    }

    /// Swift reads CRLF as one `Character`, so every line rule in the detector
    /// looked straight through it. Each of these behaves one way for LF and
    /// another for CRLF until the scan splits the pair.
    @Test func everyLineRuleTreatsCRLFExactlyAsLF() {
        // A single-dollar span may not cross a line end.
        #expect(Self.sources("Costs $20\nUSD$ each").isEmpty)
        #expect(Self.sources("Costs $20\r\nUSD$ each").isEmpty)

        // A fence hides its dollars and releases the math after it.
        #expect(Self.sources("```bash\necho $HOME\n```\n\n$$x = 1$$") == ["$$x = 1$$"])
        #expect(Self.sources("```bash\r\necho $HOME\r\n```\r\n\r\n$$x = 1$$") == ["$$x = 1$$"])

        // A blank line rejects a `$$` opener instead of protecting the rest.
        #expect(Self.sources("$$broken\n\nstill here $x$") == ["$x$"])
        #expect(Self.sources("$$broken\r\n\r\nstill here $x$") == ["$x$"])
    }

    /// The working copy is a substitution over the original bytes, never a
    /// normalized copy: whatever is not a span comes back exactly as written,
    /// CRLF included, because raw fallbacks re-emit it.
    @Test func substitutionOverCRLFKeepsTheOriginalLineEndings() {
        let source = "Line one\r\n\r\nCosts $x$ here\r\n"
        let substitution = MathSpanDetector.substitute(source)
        #expect(substitution.working == "Line one\r\n\r\nCosts \u{E000}0\u{E001} here\r\n")
        #expect(substitution.spans.map(\.source) == ["$x$"])
    }

    /// Every opener used to rescan the whole region ahead of it, so an answer
    /// full of unclosable dollars cost one pass per dollar: 180 KB of `$a `
    /// took 16.1 s and froze the transcript. The bound is generous on purpose
    /// — what it catches is the quadratic shape, not a few milliseconds.
    @Test func inlineScanningStaysLinearOnALargeAnswer() {
        let source = String(repeating: "$a ", count: 60_000)
        #expect(source.utf8.count == 180_000)
        let elapsed = ContinuousClock().measure {
            #expect(MathSpanDetector.spans(in: source).isEmpty)
        }
        #expect(elapsed < .milliseconds(100), "180 KB inline scan took \(elapsed)")
    }

    /// A model that emits the sentinel characters itself must not be able to
    /// alias a placeholder, so they are stripped before anything is indexed.
    @Test func preExistingSentinelCharactersAreStrippedBeforeSubstitution() {
        let source = "Odd \u{E000}7\u{E001} output with $x$"
        let substitution = MathSpanDetector.substitute(source)
        #expect(substitution.working == "Odd 7 output with \u{E000}0\u{E001}")
        #expect(substitution.spans.map(\.source) == ["$x$"])
    }

    // MARK: - Substitution

    @Test func substitutionIndexesEverySpanAndKeepsTheRestOfTheText() {
        let substitution = MathSpanDetector.substitute("a $x$ b $y$ c")
        #expect(substitution.working == "a \u{E000}0\u{E001} b \u{E000}1\u{E001} c")
        #expect(substitution.spans.count == 2)
    }

    /// The Gemma derivation shape: a `$$` line directly under a text line with
    /// no blank line between them. Without promotion the two share a
    /// paragraph and the soft break becomes a space.
    @Test func standaloneDisplayMathIsPromotedToItsOwnParagraph() {
        let substitution = MathSpanDetector.substitute("Subtract $c$:\n$$ax^2 + bx = -c$$\nNext")
        #expect(substitution.working
            == "Subtract \u{E000}0\u{E001}:\n\n\n\u{E000}1\u{E001}\n\n\nNext")
    }

    @Test func displayMathInsideAQuoteIsNotPromoted() {
        let substitution = MathSpanDetector.substitute("> from $$e^{i\\theta}$$ at zero")
        #expect(substitution.working == "> from \u{E000}0\u{E001} at zero")
    }

    // MARK: - Measured markdown mangling

    /// Each of these is altered by `AttributedString(markdown:)` today, so the
    /// detector has to hand the typesetter the source bytes instead.
    @Test(arguments: [
        #"$a \, b$"#,
        #"$\{x\}$"#,
        #"\[x = 1\]"#,
        #"$a*b*c$"#,
        #"$|x| = \begin{cases} x & x \ge 0 \\ -x & x < 0 \end{cases}$"#,
        #"$\begin{aligned} a &= b \\ &= c \end{aligned}$"#,
    ])
    func mangledFormsReachTheTypesetterByteIdentical(_ source: String) {
        let spans = MathSpanDetector.spans(in: source)
        #expect(spans.count == 1, "\(source.debugDescription)")
        #expect(spans.first?.source == source)
    }
}
