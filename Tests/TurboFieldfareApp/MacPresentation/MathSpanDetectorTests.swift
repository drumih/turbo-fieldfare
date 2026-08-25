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

    /// The scan walked past a `$` it had already refused as a closer, so the
    /// currency before an equation and the equation itself became one span and
    /// the whole sentence typeset as prose. A `$` after whitespace cannot
    /// close anything; it is where the next candidate starts.
    @Test(arguments: [
        ("It costs $5 (or $10 for two). The formula $E=mc^2$ applies.", ["$E=mc^2$"]),
        ("the $n$th term is $a_n$", ["$a_n$"]),
        ("Costs $20 and $30 each; $x$ is the count", ["$x$"]),
        ("Refunds under $5 today; anything over $100 needs review.", []),
        ("| Basic | $20 | $200 |", []),
        // The candidate is abandoned at the second opener rather than swallowing
        // the space between them.
        ("$a $b$", ["$b$"]),
        // A `$<digit>` candidate that spans whitespace is currency unless its
        // content reads as math.
        ("I paid $20 for the $50 item$", []),
        (#"The result is $5 \times 10^3$ exactly."#, [#"$5 \times 10^3$"#]),
    ])
    func aWhitespacePrecededDollarStartsTheNextCandidate(_ probe: (String, [String])) {
        #expect(Self.sources(probe.0) == probe.1, "\(probe.0.debugDescription)")
    }

    /// `isLetter` and `isNumber` are Unicode-wide, so a Han or Cyrillic
    /// character beside a delimiter read as a word character and the equation
    /// stayed raw. The adjacency rules exist for shell and URL text, which is
    /// ASCII; the `/` rule is unchanged.
    @Test(arguments: [
        ("\u{8D28}\u{91CF}\u{4E3A}$m$\u{7684}\u{7269}\u{4F53}", ["$m$"]),
        ("\u{044D}\u{043D}\u{0435}\u{0440}\u{0433}\u{0438}\u{044F}$E$\u{0440}\u{0430}\u{0432}\u{043D}\u{0430}", ["$E$"]),
        ("\u{516C}\u{5F0F}$E = mc^2$\u{6210}\u{7ACB}\u{3002}", ["$E = mc^2$"]),
        (#"\u{534A}\u{5F84}$r$\u{FF0C}\u{9762}\u{79EF}$A = \pi r^2$\u{3002}"#, ["$r$", #"$A = \pi r^2$"#]),
        ("caf\u{E9}$x$", ["$x$"]),
    ])
    func adjacencyRulesOnlyCountASCIIWordCharacters(_ probe: (String, [String])) {
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

    /// A `$$` was an opener anywhere, and an unclosed one shielded the rest of
    /// the answer as raw source unless a blank line happened to follow. Three
    /// or more dollars are never an opener, and only a line-start opener whose
    /// content reads as TeX earns the shield.
    @Test(arguments: [
        ("It costs $$$ a lot.\n**Bold** line\n# Heading", []),
        ("Rated $$ on the price scale.\n- item", []),
        ("See \\[ in the note.\n# Heading", []),
        ("$$$x$$$", []),
        ("a $$$ b $$ c $$", ["$$ c $$"]),
        // Line start is necessary but not sufficient: the content has to look
        // like an equation the model was cut off in the middle of.
        ("Text\n\n$$x = 1", []),
        ("Text\n\n$$x = \\frac{1", ["$$x = \\frac{1"]),
        ("- Step:\n  $$\\frac{a}{b}", ["$$\\frac{a}{b}"]),
    ])
    func onlyALineStartOpenerWithTeXContentProtectsTheRest(_ probe: (String, [String])) {
        #expect(Self.sources(probe.0) == probe.1, "\(probe.0.debugDescription)")
    }

    /// A blank line is the display block's hard boundary in both directions:
    /// nothing past one can close an opener, and an opener nothing closes
    /// protects only as far as the boundary.
    @Test func aDisplayBlockNeverCrossesABlankLine() {
        let source = "Intro.\n\n$$\n\\frac{a}{b}\n\nc = d\n$$\n\nAfter $x$"
        let spans = MathSpanDetector.spans(in: source)
        #expect(spans.map(\.source) == ["$$\n\\frac{a}{b}", "$x$"])
        #expect(spans.first?.isLiteralProtect == true)
        #expect(spans.last?.isLiteralProtect == false)
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

    /// `\[ ... \]` was math wherever it appeared, so an escaped bracket pair in
    /// prose became an equation and the words inside it disappeared into a
    /// failed typeset. It is math when the content reads as math, or when the
    /// pair owns its line.
    @Test(arguments: [
        (#"the array \[1, 2, 3\] holds"#, []),
        (#"\[1, 2, 3\]"#, [#"\[1, 2, 3\]"#]),
        (#"\[ x \]"#, [#"\[ x \]"#]),
        (#"Set \[optional\] flag"#, []),
        (#"Note \[x = 1\] inline"#, [#"\[x = 1\]"#]),
    ])
    func bracketDelimitersNeedMathContentOrTheirOwnLine(_ probe: (String, [String])) {
        #expect(Self.sources(probe.0) == probe.1, "\(probe.0.debugDescription)")
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

    /// The mask opened a fence on "```bash```" and hid every dollar below it,
    /// so an answer that names the fence syntax lost all of its math.
    @Test func aBacktickInfoStringDoesNotOpenACodeFence() {
        #expect(Self.sources("```bash``` is the fence syntax for $x$ here.") == ["$x$"])
        #expect(Self.sources("```bash```\n\n$$y$$") == ["$$y$$"])
    }

    /// The mask read every fence at column zero, so a listing indented into a
    /// nested item was never a fence and its shell variables typeset as
    /// equations. It also read `- - -` as a bullet, which kept a list open
    /// across the rule and stopped the indented block under it being code.
    @Test(arguments: [
        // A fence two levels in: only the span outside it survives. Tilde
        // fences, so the backtick-span mask cannot cover for the block mask.
        (
            "1. Outer\n    - Inner\n        ~~~bash\n        cost $a$ here\n        ~~~\n    - Next $x$",
            ["$x$"]
        ),
        ("- item\n    ~~~\n    $a$\n    ~~~\n- next $b$", ["$b$"]),
        // An indented block after a heading is code; the heading is a leaf
        // block, so there is no paragraph for it to be a continuation of.
        ("# Title\n    let x = $a$\n\nAfter $b$", ["$b$"]),
        ("- - -\n\n    $a$ code", []),
        // A deep indent that continues the item's own paragraph is lazy
        // continuation, not code.
        ("- item\n        lazy $a$", ["$a$"]),
    ])
    func theCodeMaskFollowsContainerIndentation(_ probe: (String, [String])) {
        #expect(Self.sources(probe.0) == probe.1, "\(probe.0.debugDescription)")
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
        // Generous on purpose, and it has to hold on a shared CI runner as well
        // as a fast laptop: what it catches is the quadratic shape, not a few
        // milliseconds. The defect it exists for took 16.1 s.
        #expect(elapsed < .seconds(1), "180 KB inline scan took \(elapsed)")
    }

    /// `isEscaped` counted the backslashes before every candidate, so a run of
    /// them cost one backward walk per character: 80 KB measured 0.98 s, and
    /// the open block is rescanned on every streaming tick. The parity is a
    /// mask built once per message instead.
    @Test func escapeScanningStaysLinearOnALongBackslashRun() {
        let even = String(repeating: #"\"#, count: 80_000) + "$x$"
        let odd = String(repeating: #"\"#, count: 80_001) + "$x$"
        let elapsed = ContinuousClock().measure {
            #expect(Self.sources(even) == ["$x$"])
            #expect(Self.sources(odd).isEmpty)
        }
        // Same bound and the same reason as the inline scan above: the defect
        // this exists for took 86.5 s, so a second still leaves 86x of margin
        // while tolerating a runner that is not a fast laptop.
        #expect(elapsed < .seconds(1), "80 KB of backslashes took \(elapsed)")
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
