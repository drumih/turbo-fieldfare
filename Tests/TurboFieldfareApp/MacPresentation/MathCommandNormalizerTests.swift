import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

/// The 647-string coverage sweep the pin decision was made on: the 607-entry
/// command sweep first, then the 40 longer expression probes.
///
/// Fourteen entries were stored double-encoded and measured as Latin letters
/// rather than as the mathematical characters they name. Rewritten as real
/// UTF-8 they moved the command sweep from 465 raw / 518 normalized to
/// 453 raw / 506 normalized; the Unicode symbol table brings the normalized
/// figure back to 517, and the one that stays out is a CJK word character with
/// no command to map onto.
///
/// The raw figure then moved again, from 453 to 450, when the conformer began
/// refusing what the pinned build drops without an error. `\u{221A}2`, `a~b`
/// and `\text{if}~x` had been counted as passing because an image came back;
/// the character they name was missing from it.
enum MathCoverageFixture {
    static let commandSweepCount = 607

    static let sweep: [String] = load("latex-sweep")
    static let expectedFailures: [String] = load("latex-sweep-failures")

    /// Lines are exact LaTeX, including one entry that is a lone backslash and
    /// a space, so nothing here may be trimmed.
    private static func load(_ name: String) -> [String] {
        guard let url = Bundle.module.url(
            forResource: name,
            withExtension: "txt",
            subdirectory: "Fixtures/math-coverage"),
            let text = try? String(contentsOf: url, encoding: .utf8) else {
            return []
        }
        return text
            .components(separatedBy: "\n")
            .filter { !$0.isEmpty }
    }
}

struct MathRewriteCase: Sendable {
    let input: String
    let output: String
    /// Every rewrite exists so the result typesets; a row that does not is
    /// a rule that is not earning its place.
    var typesets = true
}

/// One row per rule in the table. Each is measured twice: the rewrite text
/// itself, and the fact that the pinned typesetter accepts the result.
enum MathRewriteCases {
    static let all: [MathRewriteCase] = [
        MathRewriteCase(input: #"\boxed{x+1}"#, output: #"x+1"#),
        MathRewriteCase(input: #"\fbox{x}"#, output: #"x"#),
        MathRewriteCase(input: #"\mbox{x}"#, output: #"x"#),
        MathRewriteCase(input: #"\cancel{x}"#, output: #"x"#),
        MathRewriteCase(input: #"\xcancel{x}"#, output: #"x"#),
        MathRewriteCase(input: #"\sout{x}"#, output: #"x"#),
        MathRewriteCase(input: #"\textcolor{red}{x}"#, output: #"x"#),
        MathRewriteCase(input: #"\colorbox{red}{x}"#, output: #"x"#),
        MathRewriteCase(input: #"\color{red} x"#, output: #" x"#),
        MathRewriteCase(input: #"x \tag{1}"#, output: #"x "#),
        MathRewriteCase(input: #"x \label{eq}"#, output: #"x "#),
        MathRewriteCase(input: #"a \xrightarrow{f} b"#, output: #"a \rightarrow b"#),
        MathRewriteCase(input: #"a \xleftarrow{f} b"#, output: #"a \leftarrow b"#),
        MathRewriteCase(input: #"\underbrace{a+b}_{\text{sum}}"#, output: #"a+b"#),
        MathRewriteCase(input: #"\overbrace{a+b}^{n}"#, output: #"a+b"#),
        // The unbraced script form. Skipping only the command left its group
        // behind, so the annotation glued onto the base as "a+btotal".
        MathRewriteCase(input: #"\underbrace{a+b}_\text{total}"#, output: #"a+b"#),
        MathRewriteCase(input: #"\overbrace{a+b}^\mathrm{sum}"#, output: #"a+b"#),
        MathRewriteCase(input: #"\underbrace{a+b}_\alpha"#, output: #"a+b"#),
        MathRewriteCase(input: #"\overset{a}{b}"#, output: #"{b}^{a}"#),
        MathRewriteCase(input: #"\overset{\alpha}{b}"#, output: #"{b}^{\alpha}"#),
        // A multi-token annotation would stack an expression over the base,
        // which the replacement cannot express, so the base wins.
        MathRewriteCase(input: #"\overset{a+b}{c}"#, output: #"{c}"#),
        MathRewriteCase(input: #"\underset{a}{b}"#, output: #"{b}_{a}"#),
        MathRewriteCase(input: #"\stackrel{?}{=}"#, output: #"{=}^{?}"#),
        MathRewriteCase(input: #"x \nonumber"#, output: #"x "#),
        MathRewriteCase(
            input: #"\left( x \middle| y \right)"#,
            output: #"\left( x | y \right)"#),
        MathRewriteCase(
            input: #"\begin{array}{|c|c|} \hline a & b \end{array}"#,
            output: #"\begin{matrix}  a & b \end{matrix}"#),
        MathRewriteCase(input: #"\lvert x \rvert"#, output: #"| x |"#),
        MathRewriteCase(input: #"\lVert x \rVert"#, output: #"\| x \|"#),
        MathRewriteCase(input: #"x \dots y"#, output: #"x \ldots y"#),
        MathRewriteCase(input: #"x \dotsc y"#, output: #"x \ldots y"#),
        MathRewriteCase(input: #"x \therefore y"#, output: "x \\text{\u{2234}} y"),
        MathRewriteCase(input: #"x \because y"#, output: "x \\text{\u{2235}} y"),
        MathRewriteCase(input: #"a \: b"#, output: #"a \, b"#),
        MathRewriteCase(input: #"a \bmod b"#, output: #"a \ \mathrm{mod}\  b"#),
        MathRewriteCase(input: #"\pmb{x}"#, output: #"\mathbf{x}"#),
        MathRewriteCase(input: #"\mathscr{L}"#, output: #"\mathcal{L}"#),
        MathRewriteCase(input: #"a \coloneqq b"#, output: #"a := b"#),
        MathRewriteCase(input: #"a \eqqcolon b"#, output: #"a =: b"#),
        MathRewriteCase(
            input: #"\begin{align} a &= b \end{align}"#,
            output: #"\begin{aligned} a &= b \end{aligned}"#),
        MathRewriteCase(
            input: #"\begin{align*} a &= b \end{align*}"#,
            output: #"\begin{aligned} a &= b \end{aligned}"#),
        MathRewriteCase(
            input: #"\begin{alignat}{2} a &= b \end{alignat}"#,
            output: #"\begin{aligned} a &= b \end{aligned}"#),
        MathRewriteCase(
            input: #"\begin{alignat*}{2} a &= b \end{alignat*}"#,
            output: #"\begin{aligned} a &= b \end{aligned}"#),
        MathRewriteCase(
            input: #"\begin{aligned*} a \end{aligned*}"#,
            output: #"\begin{aligned} a \end{aligned}"#),
        MathRewriteCase(
            input: #"\begin{gather*} a \\ b \end{gather*}"#,
            output: #"\begin{aligned} a \\ b \end{aligned}"#),
        MathRewriteCase(
            input: #"\begin{equation} a = b \end{equation}"#,
            output: #"\begin{aligned} a = b \end{aligned}"#),
        MathRewriteCase(
            input: #"\begin{equation*} a = b \end{equation*}"#,
            output: #"\begin{aligned} a = b \end{aligned}"#),
        MathRewriteCase(
            input: #"\begin{array}{cc} a & b \end{array}"#,
            output: #"\begin{matrix} a & b \end{matrix}"#),
        MathRewriteCase(
            input: #"\begin{rcases} a \\ b \end{rcases}"#,
            output: #"\begin{cases} a \\ b \end{cases}"#),
        // The radical is a character, not a command: the build had no atom for
        // it, so it drew nothing and the equation looked complete without it.
        MathRewriteCase(input: "\u{221A}2", output: #"\sqrt{2}"#),
        MathRewriteCase(input: "\u{221A}16.5", output: #"\sqrt{16.5}"#),
        MathRewriteCase(input: "\u{221A}x", output: #"\sqrt{x}"#),
        MathRewriteCase(input: "\u{221A}{x+1}", output: #"\sqrt{x+1}"#),
        MathRewriteCase(input: "\u{221A}(a+b)", output: #"\sqrt{a+b}"#),
        MathRewriteCase(input: "\u{221A}\\alpha", output: #"\sqrt{\alpha}"#),
        MathRewriteCase(input: "f\u{2032}(x)", output: #"f^{\prime}(x)"#),
        MathRewriteCase(input: "f\u{2033}(x)", output: #"f^{\prime\prime}(x)"#),
        MathRewriteCase(input: "f\u{2032}\u{2032}(x)", output: #"f^{\prime\prime}(x)"#),
        MathRewriteCase(input: "90\u{00B0}", output: #"90^{\circ}"#),
        MathRewriteCase(input: "50%", output: #"50\%"#),
        MathRewriteCase(input: "a~b", output: #"a\ b"#),
        MathRewriteCase(input: "a\u{00A0}b", output: #"a\ b"#),
        MathRewriteCase(input: "a\u{2009}b", output: #"a\,b"#),
        MathRewriteCase(input: "x\u{200B}y", output: "xy"),
        MathRewriteCase(input: "\u{00AC}p \u{2228} q", output: #"\neg p \lor q"#),
        MathRewriteCase(input: "x \u{226A} y", output: #"x \ll y"#),
        MathRewriteCase(input: "\u{2308}x\u{2309}", output: #"\lceil x\rceil"#),
    ]
}

@MainActor
@Suite struct MathCommandNormalizerTests {
    static func typesets(_ latex: String) -> Bool {
        SwiftMathTypesetter().render(
            latex: latex,
            fontSize: 13,
            tint: .black,
            mode: .inline) != nil
    }

    @Test(arguments: MathRewriteCases.all)
    func eachRewriteProducesLatexThePinnedTypesetterAccepts(_ rewrite: MathRewriteCase) {
        #expect(MathCommandNormalizer.normalize(rewrite.input) == rewrite.output,
                "\(rewrite.input.debugDescription)")
        #expect(Self.typesets(rewrite.output) == rewrite.typesets,
                "\(rewrite.output.debugDescription)")
    }

    /// Every rule must be pulling its weight: if the pinned revision already
    /// renders the input, the rewrite is trading real notation for an
    /// approximation and should be deleted instead.
    @Test(arguments: MathRewriteCases.all)
    func noRewriteReplacesLatexThatAlreadyTypesets(_ rewrite: MathRewriteCase) {
        #expect(!Self.typesets(rewrite.input), "\(rewrite.input.debugDescription)")
    }

    static func nested(_ command: String, depth: Int, body: String = "x") -> String {
        String(repeating: "\\\(command){", count: depth)
            + body
            + String(repeating: "}", count: depth)
    }

    /// A rewritten argument is rewritten in turn, so nesting depth is stack
    /// depth. Without the cap this crashes the process with SIGSEGV rather
    /// than failing: the recursion died at 8000 levels with 4000 completing.
    @Test func runawayNestingReturnsTheInputInsteadOfOverflowingTheStack() {
        let runaway = Self.nested("boxed", depth: 8000)
        #expect(MathCommandNormalizer.normalize(runaway) == runaway)
    }

    /// The cap has to sit far above anything a real answer nests, or a fix for
    /// the crash becomes a rendering regression.
    @Test func nestingWellInsideTheCapIsStillRewritten() {
        #expect(MathCommandNormalizer.normalize(Self.nested("boxed", depth: 20)) == "x")
        #expect(MathCommandNormalizer.normalize(
            Self.nested("boxed", depth: 200, body: #"\frac{a}{b}"#)) == #"\frac{a}{b}"#)
    }

    @Test func unknownCommandsPassThroughUnchanged() {
        let unknown = #"\unknowncmd{x} + \frac{a}{b}"#
        #expect(MathCommandNormalizer.normalize(unknown) == unknown)
        #expect(!Self.typesets(unknown))
    }

    /// What Gemma 4 actually emits must survive the normalizer byte for byte;
    /// a rewrite that fires on working input is a mangling, not a fix.
    @Test(arguments: [
        #"ax^2 + bx + c = 0"#,
        #"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"#,
        #"|x| = \begin{cases} x & \text{if } x \geq 0 \\ -x & \text{if } x < 0 \end{cases}"#,
        #"\begin{aligned} E[X] &= \sum_{i=1}^{6} i \cdot P(X=i) \\ &= 3.5 \end{aligned}"#,
        #"\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}"#,
        #"e^{i\pi} + 1 = 0"#,
    ])
    func workingLatexIsLeftAlone(_ latex: String) {
        #expect(MathCommandNormalizer.normalize(latex) == latex)
        #expect(Self.typesets(latex))
    }

    @Test func coverageFixturesLoadAtTheRecordedSize() {
        #expect(MathCoverageFixture.sweep.count == 647)
        #expect(MathCoverageFixture.expectedFailures.count == 90)
        #expect(MathCoverageFixture.sweep.count > MathCoverageFixture.commandSweepCount)
    }

    /// Spacing has no replacement that typesets on its own — a lone space is a
    /// zero-size image, which the conformer refuses — so it is pinned in
    /// context instead of through the symbol-table row test.
    @Test func everySpacingRuleReplacesADroppedCharacterWithAnAcceptedCommand() {
        for (character, command) in MathCommandNormalizer.spacing {
            #expect(!Self.typesets("a\(character)b"), "a\(character)b already typesets")
            #expect(MathCommandNormalizer.normalize("a\(character)b") == "a\(command)b")
            #expect(Self.typesets("a\(command)b"), "\(command) does not typeset")
        }
        for character in MathCommandNormalizer.removed {
            #expect(!Self.typesets("a\(character)b"))
            #expect(MathCommandNormalizer.normalize("a\(character)b") == "ab")
        }
    }

    /// The sweep's fourteen Unicode entries were stored double-encoded, so what
    /// the fixture actually measured was a run of Latin letters: twelve of them
    /// passed for the wrong reason. Rewritten as the characters they were meant
    /// to be, they fail the pinned typesetter outright and the symbol table is
    /// what carries them.
    @Test(arguments: [
        "\u{03C0}", "\u{00D7}", "\u{2264}", "\u{2192}", "\u{2212}",
        "\u{03B1}", "\u{221E}", "\u{2211}", "\u{00BD}", "\u{2026}",
        "\u{211D}",
    ])
    func unicodeMathCharactersTypesetOnlyAfterNormalization(_ character: String) {
        #expect(!Self.typesets(character), "\(character.debugDescription)")
        #expect(Self.typesets(MathCommandNormalizer.normalize(character)),
                "\(character.debugDescription)")
    }

    /// `\u{221A}2` looked like an entry the pinned revision already drew, and
    /// it is not: the build has no atom for the radical and skipped it, so the
    /// image was `2` and nothing said so. The accented letter is drawn for
    /// real, through the table `atom(forCharacter:)` consults first.
    @Test func theRadicalOnlyLookedLikeSomethingThePinnedRevisionDrew() {
        #expect(!Self.typesets("\u{221A}2"))
        #expect(MathCommandNormalizer.normalize("\u{221A}2") == #"\sqrt{2}"#)
        #expect(Self.typesets(MathCommandNormalizer.normalize("\u{221A}2")))

        #expect(Self.typesets("\u{00E9}"))
        #expect(MathCommandNormalizer.normalize("\u{00E9}") == "\u{00E9}")
        // A CJK word character is not mathematical notation and has no command
        // to map onto; it stays in the expected-failure list.
        #expect(!Self.typesets("\u{4E2D}"))
        #expect(MathCoverageFixture.expectedFailures.contains("\u{4E2D}"))
    }

    /// Every row of the symbol table, measured in both directions: the
    /// character on its own is rejected, so the rule is earning its place, and
    /// what it rewrites to is accepted.
    @Test func everySymbolRuleReplacesARejectedCharacterWithAnAcceptedCommand() {
        for (character, command) in MathCommandNormalizer.symbols {
            #expect(!Self.typesets(String(character)), "\(character) already typesets")
            #expect(MathCommandNormalizer.normalize(String(character)) == command,
                    "\(character)")
            #expect(Self.typesets(command), "\(command) does not typeset")
        }
    }

    /// A command may not run into the letter after it, or the two spell an
    /// unknown command and the whole equation falls back to raw text.
    @Test func aSymbolFollowedByALetterKeepsItsCommandSeparate() {
        #expect(MathCommandNormalizer.normalize("\u{03B1}b") == "\\alpha b")
        #expect(MathCommandNormalizer.normalize("\u{03B1}2") == "\\alpha2")
        #expect(MathCommandNormalizer.normalize("x \u{2264} 5") == "x \\le 5")
        #expect(Self.typesets(MathCommandNormalizer.normalize("\u{03B1}b")))
    }

    /// The sweep is the pin's regression gate: moving SwiftMath, or adding a
    /// normalizer rule, shows up here as a named string that changed side.
    @Test(arguments: MathCoverageFixture.sweep)
    func sweepStringMatchesRecordedCoverage(_ latex: String) {
        let expected = !MathCoverageFixture.expectedFailures.contains(latex)
        #expect(Self.typesets(MathCommandNormalizer.normalize(latex)) == expected,
                "\(latex.debugDescription)")
    }

    /// Records the two numbers the pin decision is argued from, so a rerun
    /// prints them rather than requiring a separate probe.
    @Test func sweepCoverageImprovesWithNormalization() {
        let commands = MathCoverageFixture.sweep.prefix(
            MathCoverageFixture.commandSweepCount)
        let raw = commands.count { Self.typesets($0) }
        let normalized = commands.count { Self.typesets(MathCommandNormalizer.normalize($0)) }
        print("MATH-SWEEP commands=\(commands.count) raw=\(raw) normalized=\(normalized)")
        #expect(normalized > raw)
        // Three entries left the raw column when the conformer started refusing
        // what the build silently drops: the radical and the two tilde spaces.
        // The rewrites below carry all three, so the normalized figure holds.
        #expect(raw == 450)
        #expect(normalized == 517)
    }
}
