import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@MainActor
@Suite struct TranscriptRenderCorpusTests {
    struct Expectation {
        let usedFallback: Bool
        var mustContain: [String] = []
        var mustNotContain: [String] = []
        /// Typeset equations, counted as attachment runs. A fixture whose math
        /// stops being typeset drops to zero here rather than failing silently.
        var attachments = 0
    }

    /// One entry per corpus fixture, recording what the renderer actually does
    /// today. A fixture that flips when a renderer capability lands must flip
    /// here in the same change.
    static let expectations: [String: Expectation] = [
        "cjk": Expectation(
            usedFallback: false,
            mustContain: ["内联公式", "let x = 1", "低层级", "特性", "内置数据竞争保护。"],
            mustNotContain: ["###", "**", "| :--- |"],
            attachments: 1),
        "crlf": Expectation(
            usedFallback: false,
            mustContain: ["CRLF document", "let a = 1", "Trailing paragraph."],
            mustNotContain: ["# CRLF", "**", "```", "\r"]),
        "currency": Expectation(
            usedFallback: false,
            mustContain: ["$20 per month", "$49.99", "Basic", "$200"],
            mustNotContain: ["###", "| :--- |"]),
        "fifty-equations": Expectation(
            usedFallback: false,
            mustContain: ["Fifty equations", "Step 50.", "closed form"],
            mustNotContain: ["###"],
            attachments: 50),
        "gemma-comparison-table": Expectation(
            usedFallback: false,
            mustContain: [
                "Concurrency Comparison",
                "GCD Queues",
                "Swift Actors",
                "\u{2611}\tDefine data model",
                "\u{2610}\tOptimize performance",
                "actor BankAccount {",
            ],
            mustNotContain: ["###", "**", "```", "- [x]", "| :--- |"]),
        "gemma-derivation-table": Expectation(
            usedFallback: false,
            mustContain: [
                "1. Derivation of the Quadratic Formula",
                "completing the square",
                "4. Comparison Table",
                "Case A ($20)",
                "Target Market",
            ],
            mustNotContain: ["###", "**", "| Feature |", "| :--- |"],
            attachments: 20),
        "gemma-inline-environments": Expectation(
            usedFallback: false,
            mustContain: ["Bayes' Theorem:", "Absolute Value:", "echo $HOME", "\\d+"],
            mustNotContain: ["**", "```", "\\begin{cases}", "\\begin{aligned}"],
            attachments: 4),
        "heading-with-math": Expectation(
            usedFallback: false,
            mustContain: ["Solving", "discriminant", "heading size"],
            mustNotContain: ["##"],
            attachments: 2),
        "large-code-block": Expectation(
            usedFallback: false,
            mustContain: ["Streaming buffer implementation", "struct RingSlot057", "capacity."],
            mustNotContain: ["```", "###"]),
        "latex-paren-bracket": Expectation(
            usedFallback: false,
            mustContain: [
                "area of a circle",
                "volume of a sphere",
                "Escaped braces stay literal",
            ],
            mustNotContain: ["\\pi r^2", "\\frac{4}{3}", "\\(", "\\["],
            attachments: 2),
        "link-dollar-url": Expectation(
            usedFallback: false,
            mustContain: ["the pricing page", "the docs", "https://example.com/plain"],
            mustNotContain: ["](", "<https"]),
        "lone-display-dollars": Expectation(
            usedFallback: false,
            mustContain: [
                "Rated $$ on the price scale.",
                "Pricing notes",
                "Basic",
                "\u{2022}\tThe scale is a rating, not an equation.",
                "A single $ in prose is left alone too.",
                "That is all.",
            ],
            mustNotContain: ["# Pricing", "| :--- |", "- The scale"]),
        "nested-lists": Expectation(
            usedFallback: false,
            mustContain: [
                "Build order",
                "1.\tConfigure the toolchain",
                "\u{2022}\tInstall Xcode 26",
                "\u{2022}\tSelect the toolchain",
                "\u{2022}\txcode-select --switch",
                "2.\tBuild the package",
                "1.\tswift build",
                "2.\tswift build --build-tests",
                "3.\tRun the focused filters",
                "\u{2022}\tThree-space marker level one",
                "\u{2022}\tFour-space nested level two",
                "\u{2022}\tLevel three",
            ],
            // The marker used to come from the outermost list, so a nested
            // bullet wore its parent's number and nested ordered items
            // repeated the parent ordinal.
            mustNotContain: ["###", "*   ", "1.\t1.", "1.\tInstall Xcode 26"]),
        "prose-fence": Expectation(
            usedFallback: false,
            mustContain: [
                "Here is the fix:",
                "func apply(_ value: Int) -> Int {",
                "return value + 1",
                "Done.",
                "first line",
                "second line",
                "Both fences interrupt the sentence above them.",
            ],
            mustNotContain: ["```", "~~~"]),
        "quote-with-math": Expectation(
            usedFallback: false,
            mustContain: ["│\t", "fundamental constants", "Ordinary text after the quote."],
            mustNotContain: ["> "],
            attachments: 3),
        "shell-dollars": Expectation(
            usedFallback: false,
            mustContain: ["echo $HOME", "$HOME/$PATH", "$x=$y"],
            mustNotContain: ["```"]),
        "table-html-bold": Expectation(
            usedFallback: false,
            mustContain: [
                "Comparison",
                "GCD Queues",
                "Low-level.\u{2028}Closure based.",
                "H2O and x2",
                "ordinary paragraph",
            ],
            mustNotContain: ["###", "**", "<br>", "<br/>", "<sub>", "| :--- |"]),
        "task-lists": Expectation(
            usedFallback: false,
            mustContain: ["Release checklist", "Define data model", "plain item without a checkbox"],
            mustNotContain: ["####"]),
        "unclosed-display-math": Expectation(
            usedFallback: false,
            mustContain: ["derivative of the loss", "\\frac{\\partial L}{\\partial w}"]),
        "unclosed-fence": Expectation(
            usedFallback: true,
            mustContain: ["```metal", "kernel void reduce"]),
    ]

    static func attachments(in attributed: NSAttributedString) -> [String] {
        var sources: [String] = []
        attributed.enumerateAttribute(
            .attachment,
            in: NSRange(location: 0, length: attributed.length),
            options: []) { value, _, _ in
            if let attachment = value as? MathAttachment {
                sources.append(attachment.latexSource)
            }
        }
        return sources
    }

    @Test func everyFixtureHasAnExpectation() {
        for fixture in TranscriptCorpus.fixtures {
            #expect(Self.expectations[fixture] != nil, "missing expectation for \(fixture)")
        }
        #expect(Self.expectations.count == TranscriptCorpus.fixtures.count)
    }

    @Test(arguments: TranscriptCorpus.fixtures)
    func rendersFixtureAndEmitsFrames(_ fixture: String) throws {
        let source = try TranscriptCorpus.source(fixture)
        let expectation = try #require(Self.expectations[fixture])
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        #expect(result.usedFallback == expectation.usedFallback, "fallback for \(fixture)")
        if expectation.usedFallback {
            #expect(text == source, "raw fallback must be byte-exact for \(fixture)")
        }
        for needle in expectation.mustContain {
            #expect(text.contains(needle), "\(fixture) is missing \(needle.debugDescription)")
        }
        for needle in expectation.mustNotContain {
            #expect(!text.contains(needle), "\(fixture) still shows \(needle.debugDescription)")
        }
        // Sentinels are the substitution's own bookkeeping and must never
        // survive into the visible string; a U+FFFC is only legitimate where a
        // typeset equation replaced one.
        #expect(!text.unicodeScalars.contains { $0.value == 0xE000 })
        #expect(!text.unicodeScalars.contains { $0.value == 0xE001 })
        let attachments = Self.attachments(in: result.attributedString)
        #expect(attachments.count == expectation.attachments, "attachments for \(fixture)")
        #expect(text.unicodeScalars.count { $0.value == 0xFFFC } == expectation.attachments)

        // `plainText` is the transcript's text projection: every attachment
        // maps back to the LaTeX it replaced, so no placeholder escapes.
        let plain = ResponseMarkdownRenderer().plainText(source)
        #expect(!plain.unicodeScalars.contains { $0.value == 0xFFFC })
        var expected = text
        for latex in attachments.reversed() {
            guard let placeholder = expected.range(of: "\u{FFFC}", options: .backwards) else {
                Issue.record("attachment count does not match the placeholders")
                break
            }
            expected.replaceSubrange(placeholder, with: latex)
        }
        #expect(plain == expected)
        for latex in attachments {
            #expect(source.contains(latex), "\(fixture) lost the source of \(latex)")
        }

        for dark in [false, true] {
            try TranscriptFrameRenderer.record(
                result.attributedString,
                named: "\(fixture).final.\(dark ? "dark" : "light").png",
                dark: dark)
        }
    }
}
