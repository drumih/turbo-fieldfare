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
        "backtick-info-string": Expectation(
            usedFallback: false,
            mustContain: [
                "Fence syntax",
                "bash when you mean",
                "A mid-line ``` in a sentence",
                "Backtick span",
                "inline code",
                "Done.",
            ],
            mustNotContain: ["# Fence syntax", "| --- |"]),
        "arithmetic-prose": Expectation(
            usedFallback: false,
            mustContain: [
                "Multiply 2*3 to get 6",
                "x**2 is not emphasis",
                "a * b",
                "x *= 2",
            ],
            attachments: 1),
        "blank-line-fence": Expectation(
            usedFallback: false,
            mustContain: ["Intro.\n\n\nAfter."],
            mustNotContain: ["```"]),
        "br-only-paragraph": Expectation(
            usedFallback: true,
            mustContain: ["<br>"]),
        "cjk": Expectation(
            usedFallback: false,
            mustContain: [
                "内联公式",
                "let x = 1",
                "低层级",
                "特性",
                "内置数据竞争保护。",
                // A Han character on both sides of the delimiters: the
                // adjacency rules used to read the whole run as one word.
                "质量为\u{FFFC}的物体",
            ],
            mustNotContain: ["###", "**", "| :--- |"],
            attachments: 2),
        "crlf": Expectation(
            usedFallback: false,
            mustContain: ["CRLF document", "let a = 1", "Trailing paragraph."],
            mustNotContain: ["# CRLF", "**", "```", "\r"]),
        "currency": Expectation(
            usedFallback: false,
            mustContain: [
                "$20 per month",
                "$49.99",
                "Basic",
                "$200",
                // Only the equation typesets; the two amounts beside it stay
                // as written.
                "It costs $5 (or $10 for two). The formula \u{FFFC} applies",
            ],
            mustNotContain: ["###", "| :--- |"],
            attachments: 1),
        "display-math-blank-line": Expectation(
            usedFallback: false,
            mustContain: [
                "Intro.",
                // The opener shields only as far as the blank line under it.
                "$$\n\\frac{a}{b}",
                "c = d",
                "the answer continues.",
            ],
            attachments: 1),
        "fence-in-list-item": Expectation(
            usedFallback: false,
            mustContain: [
                "1.\tInstall the tool:",
                "brew install x",
                "2.\tCheck the version:",
                "x --version",
                "Both steps stay inside their items.",
            ],
            mustNotContain: ["```"]),
        "fence-in-quote": Expectation(
            usedFallback: false,
            mustContain: ["Run the installer first:", "brew install x", "Then continue"],
            mustNotContain: ["```", "> "]),
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
        "html-details": Expectation(
            usedFallback: true,
            mustContain: ["<details>", "Hidden body text.", "Done."]),
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
        "list-nested-fence": Expectation(
            usedFallback: false,
            mustContain: [
                "1.\tOuter step",
                "\u{2022}\tInner step",
                // Inside the listing the dollars are shell, not math.
                "cost $a$ per run",
                "export TARGET=\"$HOME/$PATH\"",
                "Then \u{FFFC} closes the answer.",
            ],
            mustNotContain: ["~~~"],
            attachments: 2),
        "list-then-fence": Expectation(
            usedFallback: false,
            mustContain: [
                "\u{2022}\tInstall the tool:",
                "brew install x",
                "brew link x",
                "Done.",
            ],
            mustNotContain: ["```", "- Install"]),
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
        "long-prose-line": Expectation(
            usedFallback: false,
            mustContain: ["word emph and more text."],
            mustNotContain: ["*"]),
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
        "prose-display-dollars": Expectation(
            usedFallback: false,
            mustContain: [
                "It costs $$$ a lot. Rated $$ on the price scale.",
                "Bold line above the heading.",
                "Heading",
                "The array [1, 2, 3] holds three values",
                "[optional] flags stay literal",
                "\u{2022}\tThe rating is not an equation.",
                "Done.",
            ],
            mustNotContain: ["**", "# Heading", "- The rating"]),
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
        "reference-links": Expectation(
            usedFallback: false,
            // The whole render resolves the definitions; the per-block one
            // cannot, which is recorded in `TranscriptCorpus.perBlockLimitations`.
            mustContain: ["See the docs and the guide for details."],
            mustNotContain: ["][", "https://example.com/docs"]),
        "shell-dollars": Expectation(
            usedFallback: false,
            mustContain: ["echo $HOME", "$HOME/$PATH", "$x=$y"],
            mustNotContain: ["```"]),
        "table-empty-cells": Expectation(
            usedFallback: false,
            // Every position is a cell, blank ones included.
            mustContain: [
                "Feature\nBasic\nPro\nPrice\n$20\n\nSSO\n\nyes\nNotes\nshort\n",
                "The blank cells keep their columns.",
            ],
            mustNotContain: ["| --- |", "|"]),
        "table-in-list": Expectation(
            usedFallback: true,
            mustContain: ["| Tier | Cost |", "- The table below sits inside this item:"]),
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
        "tab-indented-fence": Expectation(
            usedFallback: false,
            // A tab is four columns; one column of fence indent comes off it.
            mustContain: ["   tab indented\n   space indented"],
            mustNotContain: ["```", "\t"]),
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

        // Every destination that reached the text view as a real `.link` is one
        // this transcript is willing to open.
        result.attributedString.enumerateAttribute(
            .link,
            in: NSRange(location: 0, length: result.attributedString.length),
            options: []) { value, range, _ in
            guard value != nil else { return }
            let url = value as? URL
            #expect(
                ["http", "https", "mailto"].contains(url?.scheme?.lowercased() ?? ""),
                "\(fixture) links to \(String(describing: value)) at \(range)")
        }

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
