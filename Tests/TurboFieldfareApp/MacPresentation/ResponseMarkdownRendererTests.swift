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
                                                  effectiveRange: nil) as? URL
            == URL(string: "https://example.com"))
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

    /// The gate counted "```" delimiters, so one in the middle of a sentence
    /// was an odd count and the whole answer — heading, table and all — was
    /// shown as raw source. Only a line that is a fence line opens a block.
    @Test func aMidLineBacktickRunNoLongerForcesRawText() {
        let source = "# Title\n\nA mid-line ``` in a sentence.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(!result.attributedString.string.contains("# Title"))
        #expect(result.attributedString.string.contains("A mid-line ``` in a sentence."))
    }

    /// A backtick fence's info string may not contain a backtick, so this is a
    /// closed code span and the answer after it is not inside a code block.
    @Test func aBacktickInfoStringIsACodeSpanNotAnOpenFence() {
        let source = "```bash``` is the fence syntax.\n\n# Heading\n\nDone."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string
            == "bash is the fence syntax.\n\nHeading\n\nDone.")
    }

    @Test func blockLevelHTMLStaysReadableAsRawText() {
        let renderer = ResponseMarkdownRenderer()
        let samples = [
            "<div>Never execute this</div>",
            "<details>\n<summary>open</summary>\nbody\n</details>",
        ]

        for source in samples {
            let result = renderer.render(source)
            #expect(result.usedFallback)
            #expect(result.attributedString.string == source)
        }
    }

    // The old gate matched any `</?[A-Za-z][^>]*>` in the source, so a
    // comparison between two math spans read as an HTML tag and pushed the
    // whole answer to raw text.
    @Test func comparisonOperatorsBetweenMathSpansNoLongerForceRawText() {
        let typesetter = FakeMathTypesetter()
        let result = ResponseMarkdownRenderer(typesetter: typesetter)
            .render("Prove $a<b$ and $b>c$.")

        #expect(!result.usedFallback)
        #expect(result.attributedString.string == "Prove \u{FFFC} and \u{FFFC}.")
        #expect(typesetter.calls.map(\.latex) == ["a<b", "b>c"])
    }

    @Test func realHTMLTagsStillFallBackToRawText() {
        let source = "<div>markup</div>"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(result.usedFallback)
        #expect(result.attributedString.string == source)
    }

    @Test func codeBlockLinesShareOneContainerAndDropParagraphSpacing() throws {
        let source = "Before\n\n```swift\nlet a = 1\nlet b = 2\n```\n\nAfter"
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text.contains("let a = 1\nlet b = 2"))

        let first = (text as NSString).range(of: "let a = 1")
        let second = (text as NSString).range(of: "let b = 2")
        let firstStyle = try #require(result.attributedString.attribute(
            .paragraphStyle, at: first.location, effectiveRange: nil) as? NSParagraphStyle)
        let secondStyle = try #require(result.attributedString.attribute(
            .paragraphStyle, at: second.location, effectiveRange: nil) as? NSParagraphStyle)

        #expect(firstStyle.paragraphSpacing == 0)
        #expect(firstStyle.lineSpacing == 2)
        let firstBlock = try #require(firstStyle.textBlocks.first as? NSTextTableBlock)
        let secondBlock = try #require(secondStyle.textBlocks.first as? NSTextTableBlock)
        // TextKit 1 merges consecutive paragraphs into one cell only when they
        // reference the same block object, so a per-line block would draw a
        // separate box around every line.
        #expect(firstBlock === secondBlock)
        #expect(firstBlock.backgroundColor?.isEqual(NSColor.quaternarySystemFill) == true)
        #expect(firstBlock.width(for: .padding, edge: .minX) == 8)
        #expect(firstBlock.width(for: .border, edge: .minX) == 1)
        #expect(firstBlock.borderColor(for: .minX)?.isEqual(NSColor.separatorColor) == true)

        let paragraphStyle = try #require(result.attributedString.attribute(
            .paragraphStyle,
            at: (text as NSString).range(of: "Before").location,
            effectiveRange: nil) as? NSParagraphStyle)
        #expect(paragraphStyle.textBlocks.isEmpty)
    }

    @Test func twoCodeBlocksGetSeparateContainers() throws {
        let source = "```\nfirst\n```\n\ntext\n\n```\nsecond\n```"
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        let firstStyle = try #require(result.attributedString.attribute(
            .paragraphStyle,
            at: (text as NSString).range(of: "first").location,
            effectiveRange: nil) as? NSParagraphStyle)
        let secondStyle = try #require(result.attributedString.attribute(
            .paragraphStyle,
            at: (text as NSString).range(of: "second").location,
            effectiveRange: nil) as? NSParagraphStyle)
        #expect(firstStyle.textBlocks.first !== secondStyle.textBlocks.first)
    }

    @Test func inlineCodeUsesTheVisibleFillRatherThanTheControlBackground() throws {
        let result = ResponseMarkdownRenderer().render("Call `reduce(into:)` first.")
        let range = (result.attributedString.string as NSString).range(of: "reduce(into:)")
        let fill = try #require(result.attributedString.attribute(
            .backgroundColor, at: range.location, effectiveRange: nil) as? NSColor)

        #expect(fill.isEqual(NSColor.quaternarySystemFill))
    }

    @Test func inlineHTMLBreakBecomesALineSeparatorWithinTheParagraph() {
        let result = ResponseMarkdownRenderer().render("first<br>second<br/>third<br />fourth")

        #expect(!result.usedFallback)
        #expect(result.attributedString.string
            == "first\u{2028}second\u{2028}third\u{2028}fourth")
    }

    @Test func subscriptAndSuperscriptShiftTheBaselineAndShrinkTheFont() throws {
        let result = ResponseMarkdownRenderer().render("H<sub>2</sub>O and x<sup>2</sup> end")
        let text = result.attributedString.string
        #expect(text == "H2O and x2 end")

        let subscriptRange = (text as NSString).range(of: "2O")
        let superscriptRange = (text as NSString).range(of: "2 end")
        let baseFont = try #require(result.attributedString.attribute(
            .font,
            at: (text as NSString).range(of: "H").location,
            effectiveRange: nil) as? NSFont)

        let subscriptOffset = try #require(result.attributedString.attribute(
            .baselineOffset, at: subscriptRange.location, effectiveRange: nil) as? CGFloat)
        let superscriptOffset = try #require(result.attributedString.attribute(
            .baselineOffset, at: superscriptRange.location, effectiveRange: nil) as? CGFloat)
        #expect(subscriptOffset < 0)
        #expect(superscriptOffset > 0)

        let subscriptFont = try #require(result.attributedString.attribute(
            .font, at: subscriptRange.location, effectiveRange: nil) as? NSFont)
        #expect(subscriptFont.pointSize < baseFont.pointSize)
        #expect(result.attributedString.attribute(
            .baselineOffset,
            at: (text as NSString).range(of: "O and").location,
            effectiveRange: nil) == nil)
    }

    @Test func inlineHTMLEmphasisTagsMapToTraitsAndUnknownTagsKeepTheirText() throws {
        let result = ResponseMarkdownRenderer().render(
            "a<b>bold</b> c<i>it</i> d<kbd>Cmd</kbd> e<span class=\"x\">plain</span>")
        let text = result.attributedString.string
        // A tag with no mapping is shown as written: the parser reports
        // `Vec<String>` as inline HTML too, and deleting it lost the words.
        #expect(text == "abold cit dCmd e<span class=\"x\">plain</span>")

        let boldFont = try #require(result.attributedString.attribute(
            .font, at: (text as NSString).range(of: "bold").location,
            effectiveRange: nil) as? NSFont)
        #expect(boldFont.fontDescriptor.symbolicTraits.contains(.bold))

        let italicFont = try #require(result.attributedString.attribute(
            .font, at: (text as NSString).range(of: "it ").location,
            effectiveRange: nil) as? NSFont)
        #expect(italicFont.fontDescriptor.symbolicTraits.contains(.italic))

        let kbdFont = try #require(result.attributedString.attribute(
            .font, at: (text as NSString).range(of: "Cmd").location,
            effectiveRange: nil) as? NSFont)
        #expect(kbdFont.fontDescriptor.symbolicTraits.contains(.monoSpace))

        let plainFont = try #require(result.attributedString.attribute(
            .font, at: (text as NSString).range(of: "plain").location,
            effectiveRange: nil) as? NSFont)
        #expect(!plainFont.fontDescriptor.symbolicTraits.contains(.bold))
        #expect(!plainFont.fontDescriptor.symbolicTraits.contains(.monoSpace))
    }

    /// Generic parameters and comparisons are the common case: the parser
    /// calls them inline HTML, and dropping the run deleted the type name from
    /// the sentence while the reader watched.
    @Test func unrecognisedAngleBracketTextSurvivesTheHTMLPass() {
        let source = "Use Vec<String> and a <b>bold</b> word"
        let renderer = ResponseMarkdownRenderer()
        let text = renderer.render(source).attributedString.string
        #expect(text == "Use Vec<String> and a bold word")
        #expect(renderer.plainText(source) == "Use Vec<String> and a bold word")
    }

    @Test func inlineHTMLStyleDoesNotLeakPastItsBlock() throws {
        let result = ResponseMarkdownRenderer().render("<b>opened but never closed\n\nnext paragraph")
        let text = result.attributedString.string
        let font = try #require(result.attributedString.attribute(
            .font, at: (text as NSString).range(of: "next paragraph").location,
            effectiveRange: nil) as? NSFont)

        #expect(!font.fontDescriptor.symbolicTraits.contains(.bold))
    }

    @Test func imageRendersItsAltTextAsALinkAndNeverFetches() throws {
        let result = ResponseMarkdownRenderer().render(
            "Look ![remote](https://example.com/image.png) here")
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text == "Look remote here")
        let range = (text as NSString).range(of: "remote")
        #expect(result.attributedString.attribute(.link, at: range.location,
                                                  effectiveRange: nil) as? URL
            == URL(string: "https://example.com/image.png"))
        #expect(result.attributedString.attribute(.attachment, at: range.location,
                                                  effectiveRange: nil) == nil)
        let color = try #require(result.attributedString.attribute(
            .foregroundColor, at: range.location, effectiveRange: nil) as? NSColor)
        #expect(color.isEqual(NSColor.linkColor))
    }

    /// An image whose alt text is empty produced an empty run, so the picture
    /// left nothing at all on screen where the reader expected something.
    @Test func anImageWithNoAltTextShowsItsDestination() {
        let result = ResponseMarkdownRenderer().render("Look ![](https://example.com/a.png) here")
        let text = result.attributedString.string

        #expect(text == "Look https://example.com/a.png here")
        #expect(!text.unicodeScalars.contains { $0.value == 0xFFFC })
        let range = (text as NSString).range(of: "https://example.com/a.png")
        #expect(result.attributedString.attribute(
            .attachment, at: range.location, effectiveRange: nil) == nil)
        #expect(result.attributedString.attribute(
            .link, at: range.location, effectiveRange: nil) != nil)
    }

    /// The destination reaches the text view as a real `.link`, so anything but
    /// a scheme this transcript is willing to open has to be plain text.
    @Test(arguments: [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "data:text/html,<b>x</b>",
    ])
    func aLinkWithADisallowedSchemeRendersAsPlainText(_ destination: String) throws {
        let result = ResponseMarkdownRenderer().render("Click [here](\(destination)) now")
        let text = result.attributedString.string
        let range = (text as NSString).range(of: "here")

        #expect(!result.usedFallback)
        #expect(text == "Click here now")
        #expect(result.attributedString.attribute(
            .link, at: range.location, effectiveRange: nil) == nil)
        let color = try #require(result.attributedString.attribute(
            .foregroundColor, at: range.location, effectiveRange: nil) as? NSColor)
        #expect(!color.isEqual(NSColor.linkColor))
    }

    @Test func aLinkInsideATableCellKeepsItsDestinationAndItsCell() throws {
        let source = "| Name | Where |\n| --- | --- |\n| Docs | [site](https://example.com) |"
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string as NSString
        let range = text.range(of: "site")

        #expect(result.attributedString.attribute(
            .link, at: range.location, effectiveRange: nil) as? URL
            == URL(string: "https://example.com"))
        let style = try #require(result.attributedString.attribute(
            .paragraphStyle, at: range.location, effectiveRange: nil) as? NSParagraphStyle)
        #expect(style.textBlocks.first is NSTextTableBlock)
    }

    /// `<b then c>` is a valid CommonMark open tag with two bare attributes, so
    /// the run was dropped and everything after it went bold. Requiring every
    /// attribute to carry a value is what separates markup from prose.
    @Test(arguments: [
        ("if a<b then c>d holds", "if a<b then c>d holds"),
        ("<b/>x", "<b/>x"),
        ("</b y>x", "</b y>x"),
        ("<b =\"v\">x", "<b =\"v\">x"),
        ("<B>x</B> y", "x y"),
        ("<b><i>x</i></b> y", "x y"),
    ])
    func onlyCompleteTagsAreTreatedAsMarkup(_ probe: (String, String)) {
        let result = ResponseMarkdownRenderer().render(probe.0)
        #expect(result.attributedString.string == probe.1, "\(probe.0.debugDescription)")
    }

    @Test func proseThatParsesAsATagIsNeverStyled() throws {
        let source = "if a<b then c>d holds for every d"
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string as NSString
        let font = try #require(result.attributedString.attribute(
            .font, at: text.range(of: "holds").location, effectiveRange: nil) as? NSFont)
        #expect(!font.fontDescriptor.symbolicTraits.contains(.bold))
    }

    @Test func anAttributeWithAValueStillApplies() throws {
        let result = ResponseMarkdownRenderer().render(
            "a<code class=\"x\">mono</code> b<sup id='n'>up</sup>")
        let text = result.attributedString.string
        #expect(text == "amono bup")

        let mono = try #require(result.attributedString.attribute(
            .font, at: (text as NSString).range(of: "mono").location,
            effectiveRange: nil) as? NSFont)
        #expect(mono.fontDescriptor.symbolicTraits.contains(.monoSpace))
        #expect(result.attributedString.attribute(
            .baselineOffset,
            at: (text as NSString).range(of: "up").location,
            effectiveRange: nil) as? CGFloat ?? 0 > 0)
    }

    /// Marker and ordinal come from the innermost list, not the outermost.
    /// Reading the whole intent chain last-write-wins numbered a nested bullet
    /// and made every nested ordered item repeat its parent's ordinal.
    @Test func nestedListItemsWearTheirOwnMarker() {
        let source = """
            1. First
                - bullet under an ordered item
                - a second bullet
            2. Second
                1. inner one
                2. inner two
            """
        let text = ResponseMarkdownRenderer().render(source).attributedString.string
        #expect(text.contains("1.\tFirst"))
        #expect(text.contains("\u{2022}\tbullet under an ordered item"))
        #expect(text.contains("\u{2022}\ta second bullet"))
        #expect(text.contains("2.\tSecond"))
        #expect(text.contains("1.\tinner one"))
        #expect(text.contains("2.\tinner two"))
        #expect(!text.contains("1.\tbullet"))
        #expect(!text.contains("2.\t1."))
    }

    /// A bold-only line becomes its own paragraph, except inside a fence,
    /// where the bytes are code and the streamed render already drew them.
    @Test func boldHeadingPromotionStopsAtAFence() {
        let renderer = ResponseMarkdownRenderer()
        let promoted = renderer.render("**Note**\nbody text").attributedString.string
        #expect(promoted == "Note\n\nbody text")

        let fenced = renderer.render(
            "Intro.\n\n```text\n**Note**\nkeep this line attached\n```\n\nAfter.")
        #expect(!fenced.usedFallback)
        #expect(fenced.attributedString.string.contains("**Note**\nkeep this line attached"))
    }

    @Test func taskListMarkersBecomeCheckboxGlyphs() {
        let source = "- [x] done\n- [X] also done\n- [ ] open\n- plain\n\n1. [x] first\n2. [ ] second"
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text.contains("\u{2611}\tdone"))
        #expect(text.contains("\u{2611}\talso done"))
        #expect(text.contains("\u{2610}\topen"))
        #expect(text.contains("\u{2022}\tplain"))
        #expect(text.contains("1.\t\u{2611} first"))
        #expect(text.contains("2.\t\u{2610} second"))
        #expect(!text.contains("[x]"))
        #expect(!text.contains("[ ]"))
    }

    @Test func tableRendersAsNativeCellsWithColumnAlignment() throws {
        let source = "| L | C | R |\n| :--- | :---: | ---: |\n| **a** | b<br>c | d |"
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text == "L\nC\nR\na\nb\u{2028}c\nd\n")

        func style(_ needle: String) throws -> NSParagraphStyle {
            let range = (text as NSString).range(of: needle)
            return try #require(result.attributedString.attribute(
                .paragraphStyle, at: range.location, effectiveRange: nil) as? NSParagraphStyle)
        }
        #expect(try style("L").alignment == .left)
        #expect(try style("C").alignment == .center)
        #expect(try style("R").alignment == .right)
        #expect(try style("a").alignment == .left)
        #expect(try style("d").alignment == .right)
    }

    @Test func tableCellsShareOneTableAndOnlyTheHeaderRowIsFilled() throws {
        let source = "| L | C | R |\n| :--- | :---: | ---: |\n| a | b | c |"
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        func cell(_ needle: String) throws -> NSTextTableBlock {
            let range = (text as NSString).range(of: needle)
            let style = try #require(result.attributedString.attribute(
                .paragraphStyle, at: range.location, effectiveRange: nil) as? NSParagraphStyle)
            return try #require(style.textBlocks.first as? NSTextTableBlock)
        }

        let header = try cell("L")
        let body = try cell("a")
        let last = try cell("c")
        #expect(header.table === body.table)
        #expect(header.table === last.table)
        #expect(header.table.numberOfColumns == 3)
        #expect(header.table.collapsesBorders)
        #expect(header.startingRow == 0)
        #expect(header.startingColumn == 0)
        #expect(body.startingRow == 1)
        #expect(body.startingColumn == 0)
        #expect(last.startingColumn == 2)
        #expect(header.backgroundColor?.isEqual(NSColor.quaternarySystemFill) == true)
        #expect(body.backgroundColor == nil)
        #expect(header.width(for: .padding, edge: .minX) == 6)
        #expect(header.width(for: .border, edge: .minX) == 1)
        #expect(header.borderColor(for: .minX)?.isEqual(NSColor.separatorColor) == true)
    }

    /// Foundation emits no run for an empty cell, and none at all for the
    /// cells a short row never wrote, so no text block was created for those
    /// positions and TextKit laid every body row out with the wrong column
    /// count. The grid has to be complete before anything is drawn.
    static let raggedTable = """
        | Feature | Basic | Pro |
        | --- | --- | --- |
        | Price | $20 |  |
        | SSO |  | yes |
        | Notes | short |
        """

    @Test func aTableWithEmptyAndMissingCellsStillDrawsAFullGrid() throws {
        let result = ResponseMarkdownRenderer().render(Self.raggedTable)
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text == "Feature\nBasic\nPro\nPrice\n$20\n\nSSO\n\nyes\nNotes\nshort\n\n")

        var blocks: [NSTextTableBlock] = []
        result.attributedString.enumerateAttribute(
            .paragraphStyle,
            in: NSRange(location: 0, length: result.attributedString.length),
            options: []) { value, _, _ in
            guard let block = (value as? NSParagraphStyle)?.textBlocks
                .first as? NSTextTableBlock else {
                return
            }
            if blocks.last !== block { blocks.append(block) }
        }
        #expect(blocks.count == 12)
        #expect(blocks.map { ($0.startingRow, $0.startingColumn) }.map { "\($0.0),\($0.1)" }
            == ["0,0", "0,1", "0,2", "1,0", "1,1", "1,2", "2,0", "2,1", "2,2", "3,0", "3,1", "3,2"])
        #expect(Set(blocks.map { ObjectIdentifier($0.table) }).count == 1)
        #expect(blocks[0].table.numberOfColumns == 3)
        // The filler keeps its row's identity: a header fill, a body cell not.
        #expect(blocks[2].backgroundColor?.isEqual(NSColor.quaternarySystemFill) == true)
        #expect(blocks[5].backgroundColor == nil)
    }

    @Test func anEmptyHeaderCellKeepsTheHeaderFill() throws {
        let source = "| A |  | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(result.attributedString.string == "A\n\nC\n1\n2\n3\n")
        let text = result.attributedString.string as NSString
        // The empty header cell is the newline between "A" and "C".
        let style = try #require(result.attributedString.attribute(
            .paragraphStyle,
            at: text.range(of: "A\n\nC").location + 2,
            effectiveRange: nil) as? NSParagraphStyle)
        let block = try #require(style.textBlocks.first as? NSTextTableBlock)
        #expect(block.startingRow == 0)
        #expect(block.startingColumn == 1)
        #expect(block.backgroundColor?.isEqual(NSColor.quaternarySystemFill) == true)
    }

    @Test func theRaggedTablePlainTextKeepsEveryCellBoundary() {
        #expect(ResponseMarkdownRenderer().plainText(Self.raggedTable)
            == "Feature\nBasic\nPro\nPrice\n$20\n\nSSO\n\nyes\nNotes\nshort\n\n")
    }

    @Test func tableHeaderCellsAreSemiboldAndBodyCellsAreNot() throws {
        let source = "| Tier | Price |\n| --- | --- |\n| Basic | 20 |"
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string as NSString

        let header = try #require(result.attributedString.attribute(
            .font, at: text.range(of: "Tier").location, effectiveRange: nil) as? NSFont)
        let body = try #require(result.attributedString.attribute(
            .font, at: text.range(of: "Basic").location, effectiveRange: nil) as? NSFont)
        #expect(header.fontDescriptor.symbolicTraits.contains(.bold))
        #expect(!body.fontDescriptor.symbolicTraits.contains(.bold))
    }

    @Test func twoTablesInOneAnswerDoNotShareATable() throws {
        let source = "| A |\n| --- |\n| 1 |\n\ntext\n\n| B |\n| --- |\n| 2 |"
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        func table(_ needle: String) throws -> NSTextTable {
            let range = (text as NSString).range(of: needle)
            let style = try #require(result.attributedString.attribute(
                .paragraphStyle, at: range.location, effectiveRange: nil) as? NSParagraphStyle)
            return try #require((style.textBlocks.first as? NSTextTableBlock)?.table)
        }
        #expect(try table("A") !== table("B"))
    }

    @Test func tableCellKeepsBoldAndInlineHTMLBreaks() throws {
        let source = "| Feature | Detail |\n| --- | --- |\n| **Model** | one<br>two |"
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        #expect(text.contains("one\u{2028}two"))
        let boldRange = (text as NSString).range(of: "Model")
        let font = try #require(result.attributedString.attribute(
            .font, at: boldRange.location, effectiveRange: nil) as? NSFont)
        #expect(font.fontDescriptor.symbolicTraits.contains(.bold))
        let breakStyle = try #require(result.attributedString.attribute(
            .paragraphStyle,
            at: (text as NSString).range(of: "two").location,
            effectiveRange: nil) as? NSParagraphStyle)
        #expect(breakStyle.textBlocks.first is NSTextTableBlock)
    }

    /// A table nested in a quote or a list loses the quote bar and the list
    /// marker entirely, because the cell walk owns the paragraph style. Raw
    /// fallback keeps the content with its markers until the renderer can nest
    /// containers.
    @Test func aTableInsideAQuoteOrAListFallsBackToRawText() {
        let renderer = ResponseMarkdownRenderer()
        for source in [
            "> | A | B |\n> | --- | --- |\n> | 1 | 2 |",
            "- item\n\n    | A | B |\n    | --- | --- |\n    | 1 | 2 |",
        ] {
            let result = renderer.render(source)
            #expect(result.usedFallback, "\(source.debugDescription)")
            #expect(result.attributedString.string == source, "\(source.debugDescription)")
        }
        // The other direction: a table of its own still renders natively.
        let plain = renderer.render("| A | B |\n| --- | --- |\n| 1 | 2 |")
        #expect(!plain.usedFallback)
        #expect(!plain.attributedString.string.contains("| --- |"))
    }

    @Test func tableIsSeparatedFromNeighbouringParagraphs() {
        let source = "Before the table.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nAfter the table."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string
            == "Before the table.\n\nA\nB\n1\n2\n\nAfter the table.")
    }

    @Test func plainTextOfATableIsReadableCellText() {
        let renderer = ResponseMarkdownRenderer()
        let source = "| Tier | Price |\n| --- | ---: |\n| Basic | $20 |"

        #expect(renderer.plainText(source) == "Tier\nPrice\nBasic\n$20\n")
    }

    /// Replaces the old `latexRemainsReadableText` pin: with math on, the same
    /// answer typesets; with the kill switch set, it is the raw text again.
    @Test func killSwitchRestoresRawLatexText() {
        let source = "Cosine is $\\frac{u \\cdot v}{||u|| ||v||}$."
        let typesetter = FakeMathTypesetter()
        let enabled = ResponseMarkdownRenderer(
            environment: [:],
            typesetter: typesetter).render(source)
        #expect(!enabled.usedFallback)
        #expect(enabled.attributedString.string == "Cosine is \u{FFFC}.")

        let disabled = ResponseMarkdownRenderer(
            environment: ["TURBO_FIELDFARE_DISABLE_MATH": "1"],
            typesetter: typesetter).render(source)
        #expect(!disabled.usedFallback)
        #expect(disabled.attributedString.string.contains("\\frac"))
        #expect(disabled.attributedString.string.contains("\\cdot"))
        // The disabled render must not have reached the typesetter at all.
        #expect(typesetter.calls.count == 1)
    }

    @Test func boldOnlyModelHeadingStaysOnItsOwnLine() {
        let source = "**Origins**\nFieldfares arrive from northern Europe."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string
            == "Origins\n\nFieldfares arrive from northern Europe.")
    }

    // MARK: - Math

    static func attachment(
        _ result: ResponseMarkdownRenderer.Result,
        at location: Int
    ) -> MathAttachment? {
        result.attributedString.attribute(
            .attachment,
            at: location,
            effectiveRange: nil) as? MathAttachment
    }

    static func paragraphStyle(
        _ result: ResponseMarkdownRenderer.Result,
        at location: Int
    ) -> NSParagraphStyle? {
        result.attributedString.attribute(
            .paragraphStyle,
            at: location,
            effectiveRange: nil) as? NSParagraphStyle
    }

    @Test func inlineMathBecomesAnAttachmentAndLeavesTheMarkdownAroundItStyled() throws {
        let typesetter = FakeMathTypesetter()
        let result = ResponseMarkdownRenderer(typesetter: typesetter)
            .render("- **Root** of $x^2 = 4$ is two")
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text == "\u{2022}\tRoot of \u{FFFC} is two")
        let placeholder = (text as NSString).range(of: "\u{FFFC}")
        let attachment = try #require(Self.attachment(result, at: placeholder.location))
        #expect(attachment.latexSource == "$x^2 = 4$")
        #expect(attachment.image?.accessibilityDescription == "$x^2 = 4$")
        #expect(typesetter.calls.map(\.latex) == ["x^2 = 4"])

        let boldRange = (text as NSString).range(of: "Root")
        let boldFont = try #require(result.attributedString.attribute(
            .font, at: boldRange.location, effectiveRange: nil) as? NSFont)
        #expect(boldFont.fontDescriptor.symbolicTraits.contains(.bold))
    }

    /// Every one of these is altered by `AttributedString(markdown:)`: the
    /// backslash before `,` disappears, `\{`/`\}` unescape, and `\[...\]` loses
    /// its delimiters entirely. The typesetter must see the source bytes.
    @Test func mangledLatexReachesTheTypesetterByteIdentical() {
        let typesetter = FakeMathTypesetter()
        _ = ResponseMarkdownRenderer(typesetter: typesetter).render(
            #"$a \, b$ then $\{x\}$ then \[x = 1\] then $a*b*c$"#)

        #expect(typesetter.calls.map(\.latex) == [
            #"a \, b"#, #"\{x\}"#, #"x = 1"#, #"a*b*c"#,
        ])
    }

    @Test func aTypesetFailureLeavesTheRawSpanTextInPlace() throws {
        let typesetter = FakeMathTypesetter()
        typesetter.failing = [#"\bogus"#]
        let result = ResponseMarkdownRenderer(typesetter: typesetter)
            .render(#"Broken $\bogus$ but $x$ works."#)
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text == #"Broken $\bogus$ but "# + "\u{FFFC} works.")
        let rawRange = (text as NSString).range(of: #"\bogus"#)
        #expect(Self.attachment(result, at: rawRange.location) == nil)
        let font = try #require(result.attributedString.attribute(
            .font, at: rawRange.location, effectiveRange: nil) as? NSFont)
        #expect(font.pointSize == NSFont.systemFontSize)
    }

    @Test func standaloneDisplayMathGetsItsOwnCentredParagraph() throws {
        let typesetter = FakeMathTypesetter()
        let result = ResponseMarkdownRenderer(typesetter: typesetter)
            .render("Subtract $c$:\n$$ax^2 = -c$$\nDone.")
        let text = result.attributedString.string

        #expect(text == "Subtract \u{FFFC}:\n\n\u{FFFC}\n\nDone.")
        let display = (text as NSString).range(
            of: "\u{FFFC}",
            options: .backwards)
        let style = try #require(Self.paragraphStyle(result, at: display.location))
        #expect(style.alignment == .center)

        let inline = (text as NSString).range(of: "\u{FFFC}")
        let inlineStyle = try #require(Self.paragraphStyle(result, at: inline.location))
        #expect(inlineStyle.alignment != .center)
        #expect(typesetter.calls.map(\.mode) == [.inline, .display])
    }

    @Test func twoConsecutiveDisplayBlocksBecomeTwoParagraphs() {
        let typesetter = FakeMathTypesetter()
        let result = ResponseMarkdownRenderer(typesetter: typesetter)
            .render("$$a = 1$$\n$$b = 2$$")

        #expect(result.attributedString.string == "\u{FFFC}\n\n\u{FFFC}")
        #expect(typesetter.calls.map(\.latex) == ["a = 1", "b = 2"])
    }

    @Test func displayMathInsideAListItemKeepsTheListIndentAndIsNotCentred() throws {
        let typesetter = FakeMathTypesetter()
        let result = ResponseMarkdownRenderer(typesetter: typesetter)
            .render("- step $$x = 1$$ here")
        let text = result.attributedString.string

        #expect(text == "\u{2022}\tstep \u{FFFC} here")
        let placeholder = (text as NSString).range(of: "\u{FFFC}")
        let style = try #require(Self.paragraphStyle(result, at: placeholder.location))
        #expect(style.alignment != .center)
        #expect(style.headIndent == 22)
        #expect(typesetter.calls.map(\.mode) == [.display])
    }

    /// Gemma writes `cases` and `aligned` inside single dollars. Alone in a
    /// paragraph they are block notation and typeset in display style; in the
    /// middle of a sentence they stay inline.
    @Test func inlineEnvironmentsTypesetAsDisplayOnlyWhenTheyOwnTheParagraph() {
        let typesetter = FakeMathTypesetter()
        let source = #"$\begin{aligned} a &= b \end{aligned}$"#
        _ = ResponseMarkdownRenderer(typesetter: typesetter).render(source)
        _ = ResponseMarkdownRenderer(typesetter: typesetter).render("Value \(source) here")

        #expect(typesetter.calls.map(\.mode) == [.display, .inline])
    }

    @Test func mathInAHeadingAndAQuoteInheritsTheirFontSizeAndColour() throws {
        let typesetter = FakeMathTypesetter()
        _ = ResponseMarkdownRenderer(typesetter: typesetter).render("## Solve $x$")
        let headingSize = try #require(typesetter.calls.first?.fontSize)
        #expect(headingSize > NSFont.systemFontSize)

        let quoteTypesetter = FakeMathTypesetter()
        _ = ResponseMarkdownRenderer(typesetter: quoteTypesetter).render("> cite $x$")
        let quote = try #require(quoteTypesetter.calls.first)
        #expect(quote.fontSize == NSFont.systemFontSize)
        #expect(quote.tint.isEqual(NSColor.secondaryLabelColor))
    }

    @Test func attachmentSitsOnTheBaselineUsingTheReportedDescent() throws {
        let result = ResponseMarkdownRenderer(typesetter: FakeMathTypesetter())
            .render("Inline $x$ here")
        let placeholder = (result.attributedString.string as NSString).range(of: "\u{FFFC}")
        let attachment = try #require(Self.attachment(result, at: placeholder.location))

        #expect(attachment.bounds.origin.y == -FakeMathTypesetter.descent)
        #expect(attachment.bounds.height == FakeMathTypesetter.ascent + FakeMathTypesetter.descent)
    }

    /// A sentinel inside a link destination is percent-encoded into the `.link`
    /// attribute and vanishes from the character stream. A whole-message raw
    /// fallback is the only honest answer: the alternative is a silently
    /// deleted equation.
    @Test func aSentinelSwallowedByALinkDestinationFallsBackToRawText() {
        let source = "See [tier](http://example.com/?q=$a$) now."
        let result = ResponseMarkdownRenderer(typesetter: FakeMathTypesetter())
            .render(source)

        #expect(result.usedFallback)
        #expect(result.attributedString.string == source)
    }

    /// Whichever gate fires, the raw text shown must be the original source and
    /// never the sentinel working copy.
    @Test func fallbackGatesShowTheOriginalSourceNotTheWorkingCopy() {
        let renderer = ResponseMarkdownRenderer(typesetter: FakeMathTypesetter())
        for source in [
            "Math $x$ then\n\n```swift\nlet a = 1",
            "<div>block</div>\n\nMath $x$ after",
        ] {
            let result = renderer.render(source)
            #expect(result.usedFallback, "\(source.debugDescription)")
            #expect(result.attributedString.string == source)
            #expect(!result.attributedString.string.unicodeScalars
                .contains { $0.value == 0xE000 })
        }
    }

    @Test func plainTextReturnsTheLatexSourceRatherThanThePlaceholder() {
        let renderer = ResponseMarkdownRenderer(typesetter: FakeMathTypesetter())
        let plain = renderer.plainText("Roots of $x^2 = 4$ and\n$$y = 1$$\ndone.")

        #expect(plain == "Roots of $x^2 = 4$ and\n\n$$y = 1$$\n\ndone.")
        #expect(!plain.unicodeScalars.contains { $0.value == 0xFFFC })
    }

    @Test func literalProtectSpansAreRestoredVerbatimAndNeverTypeset() {
        let typesetter = FakeMathTypesetter()
        let source = "Gradient is\n\n$$\\frac{\\partial L}{\\partial w} = \\sum_i x_i"
        let result = ResponseMarkdownRenderer(typesetter: typesetter).render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string
            == "Gradient is\n\n$$\\frac{\\partial L}{\\partial w} = \\sum_i x_i")
        #expect(typesetter.calls.isEmpty)
    }

    /// The shield turned an ordinary sentence into raw source for the rest of
    /// the answer: the heading, the bold run and the list below it all showed
    /// their markers.
    @Test(arguments: [
        "It costs $$$ a lot.\n**Bold** line\n# Heading",
        "Rated $$ on the price scale.\n- item one\n- item two",
        "See \\[ in the note.\n# Heading",
    ])
    func aStrayDisplayOpenerInProseLeavesTheAnswerStyled(_ source: String) {
        let typesetter = FakeMathTypesetter()
        let result = ResponseMarkdownRenderer(typesetter: typesetter).render(source)
        let text = result.attributedString.string

        #expect(!result.usedFallback, "\(source.debugDescription)")
        #expect(!text.contains("**"), "\(source.debugDescription)")
        #expect(!text.contains("# Heading"), "\(source.debugDescription)")
        #expect(typesetter.calls.isEmpty)
    }

    /// The bracket pair carried prose into the typesetter, which failed, and
    /// the words came back as raw source with their backslashes.
    @Test func anEscapedBracketPairInProseRendersAsBrackets() {
        let typesetter = FakeMathTypesetter()
        let result = ResponseMarkdownRenderer(typesetter: typesetter)
            .render(#"the array \[1, 2, 3\] holds three values"#)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string == "the array [1, 2, 3] holds three values")
        #expect(typesetter.calls.isEmpty)
    }

    /// Two lone backticks paragraphs apart used to pair into one code span
    /// across everything between them. The equation in the middle typeset
    /// while the answer streamed, because each paragraph was rendered on its
    /// own, and then reverted to raw source at finalize when the whole message
    /// went through one pass.
    @Test func loneBackticksInSeparateParagraphsDoNotMaskTheMathBetweenThem() {
        let typesetter = FakeMathTypesetter()
        let source = """
            Press the ` key to open the console.

            The relation is $E = mc^2$ exactly.

            Use the ` character to quote a word.
            """
        let result = ResponseMarkdownRenderer(typesetter: typesetter).render(source)

        #expect(!result.usedFallback)
        #expect(typesetter.calls.map(\.latex) == ["E = mc^2"])
        #expect(!result.attributedString.string.contains("$E = mc^2$"))
        // Each paragraph keeps the backtick it wrote; only the span between
        // them was ever in question.
        #expect(result.attributedString.string.contains("Press the ` key"))
        #expect(result.attributedString.string.contains("Use the ` character"))
    }

    /// Two spans with the same LaTeX share one memoised image, so writing the
    /// accessibility description onto it gave both attachments whichever
    /// description was written last.
    @Test func eachEquationKeepsItsOwnAccessibilityDescription() throws {
        let result = ResponseMarkdownRenderer()
            .render(#"Both $x^2$ and \(x^2\) say the same thing."#)
        var descriptions: [String] = []
        result.attributedString.enumerateAttribute(
            .attachment,
            in: NSRange(location: 0, length: result.attributedString.length),
            options: []) { value, _, _ in
            guard let attachment = value as? MathAttachment else { return }
            descriptions.append(attachment.image?.accessibilityDescription ?? "")
        }
        #expect(descriptions == ["$x^2$", #"\(x^2\)"#])
    }

    @Test func currencyAndShellDollarsNeverReachTheTypesetter() {
        let typesetter = FakeMathTypesetter()
        let renderer = ResponseMarkdownRenderer(typesetter: typesetter)
        for source in [
            "The base plan is $20 per month and the pro plan is $30 per month.",
            "In prose the variables are $HOME and $PATH on one line.",
            "A shell comparison such as $x=$y must stay text.",
            "```bash\necho $HOME\nexport T=\"$HOME/$PATH\"\n```",
        ] {
            let result = renderer.render(source)
            #expect(!result.usedFallback, "\(source.debugDescription)")
        }
        #expect(typesetter.calls.isEmpty)
    }
}

@MainActor
@Suite struct InstructionTranscriptDocumentControllerTests {
    /// Streaming without the progressive render: the answer is appended as raw
    /// text and only the finalize pass styles it. These tests are what
    /// `TURBO_FIELDFARE_PROGRESSIVE_RENDER=0` has to keep doing.
    static func rawStreamingController(
        renderer: any TranscriptBlockRendering = ResponseMarkdownRenderer()
    ) -> InstructionTranscriptDocumentController {
        InstructionTranscriptDocumentController(
            renderer: renderer,
            environment: ["TURBO_FIELDFARE_PROGRESSIVE_RENDER": "0"])
    }

    @Test func appAccentMatchesProductRGB() {
        let color = TurboFieldfareMacTheme.accentNSColor
            .usingColorSpace(.sRGB)
        #expect(color != nil)
        #expect(abs((color?.redComponent ?? 0) - 106.0 / 255.0) < 0.000_001)
        #expect(abs((color?.greenComponent ?? 0) - 186.0 / 255.0) < 0.000_001)
        #expect(abs((color?.blueComponent ?? 0) - 113.0 / 255.0) < 0.000_001)
    }

    /// The raw streaming path, which `TURBO_FIELDFARE_PROGRESSIVE_RENDER=0`
    /// restores. Its progressive counterpart is in
    /// `ProgressiveTranscriptRenderTests`.
    @Test func rebuildsThenAppendsOnlyNewResponseSuffix() {
        let storage = NSMutableAttributedString()
        let controller = Self.rawStreamingController()

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

    @Test func terminalPartialOutputIsReadableAndNextRunRestoresStreamingSource() {
        let storage = NSMutableAttributedString()
        let controller = Self.rawStreamingController()
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
        #expect(storage.string.hasSuffix("Partial **answer**"))
    }

    @Test func terminalResponseRendersAgainWhenClosingFenceArrivesLate() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        let partial = "```cpp\nkernel void matmul() {}"
        let complete = partial + "\n```"

        let first = controller.synchronize(
            storage: storage,
            prompt: "Write a Metal kernel",
            response: partial,
            isTerminal: true)
        #expect(first.mutation == .finalized)
        // The unclosed fence is drawn as code, not shown as raw source: the
        // per-block finalize never puts the answer through the message-wide
        // gate that the whole-document render applies.
        #expect(storage.string.hasSuffix("kernel void matmul() {}\n"))
        #expect(!storage.string.contains("```"))

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

    @Test func streamingAppendsMultiByteAndCJKDeltasExactly() {
        let storage = NSMutableAttributedString()
        let controller = Self.rawStreamingController()
        let answer = "\u{4E2D}\u{6587}: caf\u{E9} \u{30C6}\u{30B9}\u{30C8} $x^2$ done"

        var seen = ""
        for character in answer {
            seen.append(character)
            let result = controller.synchronize(
                storage: storage,
                prompt: "Ask",
                response: seen,
                isTerminal: false)
            #expect(result.mutation == .appended || result.mutation == .rebuilt)
            #expect((storage.string as NSString).substring(with: result.assistantRange) == seen)
        }
        #expect(storage.string == "You\nAsk\n\nAnswer\n" + answer)
    }

    // A combining mark grows the response in bytes while leaving its grapheme
    // count unchanged, so neither the byte-length test nor a grapheme-length
    // test can be trusted on its own to decide that nothing was added.
    @Test func combiningMarkThatKeepsTheGraphemeCountStillReachesTheStorage() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()

        _ = controller.synchronize(
            storage: storage, prompt: "Ask", response: "caf e", isTerminal: false)
        let result = controller.synchronize(
            storage: storage, prompt: "Ask", response: "caf e\u{301}", isTerminal: false)

        #expect(result.mutation != .none)
        #expect(storage.string == "You\nAsk\n\nAnswer\ncaf e\u{301}")
        #expect((storage.string as NSString).substring(with: result.assistantRange)
            == "caf e\u{301}")
    }

    /// Clamping alone kept a selection's length over whatever now sits at
    /// those indices, so a reader with text selected watched the highlight
    /// slide onto different characters on every streaming tick.
    @Test func selectionThatReachedIntoTheRewrittenRangeStopsAtItsBoundary() {
        let adjusted = InstructionTranscriptDocumentController.adjustedRanges(
            [
                NSRange(location: 0, length: 4),
                NSRange(location: 2, length: 10),
                NSRange(location: 10, length: 6),
                NSRange(location: 40, length: 3),
                NSRange(location: 6, length: 0),
            ],
            replacing: NSRange(location: 6, length: 30),
            newLength: 5)

        #expect(adjusted == [
            // Entirely in the stable prefix: exact indices kept.
            NSRange(location: 0, length: 4),
            // Started in the prefix and ran into the rewrite: the prefix part
            // survives, the rest is gone.
            NSRange(location: 2, length: 4),
            // Entirely inside the rewrite: collapsed where it began.
            NSRange(location: 10, length: 0),
            // Below the rewrite: the same characters, moved by the delta.
            // Collapsing these too dropped a selection over text that had not
            // changed at all.
            NSRange(location: 15, length: 3),
            // A caret at the boundary is not inside the rewrite.
            NSRange(location: 6, length: 0),
        ])
    }

    /// A tail re-render rewrites the same sentence with a few characters
    /// added. Reporting the whole tail as replaced collapsed every selection
    /// inside it, including one over text that had not moved.
    @Test(arguments: [
        // old, new, expected offset into `previous`, expected old length, new length
        ("abc", "abcd", 3, 0, 1),
        ("abcd", "abc", 3, 1, 0),
        ("hello world", "hello brave world", 6, 0, 6),
        ("same", "same", -1, 0, 0),
        // A surrogate pair and a composed sequence may not be split.
        ("a\u{1F600}b", "a\u{1F600}c", 3, 1, 1),
        ("cafe\u{301} x", "cafe\u{301} y", 6, 1, 1),
    ])
    func differingTrimsWhatBothSidesShare(_ probe: (String, String, Int, Int, Int)) {
        let replaced = InstructionTranscriptDocumentController.ReplacedRange.differing(
            previous: NSRange(location: 10, length: (probe.0 as NSString).length),
            old: probe.0 as NSString,
            new: probe.1 as NSString)
        guard probe.2 >= 0 else {
            #expect(replaced == nil, "\(probe.0.debugDescription)")
            return
        }
        #expect(replaced?.previous == NSRange(location: 10 + probe.2, length: probe.3),
                "\(probe.0.debugDescription)")
        #expect(replaced?.length == probe.4, "\(probe.0.debugDescription)")
    }

    /// The whole point of trimming: a selection over a completed block above
    /// the open one keeps its exact indices through a streaming tick and
    /// through finalize.
    @Test func aSelectionOverACompletedBlockSurvivesATickAndFinalize() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController(environment: [:])
        _ = controller.synchronize(
            storage: storage, prompt: "Ask", response: "# Title\n\nBody sta", isTerminal: false)
        let heading = (storage.string as NSString).range(of: "Title")
        let selection = [heading]

        let tick = controller.synchronize(
            storage: storage, prompt: "Ask", response: "# Title\n\nBody star", isTerminal: false)
        let afterTick = InstructionTranscriptDocumentController.adjustedRanges(
            selection,
            replacing: try! #require(tick.replaced).previous,
            newLength: try! #require(tick.replaced).length)
        #expect(afterTick == selection)
        #expect((storage.string as NSString).substring(with: afterTick[0]) == "Title")

        let final = controller.synchronize(
            storage: storage, prompt: "Ask", response: "# Title\n\nBody star", isTerminal: true)
        let afterFinalize = final.replaced.map {
            InstructionTranscriptDocumentController.adjustedRanges(
                afterTick, replacing: $0.previous, newLength: $0.length)
        } ?? afterTick
        #expect((storage.string as NSString).substring(with: afterFinalize[0]) == "Title")
    }

    /// The coordinator can only adjust a selection if the controller says what
    /// it rewrote, so the range is part of the result rather than something the
    /// caller infers from `tailRange` after the fact.
    @Test func aTailReplacementReportsTheRangeItRewrote() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController(environment: [:])
        _ = controller.synchronize(
            storage: storage, prompt: "Ask", response: "Done.\n\nab", isTerminal: false)
        let before = controller.tailRange

        let update = controller.synchronize(
            storage: storage, prompt: "Ask", response: "Done.\n\nabc", isTerminal: false)
        #expect(update.mutation == .tailReplaced)
        // Only the character that arrived: the "ab" above it is the same text
        // at the same indices, so a selection over it is not disturbed.
        #expect(update.replaced?.previous
            == NSRange(location: before.upperBound, length: 0))
        #expect(update.replaced?.length == 1)

        // An append that rewrites nothing reports nothing.
        let appended = controller.synchronize(
            storage: storage, prompt: "Ask", response: "Done.\n\nabc\n\nNext.", isTerminal: false)
        #expect(appended.mutation == .tailReplaced || appended.mutation == .appended)
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

    // MARK: - Math at finalize time

    @Test func streamingShowsRawMathAndFinalizationTypesetsIt() {
        let storage = NSMutableAttributedString()
        let typesetter = FakeMathTypesetter()
        let controller = Self.rawStreamingController(
            renderer: ResponseMarkdownRenderer(typesetter: typesetter))
        let answer = "Roots:\n$$x = 1$$"

        var seen = ""
        for character in answer {
            seen.append(character)
            let streaming = controller.synchronize(
                storage: storage,
                prompt: "Solve",
                response: seen,
                isTerminal: false)
            #expect((storage.string as NSString).substring(with: streaming.assistantRange)
                == seen)
        }
        #expect(typesetter.calls.isEmpty)
        #expect(storage.string.hasSuffix("$$x = 1$$"))

        let final = controller.synchronize(
            storage: storage,
            prompt: "Solve",
            response: answer,
            isTerminal: true)
        #expect(final.mutation == .finalized)
        #expect(controller.response == answer)
        #expect(typesetter.calls.map(\.latex) == ["x = 1"])
        #expect((storage.string as NSString).substring(with: final.assistantRange)
            == "Roots:\n\n\u{FFFC}")
        #expect(final.assistantRange.upperBound == storage.length)
    }

    /// Mirror of the late-closing-fence case: a `$$` that closes after the
    /// terminal flag must re-render, not stay half typeset.
    @Test func lateClosingDisplayMathRefinalizesAndTypesets() {
        let storage = NSMutableAttributedString()
        let typesetter = FakeMathTypesetter()
        let controller = InstructionTranscriptDocumentController(
            renderer: ResponseMarkdownRenderer(typesetter: typesetter))
        let partial = "Answer\n\n$$x = \\frac{1}{2}"
        let complete = partial + "$$"

        let first = controller.synchronize(
            storage: storage,
            prompt: "Solve",
            response: partial,
            isTerminal: true)
        #expect(first.mutation == .finalized)
        #expect(typesetter.calls.isEmpty)
        #expect(storage.string.hasSuffix("$$x = \\frac{1}{2}"))

        let updated = controller.synchronize(
            storage: storage,
            prompt: "Solve",
            response: complete,
            isTerminal: true)
        #expect(updated.mutation == .finalized)
        #expect(typesetter.calls.map(\.latex) == [#"x = \frac{1}{2}"#])
        #expect((storage.string as NSString).substring(with: updated.assistantRange)
            == "Answer\n\n\u{FFFC}")
    }

    @Test func resumingAfterFinalizationRestoresTheRawStreamingSource() {
        let storage = NSMutableAttributedString()
        let typesetter = FakeMathTypesetter()
        let controller = InstructionTranscriptDocumentController(
            renderer: ResponseMarkdownRenderer(typesetter: typesetter))
        let answer = "Roots $x = 1$ found"

        _ = controller.synchronize(
            storage: storage,
            prompt: "Solve",
            response: answer,
            isTerminal: true)
        #expect(storage.string.hasSuffix("Roots \u{FFFC} found"))

        let resumed = controller.synchronize(
            storage: storage,
            prompt: "Solve",
            response: answer,
            isTerminal: false)
        #expect(resumed.mutation == .rebuilt)
        #expect(!controller.isFinalized)
        #expect(storage.string.hasSuffix(answer))
    }

    /// The mailbox can drain after the terminal flag is set, so `apply` runs
    /// the terminal path more than once; typesetting must not run again with
    /// it.
    @Test func repeatedTerminalSynchronizeTypesetsTheAnswerOnce() {
        let storage = NSMutableAttributedString()
        let typesetter = FakeMathTypesetter()
        let controller = InstructionTranscriptDocumentController(
            renderer: ResponseMarkdownRenderer(typesetter: typesetter))

        for _ in 0..<3 {
            _ = controller.synchronize(
                storage: storage,
                prompt: "Solve",
                response: "Roots $x = 1$ found",
                isTerminal: true)
        }
        #expect(typesetter.calls.count == 1)
    }

    /// Records the finalize cost of the worst corpus fixture. Fifty equations
    /// is well past one frame cold, which is why the typesetter memoises; the
    /// warm pass is what a rebuild of an already-finalised answer costs.
    @Test func fiftyEquationFinalizeCostIsRecorded() throws {
        let source = try TranscriptCorpus.source("fifty-equations")
        let clock = ContinuousClock()
        // Font registration happens on the first equation of the process.
        _ = ResponseMarkdownRenderer().render("$x$")

        func milliseconds(_ duration: Duration) -> Double {
            Double(duration.components.seconds) * 1e3
                + Double(duration.components.attoseconds) * 1e-15
        }

        // A private cache makes the cold number independent of whatever else
        // ran in this process first.
        let cache = MathRenderCache()
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController(
            renderer: ResponseMarkdownRenderer(
                typesetter: SwiftMathTypesetter(cache: cache)))
        let cold = clock.measure {
            controller.synchronize(
                storage: storage,
                prompt: "Derive",
                response: source,
                isTerminal: true)
        }
        #expect(cache.count == 50)

        // A rebuild while the answer is terminal re-renders the whole
        // transcript; this is the path the memo exists for.
        let warmStorage = NSMutableAttributedString()
        let warmController = InstructionTranscriptDocumentController(
            renderer: ResponseMarkdownRenderer(
                typesetter: SwiftMathTypesetter(cache: cache)))
        let warm = clock.measure {
            warmController.synchronize(
                storage: warmStorage,
                prompt: "Derive",
                response: source,
                isTerminal: true)
        }

        let unchanged = clock.measure {
            controller.synchronize(
                storage: storage,
                prompt: "Derive",
                response: source,
                isTerminal: true)
        }
        print("""
            MATH-FINALIZE fifty-equations \
            cold=\(String(format: "%.1f", milliseconds(cold)))ms \
            warm=\(String(format: "%.1f", milliseconds(warm)))ms \
            unchanged=\(String(format: "%.1f", milliseconds(unchanged)))ms
            """)
        #expect(milliseconds(warm) < milliseconds(cold))
        // An unchanged terminal response must not render at all.
        #expect(milliseconds(unchanged) < 1)
    }

    @Test func terminalFormattingAlwaysScrollsToBottom() {
        #expect(InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: false,
            mutation: .finalized))
        #expect(InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: true,
            mutation: .appended))
        #expect(!InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: false,
            mutation: .appended))
    }
}
