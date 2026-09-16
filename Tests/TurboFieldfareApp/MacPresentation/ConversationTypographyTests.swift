import AppKit
import Testing
import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

@MainActor @Suite struct ConversationTypographyTests {
    @Test func compactWindowKeepsRoomForTranscriptAtLargestSize() {
        let typography = ConversationTypography(.largest)
        #expect(typography.editorHeight(showsExamples: false, availableHeight: 500) == 100)
        #expect(typography.editorHeight(showsExamples: true, availableHeight: 500) == 92)
        #expect(typography.editorHeight(showsExamples: false, availableHeight: 300) == 80)
        #expect(typography.editorHeight(showsExamples: false, availableHeight: 1000) == 168)
        #expect(ConversationTypography().editorHeight(showsExamples: false, availableHeight: 500) == 84)
    }

    @Test(arguments: AppTextSize.allCases)
    func scalesEveryRunWithoutChangingCharactersOrTraits(_ size: AppTextSize) throws {
        let source = """
        # Heading

        Body **bold** *italic* `inline` <sup>script</sup> [link](https://example.com).

        > Quote

        - Item
          - Nested

        ```swift
        let value = 1
        ```

        | Name | Value |
        | --- | --- |
        | one | two |
        """
        let baseline = ResponseMarkdownRenderer().render(source).attributedString
        let scaled = ResponseMarkdownRenderer(typography: .init(size)).render(source).attributedString
        #expect(baseline.string == scaled.string)
        baseline.enumerateAttributes(in: NSRange(location: 0, length: baseline.length)) { attrs, range, _ in
            guard let baseFont = attrs[.font] as? NSFont else { return }
            let new = scaled.attributes(at: range.location, effectiveRange: nil)
            let font = new[.font] as? NSFont
            #expect(font?.pointSize == baseFont.pointSize * CGFloat(size.scale))
            #expect(font?.fontDescriptor.symbolicTraits == baseFont.fontDescriptor.symbolicTraits)
            guard let before = attrs[.paragraphStyle] as? NSParagraphStyle,
                  let after = new[.paragraphStyle] as? NSParagraphStyle else { return }
            #expect(after.lineSpacing == before.lineSpacing * CGFloat(size.scale))
            #expect(after.headIndent == before.headIndent * CGFloat(size.scale))
            #expect(after.paragraphSpacing == before.paragraphSpacing * CGFloat(size.scale))
            for (a, b) in zip(before.textBlocks, after.textBlocks) {
                #expect(b.width(for: .padding, edge: .minX)
                        == a.width(for: .padding, edge: .minX) * CGFloat(size.scale))
            }
        }
    }

    @Test(arguments: AppTextSize.allCases)
    func mathAndRawFallbackUseChosenBodySize(_ size: AppTextSize) throws {
        let fake = FakeMathTypesetter()
        let renderer = ResponseMarkdownRenderer(typography: .init(size), typesetter: fake)
        _ = renderer.render("Equation $x^2$ and $$y^2$$.")
        #expect(!fake.calls.isEmpty)
        #expect(fake.calls.allSatisfy { $0.fontSize == NSFont.systemFontSize * CGFloat(size.scale) })
        let fallback = renderer.render("<div>unsupported block</div>")
        #expect(fallback.usedFallback)
        let font = try #require(fallback.attributedString.attribute(.font, at: 0,
                                                                    effectiveRange: nil) as? NSFont)
        #expect(font.pointSize == NSFont.systemFontSize * CGFloat(size.scale))
    }
}
