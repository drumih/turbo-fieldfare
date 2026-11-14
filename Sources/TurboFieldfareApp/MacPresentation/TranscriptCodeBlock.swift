import AppKit

extension NSAttributedString.Key {
    static let transcriptCodeBlock = NSAttributedString.Key("TurboFieldfare.transcriptCodeBlock")
}

/// Shared by a block's runs for its entire streamed lifetime. The text stays
/// in NSTextStorage; copying allocates only the clicked block's current body.
final class TranscriptCodeBlock: NSObject {
    static let headerPadding: CGFloat = 34
    let id = UUID()
    var syntheticTerminator = false

    static func endsWithLineBreak(_ text: String) -> Bool {
        text.utf8.last == 10 || text.utf8.last == 13
    }

    static func body(at index: Int, in text: NSAttributedString, matching block: TranscriptCodeBlock) -> String? {
        guard index >= 0, index < text.length else { return nil }
        var range = NSRange()
        guard let current = text.attribute(.transcriptCodeBlock, at: index,
                                           longestEffectiveRange: &range,
                                           in: NSRange(location: 0, length: text.length)) as? TranscriptCodeBlock,
              current === block else { return nil }
        var body = TranscriptPlainText.string(of: text.attributedSubstring(from: range))
        if block.syntheticTerminator, body.hasSuffix("\n") { body.removeLast() }
        return body.isEmpty ? nil : body
    }

    /// Foundation terminates code paragraphs even when the source body does
    /// not. Its source ranges can extend through blank container-only lines.
    static func hasSyntheticTerminator(position: AttributedString.MarkdownSourcePosition?, source: String) -> Bool {
        guard let position, let range = Range(position, in: source), range.upperBound == source.endIndex,
              !endsWithLineBreak(source) else { return false }
        let raw = source[range]
        if raw.hasPrefix("```") || raw.hasPrefix("~~~") {
            return !TailAutoClose.endsWithClosedFence(source)
        }
        let lines = ContainerPrefix.markdownLines(String(raw))
        if lines.count > 1, let last = lines.last,
           ContainerPrefix.strippingQuoteMarkers(last).rest.allSatisfy({ $0 == " " || $0 == "\t" }) {
            // Indented code drops trailing blank container lines. Their
            // preceding line break belongs to the final code-content line.
            return false
        }
        return true
    }
}
