import AppKit

public final class TranscriptScrollView: NSScrollView {
    public var didRestoreReadingPosition: (() -> Void)?

    public override func resizeSubviews(withOldSize oldSize: NSSize) {
        guard let textView = documentView as? NSTextView else {
            super.resizeSubviews(withOldSize: oldSize)
            return
        }
        // SwiftUI measures the composer after typography reflows the text.
        // Keep the anchor through that later viewport resize as well.
        let position = TranscriptReadingPosition(
            textView: textView, scrollView: self, followsPrefill: false)
        super.resizeSubviews(withOldSize: oldSize)
        position.restore(textView: textView, scrollView: self, replacing: nil)
        didRestoreReadingPosition?()
    }
}

/// A character anchor survives reflow; the old pixel scroll offset does not.
@MainActor
public struct TranscriptReadingPosition {
    private let selections: [NSRange]
    private let character: Int
    private let lineOffset: CGFloat
    private let followsBottom: Bool

    public init(textView: NSTextView, scrollView: NSScrollView, followsPrefill: Bool) {
        selections = textView.selectedRanges.map(\.rangeValue)
        let visible = scrollView.contentView.bounds
        followsBottom = followsPrefill || visible.maxY >= textView.bounds.maxY - 24
        if let manager = textView.layoutManager, let container = textView.textContainer,
           !textView.string.isEmpty {
            manager.ensureLayout(for: container)
            let origin = textView.textContainerOrigin
            let glyph = manager.glyphIndex(
                for: NSPoint(x: max(0, visible.minX - origin.x),
                             y: max(0, visible.minY - origin.y)), in: container)
            character = manager.characterIndexForGlyph(at: glyph)
            lineOffset = visible.minY - origin.y
                - manager.lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil).minY
        } else {
            character = 0
            lineOffset = 0
        }
    }

    public func restore(
        textView: NSTextView, scrollView: NSScrollView,
        replacing: InstructionTranscriptDocumentController.ReplacedRange?
    ) {
        let count = (textView.string as NSString).length
        func adjusted(_ ranges: [NSRange]) -> [NSRange] {
            let mapped = replacing.map {
                InstructionTranscriptDocumentController.adjustedRanges(
                    ranges, replacing: $0.previous, newLength: $0.length)
            } ?? ranges
            return InstructionTranscriptDocumentController.clampedRanges(mapped, toLength: count)
        }
        textView.selectedRanges = adjusted(selections).map(NSValue.init(range:))
        guard let manager = textView.layoutManager, let container = textView.textContainer else { return }
        manager.ensureLayout(for: container)
        if followsBottom {
            textView.scrollToEndOfDocument(nil)
        } else if count > 0 {
            let index = min(adjusted([NSRange(location: character, length: 0)])[0].location,
                            count - 1)
            let glyph = manager.glyphIndexForCharacter(at: index)
            let y = manager.lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil).minY
                + textView.textContainerOrigin.y + lineOffset
            let maximum = max(0, textView.bounds.height - scrollView.contentView.bounds.height)
            scrollView.contentView.scroll(to: NSPoint(
                x: scrollView.contentView.bounds.minX, y: min(max(y, 0), maximum)))
            scrollView.reflectScrolledClipView(scrollView.contentView)
        }
    }
}
