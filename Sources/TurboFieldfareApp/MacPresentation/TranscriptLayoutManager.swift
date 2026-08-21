import AppKit
import Foundation

/// Draws a rounded, full-width container behind each fenced code block. Ranges
/// are marked by the renderer with `TranscriptCodeStyle.codeBlockAttribute`;
/// each maximal contiguous marked run becomes one rounded rect painted behind
/// the text (and behind the per-glyph background of inline code, which
/// `super` still handles).
nonisolated public final class TranscriptLayoutManager: NSLayoutManager {
    /// Extra padding painted around the code text inside the container.
    private let verticalPadding: CGFloat = 8
    private let horizontalInset: CGFloat = 2
    private let cornerRadius: CGFloat = 6

    public override func drawBackground(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
        super.drawBackground(forGlyphRange: glyphsToShow, at: origin)
        guard let storage = textStorage, let container = textContainer(forGlyphAt: glyphsToShow.location, effectiveRange: nil) else {
            return
        }
        let charRange = characterRange(forGlyphRange: glyphsToShow, actualGlyphRange: nil)

        storage.enumerateAttribute(
            TranscriptCodeStyle.codeBlockAttribute,
            in: charRange,
            options: []
        ) { value, blockCharRange, _ in
            guard value != nil else { return }
            let blockGlyphRange = glyphRange(forCharacterRange: blockCharRange, actualCharacterRange: nil)
            var rect = boundingRect(forGlyphRange: blockGlyphRange, in: container)

            // Grow to the container's full text width and add breathing room.
            let fullWidth = container.size.width - container.lineFragmentPadding * 2
            rect.origin.x = origin.x + horizontalInset
            rect.size.width = fullWidth - horizontalInset * 2
            rect.origin.y += origin.y - verticalPadding
            rect.size.height += verticalPadding * 2

            let path = NSBezierPath(roundedRect: rect, xRadius: cornerRadius, yRadius: cornerRadius)
            TranscriptCodeStyle.codeBackground.setFill()
            path.fill()
            TranscriptCodeStyle.codeBorder.setStroke()
            path.lineWidth = 1
            path.stroke()
        }
    }
}
