import AppKit
import Foundation

/// The transcript's text projection. A typeset equation occupies one
/// object-replacement character and carries its LaTeX on the attachment, so
/// anything that reads the string back — a pasteboard flavour, an
/// accessibility client, the renderer's own `plainText` — has to put the
/// source back or hand out U+FFFC where the equation was.
public enum TranscriptPlainText {
    public static func string(of attributed: NSAttributedString) -> String {
        let characters = attributed.string as NSString
        var text = ""
        attributed.enumerateAttribute(
            .attachment,
            in: NSRange(location: 0, length: attributed.length),
            options: []) { value, range, _ in
            if let attachment = value as? MathAttachment {
                text += attachment.latexSource
            } else {
                text += characters.substring(with: range)
            }
        }
        return text
    }
}

/// The transcript view.
///
/// `NSTextView` builds the plain pasteboard flavour from the character stream,
/// so a reader who selects part of an answer and pastes it anywhere plain gets
/// U+FFFC for every equation. The Copy buttons elsewhere in the app write the
/// raw response and are unaffected; this is the selection path, which has no
/// source string to fall back on.
public final class TranscriptTextView: NSTextView {
    public override func copy(_ sender: Any?) {
        copySelection(to: .general)
    }

    /// The pasteboard is a parameter so the behaviour can be measured without
    /// writing to the reader's clipboard.
    func copySelection(to pasteboard: NSPasteboard) {
        guard let selection = selectedAttributedText() else { return }
        let range = NSRange(location: 0, length: selection.length)
        // The rich flavours still come from AppKit's own serialisers, so an
        // image in the prompt strip survives a copy the way it did before.
        let rtfd = selection.rtfd(from: range, documentAttributes: [:])
        let rtf = selection.rtf(from: range, documentAttributes: [:])
        var types: [NSPasteboard.PasteboardType] = []
        if rtfd != nil { types.append(.rtfd) }
        if rtf != nil { types.append(.rtf) }
        types.append(.string)

        pasteboard.clearContents()
        pasteboard.declareTypes(types, owner: nil)
        if let rtfd { pasteboard.setData(rtfd, forType: .rtfd) }
        if let rtf { pasteboard.setData(rtf, forType: .rtf) }
        pasteboard.setString(TranscriptPlainText.string(of: selection), forType: .string)
    }

    private func selectedAttributedText() -> NSAttributedString? {
        guard let storage = textStorage else { return nil }
        let document = NSRange(location: 0, length: storage.length)
        let selected = NSMutableAttributedString()
        for value in selectedRanges {
            let range = NSIntersectionRange(value.rangeValue, document)
            guard range.length > 0 else { continue }
            if selected.length > 0 {
                selected.append(NSAttributedString(string: "\n"))
            }
            selected.append(storage.attributedSubstring(from: range))
        }
        return selected.length > 0 ? selected : nil
    }
}
