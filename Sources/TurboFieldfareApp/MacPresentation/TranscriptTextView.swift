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

    /// The same projection kept as an attributed string: each typeset equation
    /// becomes its LaTeX carrying the run's own attributes, and every other
    /// attachment — an image in the prompt strip — is left where it is.
    ///
    /// Every rich flavour is serialised from this copy. AppKit rasterises an
    /// attachment that has no file wrapper, so Cmd-C on a fifty-equation answer
    /// wrote a 12.3 MB RTFD of fifty TIFFs and logged a hundred
    /// `CGImageDestinationFinalize` lines, while the RTF flavour, which cannot
    /// carry attachments at all, dropped every equation on the floor.
    public static func replacingMath(in attributed: NSAttributedString) -> NSAttributedString {
        let replaced = NSMutableAttributedString()
        attributed.enumerateAttribute(
            .attachment,
            in: NSRange(location: 0, length: attributed.length),
            options: []) { value, range, _ in
            guard let attachment = value as? MathAttachment else {
                replaced.append(attributed.attributedSubstring(from: range))
                return
            }
            var attributes = attributed.attributes(at: range.location, effectiveRange: nil)
            attributes.removeValue(forKey: .attachment)
            replaced.append(NSAttributedString(
                string: attachment.latexSource,
                attributes: attributes))
        }
        return replaced
    }
}

/// The transcript view.
///
/// `NSTextView` builds every pasteboard flavour from the character stream and
/// the attachments in it, so a reader who selected part of an answer got U+FFFC
/// for each equation in plain text, nothing at all in RTF, and a rasterised
/// TIFF per equation in RTFD. The Copy buttons elsewhere in the app write the
/// raw response and are unaffected; this is the selection path, which has no
/// source string to fall back on.
///
/// Every route out of a selection — Cmd-C, a drag, a Services provider — goes
/// through `writeSelection`, so they cannot drift apart.
public final class TranscriptTextView: NSTextView {
    /// The answer of the turn containing a character index, if it has one.
    public var answerAtCharacterIndex: ((Int) -> String?)?
    /// The newest answer, the whole chat, and New chat. Supplied by the pane so
    /// this view owns the only context menu the transcript has.
    public var lastAnswerText: (() -> String)?
    public var conversationText: (() -> String)?
    public var startNewChat: (() -> Void)?

    /// The transcript's whole context menu, built here rather than added to
    /// AppKit's.
    ///
    /// Two reasons, both found by right-clicking the running app. The pane also
    /// carried a SwiftUI `.contextMenu`, and this view's menu wins everywhere
    /// the text is — so those commands were unreachable exactly where a reader
    /// would look for them. And AppKit's default menu for a text view offers
    /// Cut, Paste, Font, Spelling, Substitutions and Writing Tools on a
    /// transcript that cannot be edited.
    public override func menu(for event: NSEvent) -> NSMenu? {
        let menu = NSMenu()
        menu.allowsContextMenuPlugIns = false
        let point = convert(event.locationInWindow, from: nil)
        let index = characterIndexForInsertion(at: point)

        if let answer = answerAtCharacterIndex?(index), !answer.isEmpty {
            add(to: menu, title: "Copy This Answer", text: answer)
        }
        if selectedRange().length > 0 {
            let item = NSMenuItem(title: "Copy Selection",
                                  action: #selector(copy(_:)), keyEquivalent: "")
            item.target = self
            menu.addItem(item)
        }
        if let last = lastAnswerText?(), !last.isEmpty {
            add(to: menu, title: "Copy Last Answer", text: last)
        }
        if let conversation = conversationText?(), !conversation.isEmpty {
            add(to: menu, title: "Copy Conversation", text: conversation)
        }
        if startNewChat != nil {
            if !menu.items.isEmpty { menu.addItem(.separator()) }
            let item = NSMenuItem(title: "New Chat",
                                  action: #selector(newChat(_:)), keyEquivalent: "")
            item.target = self
            menu.addItem(item)
        }
        return menu.items.isEmpty ? nil : menu
    }

    private func add(to menu: NSMenu, title: String, text: String) {
        let item = NSMenuItem(title: title, action: #selector(copyText(_:)),
                              keyEquivalent: "")
        item.target = self
        item.representedObject = text
        menu.addItem(item)
    }

    @objc private func copyText(_ sender: NSMenuItem) {
        guard let text = sender.representedObject as? String else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    @objc private func newChat(_ _: NSMenuItem) {
        startNewChat?()
    }

    public override var writablePasteboardTypes: [NSPasteboard.PasteboardType] {
        selectedAttributedText() == nil ? [] : [.rtfd, .rtf, .string]
    }

    public override func writeSelection(
        to pasteboard: NSPasteboard,
        types: [NSPasteboard.PasteboardType]
    ) -> Bool {
        guard let selection = selectedAttributedText() else { return false }
        let flattened = TranscriptPlainText.replacingMath(in: selection)
        let range = NSRange(location: 0, length: flattened.length)
        var wrote = false
        for type in types {
            switch type {
            case .rtfd:
                guard let data = flattened.rtfd(from: range, documentAttributes: [:]) else {
                    continue
                }
                pasteboard.setData(data, forType: .rtfd)
                wrote = true
            case .rtf:
                guard let data = flattened.rtf(from: range, documentAttributes: [:]) else {
                    continue
                }
                pasteboard.setData(data, forType: .rtf)
                wrote = true
            case .string:
                pasteboard.setString(flattened.string, forType: .string)
                wrote = true
            default:
                continue
            }
        }
        return wrote
    }

    public override func writeSelection(
        to pasteboard: NSPasteboard,
        type: NSPasteboard.PasteboardType
    ) -> Bool {
        writeSelection(to: pasteboard, types: [type])
    }

    public override func copy(_ sender: Any?) {
        copySelection(to: .general)
    }

    /// The pasteboard is a parameter so the behaviour can be measured without
    /// writing to the reader's clipboard.
    func copySelection(to pasteboard: NSPasteboard) {
        let types = writablePasteboardTypes
        guard !types.isEmpty else { return }
        pasteboard.clearContents()
        pasteboard.declareTypes(types, owner: nil)
        _ = writeSelection(to: pasteboard, types: types)
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
