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
    private(set) var codeCopyButtons: [UUID: TranscriptCodeCopyButton] = [:]
    private var codeRefreshScheduled = false
    private weak var observedCodeStorage: NSTextStorage?
    private weak var observedCodeClip: NSClipView?
    var writeCodeToPasteboard: (String) -> Bool = { text in
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        return pasteboard.setString(text, forType: .string)
    }

    public override func viewDidMoveToSuperview() {
        super.viewDidMoveToSuperview()
        observeCodeLayout()
    }

    public override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        observeCodeLayout()
    }

    public override func setFrameSize(_ newSize: NSSize) {
        let changed = frame.size != newSize
        super.setFrameSize(newSize)
        if changed { scheduleCodeCopyRefresh() }
    }

    private func observeCodeLayout() {
        let center = NotificationCenter.default
        guard superview != nil else {
            center.removeObserver(self, name: NSTextStorage.didProcessEditingNotification, object: observedCodeStorage)
            center.removeObserver(self, name: NSView.boundsDidChangeNotification, object: observedCodeClip)
            observedCodeStorage = nil
            observedCodeClip = nil
            for button in codeCopyButtons.values { button.removeFromSuperview() }
            codeCopyButtons.removeAll()
            return
        }
        if observedCodeStorage !== textStorage {
            if let old = observedCodeStorage { center.removeObserver(self, name: NSTextStorage.didProcessEditingNotification, object: old) }
            observedCodeStorage = textStorage
            if let storage = textStorage {
                center.addObserver(self, selector: #selector(codeLayoutChanged(_:)),
                                   name: NSTextStorage.didProcessEditingNotification, object: storage)
            }
        }
        let clip = enclosingScrollView?.contentView
        if observedCodeClip !== clip {
            if let old = observedCodeClip { center.removeObserver(self, name: NSView.boundsDidChangeNotification, object: old) }
            observedCodeClip = clip
            clip?.postsBoundsChangedNotifications = true
            if let clip {
                center.addObserver(self, selector: #selector(codeLayoutChanged(_:)),
                                   name: NSView.boundsDidChangeNotification, object: clip)
            }
        }
        scheduleCodeCopyRefresh()
    }

    @objc private func codeLayoutChanged(_: Notification) {
        scheduleCodeCopyRefresh()
    }

    private func scheduleCodeCopyRefresh() {
        guard !codeRefreshScheduled else { return }
        codeRefreshScheduled = true
        // didProcessEditing precedes layout-manager invalidation. Waiting until
        // the next main-queue turn also coalesces edit, resize and scroll events.
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.codeRefreshScheduled = false
            self.refreshCodeCopyButtons()
        }
    }

    func refreshCodeCopyButtons() {
        guard superview != nil, let storage = textStorage, let layout = layoutManager, let container = textContainer else { return }
        let origin = textContainerOrigin
        let viewport = visibleRect
        var containerRect = viewport.offsetBy(dx: -origin.x, dy: -origin.y)
        // A code header precedes its first glyph. Look ahead by the largest
        // supported header plus one text line, then clip actual button frames.
        containerRect.size.height += ConversationTypography(.largest).scaled(TranscriptCodeBlock.headerPadding + NSFont.systemFontSize)
        let glyphs = layout.glyphRange(forBoundingRect: containerRect, in: container)
        let visible = layout.characterRange(forGlyphRange: glyphs, actualGlyphRange: nil)
        var retained = Set<UUID>()
        if storage.length > 0, visible.length > 0 {
            storage.enumerateAttribute(.transcriptCodeBlock, in: NSIntersectionRange(visible, NSRange(location: 0, length: storage.length))) { value, range, _ in
                guard let block = value as? TranscriptCodeBlock, !retained.contains(block.id),
                      let style = storage.attribute(.paragraphStyle, at: range.location, effectiveRange: nil) as? NSParagraphStyle,
                      let cell = style.textBlocks.first as? NSTextTableBlock else { return }
                var blockRange = NSRange()
                _ = storage.attribute(.transcriptCodeBlock, at: range.location, longestEffectiveRange: &blockRange,
                                      in: NSRange(location: 0, length: storage.length))
                if block.syntheticTerminator, blockRange.length == 1,
                   (storage.string as NSString).character(at: blockRange.location) == 10 { return }
                let runGlyphs = layout.glyphRange(forCharacterRange: blockRange, actualCharacterRange: nil)
                let rect = layout.boundsRect(for: cell, glyphRange: runGlyphs).offsetBy(dx: origin.x, dy: origin.y)
                guard rect.width > 0, rect.height > 0 else { return }
                let font = storage.attribute(.font, at: range.location, effectiveRange: nil) as? NSFont
                let scale = (font?.pointSize ?? NSFont.systemFontSize) / (NSFont.systemFontSize - 0.5)
                let side = 24 * scale
                let inset = 4 * scale
                let frame = NSRect(x: max(rect.minX + inset, rect.maxX - inset - side),
                                   y: rect.minY + inset,
                                   width: min(side, max(0, rect.width - 2 * inset)), height: side)
                guard frame.intersects(viewport), frame.width > 0 else { return }
                retained.insert(block.id)
                let button: TranscriptCodeCopyButton
                if let existing = codeCopyButtons[block.id] { button = existing }
                else {
                    button = TranscriptCodeCopyButton(block: block)
                    button.target = self
                    button.action = #selector(copyCodeBlock(_:))
                    codeCopyButtons[block.id] = button
                    addSubview(button)
                }
                button.characterIndex = blockRange.location
                button.symbolConfiguration = .init(pointSize: 12 * scale, weight: .regular)
                button.frame = frame
            }
        }
        for id in codeCopyButtons.keys where !retained.contains(id) {
            codeCopyButtons.removeValue(forKey: id)?.removeFromSuperview()
        }
    }

    @objc private func copyCodeBlock(_ sender: TranscriptCodeCopyButton) {
        refreshCodeCopyButtons()
        guard codeCopyButtons[sender.block.id] === sender, let storage = textStorage,
              let body = TranscriptCodeBlock.body(at: sender.characterIndex, in: storage, matching: sender.block) else { return }
        sender.feedbackGeneration += 1
        let generation = sender.feedbackGeneration
        guard writeCodeToPasteboard(body) else {
            sender.setCopied(false)
            sender.setCopyLabel("Copy code failed; try again")
            return
        }
        sender.setCopied(true)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak sender] in
            guard let sender, sender.feedbackGeneration == generation else { return }
            sender.setCopied(false)
        }
    }

    deinit { NotificationCenter.default.removeObserver(self) }

    public override func accessibilityChildren() -> [Any]? {
        let children = super.accessibilityChildren() ?? []
        // NSTextView exposes its text and links but omits native subviews.
        // Keep those semantics and make the visible code controls reachable.
        let buttons = codeCopyButtons.values.sorted { $0.characterIndex < $1.characterIndex }
        let elements = NSAccessibility.unignoredChildren(from: buttons)
        return children + elements.filter { element in
            !children.contains { ($0 as AnyObject) === (element as AnyObject) }
        }
    }

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

@MainActor
final class TranscriptCodeCopyButton: NSButton {
    let block: TranscriptCodeBlock
    var characterIndex = 0
    var feedbackGeneration = 0

    init(block: TranscriptCodeBlock) {
        self.block = block
        super.init(frame: .zero)
        title = ""
        imagePosition = .imageOnly
        isBordered = false
        bezelStyle = .roundRect
        setButtonType(.momentaryPushIn)
        let identifier = AccessibilityID.Prefix.transcriptCopyCode.rawValue + block.id.uuidString
        setAccessibilityIdentifier(identifier)
        (cell as? NSAccessibilityProtocol)?.setAccessibilityIdentifier(identifier)
        setCopied(false)
    }

    func setCopied(_ copied: Bool) {
        image = NSImage(systemSymbolName: copied ? "checkmark" : "doc.on.doc", accessibilityDescription: nil)
        contentTintColor = copied ? .controlAccentColor : .secondaryLabelColor
        setCopyLabel(copied ? "Code copied" : "Copy code")
    }

    func setCopyLabel(_ label: String) {
        setAccessibilityLabel(label)
        (cell as? NSAccessibilityProtocol)?.setAccessibilityLabel(label)
        toolTip = label
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
}
