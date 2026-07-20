import AppKit
import Foundation

@MainActor
public final class InstructionTranscriptDocumentController {
    public enum Mutation: Equatable {
        case none
        case rebuilt
        case appended
        case finalized
    }

    public struct UpdateResult {
        public let mutation: Mutation
        public let assistantRange: NSRange

        public init(mutation: Mutation, assistantRange: NSRange) {
            self.mutation = mutation
            self.assistantRange = assistantRange
        }
    }

    public private(set) var prompt = ""
    public private(set) var response = ""
    public private(set) var isFinalized = false
    public private(set) var showsPrefillPlaceholder = false
    public private(set) var assistantRange = NSRange(location: 0, length: 0)
    private var prefillPlaceholderRange: NSRange?
    private var prefillDotCount = 0

    private let renderer: ResponseMarkdownRenderer

    public init(renderer: ResponseMarkdownRenderer = ResponseMarkdownRenderer()) {
        self.renderer = renderer
    }

    public static func clampedRanges(
        _ ranges: [NSRange],
        toLength length: Int
    ) -> [NSRange] {
        ranges.map { range in
            let location = min(max(range.location, 0), length)
            let available = max(0, length - location)
            return NSRange(location: location, length: min(max(range.length, 0), available))
        }
    }

    public static func shouldScrollToBottom(
        wasAtBottom: Bool,
        mutation: Mutation
    ) -> Bool {
        wasAtBottom || mutation == .finalized
    }

    public static func shouldRunPrefillAnimation(
        response: String,
        isTerminal: Bool,
        requested: Bool
    ) -> Bool {
        requested && response.isEmpty && !isTerminal
    }

    @discardableResult
    public func synchronize(
        storage: NSMutableAttributedString,
        prompt: String,
        response: String,
        isTerminal: Bool,
        showsPrefillPlaceholder: Bool = false
    ) -> UpdateResult {
        let responseChanged = response != self.response
        let displaysPrefillPlaceholder = Self.shouldRunPrefillAnimation(
            response: response,
            isTerminal: isTerminal,
            requested: showsPrefillPlaceholder)
        let needsRebuild = prompt != self.prompt
            || !response.hasPrefix(self.response)
            || (isFinalized && !isTerminal)
            || displaysPrefillPlaceholder != self.showsPrefillPlaceholder

        var mutation: Mutation = .none
        if needsRebuild
            || storage.length == 0
                && (!prompt.isEmpty || !response.isEmpty || displaysPrefillPlaceholder) {
            rebuild(
                storage: storage,
                prompt: prompt,
                response: response,
                showsPrefillPlaceholder: displaysPrefillPlaceholder)
            mutation = .rebuilt
        } else if response.count > self.response.count {
            let delta = String(response.dropFirst(self.response.count))
            storage.append(NSAttributedString(
                string: delta,
                attributes: Self.responseAttributes()))
            assistantRange.length += (delta as NSString).length
            mutation = .appended
        }

        self.prompt = prompt
        self.response = response
        self.showsPrefillPlaceholder = displaysPrefillPlaceholder

        if isTerminal && (!isFinalized || responseChanged) {
            let rendered = renderer.render(response).attributedString
            storage.replaceCharacters(in: assistantRange, with: rendered)
            assistantRange.length = rendered.length
            isFinalized = true
            mutation = .finalized
        } else if !isTerminal {
            isFinalized = false
        }

        return UpdateResult(mutation: mutation, assistantRange: assistantRange)
    }

    @discardableResult
    public func advancePrefillAnimation(
        storage: NSMutableAttributedString
    ) -> Bool {
        guard showsPrefillPlaceholder, var range = prefillPlaceholderRange else {
            return false
        }
        prefillDotCount = (prefillDotCount + 1) % 4
        let replacement = NSAttributedString(
            string: Self.prefillPlaceholder(dotCount: prefillDotCount),
            attributes: Self.prefillPlaceholderAttributes())
        storage.replaceCharacters(in: range, with: replacement)
        range.length = replacement.length
        prefillPlaceholderRange = range
        return true
    }

    private func rebuild(
        storage: NSMutableAttributedString,
        prompt: String,
        response: String,
        showsPrefillPlaceholder: Bool
    ) {
        let document = NSMutableAttributedString()
        if !prompt.isEmpty {
            document.append(NSAttributedString(
                string: "You\n",
                attributes: Self.userLabelAttributes()))
            document.append(NSAttributedString(
                string: prompt,
                attributes: Self.promptAttributes()))
            document.append(NSAttributedString(
                string: "\n\n",
                attributes: Self.promptAttributes()))
        }
        document.append(NSAttributedString(
            string: "Answer\n",
            attributes: Self.assistantLabelAttributes()))
        assistantRange = NSRange(location: document.length, length: 0)
        document.append(NSAttributedString(
            string: response,
            attributes: Self.responseAttributes()))
        assistantRange.length = (response as NSString).length
        prefillDotCount = 0
        prefillPlaceholderRange = nil
        if showsPrefillPlaceholder {
            let placeholder = NSAttributedString(
                string: Self.prefillPlaceholder(dotCount: prefillDotCount),
                attributes: Self.prefillPlaceholderAttributes())
            prefillPlaceholderRange = NSRange(
                location: document.length,
                length: placeholder.length)
            document.append(placeholder)
        }
        storage.setAttributedString(document)
        isFinalized = false
    }

    private static func userLabelAttributes() -> [NSAttributedString.Key: Any] {
        labelAttributes(color: .secondaryLabelColor)
    }

    private static func assistantLabelAttributes() -> [NSAttributedString.Key: Any] {
        labelAttributes(color: TurboFieldfareMacTheme.accentNSColor)
    }

    private static func labelAttributes(
        color: NSColor
    ) -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.paragraphSpacing = 5
        return [
            .font: NSFont.systemFont(
                ofSize: NSFont.smallSystemFontSize,
                weight: .semibold),
            .foregroundColor: color,
            .paragraphStyle: style,
        ]
    }

    private static func promptAttributes() -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        return [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize),
            .foregroundColor: NSColor.secondaryLabelColor,
            .paragraphStyle: style,
        ]
    }

    private static func responseAttributes() -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 6
        return [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: style,
        ]
    }

    private static func prefillPlaceholderAttributes() -> [NSAttributedString.Key: Any] {
        var attributes = responseAttributes()
        attributes[.foregroundColor] = NSColor.secondaryLabelColor
        return attributes
    }

    private static func prefillPlaceholder(dotCount: Int) -> String {
        "Processing your prompt" + String(repeating: ".", count: dotCount)
    }
}
