import AppKit
import Foundation

/// A finished turn, as the transcript needs it. Deliberately not the app's
/// `ChatTurn`: the presentation layer does not depend on app state.
public struct TranscriptTurn: Equatable, Sendable {
    public enum Role: Sendable { case user, assistant }

    public let role: Role
    public let content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

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
    public private(set) var committed: [TranscriptTurn] = []
    private var prefillPlaceholderRange: NSRange?
    private var prefillDotCount = 0
    /// Rendered committed turns, memoised so a commit does not re-render the
    /// whole conversation's markdown. Index-aligned with `committed`.
    private var renderedCommitted: [NSAttributedString] = []

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
        committed: [TranscriptTurn] = [],
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
        // Committed turns only change when a generation finishes or the user
        // switches conversation — roughly once per run, never per token — so
        // rebuilding on that edge keeps the streaming append path untouched.
        let needsRebuild = committed != self.committed
            || prompt != self.prompt
            || !response.hasPrefix(self.response)
            || (isFinalized && !isTerminal)
            || displaysPrefillPlaceholder != self.showsPrefillPlaceholder

        var mutation: Mutation = .none
        if needsRebuild
            || storage.length == 0
                && (!committed.isEmpty || !prompt.isEmpty || !response.isEmpty
                    || displaysPrefillPlaceholder) {
            updateRenderedCommitted(committed)
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

        self.committed = committed
        self.prompt = prompt
        self.response = response
        self.showsPrefillPlaceholder = displaysPrefillPlaceholder

        // With no live section there is nothing to finalise: committed turns
        // were already rendered through the markdown path when they committed.
        let hasLiveSection = !prompt.isEmpty || !response.isEmpty
            || displaysPrefillPlaceholder || committed.isEmpty
        if isTerminal && hasLiveSection && (!isFinalized || responseChanged) {
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

    /// Re-renders only the committed turns whose content actually changed,
    /// reusing the memoised attributed strings for the unchanged prefix.
    private func updateRenderedCommitted(_ turns: [TranscriptTurn]) {
        var rendered: [NSAttributedString] = []
        rendered.reserveCapacity(turns.count)
        for (index, turn) in turns.enumerated() {
            if index < committed.count,
               index < renderedCommitted.count,
               committed[index] == turn {
                rendered.append(renderedCommitted[index])
            } else {
                rendered.append(render(turn))
            }
        }
        renderedCommitted = rendered
    }

    private func render(_ turn: TranscriptTurn) -> NSAttributedString {
        let block = NSMutableAttributedString()
        switch turn.role {
        case .user:
            block.append(NSAttributedString(
                string: "You\n", attributes: Self.userLabelAttributes()))
            block.append(NSAttributedString(
                string: turn.content, attributes: Self.promptAttributes()))
        case .assistant:
            block.append(NSAttributedString(
                string: "Answer\n", attributes: Self.assistantLabelAttributes()))
            block.append(renderer.render(turn.content).attributedString)
        }
        block.append(NSAttributedString(
            string: "\n\n", attributes: Self.promptAttributes()))
        return block
    }

    private func rebuild(
        storage: NSMutableAttributedString,
        prompt: String,
        response: String,
        showsPrefillPlaceholder: Bool
    ) {
        let document = NSMutableAttributedString()
        for block in renderedCommitted {
            document.append(block)
        }
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
        // With committed turns present and nothing in flight, the live section
        // is omitted entirely — otherwise every finished exchange would be
        // followed by a dangling "Answer" header.
        if renderedCommitted.isEmpty || !prompt.isEmpty || !response.isEmpty
            || showsPrefillPlaceholder {
            document.append(NSAttributedString(
                string: "Answer\n",
                attributes: Self.assistantLabelAttributes()))
            assistantRange = NSRange(location: document.length, length: 0)
            document.append(NSAttributedString(
                string: response,
                attributes: Self.responseAttributes()))
            assistantRange.length = (response as NSString).length
        } else {
            assistantRange = NSRange(location: document.length, length: 0)
        }
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
