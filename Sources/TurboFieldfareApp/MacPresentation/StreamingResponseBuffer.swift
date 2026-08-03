import Foundation

/// Smooths bursty streaming updates without changing the response itself.
///
/// The model remains the source of truth in `targetText`; the presentation
/// releases complete `Character` values from that text in short batches. This
/// keeps emoji and combining marks intact while avoiding a large layout jump
/// when an inference transport delivers several tokens at once.
public struct StreamingResponseBuffer: Equatable, Sendable {
    public private(set) var targetText = ""
    public private(set) var displayedText = ""

    public init() {}

    public var hasPendingText: Bool {
        targetText != displayedText
    }

    public var pendingCharacterCount: Int {
        guard targetText.hasPrefix(displayedText) else { return 0 }
        return targetText.dropFirst(displayedText.count).count
    }

    /// Starts a new response. Any source text that already arrived is kept as
    /// the target so the view can reveal it naturally from an empty response.
    public mutating func begin(with source: String = "") {
        targetText = source
        displayedText = ""
    }

    /// Incorporates a newer source snapshot. Append-only snapshots preserve
    /// the current reveal position. A non-prefix snapshot means the chat
    /// changed or the run was replaced, so it is shown atomically rather than
    /// ever mixing two responses together.
    public mutating func receive(_ source: String) {
        guard source != targetText else { return }
        guard source.hasPrefix(targetText) else {
            targetText = source
            displayedText = source
            return
        }
        targetText = source
    }

    /// Reveals the next grapheme-safe batch. Returns whether the displayed
    /// value changed.
    @discardableResult
    public mutating func revealNext(maximumCharacterCount: Int) -> Bool {
        guard maximumCharacterCount > 0, displayedText != targetText else {
            return false
        }
        guard targetText.hasPrefix(displayedText) else {
            displayedText = targetText
            return true
        }

        let next = targetText
            .dropFirst(displayedText.count)
            .prefix(maximumCharacterCount)
        guard !next.isEmpty else { return false }
        displayedText += next
        return true
    }

    /// Makes a terminal, stopped, or failed response immediately complete.
    public mutating func finish() {
        displayedText = targetText
    }

    /// Keeps a small delta feeling immediate while allowing larger transport
    /// batches to arrive over a few frames rather than as one visual jump.
    public static func recommendedBatchSize(for pendingCharacterCount: Int) -> Int {
        guard pendingCharacterCount > 0 else { return 0 }
        return min(72, max(1, (pendingCharacterCount + 2) / 3))
    }
}
