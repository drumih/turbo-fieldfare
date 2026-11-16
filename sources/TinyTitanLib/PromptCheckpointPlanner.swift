import TinyTitanCore
import Foundation
import TinyTitan

/// One interior checkpoint position in a rendered chat prompt.
struct PromptCheckpointBoundary: Sendable, Equatable {
    let kind: PromptCheckpointKind
    /// Prefix length, in tokens. The checkpoint covers `promptIDs[0..<position]`.
    let position: Int
}

/// Locates the coarse interior boundaries the checkpoint tier captures.
///
/// The positions are found on the rendered token IDs, not on a re-render of a
/// truncated message list. That distinction is load-bearing: the ChatML
/// template renders the last assistant turn differently depending on whether a
/// user turn follows it, so encoding `messages.dropLast()` does not always
/// produce a prefix of the full render. A checkpoint taken at such a
/// non-prefix would either miss (safe) or, worse, match a prompt it does not
/// describe. Locating `<|im_start|>` in the ids that were actually rendered
/// cannot do that.
///
/// Two points are captured:
///
/// - `systemEnd`: the second message start, when the chat opens with a
///   system/developer message. That is where the first user turn begins, and
///   it survives an edit to any later turn.
/// - `lastTurnStart`: the final message start when the chat ends on a user
///   turn. An edited last turn restores from here instead of re-prefilling the
///   whole conversation.
enum PromptCheckpointPlanner {
    static func boundaries(
        promptIDs: [Int32],
        messages: [GFTokenizer.Message],
        imStartID: Int32,
        gridTokens: Int? = nil,
        includeMessageBoundaries: Bool = true
    ) -> [PromptCheckpointBoundary] {
        guard promptIDs.count > 1, !messages.isEmpty else { return [] }
        let messageStarts = messageStarts(
            promptIDs: promptIDs, imStartID: imStartID)
        guard !messageStarts.isEmpty else { return [] }

        var byPosition: [Int: PromptCheckpointKind] = [:]
        let opensWithGuidance =
            messages.first.map {
                $0.role == .system || $0.role == .developer
            } ?? false
        if includeMessageBoundaries {
            if opensWithGuidance, messageStarts.count >= 2 {
                byPosition[messageStarts[1]] = .systemEnd
            }
            if messages.last?.role == .user, let last = messageStarts.last {
                // A single user turn shares its start with `systemEnd`; the
                // edited-turn point is the more useful label for that position.
                byPosition[last] = .lastTurnStart
            }
        }
        if let gridTokens, gridTokens > 0 {
            var position = gridTokens
            while position < promptIDs.count {
                if byPosition[position] == nil {
                    byPosition[position] = .grid
                }
                position += gridTokens
            }
        }
        return
            byPosition
            .filter { $0.key > 0 && $0.key < promptIDs.count }
            .map { PromptCheckpointBoundary(kind: $0.value, position: $0.key) }
            .sorted { $0.position < $1.position }
    }

    /// The message starts of a rendered chat prompt, excluding the generation
    /// prompt's assistant header (always the final `<|im_start|>`).
    ///
    /// Empty when the ids are not a chat render this planner understands: a
    /// rendered chat prompt always ends with the generation prompt, so fewer
    /// than two markers means there is no message start to locate at all.
    private static func messageStarts(
        promptIDs: [Int32],
        imStartID: Int32
    ) -> [Int] {
        var starts: [Int] = []
        for (index, token) in promptIDs.enumerated() where token == imStartID {
            starts.append(index)
        }
        guard starts.count >= 2 else { return [] }
        return Array(starts.dropLast())
    }

    /// The end of the opening system/developer message, on the rendered ids.
    ///
    /// This is the position `boundaries` labels `.systemEnd`, except that a
    /// chat with a single user turn labels the shared position
    /// `.lastTurnStart` (see below). The position itself is what the shared
    /// system-prefix cache keys on, so it is exposed separately rather than
    /// recovered from the label. Nil when the chat does not open with
    /// guidance, or when the render has no second message start.
    static func systemEndPosition(
        promptIDs: [Int32],
        messages: [GFTokenizer.Message],
        imStartID: Int32
    ) -> Int? {
        guard promptIDs.count > 1, !messages.isEmpty else { return nil }
        let opensWithGuidance =
            messages.first.map {
                $0.role == .system || $0.role == .developer
            } ?? false
        guard opensWithGuidance else { return nil }
        let messageStarts = messageStarts(
            promptIDs: promptIDs, imStartID: imStartID)
        guard messageStarts.count >= 2 else { return nil }
        let position = messageStarts[1]
        guard position > 0, position < promptIDs.count else { return nil }
        return position
    }

    /// A stable group for one conversation's checkpoints.
    ///
    /// The anchor is the system message's text plus the first user turn: both
    /// stay fixed as a conversation grows and as its last turn is edited, so
    /// every checkpoint of one conversation shares a key while two
    /// conversations that start the same way do not.
    static func conversationKey(
        domain: ServerPromptCacheDomain,
        messages: [GFTokenizer.Message]
    ) -> String {
        let system = messages.first {
            $0.role == .system || $0.role == .developer
        }
        let firstUser = messages.first { $0.role == .user }
        let anchor = [
            system?.content ?? "", firstUser?.content ?? "",
        ].joined(separator: "\u{1F}")
        let hash = Sha256Verifier.hashData(Data(anchor.utf8))
        return domain.modelID + ":" + String(hash.prefix(16))
    }
}
