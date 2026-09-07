import Foundation

/// The name a conversation gets before anyone renames it.
///
/// The first user message, trimmed on a word boundary. Not a model-generated
/// title: that needs a second generation against the one KV the app has, which
/// would either evict the conversation the user is in or make them wait for a
/// name. A rename wins forever afterwards.
public enum ConversationTitle {
    public static let maximumCharacters = 60
    public static let untitled = "New Chat"

    public static func fromFirstMessage(_ text: String) -> String {
        let collapsed = text
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !collapsed.isEmpty else { return untitled }
        guard collapsed.count > maximumCharacters else { return collapsed }
        let clipped = collapsed.prefix(maximumCharacters)
        // Cut at the last space so the title ends on a word rather than mid-way
        // through one. A single very long word has no space to cut at, and is
        // clipped as it stands rather than replaced by an ellipsis alone.
        guard let lastSpace = clipped.lastIndex(of: " ") else {
            return String(clipped) + "\u{2026}"
        }
        let trimmed = clipped[clipped.startIndex..<lastSpace]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return String(clipped) + "\u{2026}" }
        return trimmed + "\u{2026}"
    }
}
