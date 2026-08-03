import TurboFieldfareAppCore

/// Keeps prompt-edit affordances scoped to the newest completed user turn.
public enum LatestPromptEditPolicy {
    public static func editableUserMessageIndex(
        in messages: [AppChatMessage],
        canEditLastPrompt: Bool
    ) -> Int? {
        guard canEditLastPrompt else { return nil }
        return messages.lastIndex(where: { $0.role == .user })
    }
}
