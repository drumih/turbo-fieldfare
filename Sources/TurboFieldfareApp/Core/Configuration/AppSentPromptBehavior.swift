public enum AppSentPromptBehavior: String, CaseIterable, Codable, Sendable, Identifiable {
    case keep
    case clear

    public var id: String { rawValue }

    public var settingsLabel: String {
        switch self {
        case .keep: return "Keep Prompt"
        case .clear: return "Clear Prompt"
        }
    }
}
