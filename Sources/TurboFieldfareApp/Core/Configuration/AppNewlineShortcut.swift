public enum AppNewlineShortcut: String, CaseIterable, Codable, Sendable, Identifiable {
    case `return`
    case shiftReturn = "shift-return"

    public var id: String { rawValue }

    public static let sendMessageOptions: [Self] = [.shiftReturn, .return]

    public var sendMessageLabel: String {
        switch self {
        case .return: return "Command-Return"
        case .shiftReturn: return "Return"
        }
    }
}
