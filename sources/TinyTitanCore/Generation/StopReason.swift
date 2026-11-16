/// Why a generation stopped. Shared by the 7.0 engine variants and the server
/// backend contract so a completion's stop reason has one spelling across the
/// boundary (the variant translates its internal reason into this).
public enum StopReason: String, Codable, Sendable, Equatable {
    case eos
    case endOfTurn
    case maxTokens
    case stopString
    case toolCalls
    /// The caller's `shouldStop()` closure returned true (external stop
    /// signal) before any configured stop string matched.
    case external
}
