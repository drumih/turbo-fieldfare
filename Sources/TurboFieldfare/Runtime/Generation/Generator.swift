import Foundation

public enum StopReason: Sendable, Equatable {
    case eos
    case endOfTurn
    case maxTokens
    case stopString
    /// The caller's `shouldStop` asked the loop to end. Unlike `.stopString` no
    /// text was withheld: everything decoded was reported, and the final sampled
    /// token sits in `uncommittedBoundaryTokenIDs` exactly as it does for
    /// `.maxTokens`.
    case cancelled
    case toolCalls
}

enum GeneratorError: Error, CustomStringConvertible, Equatable {
    case contextOverflow(prompt: Int, maxNew: Int, maxContext: Int)
    case invalidGenerationConfig(String)
    case invalidContinuation(String)
    case emptyPrompt

    public var description: String {
        switch self {
        case .contextOverflow(let prompt, let maxNew, let maxContext):
            return "context overflow: prompt \(prompt) + maxNew \(maxNew) exceeds maxContext \(maxContext)"
        case .invalidGenerationConfig(let reason):
            return reason
        case .invalidContinuation(let reason):
            return reason
        case .emptyPrompt:
            return "empty prompt"
        }
    }
}
