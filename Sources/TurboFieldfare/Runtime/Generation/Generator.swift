import Foundation

public enum StopReason: Sendable, Equatable {
    case eos
    case endOfTurn
    case maxTokens
    case stopString
    case toolCalls
}

enum GeneratorError: Error, CustomStringConvertible, Equatable {
    case contextOverflow(prompt: Int, maxNew: Int, maxContext: Int)
    case invalidGenerationConfig(String)
    case invalidContinuation(String)
    case noVisibleOutput
    case emptyPrompt

    public var description: String {
        switch self {
        case .contextOverflow(let prompt, let maxNew, let maxContext):
            return "context overflow: prompt \(prompt) + maxNew \(maxNew) exceeds maxContext \(maxContext)"
        case .invalidGenerationConfig(let reason):
            return reason
        case .invalidContinuation(let reason):
            return reason
        case .noVisibleOutput:
            return "generation produced too many non-text control tokens without visible output"
        case .emptyPrompt:
            return "empty prompt"
        }
    }
}
