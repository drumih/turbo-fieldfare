import Foundation

public enum StopReason: Sendable, Equatable {
    case eos
    case endOfTurn
    case maxTokens
    case stopString
}

enum GeneratorError: Error, CustomStringConvertible, Equatable {
    case contextOverflow(prompt: Int, maxNew: Int, maxContext: Int)
    case invalidGenerationConfig(String)
    case emptyPrompt

    public var description: String {
        switch self {
        case .contextOverflow(let prompt, let maxNew, let maxContext):
            return "context overflow: prompt \(prompt) + maxNew \(maxNew) exceeds maxContext \(maxContext)"
        case .invalidGenerationConfig(let reason):
            return reason
        case .emptyPrompt:
            return "empty prompt"
        }
    }
}
