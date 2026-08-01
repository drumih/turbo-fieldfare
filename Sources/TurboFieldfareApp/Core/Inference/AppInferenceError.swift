import Foundation

public enum AppInferenceError: Error, Equatable, Sendable, CustomStringConvertible {
    case invalidRequest(String)
    case modelNotFound(String)
    case modelLoadFailed(String)
    case tokenizerUnavailable(String)
    case conversationOverflow(prompt: Int, maxContext: Int)
    case generationInFlight
    case modelNotLoaded
    case reloadRequired
    case cancelled
    case unknown(String)

    /// Stable wire code for the decode-service IPC. Only overflow carries one
    /// today; the UI maps it back to this enum case via `DecodeServiceEvent.errorCode`.
    public var code: String? {
        switch self {
        case .conversationOverflow: return "conversation-overflow"
        default: return nil
        }
    }

    public var description: String { userMessage }

    public var userMessage: String {
        switch self {
        case .invalidRequest(let message):
            return message
        case .modelNotFound(let path):
            return "Model directory is not loadable: \(path)"
        case .modelLoadFailed(let message):
            return "Model load failed: \(message)"
        case .tokenizerUnavailable(let message):
            return "Tokenizer unavailable: \(message)"
        case .conversationOverflow(let prompt, let maxContext):
            return "This conversation has filled the context (\(prompt) of \(maxContext) tokens). Start a new chat or raise the context length."
        case .generationInFlight:
            return "A generation is already running."
        case .modelNotLoaded:
            return "Load the model before generating."
        case .reloadRequired:
            return "Model settings changed. Reload the model before generating."
        case .cancelled:
            return "Generation cancelled."
        case .unknown(let message):
            return message
        }
    }

    public var technicalDetail: String {
        switch self {
        case .tokenizerUnavailable:
            return "The installed tokenizer sidecar is missing or invalid. A Hugging Face fallback may require network access when no local sidecar is available."
        default:
            return userMessage
        }
    }
}
