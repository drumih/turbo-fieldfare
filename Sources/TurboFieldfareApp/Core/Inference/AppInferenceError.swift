import Foundation

public enum AppInferenceError: Error, Equatable, Sendable, CustomStringConvertible {
    case invalidRequest(String)
    case modelNotFound(String)
    case modelLoadFailed(String)
    case tokenizerUnavailable(String)
    case contextOverflow(prompt: Int, maxNew: Int, maxContext: Int)
    case generationInFlight
    case modelNotLoaded
    case reloadRequired
    case cancelled
    /// The KV no longer matches the conversation it recorded, so no further
    /// turn can resume onto it. Distinct from every other failure here: those
    /// leave the conversation usable, this one can only be cleared.
    case conversationLineageLost(String)
    case unknown(String)

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
        case .contextOverflow(let prompt, let maxNew, let maxContext):
            return "Prompt (\(prompt) tokens) plus max response (\(maxNew)) exceeds the \(maxContext)-token context."
        case .generationInFlight:
            return "A generation is already running."
        case .modelNotLoaded:
            return "Load the model before generating."
        case .reloadRequired:
            return "Model settings changed. Reload the model before generating."
        case .cancelled:
            return "Generation cancelled."
        case .conversationLineageLost(let message):
            return "This conversation can no longer continue: \(message) Start a new chat."
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
