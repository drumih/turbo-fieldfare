import Foundation

public enum AppInferenceError: Error, Equatable, Sendable, CustomStringConvertible {
    case invalidRequest(String)
    case modelNotFound(String)
    case modelLoadFailed(String)
    case tokenizerUnavailable(String)
    case contextOverflow(prompt: Int, maxNew: Int, maxContext: Int)
    /// A chat that no longer leaves room for a reply. Separate from
    /// `contextOverflow` because the figure that matters is not the user's
    /// requested reply length but the floor a turn keeps free, and because the
    /// way out is a longer context or a new chat rather than a shorter prompt.
    case conversationContextExhausted(prompt: Int, reserve: Int, maxContext: Int)
    case generationInFlight
    case modelNotLoaded
    case reloadRequired
    case connectionLost(String)
    case cancelled
    /// The KV no longer matches the conversation it recorded, so no further
    /// turn can resume onto it. Distinct from every other failure here: those
    /// leave the conversation usable, this one can only be cleared.
    case conversationLineageLost(String)
    /// A stored conversation could not be replayed into the KV. The transcript
    /// is intact and readable; what failed is continuing it, which is why this
    /// is separate from a lost lineage — there is nothing to clear.
    case conversationRestoreFailed(String)
    /// A stored conversation could not be read off disk at all. Distinct from
    /// `conversationRestoreFailed`, which is a transcript that reads fine and
    /// cannot be continued: this one has nothing to show. It exists because the
    /// alternative was what the app did before — select the row, draw the empty
    /// state, and say nothing, which is indistinguishable from a chat that was
    /// never written.
    case conversationUnreadable(String)
    case conversationPersistenceFailed(String)
    case unknown(String)

    public var description: String { userMessage }

    /// The cause on its own, without the sentence `userMessage` wraps it in.
    ///
    /// For sending across a process boundary. The decode service used to
    /// forward `"\(error)"`, which is the sentence, and the client then wrapped
    /// it again — so a failed replay read "This conversation could not be
    /// reopened: This conversation could not be reopened: …".
    public var diagnosticMessage: String {
        switch self {
        case .invalidRequest(let message), .modelNotFound(let message),
             .modelLoadFailed(let message), .tokenizerUnavailable(let message),
             .conversationLineageLost(let message),
             .conversationRestoreFailed(let message),
             .conversationPersistenceFailed(let message),
             .conversationUnreadable(let message), .connectionLost(let message),
             .unknown(let message):
            return message
        default:
            return userMessage
        }
    }

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
        case .conversationContextExhausted(let prompt, let reserve, let maxContext):
            return "This chat holds \(prompt) tokens and keeps \(reserve) free for a reply, which is more than the \(maxContext)-token context. Raise the context length or start a new chat."
        case .generationInFlight:
            return "A generation is already running."
        case .modelNotLoaded:
            return "Load the model before generating."
        case .reloadRequired:
            return "Model settings changed. Reload the model before generating."
        case .connectionLost(let message):
            return "The decode service connection was lost: \(message). Retry Load to continue."
        case .cancelled:
            return "Generation cancelled."
        case .conversationLineageLost(let message):
            return "This conversation can no longer continue: \(message) Start a new chat."
        case .conversationRestoreFailed(let message):
            return "This conversation could not be reopened: \(message)"
        case .conversationUnreadable(let message):
            return "This conversation could not be read: \(message)"
        case .conversationPersistenceFailed(let message):
            return "The latest exchange could not be saved reliably. Copy the answer before leaving this chat. \(message)"
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
