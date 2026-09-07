import Foundation

struct PersistenceLineageKey: Hashable, Sendable {
    let conversationID: UUID
    let epoch: UUID
    let bindingGeneration: UInt64
}

/// The immutable inputs needed to persist one exchange in KV commit order.
struct CompletedTurnPersistence: Sendable {
    let key: PersistenceLineageKey
    let store: ConversationStore
    let userText: String
    let assistantText: String
    let diagnostics: AppDiagnostics?
    let sampling: ConversationSampling
    let boundary: ConversationBoundary
    let imageWrite: Task<TurnImageWriteOutcome, Never>?
}
