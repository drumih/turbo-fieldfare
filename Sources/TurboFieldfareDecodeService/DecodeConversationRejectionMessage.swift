import Foundation
import TurboFieldfareDecodeProtocol

extension DecodeConversationGate.Rejection {
    /// The wording the service sends back. Names what was asked for and what is
    /// actually open, because "rejected" alone is not a diagnosis.
    var message: String {
        switch self {
        case .staleConversation(let requested, let open):
            let openText = open.map(\.uuidString) ?? "none"
            return "turn belongs to conversation \(requested.uuidString), "
                + "and the open conversation is \(openText)"
        case .outOfOrderTurn(let requested, let committed):
            let requestedText = requested.map(String.init) ?? "unset"
            return "turn index \(requestedText) does not follow the "
                + "\(committed) turns committed to this conversation"
        case .oneShotDuringConversation(let open):
            return "a conversation is open (\(open.uuidString)); "
                + "a one-shot turn cannot run on it"
        }
    }

    /// The terminal event a refused turn comes back as.
    ///
    /// `.lineageLost`, not `.failed`. A refusal means the turn was composed
    /// against a conversation the KV is no longer holding, so retrying it is
    /// refused identically forever — and `.failed` reaches the app as a
    /// retryable `.unknown`, which offers exactly that retry. `.lineageLost`
    /// is the kind that says the lineage is over and only a new chat clears it.
    ///
    func terminalEvent(generationID: UUID,
                       conversationEpoch: UUID?) -> DecodeServiceEvent {
        DecodeServiceEvent(
            kind: .lineageLost,
            generationID: generationID,
            error: message,
            conversationEpoch: conversationEpoch)
    }
}
