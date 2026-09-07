import Foundation
import TurboFieldfare

/// Whether a stored conversation can be picked up again, and if not, why.
///
/// Computed from `conversation.json` alone — no transcript, no images, no model
/// — because the sidebar asks it for every row on every context change. Three
/// states, kept apart on purpose: "does not fit right now" is recoverable by
/// raising the context, "cannot be replayed" is not recoverable at all, and
/// collapsing them into one greyed-out row would tell the user neither.
public enum ConversationContinuability: Equatable, Sendable {
    case continuable
    /// The conversation is intact but does not fit the context in force.
    /// `required` is the smallest context that would hold it plus room to reply.
    case needsContext(required: Int)
    /// The stored token IDs no longer mean what they meant. Only a re-read into
    /// a new lineage can use this conversation, and that is a different chat.
    case cannotReplay(reason: Reason)

    public enum Reason: Equatable, Sendable {
        case newerFormat
        case differentModel
        case differentCheckpoint
        case differentTemplate
        case differentImageProcessing
        case tokenCountUnknown
        /// A picture of this conversation has no stored copy, while the turn
        /// that cites it still holds its placeholder tokens. Replaying it would
        /// hand the model a span of image tokens with nothing behind them.
        case imageRecordMissing
    }

    /// The rule, evaluated against the context currently loaded.
    ///
    /// `session.contextTokens` is deliberately not consulted: it records what
    /// the chat was made under, and a 3,000-token chat recorded at 8K continues
    /// perfectly well at 4K. What decides is the tokens it holds against the
    /// context in force.
    public static func evaluate(
        meta: ConversationMeta,
        currentContext: Int,
        identity: ConversationIdentity
    ) -> ConversationContinuability {
        if meta.version > ConversationMeta.currentVersion {
            return .cannotReplay(reason: .newerFormat)
        }
        if meta.identity.modelID != identity.modelID {
            return .cannotReplay(reason: .differentModel)
        }
        if meta.identity.sourceSnapshotHash != identity.sourceSnapshotHash {
            return .cannotReplay(reason: .differentCheckpoint)
        }
        if meta.identity.templateIdentity != identity.templateIdentity {
            return .cannotReplay(reason: .differentTemplate)
        }
        if meta.identity.imageProcessingVersion != identity.imageProcessingVersion,
           meta.imageCount > 0 {
            // A processing bump only matters to a conversation that has images
            // in it. Refusing a text chat for it would be a refusal with no
            // cause the user could act on.
            return .cannotReplay(reason: .differentImageProcessing)
        }
        // Checked before the count, which the projection also nils out for
        // this: "no replay record" would be true and would send the user
        // looking for the wrong thing.
        if meta.imageWriteFailed == true {
            return .cannotReplay(reason: .imageRecordMissing)
        }
        // Fail closed rather than trust a lower bound: a conversation whose
        // count was never reported would be restored, admitted, and only then
        // discovered not to fit.
        guard let kvTokens = meta.kvTokens else {
            return .cannotReplay(reason: .tokenCountUnknown)
        }
        // What the next turn's prompt starts with, and whether the smallest
        // turn then fits: the KV plus a boundary token the last run left
        // outside it. Asking whether the KV alone fit offered rows that every
        // message was then refused on.
        let prefix = kvTokens
            + (meta.boundary.needsReplay ? meta.boundary.tokens.count : 0)
        guard ConversationGenerationReserve.fitsLineage(
            tokens: prefix, maxContext: currentContext) else {
            return .needsContext(
                required: ConversationGenerationReserve.contextRequired(
                    forLineage: prefix))
        }
        return .continuable
    }
}
