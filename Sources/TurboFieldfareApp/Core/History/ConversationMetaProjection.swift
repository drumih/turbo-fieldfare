import Foundation
import TurboFieldfare

/// `conversation.json` as a function of `transcript.jsonl`.
///
/// The records are the conversation; the meta is a cache of this function over
/// them. Six fields used to be assigned by hand after every turn — the turn
/// count from the live conversation, the token count from the runtime's report,
/// the image count by `+=` — and a copy maintained by hand is a copy that
/// drifts: a row once read 1,065 tokens over a file holding 22, and the
/// continuability rule believed it.
public enum ConversationMetaProjection {
    /// Pure. Same records in, same meta out.
    ///
    /// `legacy` is the cached meta, consulted only for creation facts that
    /// neither the header nor an `origin` record carries — a store written
    /// before those existed. It is never consulted for anything the records can
    /// answer, except that a newer format stamp in either file must survive
    /// projection so the app refuses replay as well as the store refusing writes.
    public static func project(directoryID: UUID,
                               records: [TranscriptRecord],
                               legacy: ConversationMeta?) throws -> ConversationMeta {
        guard case .header(let header)? = records.first else {
            throw ConversationStoreError.missingHeader(id: directoryID)
        }
        // The directory name is the identity; the header is a claim about it.
        // A transcript that migrated into the wrong directory would otherwise
        // be projected as a different conversation.
        guard header.id == directoryID else {
            throw ConversationStoreError.headerMismatch(
                directory: directoryID.uuidString, header: header.id)
        }

        var origin: ConversationOriginRecord?
        var turns: [ConversationTurnRecord] = []
        var lastTitle: ConversationTitleRecord?
        var lastUserTitle: ConversationTitleRecord?
        for record in records {
            switch record {
            case .turn(let turn):
                turns.append(turn)
            case .title(let title):
                lastTitle = title
                if title.source == .user { lastUserTitle = title }
            case .origin(let value):
                origin = value
            // A checkpoint of a reply still being generated is superseded by
            // the `turn` that follows it, and an unknown record is a later
            // build's. Neither says anything about what the conversation holds.
            case .header, .partial, .unknown:
                continue
            }
        }

        guard let identity = header.identity ?? origin?.identity ?? legacy?.identity,
              let session = header.session ?? origin?.session ?? legacy?.session else {
            // Fail closed. Without the identity there is no continuability rule
            // to apply, and guessing one would offer the user a conversation
            // that fails on its first turn. The directory is listed as
            // unreadable, exactly as an undecodable meta is.
            throw ConversationStoreError.missingOrigin(id: directoryID)
        }

        let firstUserText = turns.first { $0.role == .user }?.text
        let title: String
        let titleSource: ConversationTitleSource
        // A name the user typed is the conversation's name and nothing
        // supersedes it. A `firstMessage` record is the stand-in `create`
        // writes so the row reads something for the length of the first reply,
        // and it stops being the answer the moment the conversation has a first
        // message of its own. Left to win outright, a chat whose very first
        // turn was stopped kept the name of a message it does not contain, and
        // every turn that followed kept it too.
        if let lastUserTitle {
            title = lastUserTitle.title
            titleSource = .user
        } else if let firstUserText {
            title = ConversationTitle.fromFirstMessage(firstUserText)
            titleSource = .firstMessage
        } else if let lastTitle {
            title = lastTitle.title
            titleSource = lastTitle.source
        } else {
            title = ConversationTitle.untitled
            titleSource = .firstMessage
        }

        // A turn whose pictures were not stored cannot be replayed either: its
        // IDs hold their placeholder runs and no file backs them, so putting it
        // back would give the model image tokens with nothing behind them.
        let imageWriteFailed = turns.contains { $0.imageWriteFailed == true }

        // Every recorded turn's tokens, which is the KV. Nil the moment one
        // turn cannot account for its own — the same refusal `lineage` makes,
        // so a row can never claim a count the replay would not accept.
        let kvTokens: Int? = turns.isEmpty || imageWriteFailed
            || turns.contains(where: { $0.tokens == nil })
            ? nil
            : turns.reduce(0) { $0 + ($1.tokens?.count ?? 0) }

        // The last turn that could carry each, not the last that does: a store
        // upgraded mid-conversation has the newest turn carrying both, and
        // reaching further back for an older turn's answer would restore a
        // boundary the conversation has already moved past.
        let sampling = turns.last(where: { $0.role == .user })?.sampling
            ?? header.sampling ?? origin?.sampling ?? legacy?.sampling
            ?? ConversationSampling.windowDefaults
        let boundary = turns.last(where: { $0.role == .assistant })?.boundary
            ?? origin?.boundary ?? legacy?.boundary ?? ConversationBoundary()

        return ConversationMeta(
            version: max(header.version, legacy?.version ?? header.version),
            id: header.id,
            title: title,
            titleSource: titleSource,
            createdAt: header.createdAt,
            updatedAt: turns.last?.at ?? header.createdAt,
            turnCount: turns.count,
            kvTokens: kvTokens,
            imageCount: turns.reduce(0) { $0 + $1.images.count },
            identity: identity,
            session: session,
            sampling: sampling,
            boundary: boundary,
            imageWriteFailed: imageWriteFailed ? true : nil,
            // Written by every build and read by nothing. Kept in the file so a
            // reader with a non-optional field keeps decoding; removing it is a
            // separate decision.
            pinned: false)
    }

    /// The creation facts a legacy transcript has to be given before it can be
    /// projected, or nil when it already carries them.
    ///
    /// Nil is the ordinary answer: a header written by this build carries the
    /// facts, and a store already hoisted has an `origin` record. A transcript
    /// with neither, and no readable meta to lift them out of, is also nil —
    /// there is nothing to hoist, and `project` refuses it rather than
    /// inventing an identity.
    public static func hoist(records: [TranscriptRecord],
                             legacy: ConversationMeta?) -> ConversationOriginRecord? {
        guard let legacy else { return nil }
        for record in records {
            switch record {
            case .header(let header) where header.identity != nil: return nil
            case .origin: return nil
            default: continue
            }
        }
        return ConversationOriginRecord(
            identity: legacy.identity,
            session: legacy.session,
            sampling: legacy.sampling,
            boundary: legacy.boundary,
            at: legacy.updatedAt)
    }
}
