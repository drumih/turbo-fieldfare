import Foundation
import TurboFieldfare

/// Why a stored conversation's records cannot be put back into a KV.
///
/// Typed rather than a string so a caller pins the case, and worded exactly as
/// the window has always worded it: these two sentences reach the user through
/// `AppInferenceError.conversationRestoreFailed`, and a conversation that was
/// refused yesterday must be refused in the same words today.
public enum ConversationRecordError: Error, Equatable, Sendable,
                                     CustomStringConvertible {
    /// A turn with no recorded IDs. Replaying the turns around it would hand
    /// the model a conversation with a hole in the middle.
    case turnWithoutTokens
    /// A stored image's placeholder run is not where the turn that cites it
    /// says. The tower's features would land on text.
    case imageSpanNotInItsTurn
    /// The turn reported that a picture's stored copy could not be written,
    /// while its recorded IDs still hold that picture's placeholders.
    case imageRecordMissing

    public var description: String {
        switch self {
        case .turnWithoutTokens:
            return "a stored turn has no recorded tokens"
        case .imageSpanNotInItsTurn:
            return "a stored image's soft tokens are not in the turn that cites it"
        case .imageRecordMissing:
            return "a stored image was never written, and its turn still cites it"
        }
    }
}

/// A stored conversation as the app uses it, built by one walk over the records.
///
/// Display and replay read the same value, so they cannot disagree about which
/// turn holds which image. They used to be three separate walks — one for the
/// chat turns, one for the pairs the transcript draws, one for the lineage —
/// and two of them disagreeing is what produced a reopened turn drawn with no
/// pictures at all, and a conversation with pictures in two turns that could
/// never be continued again.
public struct ConversationDocument: Sendable {
    public let id: UUID
    public let meta: ConversationMeta
    /// Every recorded turn, with its images resolved against the conversation's
    /// own folder: paths are stored store-relative so the folder can be moved.
    public let turns: [AppChatTurn]
    /// The same turns as the exchanges the transcript draws.
    public let pairs: [(user: AppChatTurn, assistant: AppChatTurn)]
    /// The records as the inference side needs them, or why they cannot be.
    ///
    /// A failure here is a permanent property of what is on disk, which is why
    /// it travels with the document rather than being raised from a call the
    /// display path also has to make.
    public let lineage: Result<AppConversationLineage, ConversationRecordError>

    public static func load(_ opened: ConversationOpenResult,
                            directory: URL) -> ConversationDocument {
        var turns: [AppChatTurn] = []
        var pairs: [(user: AppChatTurn, assistant: AppChatTurn)] = []
        var pendingUser: AppChatTurn?
        var tokenIDs: [Int32] = []
        var images: [AppConversationReplayImage] = []
        var committedTurns = 0
        var failure: ConversationRecordError?

        for record in opened.records {
            guard case .turn(let turn) = record else { continue }

            let chatTurn = AppChatTurn(
                id: turn.id,
                role: turn.role == .user ? .user : .assistant,
                text: turn.text,
                images: turn.images.map {
                    ChatImage.stored(storedAttachment($0, in: directory))
                },
                promptTokens: turn.promptTokens,
                cachedTokens: turn.cachedTokens,
                generatedTokens: turn.generatedTokens,
                tokenIDs: turn.tokens)
            turns.append(chatTurn)
            if chatTurn.role == .user {
                pendingUser = chatTurn
            } else if let user = pendingUser {
                pairs.append((user: user, assistant: chatTurn))
                pendingUser = nil
            }

            // The replay half of the same pass. It keeps walking after a
            // failure so the turns above are complete — the transcript is still
            // readable, and losing what the user already read helps nobody.
            guard failure == nil else { continue }
            guard turn.imageWriteFailed != true else {
                // The same refusal the projection makes, from the same fact:
                // the IDs below hold placeholder runs that no file backs.
                failure = .imageRecordMissing
                continue
            }
            guard let tokens = turn.tokens else {
                failure = .turnWithoutTokens
                continue
            }
            for (indexInTurn, image) in turn.images.enumerated() {
                // Counted within this turn, because that is what the search
                // walks. Passing the conversation-wide count made the second
                // image-bearing turn look for its own placeholders at an index
                // that only existed if every earlier image had been in the same
                // turn, so any chat with pictures in two different turns could
                // never be continued again.
                guard let span = imageSpan(
                    in: tokens, softTokens: image.softTokens,
                    offset: tokenIDs.count, alreadyPlaced: indexInTurn) else {
                    failure = .imageSpanNotInItsTurn
                    break
                }
                images.append(AppConversationReplayImage(
                    tokenLowerBound: span,
                    tokenCount: image.softTokens,
                    fileURL: directory.appendingPathComponent(image.pixelsFile),
                    expectedDigest: image.modelInputDigest))
            }
            tokenIDs += tokens
            if turn.role == .user { committedTurns += 1 }
        }

        let lineage: Result<AppConversationLineage, ConversationRecordError>
        if let failure {
            lineage = .failure(failure)
        } else {
            lineage = .success(AppConversationLineage(
                tokenIDs: tokenIDs, images: images,
                boundaryTokenIDs: opened.meta.boundary.tokens,
                boundaryNeedsReplay: opened.meta.boundary.needsReplay,
                committedTurns: committedTurns))
        }
        return ConversationDocument(
            id: opened.meta.id, meta: opened.meta, turns: turns, pairs: pairs,
            lineage: lineage)
    }

    /// Finds the nth run of image placeholders inside one turn's tokens, where
    /// n counts only the images of that same turn.
    ///
    /// Located rather than stored: a recorded offset and a recorded token
    /// sequence can disagree, and then the tower's features land on text.
    private static func imageSpan(in tokens: [Int32], softTokens: Int,
                                  offset: Int, alreadyPlaced: Int) -> Int? {
        let placeholder = MultimodalPromptRenderer.imageTokenID
        var index = 0
        var seen = 0
        while index < tokens.count {
            guard tokens[index] == placeholder else {
                index += 1
                continue
            }
            var end = index
            while end < tokens.count, tokens[end] == placeholder { end += 1 }
            if seen == alreadyPlaced {
                guard end - index == softTokens else { return nil }
                return offset + index
            }
            seen += 1
            index = end
        }
        return nil
    }

    /// The thumbnail, not the model input.
    ///
    /// The two copies exist for different readers: the PNG is the exact bytes
    /// the tower patchified and must not be resampled, while the thumbnail was
    /// written at the 720 points the transcript asks for and is lossy on
    /// purpose. `sha256` carries the source digest because that is the key the
    /// thumbnail cache uses, so a picture already on screen in the composer is
    /// not decoded a second time for the transcript.
    private static func storedAttachment(_ image: ConversationImageRecord,
                                         in directory: URL) -> StoredImage {
        let thumbnail = directory.appendingPathComponent(image.thumbnailFile)
        // The model input when the thumbnail is not there. It is the same
        // picture — 768 square, what the tower was shown — so a conversation
        // whose thumbnail was lost still shows what it was asked about rather
        // than a turn that looks like it never had an image. Reading it is a
        // read: the bytes replay needs are untouched.
        let fileURL = FileManager.default.fileExists(atPath: thumbnail.path)
            ? thumbnail
            : directory.appendingPathComponent(image.pixelsFile)
        // Display only, never a gate: this figure reaches the composer's
        // capacity arithmetic for staged pictures, and a stored one is already
        // in the conversation. A file whose size cannot be read is drawn as
        // zero bytes rather than refused, which is a decision rather than an
        // oversight.
        let bytes = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
        // The conversation store owns these, and the type says so: no release
        // path that finishes a turn can be handed one.
        return StoredImage(
            id: image.id,
            fileURL: fileURL,
            displayName: image.displayName,
            encodedBytes: bytes ?? 0,
            sha256: image.sourceDigest)
    }
}

extension ConversationDocument: Equatable {
    /// Written out rather than synthesized: `pairs` is an array of tuples, which
    /// has no `Equatable` conformance to derive from.
    public static func == (lhs: ConversationDocument,
                           rhs: ConversationDocument) -> Bool {
        lhs.id == rhs.id && lhs.meta == rhs.meta && lhs.turns == rhs.turns
            && lhs.lineage == rhs.lineage
            && lhs.pairs.count == rhs.pairs.count
            && zip(lhs.pairs, rhs.pairs).allSatisfy {
                $0.user == $1.user && $0.assistant == $1.assistant
            }
    }
}
