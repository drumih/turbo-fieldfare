import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// One walk over the records, so display and replay cannot disagree about what
/// a conversation holds.
///
/// They used to be three walks, and two of them disagreeing is what produced
/// both defects pinned here: a reopened turn drawn with no pictures at all
/// (`e68b8fe1`), and a conversation with a picture in each of two turns that
/// could never be continued again (`2cfd1d84`). Every case checks the three
/// answers against each other rather than one at a time.
@Suite struct ConversationDocumentTests {
    private static let identity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "0d77464e",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    private static let placeholder = MultimodalPromptRenderer.imageTokenID

    private static func meta(id: UUID = UUID(),
                             boundary: ConversationBoundary = ConversationBoundary())
        -> ConversationMeta {
        ConversationMeta(
            id: id, title: "stored", createdAt: Date(), updatedAt: Date(),
            identity: identity,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "on-demand"),
            sampling: ConversationSampling(
                temperature: 0.2, topKEnabled: true, topK: 64,
                topPEnabled: true, topP: 0.95, maxNewTokens: 8_192),
            boundary: boundary)
    }

    private static func opened(_ records: [TranscriptRecord],
                               boundary: ConversationBoundary = ConversationBoundary())
        -> ConversationOpenResult {
        ConversationOpenResult(
            meta: meta(boundary: boundary), records: records,
            isReadOnly: false, droppedTornFinalLine: false)
    }

    private static func image(_ name: String, softTokens: Int)
        -> ConversationImageRecord {
        ConversationImageRecord(
            id: UUID(), displayName: name,
            pixelsFile: "images/\(name).png",
            thumbnailFile: "images/\(name).thumb.jpg",
            sourceDigest: name, modelInputDigest: "digest-\(name)",
            width: 768, height: 768, softTokens: softTokens)
    }

    private static let directory = URL(
        fileURLWithPath: "/tmp/conversations/xyz", isDirectory: true)

    // MARK: - The three answers agree

    /// Turns, pairs and lineage come out of the same walk, so what the
    /// transcript draws, what it pairs, and what a replay puts back all describe
    /// the same conversation.
    @Test func turnsPairsAndLineageDescribeTheSameConversation() throws {
        let records: [TranscriptRecord] = [
            .header(ConversationHeaderRecord(id: UUID(), createdAt: Date())),
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "one", tokens: [1, 2])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "a", tokens: [3])),
            .title(ConversationTitleRecord(
                title: "renamed", source: .user, at: Date())),
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "two", tokens: [4])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "b", tokens: [5, 6])),
        ]
        let document = ConversationDocument.load(
            Self.opened(records, boundary: ConversationBoundary(
                tokens: [99], needsReplay: true)),
            directory: Self.directory)

        #expect(document.turns.map(\.text) == ["one", "a", "two", "b"])
        #expect(document.pairs.map(\.user.text) == ["one", "two"])
        #expect(document.pairs.map(\.assistant.text) == ["a", "b"])
        guard case .success(let lineage) = document.lineage else {
            Issue.record("a complete record refused to replay")
            return
        }
        // The lineage is the turns' own IDs, in the turns' own order.
        #expect(lineage.tokenIDs == document.turns.flatMap { $0.tokenIDs ?? [] })
        #expect(lineage.committedTurns == document.pairs.count)
        #expect(lineage.boundaryTokenIDs == [99])
        #expect(lineage.boundaryNeedsReplay)
        // Neither the header nor the rename is a turn.
        #expect(document.turns.count == 4)
    }

    /// An unpaired final user turn is in the turns and not in the pairs: it is
    /// the question of an exchange that has no answer yet.
    @Test func aUserTurnWithNoReplyIsNotAPair() throws {
        let records: [TranscriptRecord] = [
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "one", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "a", tokens: [2])),
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "dangling", tokens: [3])),
        ]
        let document = ConversationDocument.load(
            Self.opened(records), directory: Self.directory)
        #expect(document.turns.count == 3)
        #expect(document.pairs.count == 1)
        guard case .success(let lineage) = document.lineage else {
            Issue.record("a complete record refused to replay")
            return
        }
        #expect(lineage.committedTurns == 2, "the dangling turn is still in the KV")
    }

    // MARK: - The two refusals

    /// A turn with no recorded IDs cannot be replayed, and the wording the
    /// window shows for it does not move: it reaches the user through
    /// `AppInferenceError.conversationRestoreFailed`.
    @Test func aTurnWithoutTokensRefusesTheLineageAndStaysReadable() throws {
        let records: [TranscriptRecord] = [
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "one", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "no ids", tokens: nil)),
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "two", tokens: [2])),
        ]
        let document = ConversationDocument.load(
            Self.opened(records), directory: Self.directory)

        #expect(document.lineage == .failure(.turnWithoutTokens))
        #expect(ConversationRecordError.turnWithoutTokens.description
            == "a stored turn has no recorded tokens")
        // Refused for replay, still readable: losing what the user already read
        // helps nobody.
        #expect(document.turns.map(\.text) == ["one", "no ids", "two"])
        #expect(document.pairs.count == 1)
    }

    /// A picture whose placeholders are not where the turn says cannot be put
    /// back, because the tower's features would land on text.
    @Test func anImageWithNoPlaceholderRunRefusesTheLineage() throws {
        let records: [TranscriptRecord] = [
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "look",
                images: [Self.image("a", softTokens: 4)],
                // Three placeholders where the record claims four.
                tokens: [1] + Array(repeating: Self.placeholder, count: 3) + [2])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "ok", tokens: [9])),
        ]
        let document = ConversationDocument.load(
            Self.opened(records), directory: Self.directory)

        #expect(document.lineage == .failure(.imageSpanNotInItsTurn))
        #expect(ConversationRecordError.imageSpanNotInItsTurn.description
            == "a stored image's soft tokens are not in the turn that cites it")
        #expect(document.turns.first?.images.count == 1,
                "the picture is still drawn")
    }

    // MARK: - `2cfd1d84`: pictures in two turns

    /// A conversation with pictures in two different turns can be replayed.
    ///
    /// The span search walks one turn's tokens looking for the nth run of image
    /// placeholders, and the caller was handing it the count of images placed
    /// across the whole conversation. So the second image-bearing turn looked
    /// for its own placeholders at an index that only existed if every earlier
    /// image had been in that same turn: the lineage refused it, and the chat
    /// could never be continued again. Found by hand, sending into a chat with
    /// one image in each of two turns.
    @Test func aconversationWithImagesInTwoTurnsReplays() throws {
        // Two user turns, one picture each, with text either side of the run.
        let first = [Int32(1)] + Array(repeating: Self.placeholder, count: 4) + [2]
        let second = [Int32(3), 4] + Array(repeating: Self.placeholder, count: 4) + [5]
        let records: [TranscriptRecord] = [
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "one",
                images: [Self.image("a", softTokens: 4)], tokens: first)),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "ok", tokens: [9])),
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "two",
                images: [Self.image("b", softTokens: 4)], tokens: second)),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "ok", tokens: [9])),
        ]

        let document = ConversationDocument.load(
            Self.opened(records), directory: Self.directory)
        guard case .success(let lineage) = document.lineage else {
            Issue.record("a chat with pictures in two turns refused to replay")
            return
        }

        #expect(lineage.images.count == 2)
        // First turn: one text token, then the run.
        #expect(lineage.images[0].tokenLowerBound == 1)
        #expect(lineage.images[0].tokenCount == 4)
        // Second: six tokens of turn one, one of the reply, then two of text.
        #expect(lineage.images[1].tokenLowerBound == 6 + 1 + 2)
        #expect(lineage.images[1].tokenCount == 4)
        // And every placeholder the lineage points at really is one.
        for placed in lineage.images {
            let span = placed.tokenLowerBound..<(placed.tokenLowerBound + placed.tokenCount)
            #expect(span.allSatisfy { lineage.tokenIDs[$0] == Self.placeholder })
        }
        // The same walk drew the same two pictures, one per turn — which is the
        // agreement the separate walks did not have.
        #expect(document.turns.map(\.images.count) == [1, 0, 1, 0])
        #expect(document.turns[0].images.first?.id
            == document.pairs[0].user.images.first?.id)
    }

    // MARK: - `e68b8fe1`: the pictures the transcript dropped

    /// A reopened turn keeps the pictures that were attached to it.
    ///
    /// The images were written, recorded and replayed into the KV correctly all
    /// along; the transcript was the one reader that dropped them, because the
    /// records were mapped into chat turns without their attachments and every
    /// reopened turn came back with none. What the model still has in context
    /// and what the window shows have to agree.
    @Test func areopenedTurnKeepsItsImages() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("stored-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("images", isDirectory: true),
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        // The thumbnail is on disk, so this is the ordinary case rather than
        // the model-input fallback.
        try Data("thumbnail".utf8).write(
            to: directory.appendingPathComponent("images/1a2b.thumb.jpg"))
        let image = ConversationImageRecord(
            id: UUID(),
            displayName: "roof.heic",
            pixelsFile: "images/1a2b.png",
            thumbnailFile: "images/1a2b.thumb.jpg",
            sourceDigest: "source-digest",
            modelInputDigest: "model-input-digest",
            width: 896, height: 896, softTokens: 256)
        let records: [TranscriptRecord] = [
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "what is this?",
                images: [image],
                tokens: [1] + Array(repeating: Self.placeholder, count: 256) + [2])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "a roof.", tokens: [4, 5])),
        ]

        let document = ConversationDocument.load(
            Self.opened(records), directory: directory)
        let attached = try #require(document.turns.first?.images.first)
        #expect(document.turns.first?.images.count == 1)
        // The thumbnail, not the model input: the PNG is the exact bytes the
        // tower patchified and resampling it for display would be reading the
        // wrong copy.
        #expect(attached.fileURL == directory.appendingPathComponent("images/1a2b.thumb.jpg"))
        #expect(attached.displayName == "roof.heic")
        // The digest the thumbnail cache keys on, so a picture already drawn in
        // the composer is not decoded again for the transcript.
        #expect(attached.sha256 == "source-digest")
        #expect(attached.id == image.id)
        #expect(document.turns.last?.images.isEmpty == true,
                "the reply was given an image")
        // The pair the transcript draws is the same turn, not a second reading
        // of the record.
        #expect(document.pairs.first?.user.images.first?.fileURL == attached.fileURL)
        // And replay points at the model input, which is the other copy.
        guard case .success(let lineage) = document.lineage else {
            Issue.record("a turn with one picture refused to replay")
            return
        }
        #expect(lineage.images.first?.fileURL
            == directory.appendingPathComponent("images/1a2b.png"))
        #expect(lineage.images.first?.expectedDigest == "model-input-digest")
    }

    /// A conversation whose thumbnail is gone still shows its picture.
    ///
    /// The model input is the same image and is never resampled for replay, so
    /// reading it for the transcript costs nothing. Conversations damaged before
    /// the deletion was stopped would otherwise stay blank forever.
    @Test func amissingThumbnailFallsBackToTheModelInput() throws {
        let image = ConversationImageRecord(
            id: UUID(), displayName: "roof.heic",
            pixelsFile: "images/1a2b.png",
            thumbnailFile: "images/1a2b.thumb.jpg",
            sourceDigest: "source", modelInputDigest: "input",
            width: 768, height: 768, softTokens: 256)
        let records: [TranscriptRecord] = [
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "what is this?",
                images: [image], tokens: [1])),
        ]

        // Neither file exists under /tmp/conversations/xyz, so this is the
        // no-thumbnail case.
        let document = ConversationDocument.load(
            Self.opened(records), directory: Self.directory)
        let attached = try #require(document.turns.first?.images.first)
        #expect(attached.fileURL
            == Self.directory.appendingPathComponent("images/1a2b.png"))
        // The conversation store owns these, so no release path that finishes a
        // turn can be handed one: `AppImageAttachmentStore.remove` takes a
        // `StagedImage` and this is not one.
        #expect(attached.staged == nil)
        guard case .stored = attached else {
            Issue.record("a stored conversation's picture is not a StoredImage")
            return
        }
    }

    // MARK: - Records that are not turns

    @Test func headerTitleOriginPartialAndUnknownRecordsAreNotTurns() throws {
        let origin = ConversationOriginRecord(
            identity: Self.identity,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "on-demand"),
            sampling: ConversationSampling(
                temperature: 0.2, topKEnabled: true, topK: 64,
                topPEnabled: true, topP: 0.95, maxNewTokens: 8_192),
            boundary: ConversationBoundary(), at: Date())
        let records: [TranscriptRecord] = [
            .header(ConversationHeaderRecord(id: UUID(), createdAt: Date())),
            .origin(origin),
            .title(ConversationTitleRecord(title: "t", source: .user, at: Date())),
            // A checkpoint of a reply still being generated, superseded below.
            .partial(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "half", tokens: [9, 9])),
            .unknown(type: "summary"),
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "one", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "a", tokens: [2])),
        ]
        let document = ConversationDocument.load(
            Self.opened(records), directory: Self.directory)
        #expect(document.turns.count == 2)
        #expect(document.pairs.count == 1)
        guard case .success(let lineage) = document.lineage else {
            Issue.record("a complete record refused to replay")
            return
        }
        #expect(lineage.tokenIDs == [1, 2], "a partial reply entered the KV")
    }
}
