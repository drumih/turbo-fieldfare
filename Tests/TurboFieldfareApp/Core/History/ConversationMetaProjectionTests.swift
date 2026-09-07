import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// `conversation.json` is a cache of this function, so every field it carries
/// has to have exactly one answer here. The cases are the field table of
/// `2026-09-02-conversation-meta-projection.md`, one per row, plus the two
/// refusals: a transcript in the wrong directory, and one with no identity to
/// check a replay against.
@Suite struct ConversationMetaProjectionTests {
    private static let identity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "0d77464e",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    private static let otherIdentity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "deadbeef",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    private static let session = ConversationSessionSettings(
        contextTokens: 8_192, expertCacheSlots: 16,
        visionResidencyPolicy: "onDemand")

    private static let otherSession = ConversationSessionSettings(
        contextTokens: 16_384, expertCacheSlots: 24,
        visionResidencyPolicy: "keepReady")

    private static func sampling(_ temperature: Double) -> ConversationSampling {
        ConversationSampling(temperature: temperature, topKEnabled: true, topK: 64,
                             topPEnabled: true, topP: 0.95, maxNewTokens: 4_096)
    }

    private static let created = Date(timeIntervalSince1970: 1_756_000_000)

    private static func at(_ offset: TimeInterval) -> Date {
        created.addingTimeInterval(offset)
    }

    private static func header(
        id: UUID,
        identity: ConversationIdentity? = ConversationMetaProjectionTests.identity,
        session: ConversationSessionSettings? = ConversationMetaProjectionTests.session,
        sampling: ConversationSampling? = nil
    ) -> TranscriptRecord {
        .header(ConversationHeaderRecord(
            id: id, createdAt: created, identity: identity, session: session,
            sampling: sampling))
    }

    private static func image(_ name: String) -> ConversationImageRecord {
        ConversationImageRecord(
            id: UUID(), displayName: name,
            pixelsFile: "images/\(name).png",
            thumbnailFile: "images/\(name).thumb.jpg",
            sourceDigest: name, modelInputDigest: name,
            width: 768, height: 768, softTokens: 4)
    }

    private static func legacyMeta(id: UUID) -> ConversationMeta {
        ConversationMeta(
            id: id, title: "typed into the file", titleSource: .user,
            createdAt: at(-9_000), updatedAt: at(-9_000),
            turnCount: 99, kvTokens: 12_345, imageCount: 7,
            identity: otherIdentity, session: otherSession,
            sampling: sampling(0.9),
            boundary: ConversationBoundary(tokens: [42], needsReplay: true),
            pinned: true)
    }

    // MARK: - The shape of an empty conversation

    /// A conversation with nothing in it yet: everything comes from the header.
    @Test func aHeaderOnItsOwnProjectsAnEmptyConversation() throws {
        let id = UUID()
        let meta = try ConversationMetaProjection.project(
            directoryID: id,
            records: [Self.header(id: id, sampling: Self.sampling(0.4))],
            legacy: nil)

        #expect(meta.version == ConversationMeta.currentVersion)
        #expect(meta.id == id)
        #expect(meta.createdAt == Self.created)
        #expect(meta.updatedAt == Self.created, "an unused chat was last used at nothing")
        #expect(meta.turnCount == 0)
        #expect(meta.imageCount == 0)
        #expect(meta.kvTokens == nil, "an empty conversation cannot be replayed")
        #expect(meta.title == ConversationTitle.untitled)
        #expect(meta.titleSource == .firstMessage)
        #expect(meta.identity == Self.identity)
        #expect(meta.session == Self.session)
        #expect(meta.sampling == Self.sampling(0.4))
        #expect(meta.boundary == ConversationBoundary())
        #expect(!meta.pinned)
    }

    // MARK: - Counts

    @Test func turnsImagesAndTokensAreCountedFromTheRecords() throws {
        let id = UUID()
        let records: [TranscriptRecord] = [
            Self.header(id: id),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(10), text: "what is this?",
                images: [Self.image("a"), Self.image("b")], tokens: [1, 2, 3])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(11), text: "a roof", tokens: [4])),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(20), text: "and this?",
                images: [Self.image("c")], tokens: [5, 6])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(21), text: "a wall", tokens: [7])),
        ]

        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: nil)
        #expect(meta.turnCount == 4)
        #expect(meta.imageCount == 3)
        #expect(meta.kvTokens == 7)
        #expect(meta.updatedAt == Self.at(21), "the last turn is when it was last used")
        #expect(meta.title == "what is this?")
        #expect(meta.titleSource == .firstMessage)
    }

    /// A turn with no recorded IDs cannot be replayed, and neither can the
    /// conversation around it. The same refusal `lineage` makes, so a row can
    /// never claim a count the replay would then reject.
    @Test func aTurnWithoutTokensLeavesTheCountUnknown() throws {
        let id = UUID()
        let records: [TranscriptRecord] = [
            Self.header(id: id),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(10), text: "hello", tokens: [1, 2])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(11), text: "hi", tokens: nil)),
        ]
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: nil)
        #expect(meta.kvTokens == nil)
        #expect(meta.turnCount == 2, "the turns are still there to read")
    }

    // MARK: - Title

    @Test func aTitleRecordBeatsTheFirstMessageAndTheLastOneWins() throws {
        let id = UUID()
        let records: [TranscriptRecord] = [
            Self.header(id: id),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(10), text: "the first message",
                tokens: [1])),
            .title(ConversationTitleRecord(
                title: "renamed once", source: .user, at: Self.at(30))),
            .title(ConversationTitleRecord(
                title: "renamed again", source: .user, at: Self.at(40))),
        ]
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: nil)
        #expect(meta.title == "renamed again")
        #expect(meta.titleSource == .user)
        // Renaming is not using: the row keeps its place in the list.
        #expect(meta.updatedAt == Self.at(10), "a rename reordered the list")
    }

    @Test func aLongFirstMessageIsClippedTheWayTheTitleRuleClipsIt() throws {
        let id = UUID()
        let long = String(repeating: "word ", count: 60)
        let records: [TranscriptRecord] = [
            Self.header(id: id),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(10), text: long, tokens: [1])),
        ]
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: nil)
        #expect(meta.title == ConversationTitle.fromFirstMessage(long))
        #expect(meta.title.count <= ConversationTitle.maximumCharacters + 1)
    }

    // MARK: - Records that say nothing

    /// A checkpoint of a reply still being generated is superseded by the turn
    /// that follows it, and a record from a later build is not this build's
    /// business. Counting either would put the sidebar ahead of the transcript.
    @Test func partialAndUnknownRecordsContributeNothing() throws {
        let id = UUID()
        let counted: [TranscriptRecord] = [
            Self.header(id: id),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(10), text: "hello", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(11), text: "hi", tokens: [2])),
        ]
        let withNoise: [TranscriptRecord] = [
            counted[0],
            .partial(ConversationTurnRecord(
                role: .assistant, at: Self.at(5), text: "half a rep",
                images: [Self.image("d")], tokens: [9, 9, 9])),
            counted[1],
            .unknown(type: "summary"),
            counted[2],
            .unknown(type: "reaction"),
        ]

        #expect(try ConversationMetaProjection.project(
            directoryID: id, records: withNoise, legacy: nil)
            == ConversationMetaProjection.project(
                directoryID: id, records: counted, legacy: nil))
    }

    // MARK: - Sampling and boundary

    @Test func samplingComesFromTheLastUserTurnThatRecordedIt() throws {
        let id = UUID()
        let records: [TranscriptRecord] = [
            Self.header(id: id, sampling: Self.sampling(0.1)),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(10), text: "one", tokens: [1],
                sampling: Self.sampling(0.5))),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(11), text: "a", tokens: [2])),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(20), text: "two", tokens: [3],
                sampling: Self.sampling(0.7))),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(21), text: "b", tokens: [4])),
        ]
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: nil)
        #expect(meta.sampling == Self.sampling(0.7))
    }

    @Test func theBoundaryComesFromTheLastAssistantTurn() throws {
        let id = UUID()
        let records: [TranscriptRecord] = [
            Self.header(id: id),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(10), text: "one", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(11), text: "a", tokens: [2],
                boundary: ConversationBoundary(tokens: [77], needsReplay: true))),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(20), text: "two", tokens: [3])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(21), text: "b", tokens: [4],
                boundary: ConversationBoundary())),
        ]
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: nil)
        // The newest turn's answer, not the newest turn that had something to
        // say: the conversation has already moved past the earlier boundary.
        #expect(meta.boundary == ConversationBoundary())
    }

    // MARK: - Legacy stores

    /// The cache is consulted only for facts no record carries, and only when
    /// there is no `origin` record either.
    @Test func theCachedMetaIsReadOnlyForWhatTheRecordsCannotAnswer() throws {
        let id = UUID()
        let legacy = Self.legacyMeta(id: id)
        let records: [TranscriptRecord] = [
            .header(ConversationHeaderRecord(id: id, createdAt: Self.created)),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(10), text: "hello", tokens: [1, 2])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(11), text: "hi", tokens: [3])),
        ]
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: legacy)

        // Taken from the cache, because nothing else records them.
        #expect(meta.identity == Self.otherIdentity)
        #expect(meta.session == Self.otherSession)
        #expect(meta.sampling == Self.sampling(0.9))
        #expect(meta.boundary == ConversationBoundary(tokens: [42], needsReplay: true))
        // Every one of these the cache also claimed, and every one is wrong.
        #expect(meta.title == "hello")
        #expect(meta.titleSource == .firstMessage)
        #expect(meta.turnCount == 2)
        #expect(meta.kvTokens == 3)
        #expect(meta.imageCount == 0)
        #expect(meta.createdAt == Self.created)
        #expect(meta.updatedAt == Self.at(11))
        #expect(!meta.pinned)
    }

    /// An `origin` record is the hoisted cache, and it wins over the cache
    /// itself: after the hoist the conversation projects from its own records
    /// and the file it came from can be deleted.
    @Test func anOriginRecordSuppliesWhatALegacyHeaderDoesNot() throws {
        let id = UUID()
        let origin = ConversationOriginRecord(
            identity: Self.identity, session: Self.session,
            sampling: Self.sampling(0.3),
            boundary: ConversationBoundary(tokens: [5], needsReplay: false),
            at: Self.at(1))
        let records: [TranscriptRecord] = [
            .header(ConversationHeaderRecord(id: id, createdAt: Self.created)),
            .origin(origin),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(10), text: "hello", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(11), text: "hi", tokens: [2])),
        ]
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: Self.legacyMeta(id: id))
        #expect(meta.identity == Self.identity)
        #expect(meta.session == Self.session)
        #expect(meta.sampling == Self.sampling(0.3))
        #expect(meta.boundary == ConversationBoundary(tokens: [5], needsReplay: false))
    }

    @Test func theHeaderBeatsAnOriginRecordAndBoth() throws {
        let id = UUID()
        let origin = ConversationOriginRecord(
            identity: Self.otherIdentity, session: Self.otherSession,
            sampling: Self.sampling(0.3), boundary: ConversationBoundary(),
            at: Self.at(1))
        let records: [TranscriptRecord] = [
            Self.header(id: id),
            .origin(origin),
        ]
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: Self.legacyMeta(id: id))
        #expect(meta.identity == Self.identity)
        #expect(meta.session == Self.session)
    }

    // MARK: - Refusals

    @Test func aTranscriptInTheWrongDirectoryIsRefused() {
        let directory = UUID()
        let records = [Self.header(id: UUID())]
        #expect(throws: ConversationStoreError.self) {
            try ConversationMetaProjection.project(
                directoryID: directory, records: records, legacy: nil)
        }
    }

    @Test func aTranscriptWithNoHeaderIsRefused() {
        let id = UUID()
        #expect(throws: ConversationStoreError.missingHeader(id: id)) {
            try ConversationMetaProjection.project(
                directoryID: id,
                records: [.turn(ConversationTurnRecord(
                    role: .user, at: Self.at(1), text: "orphan", tokens: [1]))],
                legacy: nil)
        }
    }

    /// Fail closed. Without an identity there is no continuability rule to
    /// apply, and guessing one hands the user a chat that fails on its first
    /// turn instead of a row that says it cannot be read.
    @Test func aTranscriptWithNoCreationFactsAnywhereIsRefused() {
        let id = UUID()
        #expect(throws: ConversationStoreError.missingOrigin(id: id)) {
            try ConversationMetaProjection.project(
                directoryID: id,
                records: [.header(ConversationHeaderRecord(
                    id: id, createdAt: Self.created))],
                legacy: nil)
        }
    }

    // MARK: - The property that makes it a cache

    @Test func projectingTwiceGivesTheSameAnswer() throws {
        let id = UUID()
        let records: [TranscriptRecord] = [
            Self.header(id: id),
            .turn(ConversationTurnRecord(
                role: .user, at: Self.at(10), text: "hello",
                images: [Self.image("a")], tokens: [1, 2],
                sampling: Self.sampling(0.6))),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Self.at(11), text: "hi", tokens: [3],
                boundary: ConversationBoundary(tokens: [9], needsReplay: true))),
            .title(ConversationTitleRecord(
                title: "named", source: .user, at: Self.at(12))),
        ]
        let once = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: nil)
        let twice = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: once)
        #expect(once == twice, "the projection is not a function of the records")
    }

    // MARK: - Cost

    /// What the projection costs at the sizes a real conversation reaches.
    ///
    /// It runs on every append and on every open, so the shape of the curve
    /// matters more than any single number — it is one pass over the records
    /// with no per-token work beyond counting. Reported rather than asserted:
    /// a wall-clock threshold in a test suite is a flake, and the number that
    /// decides anything belongs in a benchmark note.
    @Test func theProjectionCostAtRealTranscriptSizesIsReported() throws {
        for (tokens, turns) in [(600, 8), (6_060, 40), (8_192, 16), (8_192, 60)] {
            let id = UUID()
            let records = Self.synthesized(id: id, tokens: tokens, turns: turns)
            let bytes = try Self.encodedBytes(records)

            // Warm the decoder-free path once; the measurement is the projection.
            _ = try ConversationMetaProjection.project(
                directoryID: id, records: records, legacy: nil)
            let iterations = 20
            let start = ContinuousClock.now
            for _ in 0..<iterations {
                _ = try ConversationMetaProjection.project(
                    directoryID: id, records: records, legacy: nil)
            }
            let each = (ContinuousClock.now - start) / iterations

            let meta = try ConversationMetaProjection.project(
                directoryID: id, records: records, legacy: nil)
            #expect(meta.kvTokens == tokens)
            #expect(meta.turnCount == turns)
            print("projection: \(tokens) tokens in \(turns) records "
                + "(\(bytes) bytes) took \(each)")
        }
    }

    /// A transcript of `turns` records holding `tokens` token IDs between them.
    private static func synthesized(id: UUID, tokens: Int,
                                    turns: Int) -> [TranscriptRecord] {
        var records: [TranscriptRecord] = [header(id: id)]
        let each = tokens / turns
        var remaining = tokens
        for index in 0..<turns {
            let count = index == turns - 1 ? remaining : each
            remaining -= count
            let isUser = index.isMultiple(of: 2)
            records.append(.turn(ConversationTurnRecord(
                role: isUser ? .user : .assistant,
                at: at(TimeInterval(index)),
                text: String(repeating: "word ", count: count),
                tokens: (0..<count).map { Int32(1_000 + $0) },
                sampling: isUser ? sampling(0.2) : nil,
                boundary: isUser ? nil : ConversationBoundary())))
        }
        return records
    }

    private static func encodedBytes(_ records: [TranscriptRecord]) throws -> Int {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return try records.reduce(0) { $0 + (try encoder.encode($1).count) + 1 }
    }

    // MARK: - The hoist

    @Test func theHoistFiresOnceAndOnlyForALegacyTranscript() {
        let id = UUID()
        let legacy = Self.legacyMeta(id: id)
        let legacyRecords: [TranscriptRecord] = [
            .header(ConversationHeaderRecord(id: id, createdAt: Self.created)),
        ]
        let hoisted = ConversationMetaProjection.hoist(
            records: legacyRecords, legacy: legacy)
        #expect(hoisted?.identity == Self.otherIdentity)
        #expect(hoisted?.session == Self.otherSession)
        #expect(hoisted?.sampling == Self.sampling(0.9))
        #expect(hoisted?.boundary
            == ConversationBoundary(tokens: [42], needsReplay: true))

        // Already hoisted: nothing to do.
        guard let hoisted else {
            Issue.record("a legacy transcript produced no origin record")
            return
        }
        #expect(ConversationMetaProjection.hoist(
            records: legacyRecords + [.origin(hoisted)], legacy: legacy) == nil)
        // Written by this build: the header already carries the facts.
        #expect(ConversationMetaProjection.hoist(
            records: [Self.header(id: id)], legacy: legacy) == nil)
        // Nothing to hoist from.
        #expect(ConversationMetaProjection.hoist(
            records: legacyRecords, legacy: nil) == nil)
    }
}
