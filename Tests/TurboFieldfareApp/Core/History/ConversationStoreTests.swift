import CoreGraphics
import Foundation
import ImageIO
import Metal
import Testing
import TurboFieldfare
import UniformTypeIdentifiers
@testable import TurboFieldfareAppCore

/// The store is where a conversation survives a quit, so every case here is a
/// way it could survive as something other than what was written: a torn line
/// buried in the middle, an older build stamping its schema over a newer one's
/// file, two instances appending at once, a transcript that migrated into the
/// wrong directory.
@Suite struct ConversationStoreTests {
    @Test(arguments: 0...4)
    func adversarialSaveCutPointsRecoverIdempotently(cut: Int) async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "cut point", identity: identity, session: session, sampling: sampling)
        try await store.append([
            .turn(ConversationTurnRecord(role: .user, at: Date(), text: "kept", tokens: [1])),
            .turn(ConversationTurnRecord(role: .assistant, at: Date(), text: "kept answer", tokens: [2]))
        ], to: meta.id)
        let transcript = await store.transcriptURL(for: meta.id)
        let imageDir = await store.imagesURL(for: meta.id)
        try FileManager.default.createDirectory(at: imageDir, withIntermediateDirectories: true)
        let image = imageDir.appendingPathComponent("published.png")
        // Bounded sidecar bytes: this checks transaction recovery, not decoding.
        let pixels = Data("published sidecar".utf8)
        try pixels.write(to: image)
        let user = TranscriptRecord.turn(ConversationTurnRecord(
            role: .user, at: Date(), text: "new question",
            images: [ConversationImageRecord(
                id: UUID(), displayName: "fixture", pixelsFile: "images/published.png",
                thumbnailFile: "images/published.png", sourceDigest: "source",
                modelInputDigest: "pixels", width: 48, height: 48, softTokens: 1)],
            tokens: [MultimodalPromptRenderer.imageTokenID]))
        let assistant = TranscriptRecord.turn(ConversationTurnRecord(
            role: .assistant, at: Date(), text: "new answer", tokens: [4]))
        if cut == 4 {
            try await store.append([user, assistant], to: meta.id)
        } else {
            if cut >= 1 { try appendRaw(user, to: transcript) }
            if cut == 2 {
                let handle = try FileHandle(forWritingTo: transcript)
                try handle.seekToEnd()
                try handle.write(contentsOf: Data(#"{"type":"turn","turn":{"id"#.utf8))
                try handle.close()
            }
            if cut == 3 { try appendRaw(assistant, to: transcript) }
        }
        let first = try await store.open(id: meta.id)
        let recoveredBytes = try Data(contentsOf: transcript)
        let second = try await store.open(id: meta.id)
        #expect(try Data(contentsOf: transcript) == recoveredBytes)
        #expect(first.meta.turnCount == (cut >= 3 ? 4 : 2))
        #expect(second.records == first.records)
        #expect(second.meta.kvTokens == (cut >= 3 ? 4 : 2))
        _ = try await store.sweepOrphanImages(in: meta.id)
        #expect(FileManager.default.fileExists(atPath: image.path) == (cut >= 3))
        if cut >= 3 { #expect(try Data(contentsOf: image) == pixels) }
        try await store.append([
            .turn(ConversationTurnRecord(role: .user, at: Date(), text: "retry", tokens: [5])),
            .turn(ConversationTurnRecord(role: .assistant, at: Date(), text: "retry answer", tokens: [6]))
        ], to: meta.id)
        let continued = try await store.open(id: meta.id)
        #expect(continued.meta.turnCount == (cut >= 3 ? 6 : 4))
        #expect(continued.meta.kvTokens == (cut >= 3 ? 6 : 4))
        #expect(Array(continued.records.prefix(first.records.count)) == first.records)
    }

    private enum TrashFailure: Error, CustomStringConvertible {
        case list
        case remove

        var description: String {
            switch self {
            case .list: "injected trash listing failure"
            case .remove: "injected trash removal failure"
            }
        }
    }

    private func makeRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("conversations-\(UUID().uuidString)",
                                    isDirectory: true)
    }

    private var identity: ConversationIdentity {
        ConversationIdentity(
            modelID: "google/gemma-4-26B-A4B-it",
            sourceSnapshotHash: "0d77464e",
            templateIdentity: GFTokenizer.chatTemplateIdentity,
            imageProcessingVersion: VisionImageProcessing.version)
    }

    private var session: ConversationSessionSettings {
        ConversationSessionSettings(
            contextTokens: 8_192, expertCacheSlots: 16,
            visionResidencyPolicy: "onDemand")
    }

    private var sampling: ConversationSampling {
        ConversationSampling(temperature: 0.2, topKEnabled: true, topK: 64,
                             topPEnabled: true, topP: 0.95, maxNewTokens: 4_096)
    }

    private func makeStore(_ root: URL) async throws -> ConversationStore {
        let store = ConversationStore(rootURL: root)
        try await store.activate()
        return store
    }

    private func appendRaw(_ record: TranscriptRecord, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        var data = try encoder.encode(record)
        data.append(0x0A)
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
        try handle.close()
    }

    @Test func aThreeTurnConversationRoundTripsWithItsTokenIDs() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)

        let meta = try await store.create(
            title: "Hello there", identity: identity, session: session,
            sampling: sampling)
        var lineage: [Int32] = []
        for turn in 0..<3 {
            let prompt: [Int32] = [Int32(turn * 10), Int32(turn * 10 + 1)]
            let reply: [Int32] = [Int32(turn * 10 + 2)]
            try await store.append([.turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "question \(turn)",
                tokens: prompt))], to: meta.id)
            try await store.append([.turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "answer \(turn)", tokens: reply,
                stopReason: "endOfTurn"))], to: meta.id)
            lineage += prompt + reply
        }

        let opened = try await store.open(id: meta.id)
        #expect(!opened.droppedTornFinalLine)
        // Nothing assigned these: the meta is the projection of the records
        // that were just appended.
        #expect(opened.meta.turnCount == 6)
        #expect(opened.meta.kvTokens == lineage.count)
        #expect(opened.meta.identity == identity)
        #expect(opened.meta.sampling == sampling)
        let tokens = opened.records.compactMap { record -> [Int32]? in
            guard case .turn(let turn) = record else { return nil }
            return turn.tokens
        }.flatMap { $0 }
        #expect(tokens == lineage, "the record no longer reproduces the KV")

        let listed = try await store.list()
        #expect(listed.count == 1)
        #expect(listed[0].id == meta.id)
    }

    /// A crash mid-append can only ever cut the last line. Dropping it costs one
    /// turn; refusing the file would cost the conversation.
    @Test func aTornFinalLineIsDroppedAndTruncatedAway() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "kept", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "answer", tokens: [2])),
        ], to: meta.id)

        let transcript = await store.transcriptURL(for: meta.id)
        let handle = try FileHandle(forWritingTo: transcript)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(#"{"type":"turn","turn":{"id"#.utf8))
        try handle.close()

        let opened = try await store.open(id: meta.id)
        #expect(opened.droppedTornFinalLine)
        // Header, the title, and the complete exchange survive.
        #expect(opened.records.count == 4)

        // Truncated before the next append, so the following record does not
        // land glued to the fragment and become unreadable in the middle.
        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "after", tokens: [3])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "after answer", tokens: [4])),
        ], to: meta.id)
        let reopened = try await store.open(id: meta.id)
        #expect(!reopened.droppedTornFinalLine)
        #expect(reopened.records.count == 6)
    }

    /// An append heals the torn tail itself; it cannot rely on an open having
    /// done so.
    ///
    /// The sidebar lists a conversation from its cache without opening the
    /// transcript, so a rename after a crash reached `append` with the torn
    /// line still there and glued the title record onto it. That is a torn
    /// line in the middle, which nothing recovers: the chat could be listed
    /// and never opened again.
    @Test func anAppendAfterACrashDropsTheTornTailWithoutAnOpen() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "before", identity: identity, session: session,
            sampling: sampling)
        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "kept", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "answer", tokens: [2])),
        ], to: meta.id)

        let transcript = await store.transcriptURL(for: meta.id)
        let handle = try FileHandle(forWritingTo: transcript)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(#"{"type":"turn","turn":{"id"#.utf8))
        try handle.close()

        // No open in between: this is the rename-from-the-sidebar path.
        try await store.rename(id: meta.id, to: "after")

        let opened = try await store.open(id: meta.id)
        #expect(opened.meta.title == "after")
        #expect(!opened.droppedTornFinalLine, "the fragment survived the append")
        // Header, the first title, the exchange, and the rename.
        #expect(opened.records.count == 5)
    }

    /// A rename lands after the repair, not after the damage.
    ///
    /// A crash between a turn's two records leaves a question with no answer.
    /// Only a recovering open used to drop it, so a rename first appended its
    /// title behind the orphan, and the next open truncated the orphan and the
    /// title together: the rename silently reverted. The repair belongs to the
    /// rename, not to every append: a caller appending the answer itself must
    /// not have its question truncated first.
    @Test func aRenameAfterAnUnansweredQuestionKeepsItsTitle() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "before", identity: identity, session: session,
            sampling: sampling)
        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "kept", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "answer", tokens: [2])),
        ], to: meta.id)
        try appendRaw(.turn(ConversationTurnRecord(
            role: .user, at: Date(), text: "orphan", tokens: [3])),
            to: await store.transcriptURL(for: meta.id))

        try await store.rename(id: meta.id, to: "after")

        let opened = try await store.open(id: meta.id)
        #expect(opened.meta.title == "after", "the rename was truncated away")
        #expect(opened.meta.turnCount == 2, "the orphan survived the rename")
        #expect(opened.meta.kvTokens == 2)
        // Header, the first title, the exchange, the rename: no orphan.
        #expect(opened.records.count == 5)
    }

    /// A long fragment is still found and cut before the append.
    @Test func aTornTailLongerThanOneScanWindowIsStillDropped() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)

        let transcript = await store.transcriptURL(for: meta.id)
        let handle = try FileHandle(forWritingTo: transcript)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(
            (#"{"type":"turn","turn":{"text":""# + String(repeating: "x", count: 40_000)).utf8))
        try handle.close()

        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "after", tokens: [3])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "after answer", tokens: [4])),
        ], to: meta.id)

        let opened = try await store.open(id: meta.id)
        #expect(!opened.droppedTornFinalLine)
        #expect(opened.records.count == 4)
        #expect(opened.meta.turnCount == 2)
    }

    @Test func aCompleteUserWithoutAnAssistantIsRecoveredAsNoExchange() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        let transcript = await store.transcriptURL(for: meta.id)
        let recoveredBoundary = try Data(contentsOf: transcript).count
        try appendRaw(.turn(ConversationTurnRecord(
            role: .user, at: Date(), text: "never committed", tokens: [1])),
            to: transcript)

        let opened = try await store.open(id: meta.id)

        #expect(opened.records.compactMap { record -> ConversationTurnRecord? in
            guard case .turn(let turn) = record else { return nil }
            return turn
        }.isEmpty)
        #expect(opened.meta.turnCount == 0)
        #expect(try Data(contentsOf: transcript).count == recoveredBoundary)
    }

    @Test func aCompleteUserAndTornAssistantRecoverAtTheUsersBoundary() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        let transcript = await store.transcriptURL(for: meta.id)
        let recoveredBoundary = try Data(contentsOf: transcript).count
        try appendRaw(.turn(ConversationTurnRecord(
            role: .user, at: Date(), text: "never committed", tokens: [1])),
            to: transcript)
        let handle = try FileHandle(forWritingTo: transcript)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(#"{"type":"turn","turn":{"role":"assistant""#.utf8))
        try handle.close()

        let opened = try await store.open(id: meta.id)

        #expect(opened.droppedTornFinalLine)
        #expect(opened.meta.turnCount == 0)
        #expect(try Data(contentsOf: transcript).count == recoveredBoundary)
    }

    @Test func aReadOnlyOpenOmitsAnIncompleteExchangeWithoutChangingBytes() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let writer = try await makeStore(root)
        let meta = try await writer.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        let transcript = await writer.transcriptURL(for: meta.id)
        try appendRaw(.turn(ConversationTurnRecord(
            role: .user, at: Date(), text: "never committed", tokens: [1])),
            to: transcript)
        let before = try Data(contentsOf: transcript)
        let reader = ConversationStore(rootURL: root)
        try await reader.activate()
        #expect(await reader.isReadOnly)

        let opened = try await reader.open(id: meta.id)

        #expect(opened.meta.turnCount == 0)
        #expect(try Data(contentsOf: transcript) == before)
    }

    @Test func aCompleteExchangeAppendsAtTheRecoveredBoundary() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        let transcript = await store.transcriptURL(for: meta.id)
        try appendRaw(.turn(ConversationTurnRecord(
            role: .user, at: Date(), text: "never committed", tokens: [1])),
            to: transcript)
        _ = try await store.open(id: meta.id)

        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "kept", tokens: [2])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "answer", tokens: [3])),
        ], to: meta.id)
        let opened = try await store.open(id: meta.id)
        let turns = opened.records.compactMap { record -> ConversationTurnRecord? in
            guard case .turn(let turn) = record else { return nil }
            return turn
        }
        #expect(turns.map(\.text) == ["kept", "answer"])
        #expect(opened.meta.turnCount == 2)
    }

    /// A damaged line anywhere but the end means two writers or in-place
    /// corruption. Appending to it would bury the evidence under more records.
    @Test func aTornLineInTheMiddleIsAnError() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        try await store.append([.turn(ConversationTurnRecord(
            role: .user, at: Date(), text: "one", tokens: [1]))], to: meta.id)

        let transcript = await store.transcriptURL(for: meta.id)
        var text = try String(contentsOf: transcript, encoding: .utf8)
        text = text.replacingOccurrences(of: "\"type\":\"header\"", with: "\"typ")
        try text.write(to: transcript, atomically: true, encoding: .utf8)

        await #expect(throws: ConversationStoreError.self) {
            _ = try await store.open(id: meta.id)
        }
    }

    /// The directory name is the identity; the header is a claim about it. A
    /// transcript that ended up in the wrong directory would otherwise be
    /// replayed as a different conversation.
    @Test func aHeaderNamingAnotherConversationIsRefused() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        let transcript = await store.transcriptURL(for: meta.id)
        let text = try String(contentsOf: transcript, encoding: .utf8)
            .replacingOccurrences(of: meta.id.uuidString,
                                  with: UUID().uuidString)
        try text.write(to: transcript, atomically: true, encoding: .utf8)

        await #expect(throws: ConversationStoreError.self) {
            _ = try await store.open(id: meta.id)
        }
    }

    /// An unrecognised record type is a later build's, not damage. Failing on it
    /// would make every conversation unreadable the moment one new record kind
    /// ships.
    @Test func anUnknownRecordTypeIsIgnoredRatherThanFatal() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        let transcript = await store.transcriptURL(for: meta.id)
        let handle = try FileHandle(forWritingTo: transcript)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(
            "{\"type\":\"summary\",\"summary\":{\"text\":\"later build\"}}\n".utf8))
        try handle.close()
        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "after", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "answer", tokens: [2])),
        ], to: meta.id)

        let opened = try await store.open(id: meta.id)
        // Header, the title, the later-build record, and the complete exchange.
        #expect(opened.records.count == 5)
        #expect(opened.records[2] == .unknown(type: "summary"))
        guard case .turn(let turn) = opened.records[3] else {
            Issue.record("the record after an unknown one was lost")
            return
        }
        #expect(turn.text == "after")
    }

    /// An older build stamping its schema over a newer one's file is the
    /// migration loss the field is full of. The file's bytes must not move.
    @Test func aNewerMajorOpensReadOnlyAndItsBytesAreUntouched() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        var meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        meta.version = ConversationMeta.currentVersion + 1
        let metaURL = await store.metaURL(for: meta.id)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(meta).write(to: metaURL)
        let before = try Data(contentsOf: metaURL)

        let opened = try await store.open(id: meta.id)
        #expect(opened.isReadOnly)
        #expect(opened.meta.version == meta.version)
        await #expect(throws: ConversationStoreError.self) {
            try await store.append([.turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "no", tokens: [1]))], to: meta.id)
        }
        await #expect(throws: ConversationStoreError.self) {
            try await store.rename(id: meta.id, to: "no")
        }
        // Opening it projected a meta of its own and deliberately did not write
        // it: an older build stamping its schema over a newer one's file is the
        // migration loss the field is full of.
        #expect(try Data(contentsOf: metaURL) == before)
    }

    /// The first append must respect the version it discovers itself, without
    /// requiring an earlier list or open to populate the read-only cache.
    @Test(arguments: [false, true])
    func firstAppendRefusesANewerVersionWithoutChangingEitherFile(
        incompatibleSchema: Bool
    ) async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        var meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        meta.version = ConversationMeta.currentVersion + 1
        let metaURL = await store.metaURL(for: meta.id)
        let transcriptURL = await store.transcriptURL(for: meta.id)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = incompatibleSchema
            ? try JSONSerialization.data(withJSONObject: ["version": meta.version])
            : try encoder.encode(meta)
        try data.write(to: metaURL)
        let metaBefore = try Data(contentsOf: metaURL)
        let transcriptBefore = try Data(contentsOf: transcriptURL)

        await #expect(throws: ConversationStoreError.self) {
            try await store.append([.title(ConversationTitleRecord(
                title: "no", source: .user, at: Date()))], to: meta.id)
        }
        #expect(try Data(contentsOf: metaURL) == metaBefore)
        #expect(try Data(contentsOf: transcriptURL) == transcriptBefore)
    }

    @Test(arguments: 0...2, [false, true])
    func newerHeaderProtectsTranscriptWhenMetadataCannotProtectIt(
        cacheState: Int, sweepFirst: Bool
    ) async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        let transcript = await store.transcriptURL(for: meta.id)
        let metaURL = await store.metaURL(for: meta.id)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        var bytes = try encoder.encode(TranscriptRecord.header(ConversationHeaderRecord(
            version: ConversationMeta.currentVersion + 1,
            id: meta.id, createdAt: meta.createdAt, identity: identity,
            session: session, sampling: sampling)))
        // Recovery must not truncate even a torn tail owned by a newer writer.
        bytes.append(contentsOf: "\n{\"type\":".utf8)
        try bytes.write(to: transcript)
        if cacheState == 0 { try FileManager.default.removeItem(at: metaURL) }
        if cacheState == 1 { try Data("invalid cache".utf8).write(to: metaURL) }
        let cachedBytes = FileManager.default.contents(atPath: metaURL.path)
        let image = await store.imagesURL(for: meta.id).appendingPathComponent("future.png")
        let imageBytes = Data("future format image".utf8)
        try imageBytes.write(to: image)

        if sweepFirst {
            await #expect(throws: ConversationStoreError.self) {
                try await store.sweepOrphanImages(in: meta.id)
            }
        }

        let opened = try await store.open(id: meta.id)
        #expect(opened.isReadOnly)
        #expect(opened.meta.version == ConversationMeta.currentVersion + 1)
        await #expect(throws: ConversationStoreError.self) {
            try await store.rename(id: meta.id, to: "must not write")
        }
        #expect(try Data(contentsOf: transcript) == bytes)
        #expect(FileManager.default.contents(atPath: metaURL.path) == cachedBytes)
        #expect(FileManager.default.contents(atPath: image.path) == imageBytes)
    }

    /// Two writers on one JSONL file is the documented corruption. The second
    /// instance reads and says so rather than appending beside the first.
    @Test func aSecondHolderOfTheStoreIsReadOnly() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = try await makeStore(root)
        let meta = try await first.create(
            title: "t", identity: identity, session: session, sampling: sampling)

        let second = ConversationStore(rootURL: root)
        try await second.activate()
        #expect(await second.isReadOnly)
        // Reading still works: a second window is a normal thing to open.
        #expect(try await second.list().count == 1)
        _ = try await second.open(id: meta.id)
        await #expect(throws: ConversationStoreError.self) {
            try await second.append([.turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "no", tokens: [1]))], to: meta.id)
        }
        #expect(await first.isReadOnly == false)
    }

    /// Delete means delete. A conversation used to be parked in `.trash` so an
    /// undo could restore it, which left every deleted transcript and image on
    /// disk for the life of the store with nothing to collect them.
    @Test func deleteRemovesTheConversationAndEverythingInIt() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        try await store.append([.turn(ConversationTurnRecord(
            role: .user, at: Date(), text: "keep me", tokens: [1]))], to: meta.id)
        let directory = await store.directoryURL(for: meta.id)
        let image = directory.appendingPathComponent("images/abcd.png")
        try Data("pixels".utf8).write(to: image)

        try await store.delete(id: meta.id)

        #expect(try await store.list().isEmpty)
        #expect(!FileManager.default.fileExists(atPath: directory.path),
                "the conversation was parked rather than deleted")
        #expect(!FileManager.default.fileExists(atPath: image.path))
        // Nothing anywhere in the store still holds it.
        let leftovers = try FileManager.default.contentsOfDirectory(
            atPath: directory.deletingLastPathComponent().path)
        #expect(!leftovers.contains(".trash"))
        // And deleting it again is a real error, not a silent success.
        await #expect(throws: ConversationStoreError.self) {
            try await store.delete(id: meta.id)
        }
    }

    /// A second window can take the lock once the first gives it up.
    ///
    /// `isReadOnly` was decided once, at activation, so a window that opened
    /// while another held the lock said "this one can read them but not change
    /// them" for the rest of its life — including long after the other had
    /// quit.
    @Test func asecondStoreTakesTheLockOnceTheFirstReleasesIt() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        var first: ConversationStore? = try await makeStore(root)
        let second = ConversationStore(rootURL: root)
        try await second.activate()
        #expect(await second.isReadOnly, "two writers held the store at once")

        // Retrying while the first is still there changes nothing.
        await second.retryLock()
        #expect(await second.isReadOnly)

        first = nil
        _ = first
        await second.retryLock()
        #expect(await second.isReadOnly == false, "the notice would never clear")
    }

    /// A directory whose meta names a different conversation is not listed.
    ///
    /// That is what a copy looks like, and what an undelete tool produces.
    /// Listed, its id collides with the original's and the sidebar — which keys
    /// rows by id — drew one of the two as a blank row. Found running case C27
    /// by cloning a conversation directory without fixing the id inside.
    @Test func adirectoryWhoseMetaNamesAnotherConversationIsSkipped() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "original", identity: identity, session: session,
            sampling: sampling)
        let original = await store.directoryURL(for: meta.id)

        // A copy under a new name, its meta and header still naming the original.
        let cloneID = UUID()
        let clone = original.deletingLastPathComponent()
            .appendingPathComponent(cloneID.uuidString, isDirectory: true)
        try FileManager.default.copyItem(at: original, to: clone)

        let listed = try await store.list()
        #expect(listed.count == 1, "the clone was listed under the original's id")
        #expect(listed.first?.id == meta.id)
        #expect(await store.skippedByLastList == [cloneID],
                "the clone vanished with no trace")
    }

    /// Renaming does not count as using a conversation.
    ///
    /// The list is ordered by `updatedAt`, so bumping it on a rename jumped the
    /// row to the top of the sidebar — not what someone tidying up their titles
    /// is asking for.
    @Test func renamingDoesNotBumpTheConversationsPlaceInTheList() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "before", identity: identity, session: session,
            sampling: sampling)
        let before = try await store.open(id: meta.id).meta.updatedAt

        try await store.rename(id: meta.id, to: "after")

        let after = try await store.open(id: meta.id).meta
        #expect(after.title == "after")
        #expect(after.updatedAt == before, "the rename reordered the list")
        // The rename is still recorded, with its own time.
        let titles = try await store.open(id: meta.id).records.compactMap {
            record -> String? in
            if case .title(let value) = record { return value.title }
            return nil
        }
        #expect(titles.last == "after")
    }

    /// A `.trash` left by an older build is emptied on first activation.
    ///
    /// Delete used to park conversations there for an undo to restore, and
    /// nothing ever collected them — so upgrading with the purge removed would
    /// have stranded every chat anyone had ever deleted. Deleting means
    /// deleting, including for the ones deleted before.
    @Test func activationDiscardsATrashLeftByAnOlderBuild() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        // What the old build left, in place before the app ever opens the
        // store: a conversation parked under `.trash`.
        let trash = root.appendingPathComponent(".trash", isDirectory: true)
        let parked = trash.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: parked, withIntermediateDirectories: true)
        try Data("old".utf8).write(
            to: parked.appendingPathComponent("transcript.jsonl"))

        let store = try await makeStore(root)
        #expect(await store.discardedLegacyTrashCount == 1)
        #expect(!FileManager.default.fileExists(atPath: trash.path))

        // And the store is usable afterwards.
        _ = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        #expect(try await store.list().count == 1)
    }

    @Test func activationPropagatesLegacyTrashListingFailure() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let trash = root.appendingPathComponent(".trash", isDirectory: true)
        try FileManager.default.createDirectory(
            at: trash, withIntermediateDirectories: true)
        let store = ConversationStore(
            rootURL: root,
            legacyTrashOperations: ConversationLegacyTrashOperations(
                list: { _ in throw TrashFailure.list },
                remove: { _ in throw TrashFailure.remove }))

        await #expect(throws: TrashFailure.self) {
            try await store.activate()
        }
        #expect(await store.isReadOnly)
        #expect(FileManager.default.fileExists(atPath: trash.path))
    }

    @Test func retryLockReportsLegacyTrashRemovalFailureAndLeavesItRetryable()
        async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let trash = root.appendingPathComponent(".trash", isDirectory: true)
        try FileManager.default.createDirectory(
            at: trash, withIntermediateDirectories: true)
        let store = ConversationStore(
            rootURL: root,
            legacyTrashOperations: ConversationLegacyTrashOperations(
                list: { _ in ["parked"] },
                remove: { _ in throw TrashFailure.remove }))

        await #expect(throws: TrashFailure.self) {
            try await store.activate()
        }
        let first = await store.retryLock()
        let second = await store.retryLock()

        #expect(first?.contains("injected trash removal failure") == true)
        #expect(second?.contains("injected trash removal failure") == true)
        #expect(await store.isReadOnly)
        #expect(FileManager.default.fileExists(atPath: trash.path))
    }

    // MARK: - conversation.json as a cache

    /// The meta is a projection of the records, so losing it costs nothing.
    ///
    /// Before it was one, a `conversation.json` that would not decode cost the
    /// user the whole conversation: the listing skipped the directory and
    /// nothing ever put it back.
    @Test func aMetaThatIsGoneOrDamagedIsRebuiltFromTheRecords() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "keep me", identity: identity, session: session,
            sampling: sampling)
        try await store.append([.turn(ConversationTurnRecord(
            role: .user, at: Date(), text: "keep me", tokens: [1, 2]))], to: meta.id)
        let expected = try await store.open(id: meta.id).meta
        let metaURL = await store.metaURL(for: meta.id)

        try FileManager.default.removeItem(at: metaURL)
        #expect(try await store.open(id: meta.id).meta == expected)
        #expect(FileManager.default.fileExists(atPath: metaURL.path),
                "the cache was not written back")

        try "{ not json".write(to: metaURL, atomically: true, encoding: .utf8)
        #expect(try await store.open(id: meta.id).meta == expected)
        // The rebuild is reported: the conversation survives it, but a cache
        // that could not be read is still a fault of the store.
        try "{ not json".write(to: metaURL, atomically: true, encoding: .utf8)
        #expect(try await store.open(id: meta.id).metaRebuildReason != nil)
        // And an agreeing cache is not a rebuild.
        #expect(try await store.open(id: meta.id).metaRebuildReason == nil)
    }

    /// A meta naming another conversation over a header that names this one is
    /// a damaged cache, not a damaged conversation.
    ///
    /// The directory name and the header agree, which is what identity is; the
    /// third copy was the one that drifted, and it is the one that is replaced.
    @Test func aMetaNamingAnotherConversationIsRepairedFromTheHeader() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "mine", identity: identity, session: session, sampling: sampling)
        let metaURL = await store.metaURL(for: meta.id)

        var stranger = meta
        stranger.id = UUID()
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(stranger).write(to: metaURL)

        #expect(try await store.open(id: meta.id).meta.id == meta.id)
        let listed = try await store.list()
        #expect(listed.map(\.id) == [meta.id], "the repaired chat was skipped")
        #expect(await store.skippedByLastList.isEmpty)
    }

    /// A hand edit to the cache does not survive the next open.
    ///
    /// The one behaviour the projection takes away, and stated as such in ADR
    /// 034: renaming from outside the app means appending a `title` line to the
    /// transcript, not typing into `conversation.json`.
    @Test func ahandEditToTheCacheIsReplacedRatherThanHonoured() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "as written", identity: identity, session: session,
            sampling: sampling)

        var edited = meta
        edited.title = "typed straight into the file"
        edited.turnCount = 99
        edited.kvTokens = 12_345
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(edited).write(to: await store.metaURL(for: meta.id))

        let reopened = try await store.open(id: meta.id).meta
        #expect(reopened.title == "as written")
        #expect(reopened.turnCount == 0)
        #expect(reopened.kvTokens == nil)
        // The supported way, which is a record and therefore survives.
        try await store.rename(id: meta.id, to: "renamed properly")
        #expect(try await store.open(id: meta.id).meta.title == "renamed properly")
    }

    /// A store that cannot take the writer lock projects in memory and writes
    /// nothing. Two instances repairing one store is the corruption the lock
    /// exists to prevent.
    @Test func aReadOnlyStoreRebuildsInMemoryAndLeavesTheBytesAlone() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = try await makeStore(root)
        let meta = try await first.create(
            title: "original", identity: identity, session: session,
            sampling: sampling)
        let metaURL = await first.metaURL(for: meta.id)
        try "{ not json".write(to: metaURL, atomically: true, encoding: .utf8)
        let damaged = try Data(contentsOf: metaURL)

        let second = ConversationStore(rootURL: root)
        try await second.activate()
        #expect(await second.isReadOnly)
        #expect(try await second.open(id: meta.id).meta.title == "original")
        #expect(try Data(contentsOf: metaURL) == damaged,
                "a reader repaired somebody else's store")
        // And it is left out of the reader's list rather than repaired into it.
        #expect(try await second.list().isEmpty)
        #expect(await second.skippedByLastList == [meta.id])
    }

    /// A transcript written before the header carried the creation facts gets
    /// them once, and only once.
    @Test func alegacyTranscriptIsHoistedExactlyOnce() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let id = UUID()
        let directory = await store.directoryURL(for: id)
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("images", isDirectory: true),
            withIntermediateDirectories: true)
        let created = Date(timeIntervalSince1970: 1_756_000_000)
        let legacy = ConversationMeta(
            id: id, title: "from before", titleSource: .user,
            createdAt: created, updatedAt: created,
            identity: identity, session: session, sampling: sampling,
            boundary: ConversationBoundary(tokens: [7], needsReplay: true))
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(legacy).write(
            to: directory.appendingPathComponent(ConversationMeta.fileName))
        // A header with no identity, session or sampling: what the old build
        // wrote.
        var line = try encoder.encode(TranscriptRecord.header(
            ConversationHeaderRecord(id: id, createdAt: created)))
        line.append(0x0A)
        try line.write(to: directory.appendingPathComponent("transcript.jsonl"))

        let opened = try await store.open(id: id)
        #expect(opened.records.filter {
            if case .origin = $0 { return true } else { return false }
        }.count == 1)
        #expect(opened.meta.identity == identity)
        #expect(opened.meta.boundary == ConversationBoundary(tokens: [7], needsReplay: true))
        // The title still comes from the record the old build wrote, which
        // there is not one of — so the cache's is the only source, and after
        // the hoist it is the origin record that keeps the rest.
        #expect(opened.meta.title == ConversationTitle.untitled)

        // A second open has nothing to do.
        let again = try await store.open(id: id)
        #expect(again.records.filter {
            if case .origin = $0 { return true } else { return false }
        }.count == 1)
        // And the conversation reads the same with its cache thrown away.
        try FileManager.default.removeItem(
            at: directory.appendingPathComponent(ConversationMeta.fileName))
        #expect(try await store.open(id: id).meta == again.meta)
    }

    /// A turn the runtime rewound leaves its images behind, and nothing else
    /// would ever collect them: the transcript is append-only and the files are
    /// named by digest, so they are invisible to every other pass.
    @Test func theOrphanSweepKeepsWhatATurnStillReferences() async throws {
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)
        let images = await store.imagesURL(for: meta.id)
        try Data("kept".utf8).write(to: images.appendingPathComponent("kept.png"))
        try Data("kept".utf8).write(to: images.appendingPathComponent("kept.thumb.jpg"))
        try Data("gone".utf8).write(to: images.appendingPathComponent("orphan.png"))

        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "look",
                images: [ConversationImageRecord(
                    id: UUID(), displayName: "kept.png",
                    pixelsFile: "images/kept.png",
                    thumbnailFile: "images/kept.thumb.jpg",
                    sourceDigest: "aa", modelInputDigest: "bb",
                    width: 48, height: 48, softTokens: 1)],
                tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "answer", tokens: [2])),
        ], to: meta.id)

        #expect(try await store.sweepOrphanImages(in: meta.id) == 1)
        let remaining = try FileManager.default.contentsOfDirectory(
            atPath: images.path).sorted()
        #expect(remaining == ["kept.png", "kept.thumb.jpg"])
    }

    /// The whole point of the stored copy: write it, read it back, and get the
    /// same model input. If this stops holding, a reopened conversation is
    /// looking at a different picture.
    @Test func aStoredImageReplaysToTheSameModelInput() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let root = makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await makeStore(root)
        let meta = try await store.create(
            title: "t", identity: identity, session: session, sampling: sampling)

        let source = try Self.writeSourcePNG(width: 200, height: 100)
        defer { try? FileManager.default.removeItem(at: source) }
        let attachment = StagedImage(
            fileURL: source, displayName: "photo.png",
            encodedBytes: 1_024, sha256: String(repeating: "a", count: 64))

        let images = await store.imagesURL(for: meta.id)
        let record = try ConversationImageWriter(device: device)
            .write(attachment: attachment, into: images)
        #expect(record.pixelsFile == "images/aaaaaaaaaaaaaaaa-\(record.modelInputDigest).png")
        #expect(record.thumbnailFile == "images/aaaaaaaaaaaaaaaa.thumb.jpg")
        #expect(record.softTokens > 0)

        let directory = await store.directoryURL(for: meta.id)
        let preprocessor = Gemma4ImagePreprocessor(device: device)
        let stored = directory.appendingPathComponent(record.pixelsFile)
        let plan = try preprocessor.plan(storedModelInput: stored)
        #expect(plan.geometry.processedWidth == record.width)
        #expect(plan.geometry.processedHeight == record.height)
        #expect(plan.geometry.softTokenCount == record.softTokens)
        // Accepts its own digest, and only its own.
        _ = try preprocessor.preprocess(
            storedModelInput: plan, expectedDigest: record.modelInputDigest)
        #expect(throws: VisionImageError.self) {
            try preprocessor.preprocess(
                storedModelInput: try preprocessor.plan(storedModelInput: stored),
                expectedDigest: String(repeating: "0", count: 64))
        }

        let thumbnail = directory.appendingPathComponent(record.thumbnailFile)
        #expect(FileManager.default.fileExists(atPath: thumbnail.path))

        let original = try Data(contentsOf: stored)
        let repeated = try ConversationImageWriter(device: device)
            .write(attachment: attachment, into: images)
        #expect(repeated.pixelsFile == record.pixelsFile)
        #expect(try Data(contentsOf: stored) == original)

        // Model a new preprocessing result under the same source identity.
        // Its digest must name a separate file so the older turn still replays.
        let changedSource = try Self.writeSourcePNG(width: 300, height: 100)
        defer { try? FileManager.default.removeItem(at: changedSource) }
        let changed = try ConversationImageWriter(device: device).write(
            attachment: StagedImage(
                fileURL: changedSource, displayName: "photo.png",
                encodedBytes: 1_024, sha256: attachment.sha256), into: images)
        #expect(changed.modelInputDigest != record.modelInputDigest)
        #expect(changed.pixelsFile != record.pixelsFile)
        #expect(try Data(contentsOf: stored) == original)
        #expect(try FileManager.default.contentsOfDirectory(atPath: images.path).count == 3)
    }

    private static func writeSourcePNG(width: Int, height: Int) throws -> URL {
        let rowBytes = width * 4
        var bytes = [UInt8](repeating: 255, count: rowBytes * height)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * rowBytes + x * 4
                bytes[offset] = UInt8((x * 5 + y * 3) % 256)
                bytes[offset + 1] = UInt8((x * 7) % 256)
                bytes[offset + 2] = UInt8((y * 11) % 256)
            }
        }
        let image = try bytes.withUnsafeMutableBytes { raw -> CGImage in
            let context = try #require(CGContext(
                data: raw.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: rowBytes,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue))
            return try #require(context.makeImage())
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("store-source-\(UUID().uuidString).png")
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return url
    }
}
