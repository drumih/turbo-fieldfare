import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// One test per row of the plan's "errors stay visible" table.
///
/// Every one of these was a `try?` or a `catch` that turned a real fault into a
/// benign-looking default: an empty sidebar over a store that could not be
/// listed, a picture dropped from a record while its placeholder tokens stayed
/// in the turn's IDs, a row that "could not be read" with no way to ask why, and
/// any lock failure at all reading as "another window has it". A failure that
/// changes what the user sees has to reach them with its cause.
@Suite struct ConversationErrorReportingTests {
    private static let identity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "0d77464e",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    private static let session = ConversationSessionSettings(
        contextTokens: 8_192, expertCacheSlots: 16, visionResidencyPolicy: "on-demand")

    private static let sampling = ConversationSampling(
        temperature: 0.2, topKEnabled: true, topK: 64,
        topPEnabled: true, topP: 0.95, maxNewTokens: 8_192)

    private static func header(_ id: UUID) -> TranscriptRecord {
        .header(ConversationHeaderRecord(
            id: id, createdAt: Date(timeIntervalSince1970: 1_000),
            identity: identity, session: session, sampling: sampling))
    }

    private static func makeRoot(_ label: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("error-reporting-\(label)-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return root
    }

    @MainActor
    private func readyModel(_ client: FakeInferenceClient, root: URL,
                            attachments: AppImageAttachmentStore) async throws
        -> AppModel {
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fixture = try makeCompleteModelInstall("error-reporting")
        try FileManager.default.moveItem(at: fixture, to: directory)
        let receiptURL = directory.appendingPathComponent("verified-install.json")
        var receipt = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL))
                as? [String: Any])
        receipt["modelDirectoryPath"] = directory.standardizedFileURL.path
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
            .write(to: receiptURL)
        try installVisionCompanion(forTextModel: directory)
        let model = AppModel(modelDirectory: directory, client: client,
                             attachmentStore: attachments,
                             settingsPersistenceEnabled: true)
        model.modelPathText = directory.path
        model.conversationIdentity = Self.identity
        model.installationStatus = .complete
        try await client.ensureLoaded(
            modelDirectory: directory,
            maxContextTokens: model.maxContextTokens,
            options: model.runtimeOptions,
            forceLogitsHead: true) { _ in }
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        try await model.waitForHistory()
        return model
    }

    // MARK: - Row 1: a list that cannot be read keeps the last one that could

    /// The sidebar used to empty itself. `(try? await store.list()) ?? []` made a
    /// store that could not be enumerated — a permission change, a volume that
    /// went away — look exactly like a user with no chats at all, and the window
    /// then offered New Chat over conversations that were still on disk.
    @MainActor
    @Test func alistThatCannotBeReadKeepsTheLastGoodOneAndSaysWhy() async throws {
        let root = try Self.makeRoot("list")
        defer { try? FileManager.default.removeItem(at: root) }
        let attachments = AppImageAttachmentStore(
            directoryURL: root.appendingPathComponent("staged", isDirectory: true))
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client, root: root, attachments: attachments)
        defer { model.releaseAllAttachments() }

        model.promptText = "the chat that must not vanish"
        model.send()
        await SendWaiting.turnEnds(model)
        try await waitUntil { await model.history.entries.count == 1 }
        model.historyDiagnostic = nil

        // The store's root is a file now, so enumerating it throws rather than
        // coming back empty.
        let storeRoot = await (try #require(model.conversationStore)).rootURL
        try FileManager.default.removeItem(at: storeRoot)
        try Data("not a directory".utf8).write(to: storeRoot)
        await model.refreshHistory()

        #expect(model.history.entries.count == 1,
                "the sidebar emptied itself over a store it could not read")
        #expect(model.history.unreadableCount >= 1)
        let diagnostic = try #require(model.historyDiagnostic)
        #expect(diagnostic.contains("could not be read"),
                Comment(rawValue: diagnostic))
    }

    // MARK: - Row 2: a picture that was not stored ends the replay record

    @Test func aturnWhosePicturesWereNotStoredHasNoReplayCount() throws {
        let id = UUID()
        let records: [TranscriptRecord] = [
            Self.header(id),
            .turn(ConversationTurnRecord(
                role: .user, at: Date(timeIntervalSince1970: 1_010),
                text: "what is this?", tokens: [1, 2], imageWriteFailed: true)),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(timeIntervalSince1970: 1_011),
                text: "a roof", tokens: [3])),
        ]

        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: nil)

        #expect(meta.imageWriteFailed == true)
        #expect(meta.kvTokens == nil,
                "a conversation with a picture behind nothing claimed a count")
        #expect(meta.turnCount == 2, "the turns are still there to read")
    }

    @Test func aconversationWithAnUnwrittenPictureSaysThatRatherThanNoRecord() {
        let meta = ConversationMeta(
            id: UUID(), title: "roofs", createdAt: Date(), updatedAt: Date(),
            turnCount: 2, kvTokens: nil, imageCount: 1,
            identity: Self.identity, session: Self.session, sampling: Self.sampling,
            imageWriteFailed: true)

        let state = ConversationContinuability.evaluate(
            meta: meta, currentContext: 8_192, identity: Self.identity)

        // Not `.tokenCountUnknown`: that is true as well and sends the reader
        // after the wrong thing.
        #expect(state == .cannotReplay(reason: .imageRecordMissing))
    }

    @Test func adocumentRefusesToReplayATurnWhosePictureWasNotStored() {
        let id = UUID()
        let meta = ConversationMeta(
            id: id, title: "roofs", createdAt: Date(), updatedAt: Date(),
            turnCount: 2, kvTokens: nil, imageCount: 1,
            identity: Self.identity, session: Self.session, sampling: Self.sampling,
            imageWriteFailed: true)
        let records: [TranscriptRecord] = [
            Self.header(id),
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "what is this?", tokens: [1],
                imageWriteFailed: true)),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "a roof", tokens: [2])),
        ]

        let document = ConversationDocument.load(
            ConversationOpenResult(meta: meta, records: records, isReadOnly: false,
                                   droppedTornFinalLine: false),
            directory: URL(fileURLWithPath: "/tmp/conversations/\(id.uuidString)",
                           isDirectory: true))

        // Still readable: the transcript is what the read-only view exists for.
        #expect(document.pairs.count == 1)
        #expect(document.lineage == .failure(.imageRecordMissing))
        #expect(ConversationRecordError.imageRecordMissing.description
            == "a stored image was never written, and its turn still cites it")
    }

    /// End to end: a picture the writer cannot prepare is said out loud, the
    /// record carries it, and the row stops offering a continuation it could
    /// not honour. Every one of these was silent — the image simply disappeared
    /// from the record while its placeholder tokens stayed in the turn's IDs.
    @MainActor
    @Test func apictureThatCouldNotBeStoredIsReportedAndEndsTheReplayRecord() async throws {
        let root = try Self.makeRoot("image-write")
        defer { try? FileManager.default.removeItem(at: root) }
        let attachments = AppImageAttachmentStore(
            directoryURL: root.appendingPathComponent("staged", isDirectory: true))
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client, root: root, attachments: attachments)
        defer { model.releaseAllAttachments() }

        // Bytes no preprocessor will take, staged exactly as a real picture is:
        // the write fails per image rather than per turn.
        let staged = try attachments.stage(
            data: Data("this is not a picture".utf8), displayName: "roof.png")
        model.promptText = "what is this?"
        model.setComposerAttachmentsForTesting([staged])
        model.send()
        await SendWaiting.turnEnds(model)
        // The turn is written from a task the terminal event starts, so the
        // record is the thing to wait for rather than the run.
        try await waitUntil {
            await model.history.entries.first?.imageWriteFailed == true
        }

        let diagnostic = try #require(model.historyDiagnostic)
        #expect(diagnostic.contains("1 image(s) in this turn have no stored copy"),
                Comment(rawValue: diagnostic))
        #expect(!diagnostic.contains("roof.png"),
                "the diagnostic named the user's file")
        let meta = try #require(model.history.entries.first)
        #expect(meta.imageWriteFailed == true)
        #expect(meta.kvTokens == nil)
        #expect(model.continuability(of: meta)
            == .cannotReplay(reason: .imageRecordMissing))
    }

    // MARK: - Row 5: a size that cannot be read is drawn as zero, on purpose

    /// The one discarded failure the table keeps.
    ///
    /// `encodedBytes` is display only: the capacity arithmetic it feeds is
    /// about pictures being attached, and a stored one is already in the
    /// conversation. So a file whose size cannot be read is drawn as zero bytes
    /// rather than costing the reader the turn it belongs to. Written down here
    /// so it is a decision rather than an oversight.
    @Test func astoredPictureWhoseSizeCannotBeReadIsDrawnAsZeroBytes() throws {
        let id = UUID()
        let image = ConversationImageRecord(
            id: UUID(), displayName: "roof.heic",
            pixelsFile: "images/1a2b.png",
            thumbnailFile: "images/1a2b.thumb.jpg",
            sourceDigest: "source", modelInputDigest: "input",
            width: 768, height: 768, softTokens: 256)
        let records: [TranscriptRecord] = [
            Self.header(id),
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "what is this?",
                images: [image], tokens: [1])),
        ]
        let meta = try ConversationMetaProjection.project(
            directoryID: id, records: records, legacy: nil)

        // A directory that does not exist, so neither copy can be sized.
        let document = ConversationDocument.load(
            ConversationOpenResult(meta: meta, records: records, isReadOnly: false,
                                   droppedTornFinalLine: false),
            directory: URL(fileURLWithPath: "/tmp/no-such-conversation-\(id.uuidString)",
                           isDirectory: true))

        let attached = try #require(document.turns.first?.images.first)
        #expect(attached.encodedBytes == 0)
        #expect(attached.displayName == "roof.heic",
                "the turn lost its picture over a size nobody reads")
    }

    // MARK: - Row 3: a skipped row says why it was skipped

    @Test func alistNamesTheReasonItSkippedADirectory() async throws {
        let root = try Self.makeRoot("skip")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ConversationStore(rootURL: root)
        try await store.activate()
        let meta = try await store.create(
            title: "readable", identity: Self.identity, session: Self.session,
            sampling: Self.sampling)
        let broken = try await store.create(
            title: "damaged", identity: Self.identity, session: Self.session,
            sampling: Self.sampling)

        // Damage in the middle of the transcript, which no rebuild can recover:
        // the meta goes too, so the directory has to be read from its records.
        let directory = await store.directoryURL(for: broken.id)
        try FileManager.default.removeItem(
            at: directory.appendingPathComponent(ConversationMeta.fileName))
        try Data("{not json}\n".utf8).write(
            to: directory.appendingPathComponent("transcript.jsonl"))

        let listed = try await store.list()

        #expect(listed.map(\.id) == [meta.id])
        #expect(await store.skippedByLastList == [broken.id])
        let reason = try #require(await store.skipReasons[broken.id])
        #expect(!reason.isEmpty)
        #expect(reason.contains("\(broken.id.uuidString)") || reason.contains("record"),
                Comment(rawValue: reason))
    }

    // MARK: - Row 4: a lock that failed for any other reason says so

    @Test func alockThatFailedForAnythingButAnotherWindowIsReported() async throws {
        let root = try Self.makeRoot("lock")
        defer { try? FileManager.default.removeItem(at: root) }
        let storeRoot = root.appendingPathComponent("conversations", isDirectory: true)
        let store = ConversationStore(rootURL: storeRoot)

        // A file where the store's root has to be, so taking the lock fails for
        // a reason that is not "another window holds it".
        try Data("not a directory".utf8).write(to: storeRoot)

        let reason = await store.retryLock()

        #expect(reason != nil, "a lock failure read as another window holding it")
        #expect(await store.isReadOnly)
    }

    /// And the window hears about it. A second instance holding the lock is the
    /// ordinary case and stays quiet; anything else reaches `historyDiagnostic`.
    @MainActor
    @Test func thewindowIsToldWhyItCouldNotTakeTheWriterLock() async throws {
        let root = try Self.makeRoot("lock-app")
        defer { try? FileManager.default.removeItem(at: root) }
        let attachments = AppImageAttachmentStore(
            directoryURL: root.appendingPathComponent("staged", isDirectory: true))
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = try await readyModel(client, root: root, attachments: attachments)
        defer { model.releaseAllAttachments() }
        let storeRoot = await (try #require(model.conversationStore)).rootURL

        // Another instance takes the lock first, which is the case that stays
        // quiet — and then the root becomes a file, which is the case that does
        // not.
        let peer = ConversationStore(rootURL: storeRoot)
        try FileManager.default.removeItem(at: storeRoot)
        try Data("not a directory".utf8).write(to: storeRoot)
        _ = peer
        model.history.setReadOnlyStore(true)
        model.historyDiagnostic = nil

        model.reacquireStoreIfPossible()
        try await waitUntil { await model.historyDiagnostic != nil }

        let diagnostic = try #require(model.historyDiagnostic)
        #expect(diagnostic.contains("could not be locked for writing")
                || diagnostic.contains("could not be read"),
                Comment(rawValue: diagnostic))
    }
}
