import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Every way the send path can fail, and the one thing all of them owe the
/// user: the message comes back, exactly once, with the pictures it was sent
/// with, and nothing is left behind in the conversation the model can see.
///
/// A table rather than seven hand-written tests, because the defect class here
/// is a stage nobody wrote a hand-back for. `run()` had two halves that passed
/// the composer between them, and three defects were one half not knowing what
/// the other had already done: a replay that overwrote the message that started
/// it, a hand-back that dropped the pictures, and a stop that stranded the
/// retained links its next run then overwrote.
@Suite(.serialized) struct AppModelSendPipelineTests {
    /// The stage that fails.
    enum Stage: String, CaseIterable, Sendable, CustomTestStringConvertible {
        /// What is on disk cannot be put back into the KV at all.
        case replayRefused
        /// The runtime could not put it back this time.
        case replayFailed
        /// The request does not validate.
        case requestInvalid
        /// The turn's images cannot be given a reference of their own.
        case retainFailed
        /// The stream failed after the turn had started.
        case generationFailed
        /// Stop reached the turn and the runtime rewound it.
        case generationCancelled
        /// The KV no longer holds the conversation the turn was composed
        /// against.
        case lineageLost

        var testDescription: String { rawValue }

        /// Whether the stage needs a stored conversation on screen to reach.
        var replaysFirst: Bool { self == .replayRefused || self == .replayFailed }
    }

    private static let identity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "0d77464e",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    private static let message = "the message this stage must hand back"

    /// The whole world one stage needs, and the cleanup for it.
    @MainActor
    private struct Fixture {
        let model: AppModel
        let root: URL
        let attachments: AppImageAttachmentStore

        func tearDown() {
            model.releaseAllAttachments()
            try? FileManager.default.removeItem(at: root)
        }
    }

    @MainActor
    private func makeFixture(client: any AppInferenceClient) throws -> Fixture {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("send-pipeline-\(UUID().uuidString)",
                                    isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let fixture = try makeCompleteModelInstall("send-pipeline")
        try FileManager.default.moveItem(at: fixture, to: directory)
        let receiptURL = directory.appendingPathComponent("verified-install.json")
        var receipt = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL))
                as? [String: Any])
        receipt["modelDirectoryPath"] = directory.standardizedFileURL.path
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
            .write(to: receiptURL)
        try installVisionCompanion(forTextModel: directory)
        let attachments = AppImageAttachmentStore(
            directoryURL: root.appendingPathComponent("staged", isDirectory: true))
        let model = AppModel(modelDirectory: directory, client: client,
                             attachmentStore: attachments,
                             settingsPersistenceEnabled: true)
        model.modelPathText = directory.path
        // Read from the installed model's manifest in the app; a temporary
        // directory has none, and identity is not what this covers.
        model.conversationIdentity = Self.identity
        model.installationStatus = .complete
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        return Fixture(model: model, root: root, attachments: attachments)
    }

    private func client(for stage: Stage) -> any AppInferenceClient {
        switch stage {
        case .replayRefused:
            // The one double in the tree with no model lifecycle behind it, so
            // the replay has nothing to restore through.
            return MockInferenceClient(tokenDelayNanos: 0)
        case .generationFailed:
            return TerminalEventClient(
                .failed(.unknown("the stream broke"), partial: nil))
        case .generationCancelled:
            return TerminalEventClient(.cancelled(Self.diagnostics))
        case .lineageLost:
            return TerminalEventClient(
                .failed(.conversationLineageLost("the gate refused the turn"),
                        partial: nil))
        case .replayFailed, .requestInvalid, .retainFailed:
            return FakeInferenceClient(eventDelay: .milliseconds(1))
        }
    }

    @MainActor
    @Test(arguments: Stage.allCases)
    func everyFailingStageHandsTheMessageBackOnceAndLeavesNoTurnBehind(
        _ stage: Stage
    ) async throws {
        let client = self.client(for: stage)
        let fixture = try makeFixture(client: client)
        defer { fixture.tearDown() }
        let model = fixture.model
        try await model.waitForHistory()

        if stage.replaysFirst {
            // A stored conversation to reopen, and a screen showing it rather
            // than the live chat. Written through the store rather than sent:
            // neither client here is loaded, so a send would rewind and leave
            // a row with no token record, which the composer now refuses to
            // send before either stage can be reached.
            let store = try #require(model.conversationStore)
            // Activation takes the writer lock on its own task, and
            // `waitForHistory` returns before it has; a write before then is
            // refused as read-only.
            try await waitUntil { await !store.isReadOnly }
            let stored = try await store.create(
                title: "the conversation this stage reopens",
                identity: Self.identity,
                session: ConversationSessionSettings(
                    contextTokens: model.maxContextTokens, expertCacheSlots: 16,
                    visionResidencyPolicy: "onDemand"),
                sampling: model.currentSampling())
            try await store.append([
                .turn(ConversationTurnRecord(
                    role: .user, at: Date(), text: "earlier", tokens: [1, 2])),
                .turn(ConversationTurnRecord(
                    role: .assistant, at: Date(), text: "before", tokens: [3])),
            ], to: stored.id)
            await model.refreshHistory()
            model.openConversation(id: stored.id)
            try await waitUntil { await !model.transcriptHistory.isEmpty }
            #expect(model.openedConversationState == .continuable,
                    "the reopened row is not one the composer would send")
        }
        if stage == .replayFailed {
            (client as? FakeInferenceClient)?.failNextRestore(
                with: .conversationRestoreFailed("the restore was refused"))
        }
        if stage == .requestInvalid {
            // Refused by `AppGenerationRequest.validate`, which the pipeline
            // reaches after the composer has already been handed over.
            model.maxNewTokensOverride = 0
        }
        // A real picture, not a byte string: the stored copy is a preprocess
        // and a resize, so an unreadable one would make the write below finish
        // instantly with nothing to collect and the sweep assertions vacuous.
        let source = try AppModelHistoryTests.pngFixture()
        defer { try? FileManager.default.removeItem(at: source) }
        let staged = try fixture.attachments.stage(source)
        if stage == .retainFailed {
            // A regular file where `retain` needs a directory, so every hard
            // link fails the same way. After the stage above, which is what
            // creates the staging directory it goes in.
            try Data().write(to: fixture.attachments.directoryURL
                .appendingPathComponent("retained", isDirectory: false))
        }
        model.promptText = Self.message
        model.setComposerAttachmentsForTesting([staged])
        let turnsBefore = model.conversation.turns.count

        model.send()
        // Taken on the click, before any stage runs: this is what closes the
        // window a second Generate used to fit through.
        #expect(model.promptText.isEmpty, "the composer was not handed over")
        #expect(model.imageAttachments.isEmpty, "the pictures were not handed over")
        await SendWaiting.turnEnds(model)

        #expect(model.promptText == Self.message, "the message was not handed back")
        #expect(model.imageAttachments.count == 1,
                "the pictures did not come back with it")
        #expect(model.imageAttachments.first?.id == staged.id)
        // Handed back once. A second `restoreComposer` would find the composer
        // already holding this message and delete the very files it had just
        // put there, which is invisible in the array and obvious on disk.
        for attachment in model.imageAttachments {
            #expect(FileManager.default.fileExists(atPath: attachment.fileURL.path),
                    "the hand-back ran twice and deleted its own files")
        }
        #expect(model.conversation.turns.count == turnsBefore,
                "a failed stage left its turn in the conversation")
        #expect(!model.conversation.hasTurnInFlight)
        #expect(model.error != nil, "the stage failed silently")
        #expect(!model.isTurnInFlight)
        #expect(model.outputImageAttachments.isEmpty,
                "the transcript kept a reference to the handed-back pictures")
        if stage == .lineageLost {
            #expect(model.conversation.isLineageLost)
        }
    }

    /// The message goes back once even when the composer was used again while
    /// the turn was in flight — and the turn's own copies are released rather
    /// than left behind.
    @MainActor
    @Test func acomposerUsedDuringTheTurnKeepsItsOwnDraft() async throws {
        let fixture = try makeFixture(
            client: TerminalEventClient(.failed(.unknown("broken"), partial: nil)))
        defer { fixture.tearDown() }
        let model = fixture.model
        try await model.waitForHistory()

        let sent = try fixture.attachments.stage(
            data: Data("sent".utf8), displayName: "sent.png")
        model.promptText = "the message that failed"
        model.setComposerAttachmentsForTesting([sent])
        model.send()
        model.promptText = "a new draft"
        let drafted = try fixture.attachments.stage(
            data: Data("drafted".utf8), displayName: "drafted.png")
        model.setComposerAttachmentsForTesting([drafted])
        await SendWaiting.turnEnds(model)

        #expect(model.promptText == "a new draft")
        #expect(model.imageAttachments.map(\.id) == [drafted.id])
        #expect(FileManager.default.fileExists(atPath: drafted.fileURL.path))
        #expect(model.conversation.turns.isEmpty)
    }

    /// A rewound turn's image write ends before the sweep that collects after
    /// it.
    ///
    /// `beginStoringTurnImages` set `pendingTurnImageWrite` and only
    /// `persistCompletedTurn` cleared it, which a cancelled or failed turn never
    /// reaches: the app read as permanently mid-write to anything that asked
    /// whether a turn was finished, and `sweepImagesOfRewoundTurn` raced the
    /// very write it exists to collect after — deleting nothing, or deleting a
    /// file being written. Without the fix `pendingTurnImageWrite` is still
    /// there when the send returns.
    @MainActor
    @Test func arewoundTurnEndsItsImageWriteBeforeTheSweepRuns() async throws {
        let fixture = try makeFixture(
            client: TerminalEventClient(.cancelled(Self.diagnostics)))
        defer { fixture.tearDown() }
        let model = fixture.model
        try await model.waitForHistory()

        // A committed turn first, so the conversation is on disk and the
        // rewound turn below has an images directory to be swept.
        model.promptText = "the turn that lands"
        model.send()
        await SendWaiting.turnEnds(model)

        let source = try AppModelHistoryTests.pngFixture()
        defer { try? FileManager.default.removeItem(at: source) }
        let staged = try fixture.attachments.stage(source)
        model.promptText = "the turn that is stopped"
        model.setComposerAttachmentsForTesting([staged])
        model.send()
        await SendWaiting.turnEnds(model)

        #expect(model.pendingTurnImageWrite == nil,
                "the app still reads as mid-write after a rewound turn")
        let id = try #require(model.storedConversationID)
        let images = try #require(await model.conversationFolderURL(for: id))
            .appendingPathComponent("images", isDirectory: true)
        let left = (try? FileManager.default.contentsOfDirectory(
            at: images, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])) ?? []
        #expect(left.isEmpty,
                "the sweep left \(left.count) file(s) no record names")
    }

    private static let diagnostics = AppDiagnostics(
        generatedTokens: 0, stopReason: .cancelled,
        timeToFirstTokenSeconds: 0, decodeSeconds: 0, tokensPerSecond: 0,
        peakMemoryBytes: nil,
        runtimeOptions: AppRuntimeOptions())
}

/// A client whose generation is one terminal event and nothing else.
///
/// The stages a terminal event reaches — a rewind, a stream failure, a lost
/// lineage — are otherwise arranged by racing a double, and a race is not a
/// test of what the app does with the outcome.
private final class TerminalEventClient: AppInferenceClient, @unchecked Sendable {
    private let event: AppInferenceEvent

    init(_ event: AppInferenceEvent) {
        self.event = event
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(event)
            continuation.finish()
        }
    }

    func cancel() {}
}
