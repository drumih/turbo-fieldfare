import Foundation
import Synchronization
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Persisting a conversation is only worth anything if what lands on disk is
/// what the model saw. Every case here is a way the two could come apart: a
/// turn written before it reached the KV, a rewound turn written anyway, a
/// reopen that re-renders text instead of replaying IDs, or a new chat that
/// keeps appending to the old one's file.
@Suite struct AppModelHistoryTests {
    @MainActor
    @Test func reopeningHistoryKeepsAutomaticPublicResponseLimit() async throws {
        let (model, root) = try await readyModel(FakeInferenceClient(eventDelay: .milliseconds(1)))
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(model.maxNewTokensOverride == nil)
        model.promptText = "saved exchange"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        // A recorded request budget is not a public app response-limit control.
        #expect(model.maxNewTokensOverride == nil)
    }

    @MainActor
    @Test func queuedSendKeepsTheCompletedAnswerTerminal() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "first"
        model.send()
        try await finish(model)
        let answer = model.outputText
        let run = model.runIdentity
        model.promptText = "second"
        model.send()
        // No actor hop: deliver has not published the new turn yet.
        #expect(model.isTurnInFlight)
        #expect(!model.canRun)
        #expect(model.outputText == answer)
        #expect(model.runIdentity == run)
        #expect(!model.isTranscriptTurnInFlight)
        try await finish(model)
        #expect(!model.isTranscriptTurnInFlight)
    }

    @MainActor
    @Test func queuedStoredSendDoesNotExposeTheHeldChat() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "stored A"
        model.send()
        try await finish(model)
        let a = try #require(model.storedConversationID)
        model.newChat()
        model.promptText = "held B"
        model.send()
        try await finish(model)
        model.openConversation(id: a)
        try await waitUntil { await model.screen.document?.id == a }
        model.promptText = "continue A"
        model.send()
        #expect(model.isTurnInFlight)
        #expect(!model.isTranscriptTurnInFlight)
        #expect(!model.showsLiveTurn)
        #expect(model.screen.document?.id == a)
        try await finish(model)
        #expect(model.storedConversationID == a)
    }

    @MainActor
    @Test func failedCreationCannotStartSavingHalfwayThroughLiveKV() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try #require(model.conversationStore)
        let storeRoot = await store.rootURL
        try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: storeRoot.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: storeRoot.path) }
        model.promptText = "unsaved first exchange"
        model.send()
        await SendWaiting.turnEnds(model)
        await model.awaitPendingPersistence()
        #expect(model.conversation.committedTurns == 1)
        #expect(model.storedConversationID == nil)
        #expect(model.error?.userMessage.contains("not be saved") == true)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: storeRoot.path)

        // Retrying storage must not publish a continuation without its KV prefix.
        model.promptText = "continue unsaved exchange"
        model.send()
        await SendWaiting.turnEnds(model)
        await model.awaitPendingPersistence()
        #expect(model.conversation.committedTurns == 2)
        #expect(model.storedConversationID == nil)
        #expect(try await store.list().isEmpty)
        #expect(model.error != nil)

        model.newChat()
        model.promptText = "fresh durable exchange"
        model.send()
        try await finish(model)
        #expect(try await store.list().count == 1)
    }

    @MainActor
    @Test(arguments: [false, true])
    func adversarialFailedReplayThenRetryAppendsExactlyOnce(connectionLost: Bool) async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "A"
        model.send()
        try await finish(model)
        let a = try #require(model.storedConversationID)
        model.newChat()
        model.promptText = "B"
        model.send()
        try await finish(model)
        let b = try #require(model.storedConversationID)
        let store = try #require(model.conversationStore)
        let aURL = await store.transcriptURL(for: a)
        let bURL = await store.transcriptURL(for: b)
        let aBefore = try Data(contentsOf: aURL)
        let bBefore = try Data(contentsOf: bURL)
        model.openConversation(id: a)
        try await waitUntil { await model.screen.document?.id == a }
        client.failNextRestore(with: connectionLost
            ? .connectionLost("injected EOF")
            : .conversationRestoreFailed("stored image digest mismatch"))
        model.promptText = "retry draft"
        model.send()
        await SendWaiting.turnEnds(model)
        await model.persistenceTail?.value
        #expect(model.error != nil)
        #expect(model.promptText == "retry draft")
        #expect(model.history.selection == a)
        #expect(model.serviceEpoch == nil)
        #expect(try Data(contentsOf: aURL) == aBefore)
        #expect(try Data(contentsOf: bURL) == bBefore)
        if connectionLost {
            #expect(model.canLoadModel)
            model.loadModel()
            try await waitUntil { await model.loadState.isReady }
            try await waitUntil { await model.screen.document?.id == a }
        }
        model.send()
        try await finish(model)
        #expect(model.storedConversationID == a)
        #expect(client.restoredLineages.count == 1)
        #expect(try await store.open(id: a).meta.turnCount == 4)
        #expect(try Data(contentsOf: bURL) == bBefore)
    }

    @MainActor
    @Test(arguments: [false, true])
    func adversarialContextRefusalThenRaiseAppendsOnce(boundary: Bool) async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "seed"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        let store = try #require(model.conversationStore)
        let limit = 4_096 - ConversationGenerationReserve.tokens
            - ConversationGenerationReserve.turnEnvelope
        let current = try #require(try await store.open(id: id).meta.kvTokens)
        let padding = (boundary ? limit : limit + 1) - current
        try await store.append([
            .turn(ConversationTurnRecord(role: .user, at: Date(), text: "padding",
                tokens: Array(repeating: Int32(7), count: padding - 1))),
            .turn(ConversationTurnRecord(role: .assistant, at: Date(), text: "answer",
                tokens: [8], boundary: ConversationBoundary(
                    tokens: boundary ? [7] : [], needsReplay: boundary)))
        ], to: id)
        await model.refreshHistory()
        model.newChat()
        model.setMaxContextTokens(4_096)
        client.setLoadedContext(4_096)
        model.applyLoadState(model.loadState)
        model.openConversation(id: id)
        try await waitUntil { await model.screen.document?.id == id }
        let url = await store.transcriptURL(for: id)
        let before = try Data(contentsOf: url)
        let count = try await store.open(id: id).meta.turnCount
        #expect(!model.screen.allowsSend)
        model.promptText = "continue"
        model.send()
        #expect(!model.isTurnInFlight)
        #expect(client.restoredLineages.isEmpty)
        #expect(try Data(contentsOf: url) == before)
        model.setMaxContextTokens(8_192)
        client.setLoadedContext(8_192)
        model.applyLoadState(model.loadState)
        try await waitUntil { @MainActor in model.screen.allowsSend && model.screen.document != nil }
        model.send()
        try await finish(model)
        #expect(model.storedConversationID == id)
        #expect(client.restoredLineages.count == 1)
        #expect(try await store.open(id: id).meta.turnCount == count + 2)
    }

    @MainActor
    @Test(arguments: [false, true])
    func adversarialDeletingHeldChatPreservesLatestViewedChat(delayed: Bool) async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let attachments = AppImageAttachmentStore()
        let (model, root) = try await readyModel(client, attachmentStore: attachments)
        defer { try? FileManager.default.removeItem(at: root) }
        var ids: [UUID] = []
        for label in ["A", "C", "B"] {
            model.newChat()
            model.promptText = label
            model.send()
            try await finish(model)
            ids.append(try #require(model.storedConversationID))
        }
        let viewed = delayed ? ids[1] : ids[0]
        let held = ids[2]
        let store = try #require(model.conversationStore)
        let otherURL = await store.transcriptURL(for: delayed ? ids[0] : ids[1])
        let otherBytes = try Data(contentsOf: otherURL)
        let first = ids[0]
        model.openConversation(id: first)
        try await waitUntil { await model.screen.document?.id == first }
        let gate = HistoryListGate()
        if delayed {
            model.persistenceTail = Task { await gate.wait() }
            await gate.waitUntilEntered()
        }
        model.deleteConversation(id: held)
        let deletion = try #require(model.conversationDeletionTask)
        if delayed {
            model.openConversation(id: viewed)
            try await waitUntil { await model.screen.document?.id == viewed }
        }
        model.promptText = "continue the viewed chat"
        let source = try Self.pngFixture()
        defer { try? FileManager.default.removeItem(at: source) }
        let staged = try attachments.stage(source)
        model.setComposerAttachmentsForTesting([staged])
        await gate.open()
        await deletion.value
        #expect(model.history.entry(held) == nil)
        #expect(model.history.selection == viewed)
        #expect(model.screen.conversationID == viewed)
        #expect(model.promptText == "continue the viewed chat")
        #expect(model.imageAttachments.map(\.id) == [staged.id])
        #expect(FileManager.default.fileExists(atPath: staged.fileURL.path))
        model.clearImages()
        model.send()
        try await finish(model)
        #expect(model.storedConversationID == viewed)
        #expect(client.restoredLineages.count == 1)
        #expect(try await store.open(id: viewed).meta.turnCount == 4)
        #expect(try Data(contentsOf: otherURL) == otherBytes)
    }

    @MainActor
    @Test func adversarialReadCannotResetEditsMadeBeforeItCompletes() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        model.newChat()
        model.openConversation(id: id)
        model.temperature = 0.9
        model.topK = 17
        try await waitUntil { await model.screen.document?.id == id }
        #expect(model.temperature == 0.9)
        #expect(model.topK == 17)
    }

    @MainActor
    @Test func adversarialDeletingUnrelatedRowAfterUnloadKeepsViewedChat() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        var ids: [UUID] = []
        for label in ["A", "C", "B"] {
            model.newChat()
            model.promptText = label
            model.send()
            try await finish(model)
            ids.append(try #require(model.storedConversationID))
        }
        model.unloadModel()
        try await waitUntil { await model.loadState == .notLoaded }
        let viewed = ids[0]
        model.openConversation(id: viewed)
        try await waitUntil { await model.screen.document?.id == viewed }
        model.deleteConversation(id: ids[1])
        await model.conversationDeletionTask?.value
        #expect(model.history.selection == viewed)
        #expect(model.screen.document?.id == viewed)
        #expect(model.history.entry(ids[1]) == nil)
    }

    @MainActor
    @Test func adversarialImmediateSendBeforeDocumentReadCompletes() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        model.newChat()
        // No suspension between opening and sending: the document task has
        // been scheduled but cannot yet have delivered its MainActor result.
        model.openConversation(id: id)
        #expect(model.screen.document == nil)
        model.temperature = 0.9
        model.topK = 17
        model.promptText = "second"
        model.send()
        try await finish(model)
        let store = try #require(model.conversationStore)
        let opened = try await store.open(id: id)
        #expect(opened.meta.turnCount == 4)
        #expect(model.storedConversationID == id)
        #expect(client.restoredLineages.count == 1)
        #expect(model.temperature == 0.9)
        #expect(model.topK == 17)
        #expect(opened.meta.sampling.temperature == 0.9)
        #expect(opened.meta.sampling.topK == 17)
    }

    /// KV release used to reopen the held row, silently replacing the viewed
    /// row and redirecting the next message after unload/load or reload.
    @MainActor
    @Test(arguments: [false, true], [false, true])
    func modelLifecyclePreservesViewedChat(reload: Bool, keepHeldChat: Bool) async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.applyLoadState(model.loadState)
        model.promptText = "viewed"
        model.send()
        try await finish(model)
        let viewed = try #require(model.history.selection)
        let store = try #require(model.conversationStore)
        let viewedRecord = try await store.open(id: viewed)
        let expectedTokens = viewedRecord.records.compactMap { record -> [Int32]? in
            guard case .turn(let turn) = record else { return nil }
            return turn.tokens
        }.flatMap { $0 }
        model.newChat()
        model.promptText = "held"
        model.send()
        try await finish(model)
        let held = try #require(model.history.selection)
        let heldURL = await store.transcriptURL(for: held)
        let heldBytes = try Data(contentsOf: heldURL)
        if !keepHeldChat { model.newChat() }
        model.openConversation(id: viewed)
        try await waitUntil { await model.screen.document?.id == viewed }
        model.promptText = "continue viewed"

        if reload {
            model.setMaxContextTokens(4_096)
            try #require(model.canReloadModel)
            model.reloadModel()
        } else {
            model.unloadModel()
            try await waitUntil { await model.loadState == .notLoaded }
            #expect(model.history.selection == viewed)
            model.loadModel()
        }
        try await waitUntil { await model.loadState.isReady }
        try await waitUntil { await model.screen.document != nil }
        #expect(model.history.selection == viewed)
        #expect(model.screen.conversationID == viewed)
        #expect(model.serviceEpoch == nil)
        #expect(model.conversation.turns.isEmpty)
        #expect(model.promptText == "continue viewed")
        #expect(client.restoredLineages.isEmpty, "loading must not replay before send")
        model.send()
        await SendWaiting.turnEnds(model)
        await model.persistenceTail?.value
        #expect(client.restoredLineages.count == 1)
        #expect(client.restoredLineages.first?.tokenIDs == expectedTokens)
        #expect(model.storedConversationID == viewed)
        #expect(try await store.open(id: viewed).meta.turnCount == 4)
        #expect(try Data(contentsOf: heldURL) == heldBytes)
    }

    @MainActor
    @Test(arguments: [false, true])
    func modelLifecycleKeepsUnreadableSelectionFailClosed(reload: Bool) async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.applyLoadState(model.loadState)
        model.promptText = "unreadable"
        model.send()
        try await finish(model)
        let viewed = try #require(model.history.selection)
        let store = try #require(model.conversationStore)
        model.newChat()
        model.promptText = "held"
        model.send()
        try await finish(model)
        let directory = await store.directoryURL(for: viewed)
        try FileManager.default.removeItem(at: directory)
        model.openConversation(id: viewed)
        try await waitUntil { await model.screen.shape == .unreadable }
        if reload {
            model.setMaxContextTokens(4_096)
            try #require(model.canReloadModel)
            model.reloadModel()
        } else {
            model.unloadModel()
            try await waitUntil { await model.loadState == .notLoaded }
            model.loadModel()
        }
        try await waitUntil { await model.loadState.isReady }
        try await waitUntil { await model.screen.shape != .reading }
        #expect(model.history.selection == viewed)
        #expect(model.screen.shape == .unreadable)
        #expect(!model.screen.allowsSend)
        #expect(model.serviceEpoch == nil)
        #expect(client.restoredLineages.isEmpty)
    }

    @MainActor
    @Test(arguments: [false, true])
    func modelLifecycleKeepsNewChatEmpty(reload: Bool) async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.applyLoadState(model.loadState)
        model.promptText = "previous"
        model.send()
        try await finish(model)
        let previous = try #require(model.history.selection)
        model.newChat()
        if reload {
            model.setMaxContextTokens(4_096)
            try #require(model.canReloadModel)
            model.reloadModel()
        } else {
            model.unloadModel()
            try await waitUntil { await model.loadState == .notLoaded }
            model.loadModel()
        }
        try await waitUntil { await model.loadState.isReady }
        #expect(model.history.selection == nil)
        #expect(model.screen == .live)
        #expect(model.conversation.turns.isEmpty)
        model.promptText = "new"
        model.send()
        try await finish(model)
        #expect(model.storedConversationID != previous)
        #expect(client.restoredLineages.isEmpty)
    }

    private static let identity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "0d77464e",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    private func placeCompleteModel(at directory: URL, tag: String) throws {
        let fixture = try makeCompleteModelInstall(tag)
        defer { try? FileManager.default.removeItem(at: fixture) }
        try FileManager.default.createDirectory(
            at: directory.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: fixture, to: directory)
        let receiptURL = directory.appendingPathComponent("verified-install.json")
        var receipt = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as? [String: Any])
        receipt["modelDirectoryPath"] = directory.standardizedFileURL.path
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
            .write(to: receiptURL)
    }

    @MainActor
    private func readyModel(
        _ client: FakeInferenceClient,
        attachmentStore: AppImageAttachmentStore = AppImageAttachmentStore()
    ) async throws -> (model: AppModel, root: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-\(UUID().uuidString)", isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let model = AppModel(modelDirectory: directory, client: client,
                             attachmentStore: attachmentStore,
                             settingsPersistenceEnabled: true)
        model.modelPathText = directory.path
        // The app reads this from the installed model's manifest; a temporary
        // directory has none, and the identity is not what these tests are about.
        model.conversationIdentity = Self.identity
        try await client.ensureLoaded(
            modelDirectory: directory,
            maxContextTokens: model.maxContextTokens,
            options: model.runtimeOptions,
            forceLogitsHead: true) { _ in }
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        // Otherwise `requiresModelInstallation` is true and every
        // `canStartNewChat` assertion below passes for that reason instead of
        // the one it is testing.
        model.installationStatus = .complete
        try await model.waitForHistory()
        return (model, root)
    }

    @MainActor
    private func finish(_ model: AppModel) async throws {
        await SendWaiting.turnEnds(model)
        // The turn is written from a task the terminal event starts, so the
        // assertions have to wait for the store rather than for the run.
        try await model.waitForStoredTurns(atLeast: model.conversation.turns.count)
    }

    private func imageOutcome(_ name: String) -> TurnImageWriteOutcome {
        TurnImageWriteOutcome(records: [ConversationImageRecord(
            id: UUID(), displayName: name,
            pixelsFile: "images/\(name).png",
            thumbnailFile: "images/\(name).thumb.jpg",
            sourceDigest: name, modelInputDigest: "model-\(name)",
            width: 48, height: 48, softTokens: 1)])
    }

    /// A delete waiting for the previous reply's image write must reserve its
    /// mutation before another send can start against the same directory.
    @MainActor
    @Test func deletionExcludesSendsUntilPendingPersistenceAndRemovalFinish() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(100))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        let gate = HistoryListGate()
        model.promptText = "first"
        model.send()
        try await waitUntil { client.isGenerating }
        model.pendingTurnImageWrite = Task {
            await gate.wait()
            return TurnImageWriteOutcome()
        }
        try await waitUntil { await !model.isTurnInFlight }
        try #require(!model.isTurnInFlight)
        await gate.waitUntilEntered()
        let id = try #require(model.storedConversationID)
        let folder = try #require(await model.conversationFolderURL(for: id))
        model.deleteConversation(id: id)
        model.promptText = "second"
        #expect(!model.canRun)
        #expect(!model.canMutateConversation(id: id))
        try await Task.sleep(for: .milliseconds(20))
        model.send()
        #expect(!model.isTurnInFlight)
        #expect(model.promptText == "second")
        #expect(FileManager.default.fileExists(atPath: folder.path))
        await gate.open()
        await SendWaiting.turnEnds(model)
        await model.awaitPendingPersistence()
        try await waitUntil { await model.history.entry(id) == nil }
        #expect(!FileManager.default.fileExists(atPath: folder.path))
        #expect(model.storedConversationID == nil)
        #expect(model.historyDiagnostic == nil)
        #expect(model.canRun)
    }

    /// Cancelling the waiter of an independent Task.value does not end that
    /// wait. A task-group timeout therefore kept Quit blocked on the write.
    @MainActor
    @Test func terminationDeadlineReturnsWhilePersistenceIsStillBlocked() async throws {
        let model = AppModel(client: FakeInferenceClient())
        let gate = HistoryListGate()
        model.persistenceTail = Task { await gate.wait() }
        await gate.waitUntilEntered()
        var outcome: Bool?
        let waiting = Task {
            outcome = await model.awaitPendingPersistence(timeout: .milliseconds(10))
        }
        try await waitUntil { @MainActor in outcome != nil }
        #expect(outcome == false)
        #expect(model.persistenceTail?.isCancelled == false)
        await gate.open()
        await waiting.value
    }

    @MainActor
    @Test func deletionFailureReleasesTheSendReservationAndReportsTheCause() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        let store = try #require(model.conversationStore)
        // Simulate a directory disappearing after the sidebar was populated.
        try await store.delete(id: id)
        model.deleteConversation(id: id)
        await model.conversationDeletionTask?.value
        model.promptText = "second"
        #expect(model.canRun)
        #expect(model.historyDiagnostic?.contains(id.uuidString) == true)
    }

    @MainActor
    @Test func deletionWaitingOnAnOldBindingDoesNotMutateEitherStore() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        let folder = try #require(await model.conversationFolderURL(for: id))
        let gate = HistoryListGate()
        model.persistenceTail = Task { await gate.wait() }
        await gate.waitUntilEntered()
        model.deleteConversation(id: id)
        let deletion = try #require(model.conversationDeletionTask)
        await Task.yield()
        let newDirectory = root.appendingPathComponent("other/gemma4.gturbo")
        model.setModelURL(newDirectory)
        await gate.open()
        await deletion.value
        try await model.waitForHistory(count: 0)
        #expect(FileManager.default.fileExists(atPath: folder.path))
        #expect(model.modelPathText == newDirectory.path)
        #expect(model.storedConversationID == nil)
        #expect(model.conversationDeletionTask == nil)
        #expect(model.historyDiagnostic == nil)
    }

    @MainActor
    @Test func cancelledTerminationWaitReturnsWithoutCancellingPersistence() async throws {
        let model = AppModel(client: FakeInferenceClient())
        let gate = HistoryListGate()
        model.persistenceTail = Task { await gate.wait() }
        await gate.waitUntilEntered()
        var outcome: Bool?
        let waiting = Task {
            outcome = await model.awaitPendingPersistence(timeout: .seconds(60))
        }
        await Task.yield()
        waiting.cancel()
        try await waitUntil { @MainActor in outcome != nil }
        #expect(outcome == false)
        #expect(model.persistenceTail?.isCancelled == false)
        await gate.open()
        await waiting.value
    }

    @MainActor
    @Test func terminationWaitReportsCompletedAndAbsentPersistence() async throws {
        let model = AppModel(client: FakeInferenceClient())
        #expect(await model.awaitPendingPersistence(timeout: .seconds(60)))
        let gate = HistoryListGate()
        model.persistenceTail = Task { await gate.wait() }
        await gate.waitUntilEntered()
        let waiting = Task {
            await model.awaitPendingPersistence(timeout: .seconds(60))
        }
        await gate.open()
        #expect(await waiting.value)
    }

    /// Opening restores the saved controls once. Replaying on Send must not
    /// replace the sampling choices made after that open.
    @MainActor
    @Test func replayPreservesSamplingChosenAfterOpeningTheChat() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(20))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.temperature = 0.2
        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await model.screen.document != nil }
        try #require(model.screen.document != nil)
        model.temperature = 0.9
        model.maxNewTokensOverride = 47
        model.topK = 17
        model.topP = 0.75
        model.promptText = "second"
        model.send()
        await SendWaiting.turnEnds(model)
        await model.awaitPendingPersistence()
        let store = try #require(model.conversationStore)
        let meta = try await store.open(id: id).meta
        #expect(client.restoredLineages.count == 1)
        #expect(meta.turnCount == 4)
        #expect(meta.sampling.temperature == 0.9)
        #expect(meta.sampling.maxNewTokens == 47)
        #expect(meta.sampling.topK == 17)
        #expect(meta.sampling.topP == 0.75)
        #expect(model.currentSampling() == meta.sampling)
    }

    @MainActor
    @Test func completedTurnsPersistInKVCommitOrderWithTheirCapturedState() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(20))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        let firstImageGate = HistoryListGate()

        model.temperature = 0.25
        model.maxNewTokensOverride = 31
        model.promptText = "first"
        model.send()
        try await waitUntil { client.isGenerating }
        let firstOutcome = imageOutcome("first")
        model.pendingTurnImageWrite = Task {
            await firstImageGate.wait()
            return firstOutcome
        }
        try await waitUntil { await !model.isTurnInFlight }

        model.temperature = 0.75
        model.maxNewTokensOverride = 47
        model.promptText = "second"
        model.send()
        try await waitUntil { client.isGenerating }
        let secondOutcome = imageOutcome("second")
        model.pendingTurnImageWrite = Task { secondOutcome }
        try await waitUntil { await !model.isTurnInFlight }
        #expect(model.persistenceTail != nil)

        await firstImageGate.open()
        await model.persistenceTail?.value
        let store = try #require(model.conversationStore)
        let id = try #require(model.storedConversationID)
        let turns = try await store.open(id: id).records.compactMap { record in
            if case .turn(let turn) = record { return turn }
            return nil
        }

        #expect(turns.map(\.text).filter { $0 == "first" || $0 == "second" }
            == ["first", "second"])
        #expect(turns[0].images.map(\.displayName) == ["first"])
        #expect(turns[2].images.map(\.displayName) == ["second"])
        #expect(turns[0].sampling?.temperature == 0.25)
        #expect(turns[0].sampling?.maxNewTokens == 31)
        #expect(turns[2].sampling?.temperature == 0.75)
        #expect(turns[2].sampling?.maxNewTokens == 47)
        #expect(turns[1].boundary != nil && turns[3].boundary != nil)
    }

    @MainActor
    @Test func rootSwitchCancelsAQueuedTurnBeforeItCanAppendOrReport() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(20))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        let gate = HistoryListGate()

        model.promptText = "old root"
        model.send()
        try await waitUntil { client.isGenerating }
        let outcome = imageOutcome("old-root")
        model.pendingTurnImageWrite = Task {
            await gate.wait()
            return outcome
        }
        try await waitUntil { await !model.isTurnInFlight }
        let oldTail = try #require(model.persistenceTail)
        let oldStore = try #require(model.conversationStore)
        let oldID = try #require(model.storedConversationID)

        let newDirectory = root.appendingPathComponent("other/model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: newDirectory, withIntermediateDirectories: true)
        model.setModelURL(newDirectory)
        await gate.open()
        await oldTail.value

        let oldTurns = try await oldStore.open(id: oldID).records.filter {
            if case .turn = $0 { return true }
            return false
        }
        #expect(oldTurns.isEmpty)
        #expect(model.conversationBinding?.modelDirectory == newDirectory.standardizedFileURL)
        #expect(model.historyDiagnostic == nil)
    }

    @MainActor
    @Test func rootSwitchCancelsAnOlderWriteAtTheBatchBoundary() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-batch-cancel-\(UUID())", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let gate = HistoryListGate()
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true,
            conversationStoreProvider: { root in
                ConversationStore(
                    rootURL: root,
                    appendOperations: ConversationStoreAppendOperations(beforeBatch: {
                        await gate.wait()
                    }))
            })
        model.conversationIdentity = Self.identity
        model.installationStatus = .complete
        try await client.ensureLoaded(
            modelDirectory: directory, maxContextTokens: model.maxContextTokens,
            options: model.runtimeOptions, forceLogitsHead: true) { _ in }
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        try await model.waitForHistory()

        model.promptText = "old root"
        model.send()
        try await waitUntil { await !model.isTurnInFlight }
        await gate.waitUntilEntered()
        let oldTail = try #require(model.persistenceTail)
        let oldStore = try #require(model.conversationStore)
        let oldID = try #require(model.storedConversationID)

        let newDirectory = root.appendingPathComponent("other/model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: newDirectory, withIntermediateDirectories: true)
        model.setModelURL(newDirectory)
        await gate.open()
        await oldTail.value

        let oldTurns = try await oldStore.open(id: oldID).records.filter {
            if case .turn = $0 { return true }
            return false
        }
        #expect(oldTurns.isEmpty,
                "cancellation after entering append must stop the old-root batch")
        #expect(model.conversationBinding?.modelDirectory == newDirectory.standardizedFileURL)
        #expect(model.historyDiagnostic == nil)
    }

    @MainActor
    @Test func appendUncertaintyQuarantinesTheLiveLineageUntilNewChat() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-quarantine-\(UUID())", isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let failuresRemaining = Mutex(1)
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true,
            conversationStoreProvider: { root in
                ConversationStore(
                    rootURL: root,
                    appendOperations: ConversationStoreAppendOperations(beforeBatch: {
                        let mustFail = failuresRemaining.withLock { remaining -> Bool in
                            guard remaining > 0 else { return false }
                            remaining -= 1
                            return true
                        }
                        if mustFail {
                            throw ConversationStoreError.writeFailed(
                                path: "transcript.jsonl", reason: "injected full volume")
                        }
                    }))
            })
        model.conversationIdentity = Self.identity
        model.installationStatus = .complete
        try await client.ensureLoaded(
            modelDirectory: directory, maxContextTokens: model.maxContextTokens,
            options: model.runtimeOptions, forceLogitsHead: true) { _ in }
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        try await model.waitForHistory()

        model.promptText = "first"
        model.send()
        await SendWaiting.turnEnds(model)
        await model.persistenceTail?.value
        let store = try #require(model.conversationStore)
        let afterFailure = try await store.list()
        #expect(afterFailure.count == 1)
        #expect(model.storedConversationID == nil)
        #expect(model.historyDiagnostic?.contains("injected full volume") == true)
        #expect(model.error?.userMessage.contains("not be saved") == true,
                "a completed answer whose history write failed must show a visible warning")
        #expect(model.error?.technicalDetail.contains("injected full volume") == true)

        model.promptText = "same uncertain lineage"
        model.send()
        await SendWaiting.turnEnds(model)
        await model.persistenceTail?.value
        #expect(try await store.list().count == 1,
                "a replacement file was created for the uncertain live KV")

        model.newChat()
        model.promptText = "new lineage"
        model.send()
        try await finish(model)
        #expect(try await store.list().count == 2)
    }

    @MainActor
    @Test func lateListFromTheOldModelRootCannotReplaceTheNewBinding() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-root-switch-\(UUID())", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let directoryA = root.appendingPathComponent("a/model.gturbo", isDirectory: true)
        let directoryB = root.appendingPathComponent("b/model.gturbo", isDirectory: true)
        try placeCompleteModel(at: directoryA, tag: "binding-a")
        try placeCompleteModel(at: directoryB, tag: "binding-b")
        let identityA = Self.identity
        let identityB = ConversationIdentity(
            modelID: "model-b", sourceSnapshotHash: "snapshot-b",
            templateIdentity: GFTokenizer.chatTemplateIdentity,
            imageProcessingVersion: VisionImageProcessing.version)
        let gate = HistoryListGate()
        let storeRootA = ConversationStoreLocation.url(forModelDirectory: directoryA)
        let storeA = ConversationStore(
            rootURL: storeRootA,
            listOperations: ConversationStoreListOperations(beforeList: {
                await gate.wait()
            }))
        let storeB = ConversationStore(
            rootURL: ConversationStoreLocation.url(forModelDirectory: directoryB))
        try await storeA.activate()
        try await storeB.activate()
        let metaA = try await storeA.create(
            title: "root a", identity: identityA,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "onDemand"),
            sampling: ConversationSampling(
                temperature: 0, topKEnabled: true, topK: 64,
                topPEnabled: true, topP: 0.95, maxNewTokens: 64))
        let metaB = try await storeB.create(
            title: "root b", identity: identityB,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "onDemand"),
            sampling: ConversationSampling(
                temperature: 0, topKEnabled: true, topK: 64,
                topPEnabled: true, topP: 0.95, maxNewTokens: 64))

        let model = AppModel(
            modelDirectory: directoryA,
            client: FakeInferenceClient(),
            settingsPersistenceEnabled: true,
            conversationIdentityProvider: { directory in
                directory.standardizedFileURL == directoryA.standardizedFileURL
                    ? identityA : identityB
            },
            conversationStoreProvider: { root in
                root.standardizedFileURL == storeRootA.standardizedFileURL
                    ? storeA : storeB
            })
        await gate.waitUntilEntered()
        model.setModelURL(directoryB)
        try await model.waitForHistory(count: 1)
        await gate.open()
        try await Task.sleep(for: .milliseconds(30))

        #expect(model.conversationBinding?.modelDirectory == directoryB.standardizedFileURL)
        #expect(model.conversationIdentity == identityB)
        #expect(model.history.entries.map(\.id) == [metaB.id])
        #expect(model.history.entry(metaA.id) == nil)
    }

    @MainActor
    @Test func successfulLegacyTrashCleanupReachesHistoryState() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-trash-\(UUID())", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("model.gturbo", isDirectory: true)
        let trash = ConversationStoreLocation.url(forModelDirectory: directory)
            .appendingPathComponent(".trash", isDirectory: true)
        try FileManager.default.createDirectory(
            at: trash.appendingPathComponent("one", isDirectory: true),
            withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: trash.appendingPathComponent("two", isDirectory: true),
            withIntermediateDirectories: true)

        let model = AppModel(
            modelDirectory: directory,
            client: FakeInferenceClient(),
            settingsPersistenceEnabled: true)
        try await waitUntil { await model.history.discardedLegacyTrashCount == 2 }

        #expect(!FileManager.default.fileExists(atPath: trash.path))
        #expect(model.historyDiagnostic?.contains("removed 2 conversation(s)") == true)
    }

    @MainActor
    @Test func unreadableInstalledIdentityKeepsHistoryReadableButDisablesReplayAndWrites() async throws {
        let fixture = try makeCompleteModelInstall("identity-failure")
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("identity-failure-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: fixture)
            try? FileManager.default.removeItem(at: root)
        }
        let directory = root.appendingPathComponent("model.gturbo", isDirectory: true)
        try FileManager.default.moveItem(at: fixture, to: directory)
        let receiptURL = directory.appendingPathComponent("verified-install.json")
        var receipt = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as? [String: Any])
        receipt["modelDirectoryPath"] = directory.standardizedFileURL.path
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
            .write(to: receiptURL)

        let store = ConversationStore(
            rootURL: ConversationStoreLocation.url(forModelDirectory: directory))
        try await store.activate()
        let meta = try await store.create(
            title: "readable",
            identity: Self.identity,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "onDemand"),
            sampling: ConversationSampling(
                temperature: 0, topKEnabled: true, topK: 64,
                topPEnabled: true, topP: 0.95, maxNewTokens: 64))
        _ = try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "stored", tokens: [1])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "answer", tokens: [2],
                boundary: ConversationBoundary(tokens: [3], needsReplay: false))),
        ], to: meta.id)

        let model = AppModel(
            modelDirectory: directory,
            client: FakeInferenceClient(),
            settingsPersistenceEnabled: true,
            conversationIdentityProvider: { _ in
                throw ConversationIdentityError.snapshotHashUnavailable
            },
            conversationStoreProvider: { _ in store })
        try await model.waitForHistory(count: 1)
        model.openConversation(id: meta.id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        #expect(model.conversationIdentity == nil)
        #expect(model.openedConversationState == .cannotReplay(reason: .differentModel))
        #expect(model.transcriptHistory.map(\.user.text) == ["stored"])
        #expect(model.historyDiagnostic?.contains("source snapshot hash") == true)
        #expect(await model.ensureStoredConversation(firstMessage: "private prompt") == nil)
        #expect(model.historyDiagnostic?.contains("private prompt") == false)
    }

    /// The record has to be the token IDs, not the text: re-rendering the text
    /// through the template strips historical thought spans and produces a
    /// different sequence.
    @MainActor
    @Test func acompletedTurnIsOnDiskAsTheIDsTheKVTook() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)

        let id = try #require(model.history.selection)
        let store = try #require(model.conversationStore)
        let opened = try await store.open(id: id)
        let turns = opened.records.compactMap { record -> ConversationTurnRecord? in
            guard case .turn(let turn) = record else { return nil }
            return turn
        }
        #expect(turns.count == 2)
        #expect(turns[0].role == .user)
        #expect(turns[0].text == "hello")
        #expect(turns[0].tokens?.isEmpty == false)
        #expect(turns[1].role == .assistant)
        #expect(turns[1].tokens?.isEmpty == false)
        // The transcript on disk reproduces exactly the count the KV reported.
        #expect(opened.meta.kvTokens
            == turns.compactMap(\.tokens).flatMap { $0 }.count)
        #expect(opened.meta.turnCount == 2)
        #expect(opened.meta.title == "hello")
        #expect(model.history.entries.count == 1)
    }

    @MainActor
    @Test func deleteIsRefusedWhileATurnIsInFlight() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(50))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)

        model.promptText = "kept"
        model.send()
        await SendWaiting.generationStarts(model)
        let directory = await (try #require(model.conversationStore)).directoryURL(for: id)

        model.deleteConversation(id: id)
        try await Task.sleep(for: .milliseconds(20))

        #expect(FileManager.default.fileExists(atPath: directory.path))
        #expect(model.history.entry(id) != nil)
        await SendWaiting.turnEnds(model)
    }

    @MainActor
    @Test func deleteAndRenameAreRefusedDuringReplayThenAllowedAtRest() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(80))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        let store = try #require(model.conversationStore)
        let directory = await store.directoryURL(for: id)
        let originalTitle = try await store.open(id: id).meta.title

        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await model.screen.document != nil }
        model.promptText = "after replay"
        model.send()
        try await waitUntil { await model.screen.isReplaying }
        #expect(!model.canMutateConversation(id: id))
        model.renameConversation(id: id, to: "must not land")
        model.deleteConversation(id: id)
        try await Task.sleep(for: .milliseconds(20))
        #expect(FileManager.default.fileExists(atPath: directory.path))
        #expect(try await store.open(id: id).meta.title == originalTitle)

        await SendWaiting.turnEnds(model)
        await model.persistenceTail?.value
        #expect(model.canMutateConversation(id: id))
        model.deleteConversation(id: id)
        try await model.waitForHistory(count: 0)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    @MainActor
    @Test func connectionLossRequiresReloadThenReplaysTheDurableRow() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "durable"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        let store = try #require(model.conversationStore)

        client.failNextGeneration(with: .connectionLost("unexpected EOF"))
        model.promptText = "retry me"
        model.send()
        await SendWaiting.turnEnds(model)

        guard case .failed(.connectionLost(let cause)) = model.loadState else {
            Issue.record("connection loss did not invalidate Ready")
            return
        }
        #expect(cause.contains("unexpected EOF"))
        #expect(model.serviceEpoch == nil)
        #expect(model.promptText == "retry me")
        #expect(model.history.selection == id)
        #expect(model.openedConversationState == .continuable)
        #expect(try await store.open(id: id).meta.turnCount == 2,
                "the incomplete turn reached disk")

        model.loadModel()
        try await waitUntil { await model.loadState.isReady }
        model.send()
        await SendWaiting.turnEnds(model)
        await model.persistenceTail?.value

        #expect(client.restoredLineages.count == 1)
        #expect(model.storedConversationID == id)
        #expect(try await store.open(id: id).meta.turnCount == 4)
    }

    @MainActor
    @Test func relaunchRestoresTheSelectedTranscriptWhileSidebarStaysHidden() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-selection-\(UUID())", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)

        let firstClient = FakeInferenceClient(eventDelay: .milliseconds(1))
        var first: AppModel? = AppModel(
            modelDirectory: directory,
            client: firstClient,
            settingsPersistenceEnabled: true)
        first?.conversationIdentity = Self.identity
        first?.installationStatus = .complete
        first?.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        try await firstClient.ensureLoaded(
            modelDirectory: directory, maxContextTokens: 8_192,
            options: AppRuntimeOptions(), forceLogitsHead: true) { _ in }
        try await first?.waitForHistory()
        first?.promptText = "remember this"
        first?.send()
        if let first { await SendWaiting.turnEnds(first) }
        let selected = try #require(first?.history.selection)
        first?.setSidebarVisible(false)
        first = nil
        try await Task.sleep(for: .milliseconds(20))

        let second = AppModel(
            modelDirectory: directory,
            client: FakeInferenceClient(eventDelay: .milliseconds(1)),
            settingsPersistenceEnabled: true)
        second.conversationIdentity = Self.identity
        second.installationStatus = .complete
        try await second.waitForHistory(count: 1)
        try await waitUntil { await second.screen.document != nil }

        #expect(!second.isSidebarVisible)
        #expect(second.history.selection == selected)
        #expect(second.transcriptHistory.map(\.user.text) == ["remember this"])
    }

    /// A turn the runtime rewound is in neither the KV nor the transcript. A
    /// stored copy of it would be a message the model never saw, replayed into
    /// every future reopen.
    @MainActor
    @Test func athrownTurnIsWrittenNowhere() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let store = try #require(model.conversationStore)
        let before = try await store.open(id: id).records.count

        // A load key the client no longer recognises makes the next generate
        // throw before it reaches the KV.
        await client.unload()
        model.promptText = "second"
        model.send()
        await SendWaiting.turnEnds(model)
        try await Task.sleep(for: .milliseconds(50))

        #expect(try await store.open(id: id).records.count == before)
        #expect(model.error != nil)
    }

    /// Reopening must send the stored IDs. If it ever sent re-rendered text the
    /// conversation would continue as a different lineage while claiming to be
    /// the same one. The replay itself waits for the next message.
    @MainActor
    @Test func openThenSendReplaysTheStoredIDsRatherThanTheText() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let first = try #require(model.history.selection)
        let store = try #require(model.conversationStore)
        let stored = try await store.open(id: first)
        let expected = stored.records.compactMap { record -> [Int32]? in
            guard case .turn(let turn) = record else { return nil }
            return turn.tokens
        }.flatMap { $0 }

        model.newChat()
        model.promptText = "second"
        model.send()
        try await finish(model)
        #expect(model.history.entries.count == 2)

        model.openConversation(id: first)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        // Opening costs no replay at all.
        #expect(client.restoredLineages.isEmpty)
        #expect(model.screen.conversationID == first)

        model.promptText = "third"
        model.send()
        try await waitUntil { client.restoredLineages.count == 1 }

        let replayed = try #require(client.restoredLineages.first)
        #expect(replayed.tokenIDs == expected)
        #expect(replayed.committedTurns == 1)
        #expect(model.storedConversationID == first)
        #expect(!model.isShowingStoredCopy)
    }

    /// Opening a chat is a file read, so every exchange has to be on screen
    /// straight away — including the newest one, which the live fields draw for
    /// a conversation the KV is holding and which therefore has to come from
    /// somewhere else for one it is not.
    @MainActor
    @Test func areopenedChatShowsEveryExchangeWithoutReplaying() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first question"
        model.send()
        try await finish(model)
        model.promptText = "second question"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)

        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        #expect(client.restoredLineages.isEmpty, "opening paid for a replay")
        #expect(model.transcriptHistory.map(\.user.text)
            == ["first question", "second question"])
        #expect(model.hasOutputTranscript)
        // No context break: these turns are not out of the model's context,
        // they are simply not in it yet.
        #expect(model.transcriptContextBreak == nil)
    }

    /// The replay does not overwrite the message that started it.
    ///
    /// `adoptRestoredConversation` seeded the live fields with the restored
    /// conversation's newest pair - correct when a reopen replayed on the click
    /// and the window then sat at rest showing it, wrong now that a replay only
    /// runs inside a send. It replaced the user's prompt with an older one and
    /// cleared the attachments beside it, and since those attachments are the
    /// staged files the send is about to hard-link, clearing them deleted them:
    /// the turn came back "could not retain <name>: errno 2". Found by hand,
    /// sending a second picture into a reopened chat.
    @MainActor
    @Test func areplayDoesNotOverwriteTheMessageThatStartedIt() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(20))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "the first question"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)

        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        model.promptText = "the second question"
        model.send()
        try await waitUntil { await model.screen.isReplaying }
        try await waitUntil { await !model.screen.isReplaying }

        // Not "the first question", which is what adoption used to put here.
        #expect(model.outputPromptText == "the second question")
        try await finish(model)
        #expect(model.error == nil)
        #expect(model.conversation.turns.count == 4)
        // The restored exchange is drawn from the history, and the new turn
        // from the live fields, which is the split `transcriptHistory` states.
        #expect(model.transcriptHistory.map(\.user.text) == ["the first question"])
        #expect(model.outputPromptText == "the second question")
    }

    /// New Chat takes the composer's attachments with it.
    ///
    /// It released the conversation's images and the transcript's, and left the
    /// composer's alone — so a picture attached but never sent stayed in the box
    /// and came along into the new chat, ready to be sent with a message it had
    /// nothing to do with. Its staged copy also stayed on disk until the app
    /// quit. Found running case E41 by hand.
    @MainActor
    @Test func newChatReleasesAnAttachmentThatWasNeverSent() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let staging = FileManager.default.temporaryDirectory
            .appendingPathComponent("staging-\(UUID().uuidString)", isDirectory: true)
        let store = AppImageAttachmentStore(directoryURL: staging)
        defer { try? FileManager.default.removeItem(at: staging) }
        let (model, root) = try await readyModel(client, attachmentStore: store)
        defer { try? FileManager.default.removeItem(at: root) }

        // What the composer holds after a pick: a staged file this store owns.
        let source = try Self.pngFixture()
        defer { try? FileManager.default.removeItem(at: source) }
        let staged = try store.stage(source)
        model.setComposerAttachmentsForTesting([staged])
        #expect(FileManager.default.fileExists(atPath: staged.fileURL.path))

        model.newChat()

        #expect(model.imageAttachments.isEmpty,
                "the picture followed the user into the new chat")
        #expect(!FileManager.default.fileExists(atPath: staged.fileURL.path),
                "its staged copy was left on disk")
    }

    /// Re-read is not offered when there is nothing to re-read.
    ///
    /// The prompt is built from the turns on screen, so a conversation that
    /// could not be read at all has none. Offered anyway, the button started a
    /// new chat with an empty composer, which reads as the action having done
    /// nothing.
    @MainActor
    @Test func rereadIsRefusedWhenThereIsNothingToReread() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let folder = try #require(await model.conversationFolderURL(for: id))
        model.newChat()

        // Unreadable, so nothing is drawn from it.
        try FileManager.default.removeItem(
            at: folder.appendingPathComponent("transcript.jsonl"))
        model.openConversation(id: id)
        try await waitUntil { await model.error != nil }

        #expect(model.transcriptHistory.isEmpty)
        #expect(!model.canRereadIntoNewChat, "a re-read with nothing to read")
        model.promptText = ""
        model.rereadIntoNewChat()
        #expect(model.promptText.isEmpty, "it started a chat from nothing")
    }

    /// A rewound turn's images are collected.
    ///
    /// Images are written while the reply is still generating, so a turn the
    /// runtime rewinds leaves them on disk, cited by nothing and named by
    /// digest — invisible to every other pass. `sweepOrphanImages` existed for
    /// exactly this and had no caller anywhere in the app.
    @MainActor
    @Test func arewoundTurnsImagesAreCollected() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let folder = try #require(await model.conversationFolderURL(for: id))

        // What a rewound turn leaves: image files no surviving record names.
        let images = folder.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(
            at: images, withIntermediateDirectories: true)
        let orphan = images.appendingPathComponent("deadbeef.png")
        let orphanThumb = images.appendingPathComponent("deadbeef.thumb.jpg")
        try Data("pixels".utf8).write(to: orphan)
        try Data("thumb".utf8).write(to: orphanThumb)

        // A turn that throws before it reaches the KV.
        await client.unload()
        model.promptText = "second"
        model.send()
        await SendWaiting.turnEnds(model)
        try await waitUntil {
            !FileManager.default.fileExists(atPath: orphan.path)
        }

        #expect(!FileManager.default.fileExists(atPath: orphan.path))
        #expect(!FileManager.default.fileExists(atPath: orphanThumb.path))
        // And the conversation itself is untouched: header, creation title,
        // and the one completed turn's two halves.
        let store = try #require(model.conversationStore)
        #expect(try await store.open(id: id).records.count == 4)
    }

    /// A stored conversation's pictures survive being looked at.
    ///
    /// `AppImageAttachmentStore.remove` deletes whatever a `fileURL` points at,
    /// and every path that finishes with a turn releases that turn's images
    /// through it. Once reopened turns carried attachments pointing into the
    /// conversation store, opening any other chat deleted the pictures out of
    /// the one just left, off disk. The record still named them, so the turn
    /// came back looking as though it had never had an image.
    ///
    /// Half of this is the compiler's now — a `StoredImage` cannot be handed to
    /// `remove` at all — and the half it cannot check is that the release paths
    /// walk a conversation whose pictures are stored ones and delete nothing.
    @MainActor
    @Test func openingAnotherChatDoesNotDeleteAStoredChatsImages() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let folder = try #require(await model.conversationFolderURL(for: id))

        // A stored image, written the way the store writes one, put where the
        // window holds a conversation's pictures: in a turn, in the turns a
        // released KV left on screen, and in the live fields.
        let images = folder.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(
            at: images, withIntermediateDirectories: true)
        let thumbnail = images.appendingPathComponent("abcd.thumb.jpg")
        try Data("thumbnail".utf8).write(to: thumbnail)
        let stored = ChatImage.stored(StoredImage(
            fileURL: thumbnail, displayName: "roof.heic",
            encodedBytes: 9, sha256: "abcd"))
        model.conversation.adoptRestored(
            epoch: UUID(),
            turns: [AppChatTurn(role: .user, text: "look", images: [stored]),
                    AppChatTurn(role: .assistant, text: "a roof")],
            kvTokens: 2, boundaryTokenIDs: [], boundaryNeedsReplay: false)
        model.outputImageAttachments = [stored]

        // Every release path a turn goes through.
        model.newChat()
        model.releaseAllAttachments()

        #expect(FileManager.default.fileExists(atPath: thumbnail.path),
                "a stored image was deleted by a release path")
    }

    /// A staged file is still released, or every session leaks its copies.
    @MainActor
    @Test func astagedAttachmentIsStillRemoved() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        // The model is built only so the store root exists to clean up.
        let (_, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        let store = AppImageAttachmentStore()
        let staged = try store.stage(Self.pngFixture())
        #expect(FileManager.default.fileExists(atPath: staged.fileURL.path))
        store.remove(staged)
        #expect(!FileManager.default.fileExists(atPath: staged.fileURL.path))
    }

    /// Raising the context redraws the chat that was refused for it.
    ///
    /// Continuability is a function of the conversation's size and the context
    /// in force, and the window took its answer once, when the row was clicked.
    /// So raising the context from the notice left the chat continuable but
    /// still drawn read-only, under a rule saying earlier turns were out of
    /// context when they no longer were.
    @MainActor
    @Test func raisingTheContextRedrawsAConversationThatNoLongerNeedsIt() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)

        // Recorded as bigger than a 4K context can hold. Given the tokens
        // rather than told a number: the meta is a projection of the records,
        // so a count written into it by hand would be gone at the next open.
        let store = try #require(model.conversationStore)
        try await Self.padStoredTokens(store, id: id, to: 7_000)
        await model.refreshHistory()

        model.newChat()
        model.setMaxContextTokens(4_096)
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        // Drawn read-only from disk, and drawn whole.
        #expect(model.isShowingStoredCopy)
        #expect(model.transcriptHistory.count == model.screen.document?.pairs.count)
        // And with no context break under it. The break says "the model can no
        // longer see the turns above this", which is a fact about the live
        // conversation; a stored copy is not in the context at all, so the rule
        // was drawn under every turn with nothing below it. This line asserted
        // that older behaviour.
        #expect(model.transcriptContextBreak == nil)
        guard case .needsContext = model.openedConversationState else {
            Issue.record("the chat was not refused for context")
            return
        }
        let refusedRenderID = model.displayedTranscriptID

        model.setMaxContextTokens(8_192)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        #expect(model.openedConversationState == .continuable)
        #expect(model.transcriptContextBreak == nil,
                "the boundary outlived the reason for it")
        #expect(model.conversation.outOfContextPairs.isEmpty)
        // And the renderer is told, or it keeps drawing the rule it already
        // appended: it can add to a transcript but not take anything back.
        #expect(model.displayedTranscriptID != refusedRenderID)
    }

    /// A replay that fails for a reason outside the conversation leaves the row
    /// continuable.
    ///
    /// Every restore failure used to mark the chat "It has no replay record, so
    /// there is nothing to put back into the model's context", and offer to
    /// re-read it into a new chat. For a conversation with images and the
    /// companion pack uninstalled that is simply false: the record is intact,
    /// the pack is missing, and the fix is to put the pack back. Found running
    /// case E46 with the vision pack moved aside.
    @MainActor
    @Test func afailedRestoreDoesNotBlameTheRecord() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)

        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        // The kind of failure that is about the runtime, not the record.
        client.failNextRestore(
            with: .conversationRestoreFailed("image support is unavailable"))
        model.promptText = "and again"
        model.send()
        try await waitUntil { await model.error != nil }

        #expect(model.error != nil)
        // Not `.cannotReplay`: nothing about the conversation changed, and the
        // notice that follows from it would tell the user to start again.
        #expect(model.openedConversationState == .continuable)
    }

    /// A record that genuinely cannot be replayed still says so.
    @MainActor
    @Test func aturnWithNoRecordedTokensIsReportedAsUnreplayable() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let folder = try #require(await model.conversationFolderURL(for: id))

        // Strip the tokens from the stored turns: the record cannot be put back
        // into a KV, whatever the runtime is doing.
        let transcript = folder.appendingPathComponent("transcript.jsonl")
        let lines = try String(contentsOf: transcript, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { line -> String in
                guard line.contains("\"tokens\"") else { return String(line) }
                guard let data = line.data(using: .utf8),
                      var object = try? JSONSerialization.jsonObject(
                        with: data) as? [String: Any],
                      var turn = object["turn"] as? [String: Any] else {
                    return String(line)
                }
                turn.removeValue(forKey: "tokens")
                object["turn"] = turn
                let stripped = try! JSONSerialization.data(withJSONObject: object)
                return String(data: stripped, encoding: .utf8)!
            }
        try (lines.joined(separator: "\n") + "\n").write(
            to: transcript, atomically: true, encoding: .utf8)

        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        model.promptText = "and again"
        model.send()
        try await waitUntil { await model.error != nil }

        #expect(model.openedConversationState
            == .cannotReplay(reason: .tokenCountUnknown))
        #expect(client.restoredLineages.isEmpty, "a broken record reached the runtime")
    }

    /// A conversation that cannot be read says so.
    ///
    /// The loader used to be `try?`, so a torn or missing transcript left the
    /// row selected over the ordinary empty state — which is exactly what an
    /// unwritten chat looks like. The user could not tell "your conversation is
    /// damaged" from "there is nothing here". Found by hand against a truncated
    /// `transcript.jsonl`.
    @MainActor
    @Test func atornTranscriptIsReportedRatherThanShownAsEmpty() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let folder = try #require(await model.conversationFolderURL(for: id))
        model.newChat()

        // A final record that will not decode. The store drops a torn final
        // *line* on purpose; this is the case it throws on.
        let transcript = folder.appendingPathComponent("transcript.jsonl")
        let data = try String(contentsOf: transcript, encoding: .utf8)
        try (String(data.dropLast(40)) + "\n").write(
            to: transcript, atomically: true, encoding: .utf8)

        model.openConversation(id: id)
        try await waitUntil { await model.error != nil }

        #expect(model.error != nil, "the read failure was swallowed")
        #expect(model.screen.document == nil)
        #expect(!model.hasOutputTranscript, "an unreadable chat drew a transcript")
        // And it is not left waiting to be replayed, which would fail the same
        // way on the next send.
        #expect(!model.isShowingStoredCopy)
    }

    /// A conversation the list cannot read is counted, not just skipped.
    ///
    /// Skipping is right — one bad directory must not hide the rest — but on
    /// its own a chat vanishes with nothing anywhere to say it was ever there.
    ///
    /// The transcript is what is unreadable here. A `conversation.json` that
    /// will not decode is no longer enough to lose a chat: it is a cache of the
    /// records, and the next listing rebuilds it.
    @MainActor
    @Test func aconversationWithAnUnreadableTranscriptIsCountedNotJustDropped()
        async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let folder = try #require(await model.conversationFolderURL(for: id))
        #expect(model.history.entries.count == 1)
        #expect(model.history.unreadableCount == 0)

        // Damage in the middle of the transcript, which is the case that is
        // genuinely unrecoverable: two writers, or corruption in place.
        let transcript = folder.appendingPathComponent("transcript.jsonl")
        var text = try String(contentsOf: transcript, encoding: .utf8)
        text = text.replacingOccurrences(of: "\"type\":\"header\"", with: "\"typ")
        try text.write(to: transcript, atomically: true, encoding: .utf8)
        try "{ not json".write(
            to: folder.appendingPathComponent("conversation.json"),
            atomically: true, encoding: .utf8)
        await model.refreshHistory()

        #expect(model.history.entries.isEmpty, "an unreadable chat was listed")
        #expect(model.history.unreadableCount == 1, "it vanished with no trace")
        #expect(model.historyDiagnostic?.contains(id.uuidString) == true)
    }

    /// A `conversation.json` that will not decode is a cache miss, not a lost
    /// conversation.
    ///
    /// Before the meta became a projection this cost the user the whole chat:
    /// the listing skipped it and nothing ever put it back. The records are the
    /// conversation, so the row comes back with everything it had.
    @MainActor
    @Test func adamagedMetaIsRebuiltFromTheRecordsRatherThanLosingTheChat()
        async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let before = try #require(model.history.entry(id))
        let folder = try #require(await model.conversationFolderURL(for: id))
        let metaURL = folder.appendingPathComponent("conversation.json")

        try "{ not json".write(to: metaURL, atomically: true, encoding: .utf8)
        await model.refreshHistory()

        #expect(model.history.unreadableCount == 0)
        #expect(model.history.entry(id) == before, "the rebuilt row is a different chat")

        // And a meta that is simply gone is the same case.
        try FileManager.default.removeItem(at: metaURL)
        await model.refreshHistory()
        #expect(model.history.entry(id) == before)
    }

    /// A title edited straight into `conversation.json` does not survive.
    ///
    /// Deliberate, and the one behaviour the projection takes away: the file is
    /// a cache of the records, so the only way to rename a conversation from
    /// outside the app is to append a `title` line to its transcript. Manual
    /// case F58 asserts exactly this.
    @MainActor
    @Test func ahandEditToTheStoredTitleIsReplacedAtTheNextOpen() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let store = try #require(model.conversationStore)
        let folder = try #require(await model.conversationFolderURL(for: id))
        let metaURL = folder.appendingPathComponent("conversation.json")

        var edited = try #require(model.history.entry(id))
        edited.title = "typed straight into the file"
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(edited).write(to: metaURL)

        #expect(try await store.open(id: id).meta.title == "hello")
        // Appending the record is the supported way, and it holds.
        try await store.rename(id: id, to: "renamed properly")
        #expect(try await store.open(id: id).meta.title == "renamed properly")
    }

    /// The row's Open Folder in Finder needs a folder that is actually there.
    ///
    /// A row can outlive its directory — something outside the app can move or
    /// delete it — and handing Finder a path that does not exist opens nothing
    /// and says nothing. The absence is recorded instead.
    @MainActor
    @Test func thefolderOfAStoredConversationIsFoundAndAMissingOneIsReported()
        async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)

        let folder = try #require(await model.conversationFolderURL(for: id))
        #expect(folder.lastPathComponent == id.uuidString)
        // The folder itself, holding the three things worth looking at.
        var isDirectory: ObjCBool = false
        #expect(FileManager.default.fileExists(
            atPath: folder.path, isDirectory: &isDirectory))
        #expect(isDirectory.boolValue)
        #expect(FileManager.default.fileExists(
            atPath: folder.appendingPathComponent("transcript.jsonl").path))
        #expect(FileManager.default.fileExists(
            atPath: folder.appendingPathComponent("conversation.json").path))

        model.historyDiagnostic = nil
        try FileManager.default.removeItem(at: folder)
        #expect(await model.conversationFolderURL(for: id) == nil)
        #expect(model.historyDiagnostic != nil, "the absence was swallowed")
    }

    /// One Generate is one replay, however many times it is pressed.
    ///
    /// A deferred replay is not `isRunning`, so nothing refused a second send
    /// while one was in flight — and the window gave no sign of the first, so
    /// pressing again is what a user does. Both replays restored the same
    /// conversation under a fresh epoch each, and the service ended up holding
    /// the second one while the first one's turn had already been stamped with
    /// the first: "turn belongs to conversation A, and the open conversation is
    /// B", with the message lost.
    @MainActor
    @Test func asecondSendDuringAReplayIsRefusedRatherThanStartingAnother() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(120))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)

        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        model.promptText = "second"
        model.send()
        // Synchronously, on the same run loop turn as the click: the guard has
        // to be up before the task that does the replay exists. `send()` takes
        // the composer and claims the pipeline before it returns, which is what
        // closes the window a second Generate used to fit through.
        #expect(model.isTurnInFlight)
        #expect(!model.canRun, "a second Generate was still offered")

        model.send()
        model.send()
        try await waitUntil { await model.screen.isReplaying }
        model.send()
        try await waitUntil { await !model.screen.isReplaying }
        try await finish(model)

        #expect(client.restoredLineages.count == 1, "the replay ran more than once")
        #expect(model.error == nil)
    }

    /// The message is on screen for the whole replay.
    ///
    /// The composer empties the moment it is sent, and a reopened conversation
    /// is then prefilled in full before its turn starts — tens of seconds at 8K.
    /// Keyed off `isRunning` the window drew nothing for all of it, so the app
    /// looked like it had swallowed the message.
    @MainActor
    @Test func thepromptIsShownWhileTheReplayRuns() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(120))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)

        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        model.promptText = "does this show up?"
        model.send()
        // And out of the composer on the same run loop turn as the click, or a
        // second Generate fits through underneath the first.
        #expect(model.promptText.isEmpty)
        #expect(model.isTurnInFlight)
        try await waitUntil { await model.screen.isReplaying }
        #expect(model.outputPromptText == "does this show up?")
        #expect(model.isTranscriptTurnInFlight, "the transcript would draw this as finished")
        // And the transcript is told to draw it. Gated on `isShowingStoredCopy`
        // alone this was false for the whole replay, so the window showed an
        // "Answer / Processing your prompt" with no question above it.
        #expect(model.showsLiveTurn, "the sent message would not be drawn")
        #expect(model.hasOutputTranscript)
        // And out of the composer at the same moment, or it is on screen twice.
        #expect(model.promptText.isEmpty)

        try await waitUntil { await !model.screen.isReplaying }
        try await finish(model)
        #expect(model.outputPromptText == "does this show up?")
        #expect(model.promptText.isEmpty)
    }

    /// A replay that fails takes the message back out of the transcript.
    ///
    /// The composer still holds it, so leaving a copy in the transcript would
    /// show the same message twice and imply a turn that never happened.
    @MainActor
    @Test func afailedReplayLeavesNoHalfTurnOnScreen() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)

        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        client.failNextRestore(with: .conversationRestoreFailed("digest mismatch"))
        model.promptText = "second"
        model.send()
        try await waitUntil { await !model.screen.isReplaying }

        #expect(model.error != nil)
        #expect(model.outputPromptText.isEmpty)
        #expect(model.outputImageAttachments.isEmpty)
        // The composer kept it, so the send can be retried.
        #expect(model.promptText == "second")
        #expect(model.canRun)
        #expect(model.storedConversationID == nil)
        #expect(model.serviceEpoch == nil)

        model.openConversation(id: id)
        try await waitUntil { await model.screen.document?.id == id }
        #expect(model.isShowingStoredCopy,
                "the failed restore left a fast path to a KV that no longer exists")
    }

    /// Switching between two stored chats shows the one that was clicked, and
    /// only that one.
    ///
    /// `openConversation` cleared the archive, the live conversation and the
    /// live fields, but not the stored pairs the display-only path had just
    /// filled, so the second chat's turns were drawn above the first chat's
    /// with both still on screen.
    @MainActor
    @Test func openingASecondChatReplacesTheFirstRatherThanAddingToIt() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "question in the first chat"
        model.send()
        try await finish(model)
        let first = try #require(model.history.selection)

        model.newChat()
        model.promptText = "question in the second chat"
        model.send()
        try await finish(model)
        let second = try #require(model.history.selection)
        #expect(first != second)

        model.newChat()
        model.openConversation(id: first)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        #expect(model.transcriptHistory.map(\.user.text) == ["question in the first chat"])

        model.openConversation(id: second)
        // Synchronously, before the second chat has been read off disk: the
        // first chat's turns are already gone rather than waiting to be
        // overwritten.
        #expect(model.screen.document == nil)
        #expect(model.transcriptHistory.isEmpty)

        try await waitUntil { await !model.transcriptHistory.isEmpty }
        #expect(model.transcriptHistory.map(\.user.text) == ["question in the second chat"])
    }

    /// Reading another chat and coming back costs nothing.
    ///
    /// The KV is untouched by browsing — opening a row only changes what is
    /// drawn — so the conversation the model is holding is still there when the
    /// user returns to it, and the next message continues it directly. Before
    /// this, opening anything started a new lineage on the app's side, so
    /// coming back meant replaying in full a conversation the cache had never
    /// let go of.
    @MainActor
    @Test func lookingAtAnotherChatAndComingBackCostsNoReplay() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let first = try #require(model.history.selection)

        model.newChat()
        model.temperature = 0.65
        model.topKEnabled = true
        model.topK = 23
        model.topPEnabled = true
        model.topP = 0.72
        model.maxNewTokensOverride = 91
        model.promptText = "second"
        model.send()
        try await finish(model)
        let second = try #require(model.history.selection)
        let heldEpoch = model.conversation.epoch

        // Look at the older chat, then come back to the one being held.
        model.openConversation(id: first)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        #expect(model.isShowingStoredCopy)
        #expect(model.displayedTranscriptID != heldEpoch,
                "the transcript would not know it is drawing something else")
        // The held conversation is exactly where it was.
        #expect(model.storedConversationID == second)
        #expect(model.conversation.epoch == heldEpoch)
        #expect(model.serviceEpoch == heldEpoch)

        model.temperature = 0
        model.topK = 1
        model.topP = 1
        model.maxNewTokensOverride = 2

        model.openConversation(id: second)
        #expect(model.screen == .live)
        #expect(model.conversation.epoch == heldEpoch, "the lineage was restarted")
        // The renderer appends and cannot take pairs back, so going out and
        // coming back has to read as two different things to draw. Keyed on the
        // epoch alone — which no longer changes — the chat that was read stayed
        // on screen under the row that was returned to.
        #expect(model.displayedTranscriptID == heldEpoch)
        #expect(model.temperature == 0.65)
        #expect(model.topKEnabled && model.topK == 23)
        #expect(model.topPEnabled && model.topP == 0.72)
        #expect(model.maxNewTokensOverride == 2)

        model.promptText = "third"
        model.send()
        try await finish(model)

        #expect(client.restoredLineages.isEmpty, "coming back paid for a replay")
        #expect(model.conversation.epoch == heldEpoch)
        #expect(model.conversation.turns.count == 4)
    }

    /// While another chat is being read, only that chat is on screen.
    ///
    /// The live conversation is still held, and its newest exchange is still in
    /// the live fields; drawn here it would append one chat's turn to another's
    /// transcript.
    @MainActor
    @Test func readingAnotherChatShowsOnlyThatChat() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "question in the first chat"
        model.send()
        try await finish(model)
        let first = try #require(model.history.selection)

        model.newChat()
        model.promptText = "question in the held chat"
        model.send()
        try await finish(model)

        model.openConversation(id: first)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        #expect(model.transcriptHistory.map(\.user.text)
            == ["question in the first chat"])
        #expect(model.hasOutputTranscript)
        // The held chat's newest exchange is still in the live fields, and is
        // not part of what this one shows.
        #expect(model.outputPromptText == "question in the held chat")
        #expect(!model.showsLiveTurn, "the held chat's turn would be drawn here")
    }

    /// Sending while another chat is on screen still replays that chat, and the
    /// held one is given up at that point rather than at the click.
    @MainActor
    @Test func sendingWhileReadingAnotherChatReplaysIt() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let first = try #require(model.history.selection)

        model.newChat()
        model.promptText = "second"
        model.send()
        try await finish(model)
        let heldEpoch = model.conversation.epoch

        model.openConversation(id: first)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        model.promptText = "into the older one"
        model.send()
        // The replay is a task, and `finish` only waits for a generation: it
        // would return before this one had even started.
        try await waitUntil { client.restoredLineages.count == 1 }
        try await waitUntil { await !model.screen.isReplaying }
        try await finish(model)

        #expect(client.restoredLineages.count == 1)
        #expect(model.storedConversationID == first)
        #expect(model.conversation.epoch != heldEpoch, "the KV was never handed over")
        #expect(!model.isShowingStoredCopy)
    }

    /// Opening a chat must be instant: the transcript is a file, and the
    /// model's context — a full prefill of every token, tens of seconds at 8K —
    /// is not needed until the next message. Paying it on the click made a row
    /// look dead for as long as the replay took.
    @MainActor
    @Test func openingAChatPaysNoReplayAndIsInstant() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(80))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        model.newChat()
        #expect(model.history.selection == nil)

        model.openConversation(id: id)
        // Synchronously, on the same turn of the run loop as the click.
        #expect(model.history.selection == id, "the row did not highlight")
        #expect(model.screen.conversationID == id)
        // `storedConversationID` names the chat the KV is holding, and opening
        // one to read does not make it that. It becomes this id when the replay
        // hands the cache over, not on the click.
        #expect(model.storedConversationID == nil)
        #expect(model.isShowingStoredCopy)
        // And nothing that costs time has started.
        #expect(!model.screen.isReplaying)
        #expect(model.phase == .idle)
        #expect(client.restoredLineages.isEmpty)
    }

    /// A replay is not a generation, so every guard written against `isRunning`
    /// let New Chat through while one was in flight — and the replay then landed
    /// on top of the empty chat, handing back the conversation the user had just
    /// left. The replay now happens inside a send, which is exactly when that
    /// race is reachable.
    @MainActor
    @Test func newChatIsRefusedWhileAReopenIsStillReplaying() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(120))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        // Sending is what triggers the replay.
        model.promptText = "second"
        model.send()
        try await waitUntil { await model.screen.isReplaying }
        #expect(!model.canStartNewChat, "New Chat was offered during a replay")
        // And calling it anyway must not race the replay.
        model.newChat()

        try await waitUntil { await !model.screen.isReplaying }
        #expect(model.storedConversationID == id)
        #expect(!model.isShowingStoredCopy)
    }

    /// A build with nothing to restore through refuses the send rather than
    /// retrying it forever.
    ///
    /// The replay returned successfully when the client had no model lifecycle
    /// behind it — the guard that looked for one treated its absence as "there
    /// is nothing to do" — and the send then handed the message back to the
    /// composer and called `run()` again, with the same conversation still
    /// waiting to be replayed. Nothing on screen changed and the main actor
    /// never came back. Without the fix this test does not fail, it hangs.
    @MainActor
    @Test func areplayWithNoLifecycleBehindItIsRefusedRatherThanRetried() async throws {
        let client = MockInferenceClient(tokenDelayNanos: 0)
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-\(UUID().uuidString)", isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(modelDirectory: directory, client: client,
                             settingsPersistenceEnabled: true)
        model.modelPathText = directory.path
        model.conversationIdentity = Self.identity
        model.installationStatus = .complete
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        try await model.waitForHistory()

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)

        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        model.promptText = "second"
        model.send()
        try await waitUntil { await model.error != nil }

        #expect(!model.screen.isReplaying)
        #expect(model.promptText == "second", "the message was not handed back")
        #expect(model.openedConversationState
            == .cannotReplay(reason: .tokenCountUnknown))
    }

    /// A conversation that cannot take another turn must not be replayed at
    /// all: minutes of prefill to reach a composer that refuses everything.
    @MainActor
    @Test func aconversationThatNoLongerFitsIsNeverReplayed() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let store = try #require(model.conversationStore)

        // Lower the context under what the stored conversation holds. The
        // stored files must not move for it.
        try await Self.padStoredTokens(store, id: id, to: 4_000)
        await model.refreshHistory()
        model.setMaxContextTokens(4_096)

        let listed = try #require(model.history.entry(id))
        guard case .needsContext(let required) = model.continuability(of: listed) else {
            Issue.record("a chat past the ceiling still reported as continuable")
            return
        }
        #expect(required == ConversationGenerationReserve.contextRequired(
            forLineage: 4_000))

        let bytesBefore = try Data(contentsOf: await store.transcriptURL(for: id))
        model.newChat()
        model.openConversation(id: id)
        try await Task.sleep(for: .milliseconds(100))
        #expect(client.restoredLineages.isEmpty, "a chat that cannot fit was replayed")
        #expect(model.openedConversationState == .needsContext(required: required))
        // Readable, and untouched: the row changed, the files did not.
        #expect(!model.transcriptHistory.isEmpty)
        #expect(try Data(contentsOf: await store.transcriptURL(for: id)) == bytesBefore)
    }

    /// New Chat has to stop this window appending to the previous chat's file,
    /// or the transcript on disk would claim a context the model does not have.
    @MainActor
    @Test func newChatStartsItsOwnFileOnTheNextSend() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let first = try #require(model.history.selection)

        model.newChat()
        #expect(model.storedConversationID == nil)
        #expect(model.history.selection == nil)
        // Not written until it is used: an unsent new chat takes no row.
        #expect(model.history.entries.count == 1)

        model.promptText = "second"
        model.send()
        try await finish(model)
        let second = try #require(model.history.selection)
        #expect(second != first)
        #expect(model.history.entries.count == 2)
    }

    /// Delete takes the folder with it. There is no undo because there is no
    /// copy: the alternative kept every deleted conversation on disk for the
    /// life of the store.
    @MainActor
    @Test func deleteRemovesTheChatFromDiskWithNoCopyLeftBehind() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let folder = try #require(await model.conversationFolderURL(for: id))

        model.deleteConversation(id: id)
        try await model.waitForHistory(count: 0)

        #expect(model.history.entries.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: folder.path),
                "the conversation was parked rather than deleted")
        // The open chat lost its file, not its KV, so it stops being written to.
        #expect(model.storedConversationID == nil)
        #expect(model.history.selection == nil)
        // The store keeps nothing on the side.
        let store = folder.deletingLastPathComponent()
        let leftovers = try FileManager.default.contentsOfDirectory(
            atPath: store.path)
        #expect(!leftovers.contains(".trash"))
    }

    /// Deleting the open chat ends its lineage, so the next one describes
    /// itself honestly.
    ///
    /// The KV used to be kept — the model had not forgotten anything — and the
    /// next turn opened a fresh file. But `turnCount` and `kvTokens` are taken
    /// from the live conversation, so that file held one exchange while its
    /// meta described everything behind it: a row reading 1,065 tokens over 22
    /// tokens of record, which the sidebar shows, the gauge shows, and
    /// continuability decides against. Found running case C23.
    @MainActor
    @Test func deletingTheOpenChatEndsItsLineage() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        model.promptText = "second"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        #expect(model.conversation.turns.count == 4)

        model.deleteConversation(id: id)
        try await model.waitForHistory(count: 0)

        // The conversation went with its file.
        #expect(model.conversation.turns.isEmpty, "the deleted chat is still live")
        #expect(!model.hasOutputTranscript)
        #expect(model.storedConversationID == nil)

        model.promptText = "after the delete"
        model.send()
        try await finish(model)

        let newID = try #require(model.history.selection)
        #expect(newID != id)
        let store = try #require(model.conversationStore)
        let opened = try await store.open(id: newID)
        let turns = opened.records.filter {
            if case .turn = $0 { return true } else { return false }
        }
        // What the meta claims is what the file holds.
        #expect(opened.meta.turnCount == turns.count)
        #expect(turns.count == 2)
        let recorded = opened.records.reduce(into: 0) { total, record in
            guard case .turn(let turn) = record else { return }
            total += turn.tokens?.count ?? 0
        }
        #expect(opened.meta.kvTokens == recorded,
                "the row would show a token count the file cannot back")
    }

    /// Deleting the chat being read clears what is on screen, rather than
    /// leaving a transcript with no conversation behind it.
    @MainActor
    @Test func deletingTheChatBeingReadClearsTheTranscript() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let first = try #require(model.history.selection)

        model.newChat()
        model.promptText = "second"
        model.send()
        try await finish(model)

        model.openConversation(id: first)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        #expect(model.isShowingStoredCopy)

        model.deleteConversation(id: first)
        try await model.waitForHistory(count: 1)

        #expect(model.screen == .live)
        // The window falls back to the chat the KV is still holding, so its row
        // is the one selected. Asserted as nil before, which left the second
        // chat on screen with nothing highlighted in the sidebar.
        #expect(model.history.selection == model.storedConversationID)
        #expect(model.history.selection != nil)
    }

    /// A turn carrying an epoch the service is no longer holding is refused
    /// before anything runs.
    ///
    /// The rule is the decode service's own `DecodeConversationGate`, and until
    /// the double ran it every app-side test admitted every turn: a stale epoch,
    /// a skipped position and a correct turn were indistinguishable here, so no
    /// test in this file could tell whether the app and the service agreed about
    /// which conversation was open. Without the gate in `FakeInferenceClient`
    /// this streams an ordinary reply and records nothing.
    @MainActor
    @Test func aturnStampedWithAStaleEpochIsRefusedRatherThanAdmitted() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)

        // Composed against a conversation the service has since replaced.
        model.promptText = "and again"
        let request = try model.makeRequest(
            ticket: AppConversation.Ticket(epoch: UUID(), index: 0))
        var failure: AppInferenceError?
        do {
            for try await event in client.generate(request) {
                if case .failed(let error, _) = event { failure = error }
            }
            Issue.record("the stale turn ran to completion")
        } catch {
            // The stream also throws, the way the real client's does.
        }

        guard case .conversationLineageLost = try #require(failure) else {
            Issue.record("a stale turn was not refused as a lost lineage")
            return
        }
        #expect(client.gateRejections.count == 1)
        guard case .staleConversation = try #require(client.gateRejections.first) else {
            Issue.record("the refusal was not attributed to the stale epoch")
            return
        }
        // And the position the next honest turn has to match is untouched.
        #expect(client.gateCommittedTurns == 1)
    }

    /// The position the service is holding is the position the app's next turn
    /// carries, on a live chat and on a reopened one alike.
    @MainActor
    @Test func theappAndTheGateAgreeOnThePositionAcrossAReopen() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        model.promptText = "second"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        #expect(client.gateCommittedTurns == model.conversation.committedTurns)

        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        model.promptText = "third"
        model.send()
        // A send on a reopened chat replays first, and a replay is not
        // `isRunning`: waiting only on that returns before the turn has begun.
        try await waitUntil { await !model.isTurnInFlight }
        try await finish(model)

        #expect(client.gateRejections.isEmpty,
                "the reopened conversation's turn was numbered against the wrong lineage")
        #expect(client.gateOpenEpoch == model.serviceEpoch)
        #expect(client.gateCommittedTurns == model.conversation.committedTurns)
        #expect(model.conversation.committedTurns == 3)
    }
}

extension AppModel {
    /// Waits for the history list to settle, since it is refreshed from a task.
    @MainActor
    func waitForHistory(count: Int? = nil, timeout: Duration = .seconds(5)) async throws {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if let count {
                if history.entries.count == count { return }
            } else if conversationStore != nil {
                if !history.isReadOnlyStore { return }
            } else {
                return
            }
            try await Task.sleep(for: .milliseconds(5))
        }
    }

    /// Waits for a completed turn to be fully written, records *and* meta.
    ///
    /// The records are not the whole write. `persistCompletedTurn` appends both
    /// halves, then rewrites `conversation.json`, then refreshes the list — so a
    /// helper that waited on the records alone returned inside that window, and
    /// a reopen in the next line read a meta whose `kvTokens` was still nil and
    /// came back `.cannotReplay(.tokenCountUnknown)`. It failed roughly one run
    /// in four under parallel execution and never in isolation.
    @MainActor
    func waitForStoredTurns(atLeast expected: Int) async throws {
        await persistenceTail?.value
        guard let store = conversationStore, let id = storedConversationID else { return }
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if let opened = try? await store.open(id: id),
               opened.records.filter({ if case .turn = $0 { return true } else { return false } })
                .count >= expected,
               let entry = history.entry(id), entry.turnCount >= expected {
                return
            }
            try await Task.sleep(for: .milliseconds(5))
        }
    }

}

extension AppModelHistoryTests {
    /// A one-pixel PNG on disk, for the staging path.
    /// Gives a stored conversation exactly `tokens` tokens of record.
    ///
    /// `kvTokens` is projected from the turns now, so a test that needs a chat
    /// too big for a context makes it big rather than writing a number into its
    /// `conversation.json` — which the next open would replace.
    /// A row the notice has refused cannot be sent either.
    ///
    /// `canRun` asked only the live conversation, which is always willing, so
    /// Return on a row marked "needs more context" replayed it anyway: minutes
    /// of prefill, the held chat's KV dropped, and a refusal at the end that
    /// relabelled the row continuable for the next attempt.
    @MainActor
    @Test func arowThatCannotBeContinuedCannotBeSent() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "hello"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let store = try #require(model.conversationStore)
        try await Self.padStoredTokens(store, id: id, to: 4_000)
        await model.refreshHistory()
        model.setMaxContextTokens(4_096)
        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        guard case .needsContext = model.openedConversationState else {
            Issue.record("the padded chat still reported as continuable")
            return
        }

        model.promptText = "more"
        #expect(!model.canRun, "the composer offered a send on a refused row")
        model.send()
        try await Task.sleep(for: .milliseconds(100))

        #expect(client.restoredLineages.isEmpty, "the refused row was replayed")
        #expect(!model.screen.isReplaying)
        #expect(model.screen.conversationID == id)
        #expect(model.promptText == "more", "the message was taken")
    }

    /// A row that could not be read has nowhere for a message to go.
    ///
    /// `deliver` treated only `.reading` as a stored-copy send, so with an
    /// unreadable row on screen the replay stage was skipped and the turn ran
    /// on the held conversation: written into that chat's file while the
    /// sidebar, the notice and the saved selection all named the row that
    /// could not be read.
    @MainActor
    @Test func asendWhileTheRowIsUnreadableDoesNotLandOnTheHeldChat() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "held"
        model.send()
        try await finish(model)
        let held = try #require(model.storedConversationID)
        let store = try #require(model.conversationStore)

        // A second chat, written directly, whose transcript will not read.
        let other = try await store.create(
            title: "other", identity: Self.identity,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "onDemand"),
            sampling: model.currentSampling())
        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "one", tokens: [1, 2])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "two", tokens: [3])),
        ], to: other.id)
        await model.refreshHistory()
        let folder = try #require(await model.conversationFolderURL(for: other.id))
        let transcript = folder.appendingPathComponent("transcript.jsonl")
        let data = try String(contentsOf: transcript, encoding: .utf8)
        try (String(data.dropLast(40)) + "\n").write(
            to: transcript, atomically: true, encoding: .utf8)

        model.openConversation(id: other.id)
        try await waitUntil { await model.error != nil }
        guard case .unreadable = model.screen else {
            Issue.record("the damaged row did not report as unreadable")
            return
        }

        model.promptText = "where does this go"
        #expect(!model.canRun, "the composer offered a send under an unreadable row")
        model.send()
        try await Task.sleep(for: .milliseconds(100))

        #expect(model.promptText == "where does this go", "the message was taken")
        #expect(model.storedConversationID == held)
        #expect(try await store.open(id: held).meta.turnCount == 2,
                "the turn was written into the held chat under the wrong row")
        #expect(model.history.selection == other.id)
    }

    /// Unload and Reload are closed for the whole send, not just the
    /// generation.
    ///
    /// A deferred replay is a full prefill during which `isRunning` is still
    /// false, so Model > Unload was enabled. Taking it dropped the KV under the
    /// replay, the machine discarded the message it was carrying with no
    /// hand-back, and the unload queued behind the prefill on the service
    /// until the load timeout killed the connection.
    @MainActor
    @Test func unloadIsRefusedWhileAReopenIsStillReplaying() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(120))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        model.promptText = "second"
        model.send()
        try await waitUntil { await model.screen.isReplaying }
        #expect(!model.canUnloadModel, "Unload was offered during a replay")
        #expect(!model.canReloadModel, "Reload was offered during a replay")
        #expect(!model.canLoadModel)
        // And calling it anyway must not take the KV from under the replay.
        model.unloadModel()

        await SendWaiting.turnEnds(model)
        #expect(model.loadState.isReady)
        #expect(model.storedConversationID == id)
        #expect(client.restoredLineages.count == 1)
        #expect(model.conversation.turns.count == 4, "the replayed turn did not run")
    }

    /// A replay refused on this side leaves the held conversation alone.
    ///
    /// The companion pack check runs before the service is asked for anything,
    /// and reporting its refusal as a failed replay ended the held lineage
    /// anyway: its epoch was forgotten and its staged images deleted for a
    /// restore that never happened, while the service still held it.
    @MainActor
    @Test func areplayRefusedBeforeTheServiceKeepsTheHeldChat() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "held"
        model.send()
        try await finish(model)
        let held = try #require(model.storedConversationID)
        let heldEpoch = try #require(model.serviceEpoch)
        let store = try #require(model.conversationStore)

        // A stored chat with a picture in it, at a location with no companion
        // pack: the replay is refused before the restore request is built.
        let placeholder = MultimodalPromptRenderer.imageTokenID
        let pictured = try await store.create(
            title: "pictured", identity: Self.identity,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "onDemand"),
            sampling: model.currentSampling())
        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "look",
                images: [ConversationImageRecord(
                    id: UUID(), displayName: "roof.heic",
                    pixelsFile: "images/roof.png",
                    thumbnailFile: "images/roof.thumb.jpg",
                    sourceDigest: "roof", modelInputDigest: "model-roof",
                    width: 48, height: 48, softTokens: 2)],
                tokens: [1, placeholder, placeholder, 4])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "a roof", tokens: [5])),
        ], to: pictured.id)
        await model.refreshHistory()
        #expect(!model.isVisionPackInstalled)

        model.openConversation(id: pictured.id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        #expect(model.openedConversationState == .continuable)
        model.promptText = "and this one"
        model.send()
        try await waitUntil { await model.error != nil }

        #expect(client.restoredLineages.isEmpty, "the service was asked after all")
        // The held chat is exactly where it was: same epoch, same file.
        #expect(model.serviceEpoch == heldEpoch)
        #expect(model.storedConversationID == held)
        #expect(model.conversation.epoch == heldEpoch)
        // And the row that was refused is back to what it was.
        #expect(model.screen.conversationID == pictured.id)
        #expect(model.openedConversationState == .continuable)
        #expect(model.promptText == "and this one", "the message was not handed back")
    }

    /// The decode service dying during a replay is the same loss as during a
    /// generation, and has to leave the same state.
    ///
    /// Folded into an ordinary failed replay, the window kept advertising a
    /// ready model: no Retry Load, Generate enabled, and every later send
    /// re-entering the replay to fail on a connection that was already gone.
    @MainActor
    @Test func connectionLossDuringAReplayRequiresAReload() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }

        client.failNextRestore(with: .connectionLost("unexpected EOF"))
        model.promptText = "again"
        model.send()
        try await waitUntil { await model.error != nil }

        guard case .failed(.connectionLost(let cause)) = model.loadState else {
            Issue.record("connection loss during a replay did not invalidate Ready")
            return
        }
        #expect(cause.contains("unexpected EOF"))
        #expect(model.canLoadModel, "no Retry Load after the service died")
        #expect(!model.canRun, "Generate stayed enabled with no service behind it")
        #expect(model.serviceEpoch == nil)
        #expect(model.promptText == "again", "the message was not handed back")
        #expect(model.history.selection == id)

        // After the reload the row is still the one on screen, and sending
        // replays it. Retry Load used to archive the row, clear the selection
        // and send the retried message into a brand-new chat.
        model.loadModel()
        try await waitUntil { await model.loadState.isReady }
        #expect(model.screen.conversationID == id, "Retry Load took the row off screen")
        #expect(model.history.selection == id)
        #expect(model.canRun, "the handed-back message cannot be sent after the reload")
        model.send()
        await SendWaiting.turnEnds(model)
        #expect(client.restoredLineages.count == 1)
        #expect(model.storedConversationID == id)
    }

    /// The chat the KV holds is the live chat, whatever its record says.
    ///
    /// A turn whose stored picture failed to write is still in the KV; its
    /// record just cannot be replayed later. Clicking away and back read that
    /// chat as a copy the composer refuses, with New Chat as the only way out
    /// of a conversation the model was still holding.
    @MainActor
    @Test func clickingBackToTheHeldChatIsLiveEvenWhenItsRecordCannotReplay() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "held"
        model.send()
        try await finish(model)
        let held = try #require(model.storedConversationID)
        let store = try #require(model.conversationStore)
        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "look", tokens: [1, 2],
                imageWriteFailed: true)),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "a roof", tokens: [3])),
        ], to: held)
        let other = try await store.create(
            title: "other", identity: Self.identity,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "onDemand"),
            sampling: model.currentSampling())
        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "one", tokens: [1, 2])),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "two", tokens: [3])),
        ], to: other.id)
        await model.refreshHistory()
        let listed = try #require(model.history.entry(held))
        #expect(model.continuability(of: listed)
            == .cannotReplay(reason: .imageRecordMissing))

        model.openConversation(id: other.id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        model.openConversation(id: held)

        #expect(model.screen == .live, "the held chat was read as a copy")
        #expect(model.history.selection == held)
        model.promptText = "more"
        #expect(model.canRun, "the composer refused the chat the KV holds")
    }

    /// A refusal before the model saw the turn leaves the previous turn's
    /// pictures on screen.
    ///
    /// The live fields draw the newest finished turn between sends, and every
    /// hand-back used to clear their pictures: a request that did not validate
    /// made the previous answer's photo vanish from the transcript.
    @MainActor
    @Test func aRefusedSendKeepsThePreviousTurnsPicturesOnScreen() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let folder = try #require(await model.conversationFolderURL(for: id))
        let images = folder.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(
            at: images, withIntermediateDirectories: true)
        let thumbnail = images.appendingPathComponent("abcd.thumb.jpg")
        try Data("thumbnail".utf8).write(to: thumbnail)
        model.outputImageAttachments = [ChatImage.stored(StoredImage(
            fileURL: thumbnail, displayName: "roof.heic",
            encodedBytes: 9, sha256: "abcd"))]

        // Refused by `AppGenerationRequest.validate`, after the composer has
        // been handed over.
        model.maxNewTokensOverride = 0
        model.promptText = "second"
        model.send()
        await SendWaiting.turnEnds(model)

        #expect(model.promptText == "second", "the message was not handed back")
        #expect(model.error != nil, "the refusal was silent")
        #expect(model.outputImageAttachments.count == 1,
                "the previous turn's picture was cleared from the transcript")
    }

    /// The composer's pictures are closed for the whole send, replay included.
    ///
    /// The picture guards asked `isRunning`, which a deferred replay never
    /// sets, so a picture attached during the replay met the draft-wins rule
    /// when the replay failed and the sent message's own pictures were
    /// deleted.
    @MainActor
    @Test func theComposersPicturesCannotBeChangedDuringAReplay() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(120))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        let attachments = AppImageAttachmentStore()

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        model.promptText = "second"
        model.send()
        try await waitUntil { await model.screen.isReplaying }

        let source = try Self.pngFixture()
        defer { try? FileManager.default.removeItem(at: source) }
        let staged = try attachments.stage(source)
        defer { attachments.remove(staged) }
        model.setComposerAttachmentsForTesting([staged])
        model.clearImages()
        #expect(model.imageAttachments.count == 1,
                "the composer's pictures were cleared during a replay")
        model.removeImage(id: staged.id)
        #expect(model.imageAttachments.count == 1,
                "a picture was removed during a replay")
        await SendWaiting.turnEnds(model)
        model.setComposerAttachmentsForTesting([])
    }

    /// Quit waits for the last exchange to reach disk.
    @MainActor
    @Test func awaitingPersistenceLeavesTheLastExchangeOnDisk() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "durable"
        model.send()
        await SendWaiting.turnEnds(model)
        await model.awaitPendingPersistence()

        let id = try #require(model.storedConversationID)
        let store = try #require(model.conversationStore)
        #expect(try await store.open(id: id).meta.turnCount == 2,
                "the exchange was not on disk after the wait")
    }

    /// Deleting right after the reply waits for that reply's write.
    ///
    /// The write starts when the reply lands and `isTurnInFlight` is already
    /// false, so the delete used to run under it: the writer recreated the
    /// directory and the append left a headerless transcript the list then
    /// counted as unreadable, with no row left to delete.
    @MainActor
    @Test func deletingRightAfterTheReplyLeavesNothingBehind() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "gone"
        model.send()
        await SendWaiting.turnEnds(model)
        let id = try #require(model.storedConversationID)
        let folder = try #require(await model.conversationFolderURL(for: id))

        model.deleteConversation(id: id)
        try await waitUntil { await model.history.entries.isEmpty }

        #expect(!FileManager.default.fileExists(atPath: folder.path),
                "the delete left the directory behind")
        #expect(model.history.unreadableCount == 0)
        let diagnostic: String = model.historyDiagnostic ?? ""
        #expect(model.historyDiagnostic == nil, "the delete reported: \(diagnostic)")
    }

    /// The image writer reads links of its own.
    ///
    /// It read the turn's retained links, which New Chat or a replay of
    /// another chat releases the moment the reply lands; a write still in its
    /// decode then lost its source, the turn was recorded as missing a
    /// picture, and the conversation refused every replay after.
    @MainActor
    @Test func theImageWriterKeepsItsOwnLinkToTheSource() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let attachments = AppImageAttachmentStore()
        let (model, root) = try await readyModel(client, attachmentStore: attachments)
        defer { try? FileManager.default.removeItem(at: root) }

        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.history.selection)
        let source = try Self.pngFixture()
        defer { try? FileManager.default.removeItem(at: source) }
        let staged = try attachments.stage(source)

        model.beginStoringTurnImages([staged], in: id)
        // Released as early as a hand-off or New Chat would release it.
        attachments.remove(staged)
        let outcome = await model.pendingTurnImageWrite?.value

        #expect(outcome?.failed == 0,
                "the write lost its source: \(outcome?.reason ?? "no reason")")
        #expect(outcome?.records.count == 1)
    }

    /// A store without the writer lock writes nothing, including into a chat
    /// this window adopted from another window's files.
    ///
    /// The existing id used to be returned before the read-only check, so a
    /// replayed chat had its images written into the other window's directory
    /// before the append itself was refused.
    @MainActor
    @Test func aReadOnlyStoreNeverWritesIntoAnotherWindowsChat() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-\(UUID().uuidString)", isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        // Another instance holds the writer lock first.
        let holder = ConversationStore(
            rootURL: ConversationStoreLocation.url(forModelDirectory: directory))
        try await holder.activate()

        let model = AppModel(modelDirectory: directory, client: client,
                             settingsPersistenceEnabled: true)
        model.modelPathText = directory.path
        model.conversationIdentity = Self.identity
        try await client.ensureLoaded(
            modelDirectory: directory,
            maxContextTokens: model.maxContextTokens,
            options: model.runtimeOptions,
            forceLogitsHead: true) { _ in }
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        model.installationStatus = .complete
        try await waitUntil { await model.history.isReadOnlyStore }
        #expect(model.history.isReadOnlyStore)

        model.storedConversationID = UUID()
        let written = await model.ensureStoredConversation(firstMessage: "x")
        #expect(written == nil, "a read-only store handed back a chat to write into")
        withExtendedLifetime(holder) {}
    }

    static func padStoredTokens(_ store: ConversationStore, id: UUID,
                                to tokens: Int) async throws {
        let held = try await store.open(id: id).meta.kvTokens ?? 0
        let padding = tokens - held
        guard padding >= 2 else {
            Issue.record("the conversation already holds \(held) token(s)")
            return
        }
        try await store.append([
            .turn(ConversationTurnRecord(
                role: .user, at: Date(), text: "padding",
                tokens: Array(repeating: Int32(7), count: padding - 1))),
            .turn(ConversationTurnRecord(
                role: .assistant, at: Date(), text: "ok", tokens: [8])),
        ], to: id)
    }

    static func pngFixture() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pixel-\(UUID().uuidString).png")
        let base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            + "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        try #require(Data(base64Encoded: base64)).write(to: url)
        return url
    }
}

private actor HistoryListGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var entered = false
    private var isOpen = false

    func wait() async {
        entered = true
        guard !isOpen else { return }
        await withCheckedContinuation { continuation = $0 }
    }

    func waitUntilEntered() async {
        while !entered { await Task.yield() }
    }

    func open() {
        isOpen = true
        continuation?.resume()
        continuation = nil
    }
}

/// Polls a condition the app satisfies from a task rather than synchronously.
func waitUntil(_ condition: @Sendable () async -> Bool) async throws {
    let deadline = Date().addingTimeInterval(5)
    while Date() < deadline {
        if await condition() { return }
        try await Task.sleep(for: .milliseconds(5))
    }
}


extension AppModelHistoryTests {
    @MainActor
    @Test func replayWaitsForPreviouslyCommittedTurnsToReachDisk() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(20))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        let gate = HistoryListGate()
        model.promptText = "second"
        model.send()
        try await waitUntil { client.isGenerating }
        model.pendingTurnImageWrite = Task {
            await gate.wait()
            return TurnImageWriteOutcome()
        }
        try await waitUntil { await !model.isTurnInFlight }
        try #require(!model.isTurnInFlight)
        await gate.waitUntilEntered()
        model.newChat()
        model.openConversation(id: id)
        try await waitUntil { await model.screen.document != nil }
        model.promptText = "third"
        model.send()
        // Leave the save blocked long enough for an incorrect replay to finish.
        try await Task.sleep(for: .milliseconds(300))
        #expect(client.restoredLineages.isEmpty)
        await gate.open()
        await SendWaiting.turnEnds(model)
        await model.awaitPendingPersistence()
        let store = try #require(model.conversationStore)
        let disk = try await store.open(id: id).records.compactMap { record -> String? in
            if case .turn(let turn) = record, turn.role == .user { return turn.text }
            return nil
        }
        let live = model.conversation.turns.filter { $0.role == .user }.map(\.text)
        #expect(live == disk)
    }

    @MainActor
    @Test func rewindWaitsForPreviouslyCommittedImagesBeforeSweeping() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(20))
        let (model, root) = try await readyModel(client)
        defer { try? FileManager.default.removeItem(at: root) }
        model.promptText = "first"
        model.send()
        try await finish(model)
        let id = try #require(model.storedConversationID)
        let store = try #require(model.conversationStore)
        let directory = await store.imagesURL(for: id)
        let pixels = directory.appendingPathComponent("second.png")
        let thumbnail = directory.appendingPathComponent("second.thumb.jpg")
        try Data("pixels".utf8).write(to: pixels)
        try Data("thumb".utf8).write(to: thumbnail)
        let gate = HistoryListGate()
        let outcome = imageOutcome("second")
        model.promptText = "second"
        model.send()
        try await waitUntil { client.isGenerating }
        model.pendingTurnImageWrite = Task {
            await gate.wait()
            return outcome
        }
        try await waitUntil { await !model.isTurnInFlight }
        try #require(!model.isTurnInFlight)
        await gate.waitUntilEntered()
        let orphan = directory.appendingPathComponent("rewound.png")
        try Data("orphan".utf8).write(to: orphan)
        client.failNextGeneration(with: .unknown("injected rewind"))
        model.promptText = "third"
        model.send()
        try await Task.sleep(for: .milliseconds(300))
        #expect(FileManager.default.fileExists(atPath: pixels.path))
        #expect(FileManager.default.fileExists(atPath: thumbnail.path))
        await gate.open()
        await SendWaiting.turnEnds(model)
        let saved = try await store.open(id: id)
        #expect(saved.meta.imageCount == 1)
        #expect(model.promptText == "third")
        #expect(!FileManager.default.fileExists(atPath: orphan.path))
        #expect(FileManager.default.fileExists(atPath: pixels.path))
    }
}
