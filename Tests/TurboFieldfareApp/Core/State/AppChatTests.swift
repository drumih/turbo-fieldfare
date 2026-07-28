import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppChatTests {
    @Test func chatPreviewUsesLatestMessageThenDraftThenEmptyFallback() {
        var chat = AppChat(draft: "  draft\non two lines  ")
        #expect(chat.preview == "draft on two lines")

        chat.messages = [
            AppChatMessage(role: .user, content: "Question"),
            AppChatMessage(role: .assistant, content: "  final\nanswer  "),
        ]
        #expect(chat.preview == "final answer")

        chat.messages.removeAll()
        chat.draft = ""
        #expect(chat.preview == "No messages yet")
    }

    @Test func archiveValidationRejectsBrokenIdentityAndSummaryInvariants() {
        let first = AppChat()
        let otherID = UUID()
        let message = AppChatMessage(role: .assistant, content: "answer")
        let validSummaryChat = AppChat(
            messages: [message],
            contextSummary: "memory",
            summarizedThroughMessageID: message.id)

        #expect(AppChatArchive(
            selectedChatID: first.id,
            chats: [first]).isValid)
        #expect(!AppChatArchive(
            version: 999,
            selectedChatID: first.id,
            chats: [first]).isValid)
        #expect(!AppChatArchive(
            selectedChatID: first.id,
            chats: []).isValid)
        #expect(!AppChatArchive(
            selectedChatID: otherID,
            chats: [first]).isValid)
        #expect(!AppChatArchive(
            selectedChatID: first.id,
            chats: [first, first]).isValid)
        #expect(!AppChatArchive(
            selectedChatID: first.id,
            chats: [
                AppChat(
                    id: first.id,
                    contextSummary: "memory",
                    summarizedThroughMessageID: otherID),
            ]).isValid)
        #expect(!AppChatArchive(
            selectedChatID: first.id,
            chats: [
                AppChat(
                    id: first.id,
                    messages: [message],
                    contextSummary: " ",
                    summarizedThroughMessageID: message.id),
            ]).isValid)
        #expect(AppChatArchive(
            selectedChatID: validSummaryChat.id,
            chats: [validSummaryChat]).isValid)
    }

    @Test func invalidArchiveCannotBePersisted() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "AppChatInvalidSaveTests-\(UUID().uuidString)",
                isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent(
            "model.gturbo",
            isDirectory: true)
        let chat = AppChat()
        let invalid = AppChatArchive(
            selectedChatID: UUID(),
            chats: [chat])

        #expect(throws: (any Error).self) {
            try AppChatFileStore.save(
                invalid,
                forModelDirectory: modelDirectory)
        }
        #expect(!FileManager.default.fileExists(
            atPath: AppChatFileStore.fileURL(
                forModelDirectory: modelDirectory).path))
    }

    @MainActor
    @Test func newChatsKeepIndependentDraftsAndAttachments() {
        let model = AppModel()
        let firstChatID = model.selectedChatID
        model.promptText = "First draft"
        model.addPromptAttachment(AppPromptAttachment(
            fileName: "first.pdf",
            formatLabel: "PDF",
            extractedText: "First document"))

        let secondChatID = model.createChat()
        #expect(secondChatID != firstChatID)
        #expect(model.promptText.isEmpty)
        #expect(model.promptAttachments.isEmpty)
        model.promptText = "Second draft"

        model.selectChat(id: firstChatID)
        #expect(model.promptText == "First draft")
        #expect(model.promptAttachments.map(\.fileName) == ["first.pdf"])

        model.selectChat(id: secondChatID)
        #expect(model.promptText == "Second draft")
        #expect(model.promptAttachments.isEmpty)
    }

    @MainActor
    @Test func chatCRUDMaintainsSelectionOrderingAndSafeFallback() {
        let model = AppModel()
        let originalID = model.selectedChatID
        model.renameChat(id: originalID, title: "  Original  ")
        #expect(model.selectedChat.title == "Original")

        let secondID = model.createChat()
        #expect(model.chats.map(\.id) == [secondID, originalID])
        model.renameChat(
            id: secondID,
            title: String(repeating: "x", count: 100))
        #expect(model.selectedChat.title == String(repeating: "x", count: 80))

        model.renameChat(id: secondID, title: " \n ")
        #expect(model.selectedChat.title == String(repeating: "x", count: 80))
        model.renameChat(id: UUID(), title: "Unknown")
        #expect(model.chats.count == 2)

        model.deleteChat(id: originalID)
        #expect(model.selectedChatID == secondID)
        #expect(model.chats.map(\.id) == [secondID])

        model.deleteChat(id: secondID)
        #expect(model.chats.count == 1)
        #expect(model.selectedChatID == model.chats[0].id)
        #expect(model.selectedChatID != secondID)
        #expect(model.selectedChat.title == "New chat")
    }

    @MainActor
    @Test func deletingSelectedChatChoosesTheNearestRemainingChat() {
        let model = AppModel()
        let oldestID = model.selectedChatID
        let middleID = model.createChat()
        let newestID = model.createChat()
        #expect(model.chats.map(\.id) == [newestID, middleID, oldestID])

        model.deleteChat(id: middleID)
        #expect(model.selectedChatID == newestID)
        model.selectChat(id: oldestID)
        model.deleteChat(id: oldestID)
        #expect(model.selectedChatID == newestID)

        let selectionBeforeUnknown = model.selectedChatID
        model.selectChat(id: UUID())
        model.deleteChat(id: UUID())
        #expect(model.selectedChatID == selectionBeforeUnknown)
    }

    @MainActor
    @Test func attachmentMutationsAreScopedAndIdempotent() {
        let model = AppModel()
        let first = AppPromptAttachment(
            fileName: "one.pdf",
            formatLabel: "PDF",
            extractedText: "one")
        let second = AppPromptAttachment(
            fileName: "two.xlsx",
            formatLabel: "Excel",
            extractedText: "two")
        model.addPromptAttachment(first)
        model.addPromptAttachment(second)

        model.removePromptAttachment(id: UUID())
        #expect(model.promptAttachments == [first, second])
        model.removePromptAttachment(id: first.id)
        #expect(model.promptAttachments == [second])
        model.clearPromptAttachments()
        #expect(model.promptAttachments.isEmpty)
        model.clearPromptAttachments()
        #expect(model.promptAttachments.isEmpty)
    }

    @MainActor
    @Test func examplesOnlyAppearInACompletelyEmptyChat() {
        let model = AppModel()
        #expect(model.showsPromptExamples)

        model.promptText = "Draft"
        #expect(!model.showsPromptExamples)

        model.promptText = ""
        model.addPromptAttachment(AppPromptAttachment(
            fileName: "notes.docx",
            formatLabel: "Word",
            extractedText: "Notes"))
        #expect(!model.showsPromptExamples)
    }

    @MainActor
    @Test func examplesStayHiddenWhenAnExistingChatHasHistory() async {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        client.prefillSteps = 0
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.maxNewTokensOverride = 1
        model.promptText = "hello"

        model.run()
        await waitForIdle(model)

        #expect(model.promptText.isEmpty)
        #expect(!model.showsPromptExamples)
        model.clearOutput()
        #expect(model.showsPromptExamples)
    }

    @MainActor
    @Test func successfulRunCreatesTitleAndKeepsDocumentTextOutOfVisibleTranscript() async {
        let client = CompressionInferenceClient()
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.maxNewTokensOverride = 1
        let prompt = String(repeating: "p", count: 60)
        model.promptText = prompt
        model.addPromptAttachment(AppPromptAttachment(
            fileName: "private.pdf",
            formatLabel: "PDF",
            extractedText: "DOCUMENT-ONLY-SECRET"))

        model.run()
        await waitForIdle(model)

        #expect(model.selectedChat.title == String(repeating: "p", count: 48))
        #expect(model.selectedChat.messages.first?.content
            == "\(prompt)\n\nAttachments: private.pdf")
        #expect(model.selectedChat.messages.first?.contextContent.contains(
            "DOCUMENT-ONLY-SECRET") == true)
        #expect(!model.outputConversationPlainText.contains(
            "DOCUMENT-ONLY-SECRET"))
        #expect(model.promptAttachments.isEmpty)
    }

    @MainActor
    @Test func completedTurnBecomesContextForTheNextRequest() async {
        let client = MockInferenceClient(response: "first answer", tokenDelayNanos: 1)
        client.prefillSteps = 0
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.maxNewTokensOverride = 2
        model.promptText = "first question"
        model.run()
        await waitForIdle(model)

        model.promptText = "follow-up"
        let request = try? model.makeRequest()

        #expect(request?.messages.map(\.role) == [.user, .assistant, .user])
        #expect(request?.messages.first?.content == "first question")
        #expect(request?.messages.last?.content == "follow-up")
        #expect(model.outputConversationPlainText.contains("first answer"))
    }

    @Test func chatArchiveRoundTripKeepsDocumentContext() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppChatTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("model.gturbo", isDirectory: true)
        let chat = AppChat(
            title: "Research",
            messages: [
                AppChatMessage(
                    role: .user,
                    content: "Summarize report.pdf",
                    contextContent: "Document text\n\nSummarize it"),
                AppChatMessage(role: .assistant, content: "Summary"),
            ],
            draftAttachments: [
                AppPromptAttachment(
                    fileName: "table.xlsx",
                    formatLabel: "Excel",
                    extractedText: "A1: Revenue"),
            ])
        let archive = AppChatArchive(selectedChatID: chat.id, chats: [chat])

        try AppChatFileStore.save(archive, forModelDirectory: modelDirectory)
        let loaded = AppChatFileStore.loadOrCreate(forModelDirectory: modelDirectory)

        #expect(loaded == archive)
    }

    @Test func chatArchiveRoundTripKeepsRollingSummaryBoundary() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppChatSummaryTests-\(UUID().uuidString)",
                                    isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent(
            "model.gturbo",
            isDirectory: true)
        let user = AppChatMessage(role: .user, content: "old question")
        let assistant = AppChatMessage(role: .assistant, content: "old answer")
        let chat = AppChat(
            title: "Summary",
            messages: [user, assistant],
            contextSummary: "The user chose option A.",
            summarizedThroughMessageID: assistant.id)
        let archive = AppChatArchive(
            selectedChatID: chat.id,
            chats: [chat])

        try AppChatFileStore.save(
            archive,
            forModelDirectory: modelDirectory)

        #expect(AppChatFileStore.loadOrCreate(
            forModelDirectory: modelDirectory) == archive)
    }

    @Test func legacyArchiveWithoutSummaryFieldsStillLoads() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppChatLegacyTests-\(UUID().uuidString)",
                                    isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        let modelDirectory = root.appendingPathComponent(
            "model.gturbo",
            isDirectory: true)
        let chatID = UUID()
        let legacy = LegacyChatArchive(
            selectedChatID: chatID,
            chats: [LegacyChat(id: chatID)])
        try JSONEncoder().encode(legacy).write(
            to: AppChatFileStore.fileURL(
                forModelDirectory: modelDirectory))

        let loaded = AppChatFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)

        #expect(loaded.selectedChatID == chatID)
        #expect(loaded.chats.first?.contextSummary == nil)
        #expect(loaded.chats.first?.summarizedThroughMessageID == nil)
    }

    @Test func corruptArchiveIsQuarantinedInsteadOfDeleted() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppChatRecoveryTests-\(UUID().uuidString)",
                                    isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        let modelDirectory = root.appendingPathComponent(
            "model.gturbo",
            isDirectory: true)
        let archiveURL = AppChatFileStore.fileURL(
            forModelDirectory: modelDirectory)
        try Data("{not valid json".utf8).write(to: archiveURL)

        let result = AppChatFileStore.loadOrCreateWithRecovery(
            forModelDirectory: modelDirectory)

        let recoveryURL = try #require(result.recoveryURL)
        #expect(FileManager.default.fileExists(atPath: recoveryURL.path))
        #expect(String(
            data: try Data(contentsOf: recoveryURL),
            encoding: .utf8) == "{not valid json")
        #expect(result.archive.isValid)
        #expect(AppChatFileStore.loadOrCreate(
            forModelDirectory: modelDirectory) == result.archive)
    }

    @Test func persistenceCoordinatorNeverLetsAStaleDraftWin() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppChatWriterTests-\(UUID().uuidString)",
                                    isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent(
            "model.gturbo",
            isDirectory: true)
        let coordinator = AppChatPersistenceCoordinator(
            label: "AppChatWriterTests.\(UUID().uuidString)")
        let oldChat = AppChat(title: "Old", draft: "stale")
        let oldArchive = AppChatArchive(
            selectedChatID: oldChat.id,
            chats: [oldChat])
        let newChat = AppChat(title: "New", draft: "latest")
        let newArchive = AppChatArchive(
            selectedChatID: newChat.id,
            chats: [newChat])

        coordinator.save(
            revision: 1,
            archive: oldArchive,
            modelDirectory: modelDirectory,
            delay: 0.05,
            onFailure: { _ in })
        coordinator.save(
            revision: 2,
            archive: newArchive,
            modelDirectory: modelDirectory,
            delay: 0,
            onFailure: { _ in })
        try await Task.sleep(for: .milliseconds(100))

        #expect(AppChatFileStore.loadOrCreate(
            forModelDirectory: modelDirectory) == newArchive)
    }

    @MainActor
    @Test func appModelFlushRestoresMultipleChatsAndSelection() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppChatRestoreTests-\(UUID().uuidString)",
                                    isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent(
            "model.gturbo",
            isDirectory: true)
        let model = AppModel(
            modelDirectory: modelDirectory,
            settingsPersistenceEnabled: true)
        let firstChatID = model.selectedChatID
        model.promptText = "first draft"
        let secondChatID = model.createChat()
        model.promptText = "second draft"
        model.flushChatPersistence()

        let restored = AppModel(
            modelDirectory: modelDirectory,
            settingsPersistenceEnabled: true)

        #expect(restored.chats.count == 2)
        #expect(restored.selectedChatID == secondChatID)
        #expect(restored.promptText == "second draft")
        restored.selectChat(id: firstChatID)
        #expect(restored.promptText == "first draft")
        restored.flushChatPersistence()
    }

    @MainActor
    @Test func changingModelLocationKeepsIndependentChatArchives() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "AppChatModelIsolationTests-\(UUID().uuidString)",
                isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let firstModel = root
            .appendingPathComponent("first", isDirectory: true)
            .appendingPathComponent("model.gturbo", isDirectory: true)
        let secondModel = root
            .appendingPathComponent("second", isDirectory: true)
            .appendingPathComponent("model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: firstModel.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: secondModel.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let model = AppModel(
            modelDirectory: firstModel,
            client: MockInferenceClient(),
            settingsPersistenceEnabled: true)
        let firstChatID = model.selectedChatID
        model.promptText = "first-model draft"
        model.flushChatPersistence()

        model.setModelURL(secondModel)
        #expect(model.promptText.isEmpty)
        #expect(model.chats.count == 1)
        let secondChatID = model.selectedChatID
        model.promptText = "second-model draft"
        model.flushChatPersistence()

        model.setModelURL(firstModel)
        #expect(model.selectedChatID == firstChatID)
        #expect(model.promptText == "first-model draft")
        model.setModelURL(secondModel)
        #expect(model.selectedChatID == secondChatID)
        #expect(model.promptText == "second-model draft")
        model.flushChatPersistence()
    }

    @MainActor
    @Test func chatAndAttachmentMutationsAreBlockedDuringPreparation() async {
        let client = SlowRequestPreparationClient()
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        let otherChatID = model.createChat()
        let activeChatID = model.chats.last!.id
        model.selectChat(id: activeChatID)
        let attachment = AppPromptAttachment(
            fileName: "locked.pdf",
            formatLabel: "PDF",
            extractedText: "locked")
        model.addPromptAttachment(attachment)
        model.promptText = "pending"

        model.run()
        #expect(model.isRunning)
        #expect(model.createChat() == activeChatID)
        model.selectChat(id: otherChatID)
        model.renameChat(id: activeChatID, title: "Renamed")
        model.deleteChat(id: otherChatID)
        model.removePromptAttachment(id: attachment.id)
        model.clearPromptAttachments()
        model.clearOutput()

        #expect(model.selectedChatID == activeChatID)
        #expect(model.chats.count == 2)
        #expect(model.selectedChat.title == "New chat")
        #expect(model.promptAttachments == [attachment])
        #expect(model.promptText == "pending")

        model.cancel()
        await waitForIdle(model)
    }

    @MainActor
    @Test func clearingHistoryOnlyAffectsTheSelectedChatAndKeepsItsDraft() async {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        client.prefillSteps = 0
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.maxNewTokensOverride = 1

        let firstChatID = model.selectedChatID
        model.promptText = "first"
        model.run()
        await waitForIdle(model)

        let secondChatID = model.createChat()
        model.promptText = "second"
        model.run()
        await waitForIdle(model)
        model.promptText = "kept draft"
        model.clearOutput()

        #expect(model.selectedChatID == secondChatID)
        #expect(model.selectedChat.messages.isEmpty)
        #expect(model.promptText == "kept draft")
        #expect(!model.showsPromptExamples)

        model.selectChat(id: firstChatID)
        #expect(model.selectedChat.messages.count == 2)
        #expect(model.outputText.contains("answer"))
    }

    @MainActor
    @Test func completedChatsNeverShareModelContext() async {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        client.prefillSteps = 0
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.maxNewTokensOverride = 1

        let firstChatID = model.selectedChatID
        model.promptText = "alpha-only"
        model.run()
        await waitForIdle(model)

        let secondChatID = model.createChat()
        model.promptText = "beta-only"
        model.run()
        await waitForIdle(model)

        model.promptText = "second follow-up"
        let secondRequest = try? model.makeRequest()
        #expect(secondRequest?.messages.map(\.content).contains {
            $0.contains("beta-only")
        } == true)
        #expect(secondRequest?.messages.contains {
            $0.content.contains("alpha-only")
        } == false)

        model.selectChat(id: firstChatID)
        model.promptText = "first follow-up"
        let firstRequest = try? model.makeRequest()
        #expect(firstRequest?.messages.contains {
            $0.content.contains("alpha-only")
        } == true)
        #expect(firstRequest?.messages.contains {
            $0.content.contains("beta-only")
        } == false)
        #expect(secondChatID != firstChatID)
    }

    @MainActor
    @Test func preparationFailureKeepsDraftOutOfHistory() async {
        let client = RejectingRequestPreparationClient()
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.promptText = "keep this draft"

        model.run()
        await waitForIdle(model)

        #expect(model.promptText == "keep this draft")
        #expect(model.selectedChat.messages.isEmpty)
        guard case .contextOverflow = model.error else {
            Issue.record("Expected contextOverflow, got \(String(describing: model.error))")
            return
        }
    }

    @MainActor
    @Test func cancellingPreparationKeepsDraftAndReturnsToIdle() async {
        let client = SlowRequestPreparationClient()
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.promptText = "still editable"

        model.run()
        #expect(model.isRunning)
        model.cancel()
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(model.error == .cancelled)
        #expect(model.promptText == "still editable")
        #expect(model.selectedChat.messages.isEmpty)
    }

    @MainActor
    @Test func overflowingHistoryBecomesRollingMemoryWithoutHidingTranscript() async {
        let client = CompressionInferenceClient()
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.maxContextTokens = 512
        model.maxNewTokensOverride = 1
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)

        model.promptText = String(repeating: "alpha fact ", count: 120)
        model.addPromptAttachment(AppPromptAttachment(
            fileName: "facts.pdf",
            formatLabel: "PDF",
            extractedText: "DOCUMENT-MEMORY-MARKER: the launch date is Tuesday."))
        model.run()
        await waitForIdle(model)
        let firstAssistantID = model.selectedChat.messages.last?.id

        model.promptText = String(repeating: "beta detail ", count: 70)
        model.run()
        await waitForIdle(model)

        #expect(model.selectedChat.contextSummary == "compressed conversation memory")
        #expect(model.selectedChat.summarizedThroughMessageID == firstAssistantID)
        #expect(model.selectedChat.messages.count == 4)
        #expect(client.compressionGenerationCount() > 0)
        #expect(client.compressionRequestsContain("DOCUMENT-MEMORY-MARKER"))
        let firstCompressionCount = client.compressionGenerationCount()
        let secondAssistantID = model.selectedChat.messages.last?.id

        model.promptText = String(repeating: "gamma constraint ", count: 70)
        model.run()
        await waitForIdle(model)

        #expect(model.selectedChat.summarizedThroughMessageID == secondAssistantID)
        #expect(model.selectedChat.messages.count == 6)
        #expect(client.compressionGenerationCount() > firstCompressionCount)
        let lastMainRequest = client.lastMainRequest()
        #expect(lastMainRequest?.messages.first?.role == .system)
        #expect(lastMainRequest?.messages.first?.content.contains(
            "compressed conversation memory") == true)
        #expect(lastMainRequest?.messages.last?.content.contains(
            "gamma constraint") == true)

        let compressedChatID = model.selectedChatID
        let freshChatID = model.createChat()
        model.promptText = "fresh context"
        let freshRequest = try? model.makeRequest()
        #expect(freshChatID != compressedChatID)
        #expect(model.selectedChat.contextSummary == nil)
        #expect(freshRequest?.messages.map(\.role) == [.user])
        #expect(freshRequest?.messages.first?.content == "fresh context")

        model.selectChat(id: compressedChatID)
        #expect(model.selectedChat.contextSummary
            == "compressed conversation memory")
        model.clearOutput()
        #expect(model.selectedChat.messages.isEmpty)
        #expect(model.selectedChat.contextSummary == nil)
        #expect(model.selectedChat.summarizedThroughMessageID == nil)
    }

    @MainActor
    @Test func cancellingHistoryCompressionKeepsDraftAndExistingTranscript() async {
        let client = CompressionInferenceClient(
            compressionDelayNanoseconds: 5_000_000_000)
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.maxContextTokens = 512
        model.maxNewTokensOverride = 1
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)

        model.promptText = String(repeating: "earlier fact ", count: 120)
        model.run()
        await waitForIdle(model)
        #expect(model.selectedChat.messages.count == 2)

        let pendingDraft = String(repeating: "new detail ", count: 80)
        model.promptText = pendingDraft
        model.run()
        for _ in 0..<200 where model.phase != .compressing {
            try? await Task.sleep(nanoseconds: 1_000_000)
        }
        #expect(model.phase == .compressing)

        model.cancel()
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(model.error == .cancelled)
        #expect(model.promptText == pendingDraft)
        #expect(model.selectedChat.messages.count == 2)
        #expect(model.selectedChat.contextSummary == nil)
        #expect(model.selectedChat.summarizedThroughMessageID == nil)
    }

    @MainActor
    @Test func emptyCompressionResultPreservesHistoryAndPendingDraft() async {
        let client = CompressionInferenceClient(compressionResponse: "")
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.maxContextTokens = 512
        model.maxNewTokensOverride = 1
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)

        model.promptText = String(repeating: "earlier fact ", count: 120)
        model.run()
        await waitForIdle(model)
        let originalMessages = model.selectedChat.messages

        let pendingDraft = String(repeating: "new detail ", count: 80)
        model.promptText = pendingDraft
        model.run()
        await waitForIdle(model)

        #expect(model.promptText == pendingDraft)
        #expect(model.selectedChat.messages == originalMessages)
        #expect(model.selectedChat.contextSummary == nil)
        #expect(model.selectedChat.summarizedThroughMessageID == nil)
        guard case .unknown(let detail) = model.error else {
            Issue.record("Expected an unknown compression error")
            return
        }
        #expect(detail.contains("produced no text"))
    }

    @MainActor
    @Test func chatEndingInCancelledUserTurnKeepsEarlierAssistantInPlace() async {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        client.prefillSteps = 0
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.maxNewTokensOverride = 1
        model.promptText = "first"
        model.run()
        await waitForIdle(model)

        client.prefillSteps = 20
        client.tokenDelayNanos = 1_000_000
        model.promptText = "cancelled follow-up"
        model.run()
        for _ in 0..<200 where model.livePrefillDone == 0 {
            try? await Task.sleep(nanoseconds: 1_000_000)
        }
        model.cancel()
        await waitForIdle(model)
        let originalChatID = model.selectedChatID

        model.createChat()
        model.selectChat(id: originalChatID)

        #expect(model.outputText.isEmpty)
        #expect(model.transcriptBaseMessages.map(\.role) == [.user, .assistant, .user])
        #expect(model.transcriptBaseMessages.last?.content == "cancelled follow-up")
    }

    @MainActor
    private func waitForIdle(_ model: AppModel) async {
        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}

private final class RejectingRequestPreparationClient:
    AppGenerationRequestPreparing, @unchecked Sendable {
    func prepare(_ request: AppGenerationRequest) async throws
        -> AppGenerationRequest {
        throw AppInferenceError.contextOverflow(
            prompt: 10,
            maxNew: request.maxNewTokens,
            maxContext: 8)
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }

    func cancel() {}
}

private final class SlowRequestPreparationClient:
    AppGenerationRequestPreparing, @unchecked Sendable {
    func prepare(_ request: AppGenerationRequest) async throws
        -> AppGenerationRequest {
        try await Task.sleep(for: .seconds(5))
        return request
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }

    func cancel() {}
}

private final class CompressionInferenceClient:
    AppGenerationContextReporting, @unchecked Sendable {
    private let lock = NSLock()
    private let compressionDelayNanoseconds: UInt64
    private let compressionResponse: String
    private var compressionCount = 0
    private var compressionRequests: [AppGenerationRequest] = []
    private var mainRequests: [AppGenerationRequest] = []
    private var delayedCompressionTask: Task<Void, Never>?

    init(
        compressionDelayNanoseconds: UInt64 = 0,
        compressionResponse: String = "compressed conversation memory"
    ) {
        self.compressionDelayNanoseconds = compressionDelayNanoseconds
        self.compressionResponse = compressionResponse
    }

    func prepare(_ request: AppGenerationRequest) async throws
        -> AppGenerationRequest {
        try await prepareWithContextReport(request).request
    }

    func prepareWithContextReport(_ request: AppGenerationRequest) async throws
        -> AppPreparedGenerationRequest {
        let fit = try AppGenerationContextWindow.fit(
            messages: request.messages,
            maximumPromptTokens: request.maxContextTokens,
            maxNewTokens: request.maxNewTokens
        ) { messages in
            4 + messages.reduce(0) {
                $0 + max(1, $1.content.count / 4)
            }
        }
        var prepared = request
        prepared.messages = fit.messages
        return AppPreparedGenerationRequest(
            request: prepared,
            removedMessages: Array(
                request.messages.prefix(fit.removedMessageCount)))
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        let isCompression = request.messages.last?.content.contains(
            "Update a compact memory") == true
        lock.lock()
        if isCompression {
            compressionCount += 1
            compressionRequests.append(request)
        } else {
            mainRequests.append(request)
        }
        lock.unlock()
        let response = isCompression
            ? compressionResponse
            : "normal answer"

        return AsyncThrowingStream { [weak self] continuation in
            guard isCompression,
                  let self,
                  compressionDelayNanoseconds > 0 else {
                Self.finish(
                    continuation,
                    response: response,
                    request: request)
                return
            }

            let task = Task { [weak self] in
                do {
                    try await Task.sleep(
                        nanoseconds: self?.compressionDelayNanoseconds ?? 0)
                    try Task.checkCancellation()
                    Self.finish(
                        continuation,
                        response: response,
                        request: request)
                } catch {
                    continuation.finish(throwing: CancellationError())
                }
                self?.clearDelayedCompressionTask()
            }
            lock.lock()
            delayedCompressionTask = task
            lock.unlock()
            continuation.onTermination = { [weak self] _ in
                self?.cancelDelayedCompressionTask()
            }
        }
    }

    func cancel() {
        cancelDelayedCompressionTask()
    }

    func compressionGenerationCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return compressionCount
    }

    func lastMainRequest() -> AppGenerationRequest? {
        lock.lock()
        defer { lock.unlock() }
        return mainRequests.last
    }

    func compressionRequestsContain(_ text: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return compressionRequests.contains { request in
            request.messages.contains { $0.content.contains(text) }
        }
    }

    private static func finish(
        _ continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation,
        response: String,
        request: AppGenerationRequest
    ) {
        continuation.yield(.token(AppTokenEvent(
            index: 0,
            textDelta: response,
            elapsedDecodeSeconds: 0.001)))
        continuation.yield(.finished(AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            promptTokenCount: 1,
            prefillSeconds: 0.001,
            timeToFirstTokenSeconds: 0.001,
            decodeSeconds: 0.001,
            tokensPerSecond: 1_000,
            peakMemoryBytes: nil,
            runtimeOptions: request.runtimeOptions)))
        continuation.finish()
    }

    private func cancelDelayedCompressionTask() {
        let task: Task<Void, Never>?
        lock.lock()
        task = delayedCompressionTask
        delayedCompressionTask = nil
        lock.unlock()
        task?.cancel()
    }

    private func clearDelayedCompressionTask() {
        lock.lock()
        delayedCompressionTask = nil
        lock.unlock()
    }
}

private struct LegacyChatArchive: Codable {
    var version = 1
    var selectedChatID: UUID
    var chats: [LegacyChat]
}

private struct LegacyChat: Codable {
    var id: UUID
    var title = "Legacy"
    var messages: [AppChatMessage] = []
    var draft = "restored draft"
    var draftAttachments: [AppPromptAttachment] = []
    var createdAt = Date()
    var updatedAt = Date()
}
