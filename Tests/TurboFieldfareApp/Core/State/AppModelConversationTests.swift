import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite @MainActor struct AppModelConversationTests {
    /// Creates an AppModel backed by a store on a unique temporary file,
    /// so history tests never touch the real Application Support data.
    private func makeModel() -> (AppModel, URL) {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("history-\(UUID().uuidString).json")
        let store = ConversationStore(storageURL: url)
        return (AppModel(conversationStore: store), url)
    }

    private func attachment(filename: String = "notes.txt") -> DocumentAttachment {
        DocumentAttachment(filename: filename,
                           type: .txt,
                           fileSize: 10,
                           extractedText: "document body")
    }

    @Test
    func saveStoresConversationWithDiagnostics() {
        let (model, _) = makeModel()
        model.promptText = "Explain Metal"
        model.outputText = "Metal is a GPU API."

        model.saveCurrentConversation(diagnostic: makeDiagnostics())

        let saved = model.conversationStore.conversations[0]
        #expect(saved.prompt == "Explain Metal")
        #expect(saved.response == "Metal is a GPU API.")
        #expect(saved.promptTokenCount == 12)
        #expect(saved.generatedTokenCount == 34)
        #expect(saved.stopReason == "maxTokens")
        #expect(model.activeConversationID == saved.id)
    }

    @Test
    func saveIgnoresEmptyPrompt() {
        let (model, _) = makeModel()
        model.promptText = "   "
        model.outputText = "ignored"

        model.saveCurrentConversation()

        #expect(model.conversationStore.conversations.isEmpty)
        #expect(model.activeConversationID == nil)
    }

    @Test
    func regeneratingSamePromptUpdatesActiveConversation() {
        let (model, _) = makeModel()
        model.promptText = "same prompt"
        model.outputText = "first response"
        model.saveCurrentConversation()
        let firstID = model.activeConversationID
        let firstCreatedAt = model.conversationStore.conversations[0].createdAt

        // Regenerate the same prompt: same entry, updated response, no duplicate.
        model.outputText = "second response"
        model.saveCurrentConversation()

        #expect(model.activeConversationID == firstID)
        #expect(model.conversationStore.conversations.count == 1)
        #expect(model.conversationStore.conversations[0].response == "second response")
        #expect(model.conversationStore.conversations[0].createdAt == firstCreatedAt)
        #expect(model.conversationStore.conversations[0].updatedAt >= firstCreatedAt)
    }

    @Test
    func differentPromptCreatesNewConversation() {
        let (model, _) = makeModel()
        model.promptText = "first prompt"
        model.saveCurrentConversation()
        let firstID = model.activeConversationID

        model.promptText = "second prompt"
        model.saveCurrentConversation()

        #expect(model.activeConversationID != firstID)
        #expect(model.conversationStore.conversations.count == 2)
    }

    @Test
    func saveStoresAttachmentsWithConversation() {
        let (model, _) = makeModel()
        model.promptText = "Summarize this"
        model.attachments = [attachment()]

        model.saveCurrentConversation()

        let saved = model.conversationStore.conversations[0]
        #expect(saved.attachments.count == 1)
        #expect(saved.attachments[0].extractedText == "document body")
    }

    @Test
    func loadRestoresPromptResponseAndAttachments() {
        let (model, _) = makeModel()
        let stored = Conversation(title: "t",
                                  prompt: "original prompt",
                                  response: "original response",
                                  attachments: [attachment()])
        model.conversationStore.upsert(stored)

        model.loadConversation(stored)

        #expect(model.activeConversationID == stored.id)
        #expect(model.promptText == "original prompt")
        #expect(model.outputText == "original response")
        #expect(model.attachments == stored.attachments)
        #expect(model.diagnostics == nil)
        #expect(model.error == nil)
    }

    @Test
    func loadIsIgnoredWhileRunning() {
        let (model, _) = makeModel()
        let stored = Conversation(title: "t", prompt: "p", response: "r")
        model.conversationStore.upsert(stored)
        model.promptText = "in-progress text"
        model.runState = .running

        model.loadConversation(stored)

        // The running prompt must not be clobbered.
        #expect(model.promptText == "in-progress text")
    }

    @Test
    func newConversationClearsEverything() {
        let (model, _) = makeModel()
        model.promptText = "old prompt"
        model.outputText = "old response"
        model.attachments = [attachment()]
        model.saveCurrentConversation()
        #expect(model.activeConversationID != nil)

        model.newConversation()

        #expect(model.activeConversationID == nil)
        #expect(model.promptText.isEmpty)
        #expect(model.outputPromptText.isEmpty)
        #expect(model.outputText.isEmpty)
        #expect(model.attachments.isEmpty)
    }

    @Test
    func deleteConversationRemovesAndResetsWhenActive() {
        let (model, _) = makeModel()
        model.promptText = "to delete"
        model.outputText = "response"
        model.saveCurrentConversation()
        let saved = model.conversationStore.conversations[0]

        model.deleteConversation(saved)

        #expect(model.conversationStore.conversations.isEmpty)
        #expect(model.activeConversationID == nil)
        #expect(model.promptText.isEmpty)
    }

    @Test
    func deleteInactiveConversationKeepsCurrent() {
        let (model, _) = makeModel()
        model.promptText = "active"
        model.outputText = "r"
        model.saveCurrentConversation()
        let active = model.conversationStore.conversations[0]

        let other = Conversation(title: "other", prompt: "other", response: "r")
        model.conversationStore.upsert(other)
        model.deleteConversation(other)

        #expect(model.conversationStore.conversations == [active])
        #expect(model.activeConversationID == active.id)
        #expect(model.promptText == "active")
    }

    @Test
    func historySurvivesModelRecreation() throws {
        let (model, url) = makeModel()
        model.promptText = "persisted prompt"
        model.outputText = "persisted response"
        model.attachments = [attachment(filename: "report.txt")]
        model.saveCurrentConversation()

        // A fresh model over the same store file must restore the history,
        // including attachments, and load them back into the composer.
        let reloadedStore = ConversationStore(storageURL: url)
        let reloaded = AppModel(conversationStore: reloadedStore)
        #expect(reloaded.conversationStore.conversations.count == 1)
        let restored = reloaded.conversationStore.conversations[0]
        #expect(restored.prompt == "persisted prompt")
        #expect(restored.attachments.count == 1)

        reloaded.loadConversation(restored)
        #expect(reloaded.promptText == "persisted prompt")
        #expect(reloaded.attachments[0].filename == "report.txt")
    }

    @Test
    func renameUpdatesTitle() {
        let (model, _) = makeModel()
        model.promptText = "prompt"
        model.outputText = "response"
        model.saveCurrentConversation()
        let saved = model.conversationStore.conversations[0]

        model.renameConversation(saved, title: "  Custom title  ")

        #expect(model.conversationStore.conversations[0].title == "Custom title")
        #expect(model.conversationStore.conversations[0].id == saved.id)
    }

    @Test
    func renameIgnoresBlankTitle() {
        let (model, _) = makeModel()
        model.promptText = "prompt"
        model.outputText = "response"
        model.saveCurrentConversation()
        let saved = model.conversationStore.conversations[0]

        model.renameConversation(saved, title: "   ")

        #expect(model.conversationStore.conversations[0].title == "prompt")
    }

    @Test
    func togglePinFlipsConversation() {
        let (model, _) = makeModel()
        model.promptText = "p"
        model.outputText = "r"
        model.saveCurrentConversation()
        let saved = model.conversationStore.conversations[0]
        #expect(!saved.isPinned)

        model.togglePin(saved)
        #expect(model.conversationStore.conversations[0].isPinned)

        model.togglePin(model.conversationStore.conversations[0])
        #expect(!model.conversationStore.conversations[0].isPinned)
    }

    @Test
    func exportAndImportThroughModel() throws {
        let (model, _) = makeModel()
        model.promptText = "export me"
        model.outputText = "done"
        model.saveCurrentConversation()

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("export-\(UUID().uuidString).json")
        try model.exportConversations(to: url)

        let (otherModel, _) = makeModel()
        let count = try otherModel.importConversations(from: url)

        #expect(count == 1)
        #expect(otherModel.conversationStore.conversations[0].prompt == "export me")
    }

    /// Minimal diagnostics for token-count assertions.
    private func makeDiagnostics() -> AppDiagnostics {
        AppDiagnostics(generatedTokens: 34,
                       stopReason: .maxTokens,
                       promptTokenCount: 12,
                       timeToFirstTokenSeconds: 0,
                       decodeSeconds: 0,
                       tokensPerSecond: 0,
                       peakMemoryBytes: nil,
                       runtimeOptions: AppRuntimeOptions())
    }
}
