import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite @MainActor struct AppModelMaxNewTokensTests {
    private func makeModel() -> AppModel {
        AppModel(conversationStore: ConversationStore(storageURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("tokens-\(UUID().uuidString).json")))
    }

    @Test func saveStoresMaxNewTokens() {
        let model = makeModel()
        model.promptText = "prompt"
        model.outputText = "response"
        model.maxNewTokensOverride = 256
        model.saveCurrentConversation()
        #expect(model.conversationStore.conversations[0].maxNewTokens == 256)
    }

    @Test func loadRestoresMaxNewTokens() {
        let model = makeModel()
        let stored = Conversation(title: "t", prompt: "p", response: "r", maxNewTokens: 512)
        model.conversationStore.upsert(stored)

        model.loadConversation(stored)
        #expect(model.maxNewTokensOverride == 512)
    }

    @Test func clearAttachmentErrorsClearsAll() {
        let model = AppModel()
        model.attachmentErrors = ["a", "b"]
        model.clearAttachmentErrors()
        #expect(model.attachmentErrors.isEmpty)
    }

    @Test func dismissAttachmentErrorRemovesOnlySelected() {
        let model = AppModel()
        model.attachmentErrors = ["a", "b", "c"]
        model.dismissAttachmentError(at: 1)
        #expect(model.attachmentErrors == ["a", "c"])
    }

    @Test func cognitiveTranscriptIsEmptyForStandardRun() async throws {
        let model = makeModel()
        model.promptText = "plain prompt"
        model.outputText = "plain answer"
        // Standard run does not populate the cognitive transcript.
        #expect(model.cognitiveTranscript.isEmpty)
    }
}
