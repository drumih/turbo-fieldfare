import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite @MainActor struct AppModelMultiTurnTests {
    private func makeModel() -> AppModel {
        AppModel(conversationStore: ConversationStore(storageURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("multiturn-\(UUID().uuidString).json")))
    }

    @Test func conversationAsPromptFormatsTurns() {
        let conversation = Conversation(title: "t",
                                        turns: [
                                            Turn(role: .user, text: "first"),
                                            Turn(role: .model, text: "answer"),
                                            Turn(role: .user, text: "second"),
                                        ])

        let prompt = conversation.asPrompt()

        #expect(prompt.contains("User:\nfirst"))
        #expect(prompt.contains("Model:\nanswer"))
        #expect(prompt.contains("User:\nsecond"))
        #expect(prompt.hasSuffix("Model:"))
    }

    @Test func displayTranscriptLabelsTurns() {
        let conversation = Conversation(title: "t",
                                        turns: [
                                            Turn(role: .user, text: "hello"),
                                            Turn(role: .model, text: "hi there"),
                                        ])

        let transcript = conversation.displayTranscript

        #expect(transcript.contains("You:\nhello"))
        #expect(transcript.contains("Answer:\nhi there"))
    }

    @Test
    func continuationPromptIncludesHistoryAndNewMessage() {
        let model = makeModel()
        let stored = Conversation(title: "t",
                                  turns: [
                                      Turn(role: .user, text: "first"),
                                      Turn(role: .model, text: "answer"),
                                  ])
        model.loadConversation(stored)
        model.promptText = "second"

        let prompt = model.continuationPromptText

        #expect(prompt.contains("User:\nfirst"))
        #expect(prompt.contains("Model:\nanswer"))
        #expect(prompt.hasSuffix("User:\nsecond"))
    }

    @Test
    func continuationAddsAttachmentsToNewMessage() {
        let model = makeModel()
        let stored = Conversation(title: "t",
                                  turns: [Turn(role: .user, text: "hi", createdAt: Date())])
        model.loadConversation(stored)
        model.promptText = "summarize"
        model.attachments = [DocumentAttachment(filename: "f.txt",
                                                type: .txt,
                                                fileSize: 10,
                                                extractedText: "body")]

        let prompt = model.continuationPromptText

        #expect(prompt.contains("[Document: f.txt]"))
        #expect(prompt.contains("body"))
    }

    @Test
    func makeRequestUsesContinuationWhenConversationActive() throws {
        let model = makeModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        let stored = Conversation(title: "t",
                                  turns: [Turn(role: .user, text: "old", createdAt: Date())])
        model.loadConversation(stored)
        model.promptText = "new question"

        let request = try model.makeRequest()

        #expect(request.prompt.contains("User:\nold"))
        #expect(request.prompt.hasSuffix("User:\nnew question"))
    }

    @Test
    func legacySingleExchangeFileDecodesIntoTurns() throws {
        // A history file written before multi-turn support.
        let json = """
        [{
          "id": "\(UUID().uuidString)",
          "title": "legacy",
          "prompt": "old prompt",
          "response": "old answer",
          "createdAt": "2026-01-01T10:00:00Z",
          "updatedAt": "2026-01-01T10:00:00Z"
        }]
        """.data(using: .utf8)!
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("legacy-mt-\(UUID().uuidString).json")
        try json.write(to: url)

        let store = ConversationStore(storageURL: url)

        #expect(store.conversations.count == 1)
        #expect(store.conversations[0].turns.count == 2)
        #expect(store.conversations[0].firstPrompt == "old prompt")
        #expect(store.conversations[0].lastResponse == "old answer")
    }

    @Test
    func savingAfterContinuationAppendsTurns() {
        let model = makeModel()
        model.promptText = "turn one"
        model.outputText = "answer one"
        model.saveCurrentConversation()
        let id = model.activeConversationID
        #expect(model.conversationStore.conversations[0].turns.count == 2)

        // Continue the same conversation with a new message.
        model.promptText = "turn two"
        model.outputText = "answer two"
        model.saveCurrentConversation()

        #expect(model.activeConversationID == id)
        #expect(model.conversationStore.conversations.count == 1)
        let turns = model.conversationStore.conversations[0].turns
        #expect(turns.count == 4)
        #expect(turns[2].text == "turn two")
        #expect(turns[3].text == "answer two")
    }

    @Test
    func regeneratingLastPromptReplacesModelAnswer() {
        let model = makeModel()
        model.promptText = "same"
        model.outputText = "answer one"
        model.saveCurrentConversation()

        model.outputText = "answer two"
        model.saveCurrentConversation()

        let turns = model.conversationStore.conversations[0].turns
        #expect(turns.count == 2)
        #expect(turns[1].text == "answer two")
    }

    // MARK: - Fork

    @Test
    func forkCreatesLinkedCopyAsActiveConversation() {
        let (model, _) = AppModelConversationTestsHelper.makeModel()
        model.promptText = "original"
        model.outputText = "answer"
        model.saveCurrentConversation()
        let source = model.conversationStore.conversations[0]

        model.forkConversation(source)

        #expect(model.conversationStore.conversations.count == 2)
        let fork = model.conversationStore.conversations[0]
        #expect(fork.id != source.id)
        #expect(fork.parentConversationID == source.id)
        #expect(fork.title == "Fork of \(source.title)")
        #expect(fork.turns == source.turns)
        #expect(model.activeConversationID == fork.id)
    }

    @Test
    func forkUpToTurnTruncatesHistory() {
        let (model, _) = AppModelConversationTestsHelper.makeModel()
        model.promptText = "one"
        model.outputText = "answer one"
        model.saveCurrentConversation()
        model.promptText = "two"
        model.outputText = "answer two"
        model.saveCurrentConversation()
        let source = model.conversationStore.conversations[0]
        let secondUserTurn = source.turns[2]

        model.forkConversation(source, upTo: secondUserTurn)

        let fork = model.conversationStore.conversations[0]
        #expect(fork.turns.count == 3)
        #expect(fork.turns.last?.text == "two")
    }

    // MARK: - Markdown export

    @Test
    func exportMarkdownRendersTurns() {
        let (model, _) = AppModelConversationTestsHelper.makeModel()
        let conversation = Conversation(title: "My chat",
                                        turns: [
                                            Turn(role: .user, text: "hello"),
                                            Turn(role: .model, text: "hi"),
                                        ])

        let markdown = model.exportConversationMarkdown(conversation)

        #expect(markdown.hasPrefix("# My chat"))
        #expect(markdown.contains("## User\n\nhello"))
        #expect(markdown.contains("## Assistant\n\nhi"))
    }

    // MARK: - Templates & tags

    @Test
    func setTemplateMarksConversation() {
        let (model, _) = AppModelConversationTestsHelper.makeModel()
        let conversation = Conversation(title: "t", prompt: "p", response: "r")
        model.conversationStore.upsert(conversation)

        model.setTemplate(conversation, isTemplate: true)

        #expect(model.conversationStore.conversations[0].isTemplate)

        model.setTemplate(model.conversationStore.conversations[0], isTemplate: false)
        #expect(!model.conversationStore.conversations[0].isTemplate)
    }

    @Test
    func setTagsNormalizesAndDeduplicates() {
        let (model, _) = AppModelConversationTestsHelper.makeModel()
        let conversation = Conversation(title: "t", prompt: "p", response: "r")
        model.conversationStore.upsert(conversation)

        model.setTags(conversation, tags: [" work ", "code", "work", "  "])

        #expect(model.conversationStore.conversations[0].tags == ["work", "code"])
    }

    @Test
    func savingActiveConversationPreservesTemplateAndTags() {
        let (model, _) = AppModelConversationTestsHelper.makeModel()
        model.promptText = "prompt"
        model.outputText = "answer"
        model.saveCurrentConversation()
        let saved = model.conversationStore.conversations[0]
        model.setTemplate(saved, isTemplate: true)
        model.setTags(saved, tags: ["meta"])

        // Regenerate the same prompt: the flags must survive the upsert.
        model.outputText = "answer two"
        model.saveCurrentConversation()

        let updated = model.conversationStore.conversations[0]
        #expect(updated.isTemplate)
        #expect(updated.tags == ["meta"])
        #expect(updated.lastResponse == "answer two")
    }
}

/// Test helper sharing the isolated-store model factory.
enum AppModelConversationTestsHelper {
    @MainActor
    static func makeModel() -> (AppModel, URL) {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("helper-\(UUID().uuidString).json")
        return (AppModel(conversationStore: ConversationStore(storageURL: url)), url)
    }
}
