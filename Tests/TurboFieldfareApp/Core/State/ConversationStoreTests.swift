import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite @MainActor struct ConversationStoreTests {
    /// Creates an isolated store backed by a unique temporary file.
    private func makeStore() -> (ConversationStore, URL) {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("conversations-\(UUID().uuidString).json")
        return (ConversationStore(storageURL: url), url)
    }

    private func conversation(title: String,
                              updatedAt: Date = Date(),
                              promptTokenCount: Int? = nil,
                              generatedTokenCount: Int? = nil,
                              stopReason: String? = nil) -> Conversation {
        Conversation(id: UUID(),
                     title: title,
                     prompt: "Prompt for \(title)",
                     response: "Response for \(title)",
                     updatedAt: updatedAt,
                     promptTokenCount: promptTokenCount,
                     generatedTokenCount: generatedTokenCount,
                     stopReason: stopReason)
    }

    @Test func insertsNewConversationAtTop() {
        let (store, _) = makeStore()
        let older = conversation(title: "older", updatedAt: Date().addingTimeInterval(-60))
        let newer = conversation(title: "newer", updatedAt: Date())

        store.upsert(older)
        store.upsert(newer)

        #expect(store.conversations.count == 2)
        #expect(store.conversations.map(\.title) == ["newer", "older"])
    }

    @Test func upsertReplacesExistingConversation() {
        let (store, _) = makeStore()
        let id = UUID()
        let first = Conversation(id: id, title: "first", prompt: "p1", response: "r1")
        store.upsert(first)

        let second = Conversation(id: id, title: "second", prompt: "p2", response: "r2",
                                  updatedAt: Date().addingTimeInterval(10))
        store.upsert(second)

        #expect(store.conversations.count == 1)
        #expect(store.conversations.first?.title == "second")
        #expect(store.conversations.first?.prompt == "p2")
        #expect(store.conversations.first?.response == "r2")
    }

    @Test func deleteRemovesConversation() {
        let (store, _) = makeStore()
        let a = conversation(title: "a")
        let b = conversation(title: "b")
        store.upsert(a)
        store.upsert(b)

        store.delete(a)

        #expect(store.conversations.count == 1)
        #expect(store.conversations.first?.title == "b")
    }

    @Test func clearAllEmptiesStore() {
        let (store, _) = makeStore()
        store.upsert(conversation(title: "a"))
        store.upsert(conversation(title: "b"))

        store.clearAll()

        #expect(store.conversations.isEmpty)
    }

    @Test func persistsAcrossStoreInstances() {
        let (store, url) = makeStore()
        let saved = conversation(title: "persisted",
                                 promptTokenCount: 12,
                                 generatedTokenCount: 34,
                                 stopReason: "length")
        store.upsert(saved)

        // A fresh store over the same file must reload the conversation.
        let reloaded = ConversationStore(storageURL: url)
        #expect(reloaded.conversations.count == 1)
        let restored = reloaded.conversations[0]
        #expect(restored.id == saved.id)
        #expect(restored.title == "persisted")
        #expect(restored.prompt == "Prompt for persisted")
        #expect(restored.response == "Response for persisted")
        #expect(restored.promptTokenCount == 12)
        #expect(restored.generatedTokenCount == 34)
        #expect(restored.stopReason == "length")
    }

    @Test func missingFileLoadsEmptyStore() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("does-not-exist-\(UUID().uuidString).json")
        let store = ConversationStore(storageURL: url)
        #expect(store.conversations.isEmpty)
    }

    @Test func titleFromEmptyPromptUsesPlaceholder() {
        #expect(Conversation.title(from: "   \n  ") == "New conversation")
    }

    @Test func titleKeepsShortPrompt() {
        #expect(Conversation.title(from: "The capital of France") == "The capital of France")
    }

    @Test func titleTruncatesLongPrompt() {
        let long = String(repeating: "x", count: 120)
        let title = Conversation.title(from: long)
        #expect(title.count == 61)  // 60 characters + ellipsis
        #expect(title.hasSuffix("…"))
        #expect(title.dropLast() == String(repeating: "x", count: 60))
    }

    @Test func responsePreviewTruncatesLongResponse() {
        let long = String(repeating: "y", count: 250)
        let preview = Conversation(title: "t", prompt: "p", response: long).responsePreview
        #expect(preview.count == 101)  // 100 characters + ellipsis
        #expect(preview.hasSuffix("…"))
    }

    @Test func responsePreviewKeepsShortResponse() {
        let preview = Conversation(title: "t", prompt: "p", response: "short").responsePreview
        #expect(preview == "short")
    }
}
