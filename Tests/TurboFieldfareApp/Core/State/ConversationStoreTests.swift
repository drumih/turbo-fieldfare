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

    @Test func pinnedConversationsSortFirst() {
        let (store, _) = makeStore()
        let olderPinned = conversation(title: "pinned old",
                                       updatedAt: Date().addingTimeInterval(-3600))
            .replacing(isPinned: true)
        let recent = conversation(title: "recent", updatedAt: Date())

        store.upsert(recent)
        store.upsert(olderPinned)

        #expect(store.conversations.map(\.title) == ["pinned old", "recent"])
    }

    @Test func exportAndImportRoundTrip() throws {
        let (store, _) = makeStore()
        store.upsert(conversation(title: "a", promptTokenCount: 1))
        store.upsert(conversation(title: "b"))

        let data = try store.exportJSON()

        let (otherStore, _) = makeStore()
        let count = try otherStore.importJSON(data)

        #expect(count == 2)
        #expect(otherStore.conversations.map(\.title) == store.conversations.map(\.title))
        #expect(otherStore.conversations.first { $0.title == "a" }?.promptTokenCount == 1)
    }

    @Test func importReplacesSameId() throws {
        let (store, _) = makeStore()
        store.upsert(conversation(title: "original"))

        // An exported file with the same id but a renamed title.
        let renamed = store.conversations[0].replacing(title: "renamed")
        let (otherStore, _) = makeStore()
        otherStore.upsert(renamed)
        let data = try otherStore.exportJSON()

        let count = try store.importJSON(data)

        #expect(count == 1)
        #expect(store.conversations.count == 1)
        #expect(store.conversations[0].title == "renamed")
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

    @Test func decodesLegacyFileWithoutAttachments() throws {
        // History files written before document attachments existed have no
        // "attachments" key; they must still load with an empty list.
        let legacyJSON = """
        [
          {
            "id": "\(UUID().uuidString)",
            "title": "legacy",
            "prompt": "old prompt",
            "response": "old response",
            "createdAt": "2026-01-01T10:00:00Z",
            "updatedAt": "2026-01-01T10:00:00Z"
          }
        ]
        """.data(using: .utf8)!
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("legacy-\(UUID().uuidString).json")
        try legacyJSON.write(to: url)

        let store = ConversationStore(storageURL: url)

        #expect(store.conversations.count == 1)
        #expect(store.conversations[0].title == "legacy")
        #expect(store.conversations[0].attachments.isEmpty)
    }

    @Test func decodesFractionalSecondDates() throws {
        // Files written by other tools may use fractional-second timestamps;
        // they must still load (whole-second ones are produced by the app).
        let json = """
        [
          {
            "id": "\(UUID().uuidString)",
            "title": "fractional",
            "prompt": "p",
            "response": "r",
            "createdAt": "2026-01-01T10:00:00.123Z",
            "updatedAt": "2026-01-01T10:00:00Z"
          }
        ]
        """.data(using: .utf8)!
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("fractional-\(UUID().uuidString).json")
        try json.write(to: url)

        let store = ConversationStore(storageURL: url)

        #expect(store.conversations.count == 1)
        #expect(store.conversations[0].title == "fractional")
    }

    @Test func persistsAttachmentsAcrossStoreInstances() {
        let (store, url) = makeStore()
        let attachment = DocumentAttachment(filename: "report.pdf",
                                            type: .pdf,
                                            fileSize: 2048,
                                            extractedText: "report body",
                                            pageCount: 3,
                                            truncated: false,
                                            originalLength: 11)
        store.upsert(Conversation(title: "with docs",
                                  prompt: "p",
                                  response: "r",
                                  attachments: [attachment]))

        let reloaded = ConversationStore(storageURL: url)

        #expect(reloaded.conversations.count == 1)
        let restored = reloaded.conversations[0].attachments
        #expect(restored.count == 1)
        #expect(restored[0].filename == "report.pdf")
        #expect(restored[0].type == .pdf)
        #expect(restored[0].pageCount == 3)
        #expect(restored[0].extractedText == "report body")
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
