import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite @MainActor struct ConversationStoreNewTests {
    private struct TestAttachment: Codable {
        let filename: String
    }

    private func makeStore(fileName: String = "conversations-\(UUID().uuidString).json") -> (ConversationStore, URL) {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        return (ConversationStore(storageURL: url), url)
    }

    @Test func persistsMaxNewTokensAcrossInstances() {
        let (store, url) = makeStore()
        store.upsert(Conversation(title: "t", prompt: "p", response: "r", maxNewTokens: 128))

        let reloaded = ConversationStore(storageURL: url)
        #expect(reloaded.conversations[0].maxNewTokens == 128)
    }

    @Test func corruptedFileIsPreservedAndFlagSet() throws {
        let (store, url) = makeStore()
        store.upsert(Conversation(title: "t", prompt: "p", response: "r"))

        try "!not json".data(using: .utf8)!.write(to: url)

        let recovered = ConversationStore(storageURL: url)
        #expect(recovered.didLoadFromCorrupted)
        #expect(recovered.conversations.isEmpty)
        #expect(FileManager.default.fileExists(atPath: url.appendingPathExtension("corrupted").path))
    }

    @Test func preservedCorruptedFileVisible() throws {
        let (store, url) = makeStore()
        _ = store
        try "!corrupt".data(using: .utf8)!.write(to: url)
        let recovered = ConversationStore(storageURL: url)
        // The .corrupted backup exists so the original data is not lost.
        #expect(recovered.didLoadFromCorrupted)
        #expect(FileManager.default.fileExists(atPath: url.appendingPathExtension("corrupted").path))
    }

    @Test func fractionalDatesLoad() throws {
        let json = """
        [{"id": "\(UUID().uuidString)", "title": "t", "prompt": "p", "response": "r",
          "createdAt": "2026-01-01T10:00:00.123Z", "updatedAt": "2026-01-01T10:00:00Z"}]
        """.data(using: .utf8)!
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fractional-\(UUID().uuidString).json")
        try json.write(to: url)

        let store = ConversationStore(storageURL: url)
        #expect(store.conversations.count == 1)
        #expect(store.conversations[0].title == "t")
    }
}
