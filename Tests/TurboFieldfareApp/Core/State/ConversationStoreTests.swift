import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ConversationStoreTests {
    @Test func roundTripsThroughDisk() throws {
        let directory = try makeModelDirectory("round-trip")
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }

        let conversation = Conversation(
            title: "Chunked prefill",
            turns: [
                ChatTurn(role: .user, content: "Explain chunked prefill."),
                ChatTurn(role: .assistant, content: "It splits the prompt."),
            ],
            lastPromptTokenCount: 128)
        try ConversationFileStore.save(
            ConversationStoreFile(conversations: [conversation]),
            forModelDirectory: directory)

        let loaded = ConversationFileStore.load(forModelDirectory: directory)
        #expect(loaded.conversations == [conversation])
    }

    @Test func storeLivesBesideTheModelDirectory() {
        let directory = URL(fileURLWithPath: "/tmp/example/gemma4.gturbo", isDirectory: true)
        #expect(ConversationFileStore.fileURL(forModelDirectory: directory).path
            == "/tmp/example/conversations.json")
    }

    @Test func missingStoreLoadsEmptyWithoutCreatingAFile() throws {
        let directory = try makeModelDirectory("missing")
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }

        #expect(ConversationFileStore.load(forModelDirectory: directory).conversations.isEmpty)
        #expect(!FileManager.default.fileExists(
            atPath: ConversationFileStore.fileURL(forModelDirectory: directory).path))
    }

    /// Chat history is recreatable; a wedged app is not. A corrupt or
    /// future-version file is discarded rather than partially migrated.
    @Test func corruptStoreIsDiscardedRatherThanThrowing() throws {
        let directory = try makeModelDirectory("corrupt")
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let url = ConversationFileStore.fileURL(forModelDirectory: directory)
        try Data("{ not json".utf8).write(to: url)

        #expect(ConversationFileStore.load(forModelDirectory: directory).conversations.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test func futureVersionIsDiscarded() throws {
        let directory = try makeModelDirectory("future")
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let url = ConversationFileStore.fileURL(forModelDirectory: directory)
        try Data(#"{"version": 99, "conversations": []}"#.utf8).write(to: url)

        #expect(ConversationFileStore.load(forModelDirectory: directory).conversations.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test func duplicateIdentifiersAreRejected() {
        let id = UUID()
        let store = ConversationStoreFile(conversations: [
            Conversation(id: id),
            Conversation(id: id),
        ])
        #expect(!store.isValid())
    }

    @Test(arguments: [
        ("Explain chunked prefill", "Explain chunked prefill"),
        ("  spaced   out  words ", "spaced out words"),
        ("multi\nline\nprompt", "multi line prompt"),
        ("", Conversation.untitled),
        ("   ", Conversation.untitled),
    ])
    func derivesTitlesFromTheFirstUserTurn(_ input: String, _ expected: String) {
        #expect(Conversation.derivedTitle(from: input) == expected)
    }

    @Test func longTitlesAreClippedAtAWordBoundary() {
        let title = Conversation.derivedTitle(
            from: "Explain why chunked prefill reduces time to first token while bounding memory")
        #expect(title.hasSuffix("…"))
        #expect(title.count <= 41)
        #expect(!title.contains("  "))
    }

    @Test func renameSticksButDerivedTitlesTrackTheFirstTurn() {
        var conversation = Conversation()
        conversation.turns = [ChatTurn(role: .user, content: "first question")]
        conversation.retitleIfNeeded()
        #expect(conversation.title == "first question")

        conversation.title = "My chat"
        conversation.titleIsCustom = true
        conversation.retitleIfNeeded()
        #expect(conversation.title == "My chat")
    }

    private func makeModelDirectory(_ name: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("conversation-store-\(name)-\(UUID().uuidString)",
                                    isDirectory: true)
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        return modelDirectory
    }
}
