import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ConversationMigrationTests {
    private func makeTree(_ name: String) throws -> (support: URL, modelDirectory: URL) {
        let support = FileManager.default.temporaryDirectory
            .appendingPathComponent("conv-migrate-\(name)-\(UUID().uuidString)",
                                    isDirectory: true)
        let modelDirectory = support
            .appendingPathComponent("models/owner--model/model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDirectory,
                                                withIntermediateDirectories: true)
        return (support, modelDirectory)
    }

    private func seedLegacyStore(at modelDirectory: URL, title: String) throws {
        let conversation = Conversation(
            title: title,
            turns: [
                ChatTurn(role: .user, content: "Hello"),
                ChatTurn(role: .assistant, content: "Hi"),
            ])
        try ConversationFileStore.save(
            ConversationStoreFile(conversations: [conversation]),
            forModelDirectory: modelDirectory)
    }

    @Test func chatTurnCarriesAnOptionalModelID() {
        let turn = ChatTurn(role: .assistant, content: "Hi", modelID: "owner/model")
        #expect(turn.modelID == "owner/model")
        #expect(ChatTurn(role: .user, content: "Hello").modelID == nil)
    }

    @Test func globalStoreRoundTripsThroughDisk() throws {
        let (support, _) = try makeTree("round-trip")
        defer { try? FileManager.default.removeItem(at: support) }

        let store = ConversationStoreFile(conversations: [
            Conversation(title: "Kept", turns: [
                ChatTurn(role: .user, content: "Hello", modelID: "owner/model"),
            ]),
        ])
        try ConversationFileStore.saveGlobal(store, inSupportDirectory: support)
        #expect(ConversationFileStore.loadGlobal(inSupportDirectory: support) == store)
    }

    @Test func migrationImportsLegacyStoreAndTagsTurns() throws {
        let (support, modelDirectory) = try makeTree("import")
        defer { try? FileManager.default.removeItem(at: support) }
        try seedLegacyStore(at: modelDirectory, title: "Legacy")

        let imported = try ConversationFileStore.migrate(
            fromModelDirectories: [modelDirectory],
            modelIDsByDirectory: [modelDirectory: "owner/model"],
            inSupportDirectory: support)
        #expect(imported == 1)

        let global = ConversationFileStore.loadGlobal(inSupportDirectory: support)
        #expect(global.conversations.count == 1)
        #expect(global.conversations[0].title == "Legacy")
        #expect(global.conversations[0].turns.allSatisfy { $0.modelID == "owner/model" })
    }

    @Test func migrationIsIdempotent() throws {
        let (support, modelDirectory) = try makeTree("idempotent")
        defer { try? FileManager.default.removeItem(at: support) }
        try seedLegacyStore(at: modelDirectory, title: "Legacy")

        let first = try ConversationFileStore.migrate(
            fromModelDirectories: [modelDirectory],
            modelIDsByDirectory: [modelDirectory: "owner/model"],
            inSupportDirectory: support)
        let second = try ConversationFileStore.migrate(
            fromModelDirectories: [modelDirectory],
            modelIDsByDirectory: [modelDirectory: "owner/model"],
            inSupportDirectory: support)

        #expect(first == 1)
        #expect(second == 0)
        #expect(ConversationFileStore.loadGlobal(inSupportDirectory: support)
            .conversations.count == 1)
    }

    @Test func migrationRenamesTheLegacyFile() throws {
        let (support, modelDirectory) = try makeTree("rename")
        defer { try? FileManager.default.removeItem(at: support) }
        try seedLegacyStore(at: modelDirectory, title: "Legacy")

        let legacyURL = ConversationFileStore.fileURL(forModelDirectory: modelDirectory)
        _ = try ConversationFileStore.migrate(
            fromModelDirectories: [modelDirectory],
            modelIDsByDirectory: [modelDirectory: "owner/model"],
            inSupportDirectory: support)

        #expect(FileManager.default.fileExists(atPath: legacyURL.path) == false)
        let renamed = legacyURL.deletingLastPathComponent()
            .appendingPathComponent(ConversationFileStore.migratedFileName)
        #expect(FileManager.default.fileExists(atPath: renamed.path))
    }

    @Test func migrationMergesMultipleModels() throws {
        let (support, firstModel) = try makeTree("merge")
        defer { try? FileManager.default.removeItem(at: support) }
        let secondModel = support
            .appendingPathComponent("models/owner--other/model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: secondModel,
                                                withIntermediateDirectories: true)
        try seedLegacyStore(at: firstModel, title: "From first")
        try seedLegacyStore(at: secondModel, title: "From second")

        let imported = try ConversationFileStore.migrate(
            fromModelDirectories: [firstModel, secondModel],
            modelIDsByDirectory: [firstModel: "owner/model", secondModel: "owner/other"],
            inSupportDirectory: support)

        #expect(imported == 2)
        let titles = Set(ConversationFileStore.loadGlobal(inSupportDirectory: support)
            .conversations.map(\.title))
        #expect(titles == ["From first", "From second"])
    }
}
