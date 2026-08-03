import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppChatHistoryTests {
    @Test func messageWithoutImagesDecodesAsLegacyTextMessage() throws {
        let data = Data(#"{"role":"user","content":"legacy question"}"#.utf8)

        let message = try JSONDecoder().decode(AppChatMessage.self, from: data)

        #expect(message == AppChatMessage(role: .user, content: "legacy question"))
        #expect(message.images.isEmpty)
    }

    @Test func historyFileLivesBesideModelDirectory() {
        let model = URL(fileURLWithPath: "/tmp/TurboFieldfare/gemma4.gturbo",
                        isDirectory: true)

        #expect(AppChatHistoryFileStore.fileURL(forModelDirectory: model).path
            == "/tmp/TurboFieldfare/mac-app-chat-history.json")
    }

    @Test func savedHistoryRoundTripsWithTheSelectedChat() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let timestamp = Date(timeIntervalSinceReferenceDate: 1_234_567.5)
        let older = AppChatThread(
            title: "Older chat",
            messages: [AppChatMessage(role: .user, content: "Earlier question")],
            createdAt: timestamp,
            updatedAt: timestamp)
        let selected = AppChatThread(
            title: "Selected chat",
            systemPrompt: "Be concise.",
            messages: [
                AppChatMessage(role: .user, content: "Current question"),
                AppChatMessage(role: .assistant, content: "Current answer"),
            ],
            createdAt: timestamp,
            updatedAt: timestamp)
        let history = AppChatHistoryDocument(
            selectedChatID: selected.id,
            chats: [selected, older])

        try AppChatHistoryFileStore.save(history, forModelDirectory: model)
        let loaded = AppChatHistoryFileStore.load(forModelDirectory: model)

        #expect(loaded == history)
    }

    @Test func savedHistoryRoundTripsImageMetadataWithoutImageBytes() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let chatID = UUID()
        let image = AppImageAttachment(
            relativePath: "\(chatID.uuidString.lowercased())/image.png",
            originalFilename: "diagram.png",
            mediaTypeIdentifier: "public.png",
            pixelWidth: 640,
            pixelHeight: 480,
            byteCount: 1_024,
            sha256: String(repeating: "a", count: 64))
        let chat = AppChatThread(
            id: chatID,
            title: "Image question",
            messages: [
                AppChatMessage(
                    role: .user,
                    content: "What is shown?",
                    images: [image]),
            ])
        let history = AppChatHistoryDocument(
            selectedChatID: chat.id,
            chats: [chat])

        try AppChatHistoryFileStore.save(history, forModelDirectory: model)
        let loaded = AppChatHistoryFileStore.load(forModelDirectory: model)
        let historyData = try Data(contentsOf:
            AppChatHistoryFileStore.fileURL(forModelDirectory: model))

        #expect(loaded == history)
        #expect(historyData.count < 4_096)
        #expect(!String(decoding: historyData, as: UTF8.self)
            .contains("data:image"))
    }

    private func makeTemporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppChatHistoryTests-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        return root
    }
}
