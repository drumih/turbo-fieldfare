import Foundation

public struct AppChatThread: Codable, Equatable, Identifiable, Sendable {
    public let id: UUID
    public var title: String
    public var systemPrompt: String
    public var messages: [AppChatMessage]
    public let createdAt: Date
    public var updatedAt: Date

    public init(id: UUID = UUID(),
                title: String = "New chat",
                systemPrompt: String = "",
                messages: [AppChatMessage] = [],
                createdAt: Date = Date(),
                updatedAt: Date = Date()) {
        self.id = id
        self.title = title
        self.systemPrompt = systemPrompt
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    public var preview: String {
        guard let message = messages.last else { return "" }
        let content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
        if !content.isEmpty { return content }
        return message.images.isEmpty ? "" : "Image"
    }
}

struct AppChatHistoryDocument: Codable, Equatable, Sendable {
    static let currentVersion = 1

    var version: Int = currentVersion
    var selectedChatID: UUID
    var chats: [AppChatThread]

    static func fresh() -> AppChatHistoryDocument {
        let chat = AppChatThread()
        return AppChatHistoryDocument(selectedChatID: chat.id, chats: [chat])
    }

    func isValid() -> Bool {
        version == Self.currentVersion
            && !chats.isEmpty
            && Set(chats.map(\.id)).count == chats.count
            && chats.contains(where: { $0.id == selectedChatID })
            && chats.allSatisfy { chat in
                let attachmentDirectory = chat.id.uuidString.lowercased()
                return !chat.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    && !chat.messages.contains(where: { message in
                        message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || message.images.count > 1
                            || !message.images.allSatisfy(\.hasValidStoredMetadata)
                            || (!message.images.isEmpty && message.role != .user)
                            || message.images.contains { image in
                                guard let imageDirectory = image.relativePath
                                    .split(separator: "/").first else { return true }
                                return String(imageDirectory) != attachmentDirectory
                            }
                    })
            }
    }
}

enum AppChatHistoryFileStore {
    private static let fileName = "mac-app-chat-history.json"

    static func load(forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) -> AppChatHistoryDocument {
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        guard fileManager.fileExists(atPath: fileURL.path),
              let data = try? Data(contentsOf: fileURL),
              let document = try? JSONDecoder().decode(AppChatHistoryDocument.self, from: data),
              document.isValid() else {
            return .fresh()
        }
        return document
    }

    static func save(_ document: AppChatHistoryDocument,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        guard document.isValid() else { throw InvalidHistory() }
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(document)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(fileName, isDirectory: false)
    }

    private struct InvalidHistory: Error {}
}
