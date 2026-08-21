import Foundation

public struct AppChatMessage: Identifiable, Codable, Equatable, Sendable {
    public enum Role: String, Codable, Equatable, Sendable {
        case user
        case assistant
    }

    public let id: UUID
    public var role: Role
    public var content: String
    public var contextContent: String
    public var createdAt: Date

    public init(id: UUID = UUID(),
                role: Role,
                content: String,
                contextContent: String? = nil,
                createdAt: Date = Date()) {
        self.id = id
        self.role = role
        self.content = content
        self.contextContent = contextContent ?? content
        self.createdAt = createdAt
    }
}

public struct AppChat: Identifiable, Codable, Equatable, Sendable {
    public let id: UUID
    public var title: String
    public var messages: [AppChatMessage]
    public var draft: String
    public var draftAttachments: [AppPromptAttachment]
    public var contextSummary: String?
    public var summarizedThroughMessageID: AppChatMessage.ID?
    public let createdAt: Date
    public var updatedAt: Date

    public init(id: UUID = UUID(),
                title: String = "New chat",
                messages: [AppChatMessage] = [],
                draft: String = "",
                draftAttachments: [AppPromptAttachment] = [],
                contextSummary: String? = nil,
                summarizedThroughMessageID: AppChatMessage.ID? = nil,
                createdAt: Date = Date(),
                updatedAt: Date = Date()) {
        self.id = id
        self.title = title
        self.messages = messages
        self.draft = draft
        self.draftAttachments = draftAttachments
        self.contextSummary = contextSummary
        self.summarizedThroughMessageID = summarizedThroughMessageID
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    public var preview: String {
        if let lastMessage = messages.last {
            return lastMessage.content
                .replacingOccurrences(of: "\n", with: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if !draft.isEmpty {
            return draft
                .replacingOccurrences(of: "\n", with: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return "No messages yet"
    }

    var hasValidSummaryBoundary: Bool {
        let summary = contextSummary?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if summary.isEmpty {
            return summarizedThroughMessageID == nil
        }
        guard let summarizedThroughMessageID else { return false }
        return messages.contains { $0.id == summarizedThroughMessageID }
    }
}

struct AppChatArchive: Codable, Equatable, Sendable {
    static let currentVersion = 1

    var version = currentVersion
    var selectedChatID: UUID
    var chats: [AppChat]

    static func empty() -> AppChatArchive {
        let chat = AppChat()
        return AppChatArchive(selectedChatID: chat.id, chats: [chat])
    }

    var isValid: Bool {
        version == Self.currentVersion
            && !chats.isEmpty
            && Set(chats.map(\.id)).count == chats.count
            && chats.contains { $0.id == selectedChatID }
            && chats.allSatisfy(\.hasValidSummaryBoundary)
    }
}

struct AppChatLoadResult: Equatable, Sendable {
    var archive: AppChatArchive
    var recoveryURL: URL?
}

enum AppChatFileStore {
    static let fileName = "mac-app-chats.json"
    static let recoveryFilePrefix = "mac-app-chats.recovery-"

    static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(fileName, isDirectory: false)
    }

    static func loadOrCreate(forModelDirectory modelDirectory: URL,
                             fileManager: FileManager = .default) -> AppChatArchive {
        loadOrCreateWithRecovery(
            forModelDirectory: modelDirectory,
            fileManager: fileManager).archive
    }

    static func loadOrCreateWithRecovery(
        forModelDirectory modelDirectory: URL,
        fileManager: FileManager = .default
    ) -> AppChatLoadResult {
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                let archive = try JSONDecoder().decode(AppChatArchive.self, from: data)
                guard archive.isValid else { throw InvalidArchive() }
                return AppChatLoadResult(archive: archive, recoveryURL: nil)
            } catch {
                let recoveryURL = quarantine(
                    fileURL,
                    fileManager: fileManager)
                let archive = AppChatArchive.empty()
                if recoveryURL != nil {
                    try? save(
                        archive,
                        forModelDirectory: modelDirectory,
                        fileManager: fileManager)
                }
                return AppChatLoadResult(
                    archive: archive,
                    recoveryURL: recoveryURL)
            }
        }

        let archive = AppChatArchive.empty()
        try? save(archive, forModelDirectory: modelDirectory, fileManager: fileManager)
        return AppChatLoadResult(archive: archive, recoveryURL: nil)
    }

    static func save(_ archive: AppChatArchive,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        guard archive.isValid else { throw InvalidArchive() }
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(archive)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    private static func quarantine(
        _ fileURL: URL,
        fileManager: FileManager
    ) -> URL? {
        let recoveryURL = fileURL
            .deletingLastPathComponent()
            .appendingPathComponent(
                "\(recoveryFilePrefix)\(UUID().uuidString.lowercased()).json",
                isDirectory: false)
        do {
            try fileManager.moveItem(at: fileURL, to: recoveryURL)
            return recoveryURL
        } catch {
            return nil
        }
    }

    private struct InvalidArchive: Error {}
}

final class AppChatPersistenceCoordinator: @unchecked Sendable {
    typealias FailureHandler = @Sendable (String) -> Void

    private let queue: DispatchQueue
    private var latestRevision: UInt64 = 0

    init(label: String = "com.turbofieldfare.chat-persistence") {
        self.queue = DispatchQueue(label: label, qos: .utility)
    }

    func save(
        revision: UInt64,
        archive: AppChatArchive,
        modelDirectory: URL,
        delay: TimeInterval,
        onFailure: @escaping FailureHandler
    ) {
        queue.async { [self] in
            latestRevision = max(latestRevision, revision)
            if delay > 0 {
                queue.asyncAfter(deadline: .now() + delay) { [self] in
                    writeIfLatest(
                        revision: revision,
                        archive: archive,
                        modelDirectory: modelDirectory,
                        onFailure: onFailure)
                }
            } else {
                writeIfLatest(
                    revision: revision,
                    archive: archive,
                    modelDirectory: modelDirectory,
                    onFailure: onFailure)
            }
        }
    }

    func flush(
        revision: UInt64,
        archive: AppChatArchive,
        modelDirectory: URL
    ) throws {
        try queue.sync { [self] in
            latestRevision = max(latestRevision, revision)
            guard revision == latestRevision else { return }
            try AppChatFileStore.save(
                archive,
                forModelDirectory: modelDirectory)
        }
    }

    private func writeIfLatest(
        revision: UInt64,
        archive: AppChatArchive,
        modelDirectory: URL,
        onFailure: @escaping FailureHandler
    ) {
        guard revision == latestRevision else { return }
        do {
            try AppChatFileStore.save(
                archive,
                forModelDirectory: modelDirectory)
        } catch {
            onFailure("\(error)")
        }
    }
}
