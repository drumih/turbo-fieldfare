import Foundation

public enum AppChatRole: String, Codable, Equatable, Sendable {
    case system
    case user
    case assistant
}

/// A compact reference to an app-managed local image. Image bytes never enter
/// chat-history JSON or the decode-service command frame.
public struct AppImageAttachment: Codable, Equatable, Identifiable, Sendable {
    public let id: UUID
    public let relativePath: String
    public let originalFilename: String
    public let mediaTypeIdentifier: String
    public let pixelWidth: Int
    public let pixelHeight: Int
    public let byteCount: UInt64
    public let sha256: String

    public init(id: UUID = UUID(),
                relativePath: String,
                originalFilename: String,
                mediaTypeIdentifier: String,
                pixelWidth: Int,
                pixelHeight: Int,
                byteCount: UInt64,
                sha256: String) {
        self.id = id
        self.relativePath = relativePath
        self.originalFilename = originalFilename
        self.mediaTypeIdentifier = mediaTypeIdentifier
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
        self.byteCount = byteCount
        self.sha256 = sha256
    }

    var hasValidStoredMetadata: Bool {
        let pathComponents = relativePath.split(
            separator: "/",
            omittingEmptySubsequences: false)
        guard pixelWidth > 0, pixelHeight > 0 else { return false }
        let pixels = UInt64(pixelWidth).multipliedReportingOverflow(
            by: UInt64(pixelHeight))
        return pathComponents.count == 2
            && pathComponents.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
            && !originalFilename.isEmpty
            && !mediaTypeIdentifier.isEmpty
            && !pixels.overflow
            && pixels.partialValue <= AppChatAttachmentStore.maximumPixelCount
            && byteCount > 0
            && byteCount <= AppChatAttachmentStore.maximumFileBytes
            && sha256.count == 64
            && sha256.allSatisfy(\.isHexDigit)
            && sha256 == sha256.lowercased()
    }
}

public struct AppChatMessage: Codable, Equatable, Sendable {
    public var role: AppChatRole
    public var content: String
    public var images: [AppImageAttachment]

    public init(role: AppChatRole,
                content: String,
                images: [AppImageAttachment] = []) {
        self.role = role
        self.content = content
        self.images = images
    }

    private enum CodingKeys: String, CodingKey {
        case role
        case content
        case images
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(AppChatRole.self, forKey: .role)
        content = try container.decode(String.self, forKey: .content)
        // Histories written before image support have no `images` key.
        images = try container.decodeIfPresent(
            [AppImageAttachment].self,
            forKey: .images) ?? []
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encode(content, forKey: .content)
        if !images.isEmpty {
            try container.encode(images, forKey: .images)
        }
    }
}

public struct AppGenerationRequest: Equatable, Sendable {
    /// Vision projections coexist for the duration of a generation. Keep the
    /// per-request count explicitly bounded instead of allowing chat history
    /// to grow image memory without limit.
    public static let maximumImageAttachments = 8

    public var modelDirectory: URL
    public var prompt: String
    /// Complete conversation ending with the user message to answer.
    public var messages: [AppChatMessage]
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var runtimeOptions: AppRuntimeOptions

    public init(modelDirectory: URL,
                prompt: String,
                maxNewTokens: Int = 4_096,
                maxContextTokens: Int = 4096,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                runtimeOptions: AppRuntimeOptions = AppRuntimeOptions(),
                messages: [AppChatMessage]? = nil) {
        self.modelDirectory = modelDirectory
        self.prompt = prompt
        self.messages = messages ?? [AppChatMessage(role: .user, content: prompt)]
        self.maxNewTokens = maxNewTokens
        self.maxContextTokens = maxContextTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.runtimeOptions = runtimeOptions
    }

    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

    public func validate(fileManager: FileManager = .default,
                         requireModelDirectory: Bool = true) throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AppInferenceError.invalidRequest("Prompt cannot be empty.")
        }
        guard !messages.isEmpty, messages.last?.role == .user else {
            throw AppInferenceError.invalidRequest("Conversation must end with a user message.")
        }
        guard !messages.dropFirst().contains(where: { $0.role == .system }) else {
            throw AppInferenceError.invalidRequest("System instructions must be the first message.")
        }
        guard messages.allSatisfy({ !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw AppInferenceError.invalidRequest("Conversation messages cannot be empty.")
        }
        guard messages.allSatisfy({ message in
            message.images.isEmpty || message.role == .user
        }) else {
            throw AppInferenceError.invalidRequest(
                "Images can only be attached to user messages.")
        }
        guard messages.allSatisfy({ $0.images.count <= 1 }) else {
            throw AppInferenceError.invalidRequest(
                "Only one image can be attached to each user message.")
        }
        let imageCount = messages.reduce(0) { $0 + $1.images.count }
        guard imageCount <= Self.maximumImageAttachments else {
            throw AppInferenceError.invalidRequest(
                "A chat can include at most \(Self.maximumImageAttachments) images.")
        }
        guard messages.flatMap(\.images).allSatisfy(\.hasValidStoredMetadata) else {
            throw AppInferenceError.invalidRequest(
                "The conversation contains an invalid image attachment.")
        }
        guard maxNewTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max response length must be greater than zero.")
        }
        guard maxContextTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max context must be greater than zero.")
        }
        guard temperature >= 0 else {
            throw AppInferenceError.invalidRequest("Temperature cannot be negative.")
        }
        if let topK {
            guard (1...256).contains(topK) else {
                throw AppInferenceError.invalidRequest("Top-K must be between 1 and 256.")
            }
        }
        if let topP {
            guard topP > 0, topP <= 1 else {
                throw AppInferenceError.invalidRequest("Top-P must be greater than 0 and at most 1.")
            }
            if temperature > 0, topP < 1, topK == nil {
                throw AppInferenceError.invalidRequest(
                    "Top-P below 1 requires Top-K to be enabled.")
            }
        }
        guard repetitionPenalty >= 1 else {
            throw AppInferenceError.invalidRequest("Repetition penalty must be at least 1.")
        }
        try runtimeOptions.validate()

        if requireModelDirectory {
            var isDirectory = ObjCBool(false)
            guard fileManager.fileExists(atPath: modelDirectory.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw AppInferenceError.modelNotFound(modelDirectory.path)
            }
            for image in messages.flatMap(\.images) {
                do {
                    try AppChatAttachmentStore.validateStoredImage(
                        image,
                        forModelDirectory: modelDirectory,
                        fileManager: fileManager)
                } catch let attachmentError as AppChatAttachmentStoreError {
                    throw AppInferenceError.invalidRequest(
                        attachmentError.description)
                } catch {
                    throw AppInferenceError.invalidRequest(
                        "The attached image could not be validated: \(error)")
                }
            }
        }
    }
}
