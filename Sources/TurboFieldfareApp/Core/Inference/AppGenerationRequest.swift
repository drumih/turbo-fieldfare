import Foundation

public enum AppChatRole: String, Codable, Equatable, Sendable {
    case system
    case user
    case assistant
}

public struct AppChatMessage: Codable, Equatable, Sendable {
    public var role: AppChatRole
    public var content: String

    public init(role: AppChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

public struct AppGenerationRequest: Equatable, Sendable {
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
        }
    }
}
