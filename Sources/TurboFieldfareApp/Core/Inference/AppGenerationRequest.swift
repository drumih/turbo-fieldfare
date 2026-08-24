import Foundation
import TurboFieldfare

public struct AppGenerationRequest: Equatable, Sendable {
    public var modelDirectory: URL
    public var prompt: String
    public var imageAttachments: [AppImageAttachment]
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var runtimeOptions: AppRuntimeOptions

    public init(modelDirectory: URL,
                prompt: String,
                imageAttachments: [AppImageAttachment] = [],
                maxNewTokens: Int = 4_096,
                maxContextTokens: Int = 4096,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                runtimeOptions: AppRuntimeOptions = AppRuntimeOptions()) {
        self.modelDirectory = modelDirectory
        self.prompt = prompt
        self.imageAttachments = imageAttachments
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
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !imageAttachments.isEmpty else {
            throw AppInferenceError.invalidRequest("Prompt or image cannot be empty.")
        }
        // The same context-derived rule the server uses. A fixed four here
        // meant a request the API accepted was refused in the app.
        guard Set(imageAttachments.map(\.id)).count == imageAttachments.count else {
            throw AppInferenceError.invalidRequest("Images must be distinct.")
        }
        let capacity = VisionImageTokenBudget.capacity(
            maxContext: maxContextTokens, reservedTextTokens: 0)
        guard imageAttachments.count <= capacity else {
            throw AppInferenceError.invalidRequest(
                "\(imageAttachments.count) images need up to "
                    + "\(imageAttachments.count * VisionImageTokenBudget.maximumTokensPerImage) "
                    + "tokens, beyond the \(maxContextTokens)-token context.")
        }
        for attachment in imageAttachments {
            guard attachment.fileURL.isFileURL,
                  attachment.encodedBytes >= 0,
                  attachment.encodedBytes <= VisionImageLimits().maximumEncodedBytes,
                  attachment.sha256.count == 64,
                  attachment.sha256.unicodeScalars.allSatisfy({
                      (48...57).contains($0.value) || (97...102).contains($0.value)
                  }) else {
                throw AppInferenceError.invalidRequest(
                    "Invalid image attachment \(attachment.displayName).")
            }
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
