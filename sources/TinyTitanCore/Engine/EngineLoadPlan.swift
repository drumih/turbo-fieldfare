import Foundation

public enum ModelIntegrityPolicy: Sendable, Equatable {
    case fullSha256
    case sizeCheckTrustedReceipt
}

public enum RDAdvicePolicyMode: String, Codable, Sendable, Equatable {
    case `default`
    case off
    case bounded
    case adaptive

    public static func parse(_ raw: String?) -> RDAdvicePolicyMode {
        switch raw?.lowercased() {
        case "off", "none", "disabled": .off
        case "bounded": .bounded
        case "adaptive": .adaptive
        default: .default
        }
    }
}

public struct EngineModelMetadata: Sendable, Equatable {
    public let modelID: String
    public let family: String
    public let weightBits: Int
    public let sampling: SamplingProfile
    public let reasoningControl: ReasoningControl

    public init(
        modelID: String, family: String, weightBits: Int, sampling: SamplingProfile,
        reasoningControl: ReasoningControl
    ) {
        self.modelID = modelID
        self.family = family
        self.weightBits = weightBits
        self.sampling = sampling
        self.reasoningControl = reasoningControl
    }
}

public struct EngineLoadPlan: Sendable {
    public let architecture: ArchConfig
    public let expertCacheSlots: Int
    public let tunedExpertCacheBudgetBytes: Int
    public let prefillConfiguration: PrefillRuntimeConfig
    public let fp16RingEnabled: Bool
    public let yarnContextTokens: Int
    public let runtimeIdentity: String
    public let engineConfiguration: any Sendable

    public init(
        architecture: ArchConfig, expertCacheSlots: Int, tunedExpertCacheBudgetBytes: Int,
        prefillConfiguration: PrefillRuntimeConfig, fp16RingEnabled: Bool,
        yarnContextTokens: Int, runtimeIdentity: String, engineConfiguration: any Sendable
    ) {
        self.architecture = architecture
        self.expertCacheSlots = expertCacheSlots
        self.tunedExpertCacheBudgetBytes = tunedExpertCacheBudgetBytes
        self.prefillConfiguration = prefillConfiguration
        self.fp16RingEnabled = fp16RingEnabled
        self.yarnContextTokens = yarnContextTokens
        self.runtimeIdentity = runtimeIdentity
        self.engineConfiguration = engineConfiguration
    }
}

public struct EngineSessionPreview: Sendable, Equatable {
    public let modelID: String
    public let prefillChunkTokens: Int

    public init(modelID: String, prefillChunkTokens: Int) {
        self.modelID = modelID
        self.prefillChunkTokens = prefillChunkTokens
    }
}

extension SamplingProfile {
    public static let house = SamplingProfile(
        temperature: 0.6, topK: 20, topP: 0.95, presencePenalty: 0)

    public func effective(thinking: Bool) -> SamplingProfile {
        guard thinking, let row = self.thinking else { return self }
        return SamplingProfile(
            temperature: row.temperature, topK: row.topK, topP: row.topP,
            presencePenalty: row.presencePenalty)
    }
}
