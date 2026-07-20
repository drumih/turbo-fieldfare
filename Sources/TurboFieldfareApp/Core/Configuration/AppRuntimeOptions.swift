import Foundation
import TurboFieldfare

public enum AppExpertCachePolicy: String, CaseIterable, Sendable, Identifiable {
    case lfu
    case lru

    public var id: String { rawValue }
    public var label: String { rawValue.uppercased() }
}

public enum AppRDAdvicePolicy: String, CaseIterable, Sendable, Identifiable {
    case off
    case `default`
    case bounded
    case adaptive

    public var id: String { rawValue }
    public var label: String { rawValue.capitalized }

    var runtimeValue: RDAdvicePolicyMode {
        switch self {
        case .off: return .off
        case .default: return .default
        case .bounded: return .bounded
        case .adaptive: return .adaptive
        }
    }
}

public enum AppModelVerification: String, CaseIterable, Sendable, Identifiable {
    case fullSha256 = "full-sha256"
    case trustedInstall = "trusted-install"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .fullSha256: return "Full SHA-256"
        case .trustedInstall: return "Trust verified install"
        }
    }

    var runtimeValue: ModelIntegrityPolicy {
        switch self {
        case .fullSha256: return .fullSha256
        case .trustedInstall: return .sizeCheckTrustedReceipt
        }
    }
}

public struct AppRuntimeOptions: Equatable, Sendable {
    public static let allowedSlotCounts = RuntimeConfiguration.allowedExpertCacheSlots
    public static let allowedPrefillChunkTokens = RuntimeConfiguration.allowedPrefillChunkTokens

    public var expertCacheSlots: Int
    public var expertCachePolicy: AppExpertCachePolicy
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int
    public var rdadvisePolicy: AppRDAdvicePolicy
    public var modelVerification: AppModelVerification

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: AppExpertCachePolicy = .lfu,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                rdadvisePolicy: AppRDAdvicePolicy = .off,
                modelVerification: AppModelVerification = .fullSha256) {
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillEnabled = prefillEnabled
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
        self.modelVerification = modelVerification
    }

    public func validate() throws {
        guard Self.allowedSlotCounts.contains(expertCacheSlots) else {
            throw AppInferenceError.invalidRequest(
                "expert cache slots must be one of \(Self.allowedSlotCounts)")
        }
        guard Self.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw AppInferenceError.invalidRequest(
                "prefill chunk size must be one of \(Self.allowedPrefillChunkTokens)")
        }
    }

    public var prefillConfig: PrefillRuntimeConfig {
        prefillEnabled ? .production(chunkTokens: prefillChunkTokens) : .off
    }

    public var resultSummary: String {
        let prefill = prefillEnabled ? "prefill \(prefillChunkTokens)" : "prefill off"
        let verification = modelVerification == .fullSha256 ? "full SHA-256" : "trusted receipt"
        return "Cache \(expertCacheSlots) \(expertCachePolicy.label), \(prefill), FP16 KV, RDADVISE \(rdadvisePolicy.label.lowercased()), \(verification)"
    }

    public static func slotsLabel(for slots: Int) -> String {
        switch slots {
        case 8: "8, -0.8 GB"
        case 16: "16, Default"
        case 24: "24, +0.8 GB"
        case 32: "32, +1.61 GB"
        default: "\(slots)"
        }
    }

    public func resolvedRuntimeConfiguration(forceLogitsHead: Bool) throws -> RuntimeConfiguration {
        try validate()
        return RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy == .lru ? .lru : .lfu,
            rdadvisePolicy: rdadvisePolicy.runtimeValue,
            prefillEnabled: prefillEnabled,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead)
    }
}

public struct AppLoadedRuntimeKey: Equatable, Sendable {
    public var modelDirectory: URL
    public var maxContextTokens: Int
    public var expertCacheSlots: Int
    public var expertCachePolicy: AppExpertCachePolicy
    public var rdadvisePolicy: AppRDAdvicePolicy
    public var modelVerification: AppModelVerification
    public var forceLogitsHead: Bool

    public init(modelDirectory: URL,
                maxContextTokens: Int,
                options: AppRuntimeOptions,
                forceLogitsHead: Bool = false) {
        self.modelDirectory = modelDirectory.standardizedFileURL
        self.maxContextTokens = maxContextTokens
        self.expertCacheSlots = options.expertCacheSlots
        self.expertCachePolicy = options.expertCachePolicy
        self.rdadvisePolicy = options.rdadvisePolicy
        self.modelVerification = options.modelVerification
        self.forceLogitsHead = forceLogitsHead
    }
}
