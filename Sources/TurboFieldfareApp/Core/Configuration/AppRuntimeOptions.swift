import Foundation
import TurboFieldfare

public enum AppExpertCachePolicy: String, CaseIterable, Sendable, Identifiable {
    case lfu
    case lru

    public var id: String { rawValue }
    public var label: String { rawValue.uppercased() }
}

public enum AppRDAdvicePolicy: String, Codable, CaseIterable, Sendable, Identifiable {
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
    public var visionResidencyPolicy: VisionResidencyPolicy

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: AppExpertCachePolicy = .lfu,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                rdadvisePolicy: AppRDAdvicePolicy = .off,
                modelVerification: AppModelVerification = .fullSha256,
                visionResidencyPolicy: VisionResidencyPolicy = .onDemand) {
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillEnabled = prefillEnabled
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
        self.modelVerification = modelVerification
        self.visionResidencyPolicy = visionResidencyPolicy
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
        // Refused here rather than inside the first prefill chunk, which is
        // after the model load and, on an image turn, after every attachment
        // has already been encoded on the GPU.
        let minimumSlots = RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        guard !prefillEnabled || expertCacheSlots >= minimumSlots else {
            throw AppInferenceError.invalidRequest(
                "Prefill needs at least \(minimumSlots) expert cache slots, and Slots is set to "
                    + "\(expertCacheSlots). Raise Slots to \(minimumSlots) or turn Prefill off.")
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
    public var visionResidencyPolicy: VisionResidencyPolicy
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
        self.visionResidencyPolicy = options.visionResidencyPolicy
        self.forceLogitsHead = forceLogitsHead
    }

    /// The options this session was actually loaded with. A run has to be built
    /// from these rather than from the current settings, or a pending change the
    /// user has not reloaded for is refused by the loaded session.
    ///
    /// Prefill is passed in because it is deliberately not part of this key:
    /// it is chosen per request and changing it must not make a loaded
    /// session stale.
    public func options(prefillEnabled: Bool,
                        prefillChunkTokens: Int) -> AppRuntimeOptions {
        AppRuntimeOptions(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            prefillEnabled: prefillEnabled,
            prefillChunkTokens: prefillChunkTokens,
            rdadvisePolicy: rdadvisePolicy,
            modelVerification: modelVerification,
            visionResidencyPolicy: visionResidencyPolicy)
    }
}
