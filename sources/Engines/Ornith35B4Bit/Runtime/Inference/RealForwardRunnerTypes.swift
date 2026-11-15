import Foundation

public enum RDAdvicePolicyMode: String, Codable, Sendable, Equatable {
    case `default`
    case off
    case bounded
    case adaptive

    public static func parse(_ raw: String?) -> RDAdvicePolicyMode {
        switch raw?.lowercased() {
        case "off", "none", "disabled":
            return .off
        case "bounded":
            return .bounded
        case "adaptive":
            return .adaptive
        default:
            return .default
        }
    }
}

public struct RDAdviceAdaptivePolicyConfig: Sendable, Equatable {
    public var missCap: Int
    public var byteCap: UInt64
    public var slowCallNanos: UInt64

    public init(
        missCap: Int,
        byteCap: UInt64,
        slowCallNanos: UInt64
    ) {
        self.missCap = missCap
        self.byteCap = byteCap
        self.slowCallNanos = slowCallNanos
    }

    public static let conservative = RDAdviceAdaptivePolicyConfig(
        missCap: 12,
        byteCap: 384 * 1_048_576,
        slowCallNanos: 1_000_000)
}

struct RDAdviceAdaptivePolicyState: Sendable, Equatable {
    var config: RDAdviceAdaptivePolicyConfig
    var skipUntilPosition: Int = -1
    var recentSlowCallNanos: UInt64 = 0

    init(config: RDAdviceAdaptivePolicyConfig = .conservative) {
        self.config = config
    }

    mutating func reset() {
        skipUntilPosition = -1
        recentSlowCallNanos = 0
    }

    func shouldSkip(
        position: Int,
        requestedMisses: Int,
        estimatedBytes: UInt64,
        canOverlapUsefulGPUWork: Bool
    ) -> Bool {
        position <= skipUntilPosition || !canOverlapUsefulGPUWork
            || requestedMisses > config.missCap || estimatedBytes > config.byteCap
    }

    mutating func update(
        after result: ExpertIOAdviceResult,
        position: Int
    ) {
        recentSlowCallNanos = max(recentSlowCallNanos, result.maxCallNanos)
        if result.maxCallNanos >= config.slowCallNanos {
            skipUntilPosition = max(skipUntilPosition, position)
        }
    }
}

internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    static func selectedDispatch(
        for family: PrefillProjectionFamily,
        chunkTokens: Int
    ) -> PrefillProjectionDispatch {
        guard chunkTokens >= 32 else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return .repeatedGEMV
        case .kv, .o:
            return .qmm
        }
    }
}
