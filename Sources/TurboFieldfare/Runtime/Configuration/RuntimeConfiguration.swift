public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
}

public struct RuntimeConfiguration: Sendable, Equatable {
    public static let allowedExpertCacheSlots = [8, 16, 24, 32]
    public static let allowedPrefillChunkTokens = [32, 64, 128]
    public static let minimumExpertCacheSlotsForChunkedPrefill = 16

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath
    public let profilingEnabled: Bool
    public let profilingJson: Bool
    public let telemetry: RuntimeTelemetry?

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                forceLogitsHead: Bool = false,
                profilingEnabled: Bool = false,
                profilingJson: Bool = false,
                telemetry: RuntimeTelemetry? = nil) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.profilingEnabled = profilingEnabled || profilingJson
        self.profilingJson = profilingJson
        if let telemetry {
            self.telemetry = telemetry
        } else if profilingEnabled || profilingJson {
            self.telemetry = RuntimeTelemetry(isEnabled: true, cacheSlotsPerLayer: expertCacheSlots)
        } else {
            self.telemetry = nil
        }
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        expertCachePolicy == .lru ? .lru : .lfu
    }

    public static func == (lhs: RuntimeConfiguration, rhs: RuntimeConfiguration) -> Bool {
        lhs.expertCacheSlots == rhs.expertCacheSlots &&
        lhs.expertCachePolicy == rhs.expertCachePolicy &&
        lhs.rdadvisePolicy == rhs.rdadvisePolicy &&
        lhs.prefillPolicy == rhs.prefillPolicy &&
        lhs.prefillChunkTokens == rhs.prefillChunkTokens &&
        lhs.prefillAttentionPath == rhs.prefillAttentionPath &&
        lhs.headPath == rhs.headPath &&
        lhs.profilingEnabled == rhs.profilingEnabled &&
        lhs.profilingJson == rhs.profilingJson &&
        lhs.telemetry === rhs.telemetry
    }
}
