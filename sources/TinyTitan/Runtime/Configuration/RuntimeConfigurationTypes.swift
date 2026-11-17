import Foundation
import TinyTitanCore

// The runtime configuration's enums: the head, prefill, expert-cache, expert-
// IO and KV-precision choices, and the configuration error type.
//
// Split out of `RuntimeConfiguration.swift` (2026-09-28) under the 500-line-per-
// file rule (Task 8 of the cleanup runbook) as pure code motion.
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

/// Decode scheduling for SSD-backed routed experts.
///
/// `hitFixup` commits phase 1 for resident experts while cache misses are read,
/// then computes only the missed experts before the common reduction. `barrier`
/// preserves the former all-experts-after-I/O path as a correctness/performance
/// control for A/B measurements.
public enum RuntimeDecodeExpertExecution: String, Codable, Sendable {
    case hitFixup = "hit-fixup"
    case barrier
    case gpuResidency = "gpu-residency"

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeDecodeExpertExecution {
        guard let raw = environment["TINYTITAN_DECODE_EXPERT_EXECUTION"] else {
            return .hitFixup
        }
        guard let value = RuntimeDecodeExpertExecution(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidDecodeExpertExecution(raw)
        }
        return value
    }
}

public enum RuntimeExpertIOSynchronization: String, Codable, Sendable {
    case host
    case event

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeExpertIOSynchronization {
        guard let raw = environment["TINYTITAN_EXPERT_IO_SYNC"] else { return .host }
        guard let value = RuntimeExpertIOSynchronization(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidExpertIOSynchronization(raw)
        }
        return value
    }
}

public enum RuntimeExpertIOSubmission: String, Codable, Sendable {
    case deferred
    case immediate

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeExpertIOSubmission {
        guard let raw = environment["TINYTITAN_EXPERT_IO_SUBMISSION"] else { return .deferred }
        guard let value = RuntimeExpertIOSubmission(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidExpertIOSubmission(raw)
        }
        return value
    }
}

/// How many completion timelines the event-driven expert I/O path uses.
///
/// `.shared` is the shipped wiring: one timeline for the whole model, which is
/// what makes an out-of-order completion safe but also lets one slow read gate
/// every later release. `.perLayer` keeps the same rule inside one layer, where
/// the streamer issues its values in order anyway.
public enum RuntimeExpertIOTimeline: String, Codable, Sendable {
    case shared
    case perLayer = "per-layer"

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeExpertIOTimeline {
        guard let raw = environment["TINYTITAN_EXPERT_IO_TIMELINE"] else { return .shared }
        guard let value = RuntimeExpertIOTimeline(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidExpertIOTimeline(raw)
        }
        return value
    }
}

public typealias KVCachePrecision = TinyTitanCore.KVCachePrecision
public typealias RuntimeRoPEScalingMode = TinyTitanCore.RuntimeRoPEScalingMode

public enum RuntimeConfigurationError: Error, CustomStringConvertible, Equatable {
    case invalidExpertCacheSlots(Int)
    case invalidPrefillChunkTokens(Int)
    case invalidYaRNContextTokens(Int)
    case contextRequiresYaRN(Int)
    case yaRNContextMismatch(maxContext: Int, configured: Int)
    case yaRNUnsupportedArchitecture
    case invalidDecodeExpertExecution(String)
    case invalidExpertIOSynchronization(String)
    case invalidExpertIOSubmission(String)
    case invalidExpertIOTimeline(String)
    case invalidHostKVWindowTokens(Int)

    public var description: String {
        switch self {
        case .invalidExpertCacheSlots(let value):
            return
                "unsupported expert-cache slot count \(value); allowed: \(RuntimeConfiguration.allowedExpertCacheSlots)"
        case .invalidPrefillChunkTokens(let value):
            return
                "unsupported prefill chunk size \(value); allowed: \(RuntimeConfiguration.allowedPrefillChunkTokens)"
        case .invalidYaRNContextTokens(let value):
            return
                "unsupported YaRN context \(value); allowed: \(RuntimeConfiguration.supportedYaRNContextTokens)"
        case .contextRequiresYaRN(let value):
            return
                "context \(value) exceeds the native \(RuntimeConfiguration.nativeMaximumContextTokens)-token limit; enable YaRN"
        case .yaRNContextMismatch(let maxContext, let configured):
            return "YaRN is configured for \(configured) tokens, but max context is \(maxContext)"
        case .yaRNUnsupportedArchitecture:
            return "YaRN requires the Qwen3.5-MoE NeoX sub-dimension RoPE architecture"
        case .invalidDecodeExpertExecution(let value):
            return
                "unsupported decode expert execution '\(value)'; allowed: hit-fixup, barrier, gpu-residency"
        case .invalidExpertIOSynchronization(let value):
            return "unsupported expert I/O synchronization '\(value)'; allowed: host, event"
        case .invalidExpertIOSubmission(let value):
            return "unsupported expert I/O submission '\(value)'; allowed: deferred, immediate"
        case .invalidExpertIOTimeline(let value):
            return "unsupported expert I/O timeline '\(value)'; allowed: shared, per-layer"
        case .invalidHostKVWindowTokens(let value):
            return "host-KV window must be a positive token count, not \(value)"
        }
    }
}
