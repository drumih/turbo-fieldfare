import Foundation
import TinyTitanCore

/// Whether an engine honours one user-facing parameter (TPP/1 §4).
///
/// Nothing a user sets may be accepted and silently ignored: a value the engine
/// cannot honour is either refused (when the user asked for it) or replaced by
/// the engine's own and announced (when it was only a default).
public enum Support<Value: Sendable & Equatable>: Sendable {
    /// Every value works.
    case supported
    /// Only these values work; `values.first` is the engine's own default.
    case only([Value], reason: String)
    /// The parameter does not apply to this engine at all.
    case refused(reason: String)

    public func accepts(_ value: Value) -> Bool {
        switch self {
        case .supported: true
        case .only(let values, _): values.contains(value)
        case .refused: false
        }
    }

    /// The refusal for an explicit `value` of `flag`, or nil when it works.
    /// `spelled` writes a value the way the user typed it (`8`, not `int8`).
    public func refusal(
        for value: Value, flag: String, plugin: String,
        spelled: (Value) -> String = { "\($0)" }
    ) -> String? {
        switch self {
        case .supported: return nil
        case .only(let values, let reason):
            return values.contains(value)
                ? nil : "\(flag) \(spelled(value)) is not supported by \(plugin): \(reason)"
        case .refused(let reason):
            return "\(flag) does not apply to \(plugin): \(reason)"
        }
    }

    /// The value to run with when `value` was only a default: itself when it
    /// works, else the engine's own, with the notice the host logs.
    public func resolveDefault(
        _ value: Value, flag: String, plugin: String,
        spelled: (Value) -> String = { "\($0)" }
    ) -> (value: Value, notice: String?) {
        switch self {
        case .supported: return (value, nil)
        case .only(let values, let reason):
            guard !values.contains(value), let own = values.first else { return (value, nil) }
            return (
                own,
                "\(plugin): default \(flag) \(spelled(value)) does not apply; using "
                    + "\(spelled(own)) (\(reason))"
            )
        case .refused: return (value, nil)
        }
    }
}

/// The parameter table a plugin declares (TPP/1 §4). A new parameter in a
/// later minor version defaults to `.refused("not implemented by …")` in the
/// host, so an older plugin reports the gap instead of hiding it.
public struct ParameterSupport: Sendable {
    /// `--kv-bits` / `EngineConfiguration.cachePrecision`.
    public var kvCachePrecision: Support<KVCachePrecision>
    /// `--rope-scaling`.
    public var ropeScaling: Support<RuntimeRoPEScalingMode>
    /// `--host-kv-window`.
    public var hostKVWindow: Support<Int>
    /// `--max-concurrent-sequences`: the widths the engine runs.
    public var concurrentSequences: Support<Int>
    /// `--mtp-model`: a multi-token-prediction draft head.
    public var multiTokenPrediction: Support<Bool>
    /// `--ram-budget` / `--expert-cache-slots`: routed experts streamed into
    /// a slot cache sized by the budget.
    public var expertStreaming: Support<Bool>

    public init(
        kvCachePrecision: Support<KVCachePrecision> = .supported,
        ropeScaling: Support<RuntimeRoPEScalingMode> = .supported,
        hostKVWindow: Support<Int> = .supported,
        concurrentSequences: Support<Int> = .supported,
        multiTokenPrediction: Support<Bool> = .supported,
        expertStreaming: Support<Bool> = .supported
    ) {
        self.kvCachePrecision = kvCachePrecision
        self.ropeScaling = ropeScaling
        self.hostKVWindow = hostKVWindow
        self.concurrentSequences = concurrentSequences
        self.multiTokenPrediction = multiTokenPrediction
        self.expertStreaming = expertStreaming
    }
}
