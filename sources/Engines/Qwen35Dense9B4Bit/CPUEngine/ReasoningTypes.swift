import Foundation

public enum ModelThinkingMode: String, Codable, CaseIterable, Sendable {
    case off
    case on

    public var isEnabled: Bool { self == .on }

    /// Backwards-compatible resolution for processes that still configure the
    /// runtime through `TINYTITAN_THINKING_MODE`. Unknown values retain the
    /// historical safe default of off.
    public static func resolved(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ModelThinkingMode {
        switch environment["TINYTITAN_THINKING_MODE"]?.lowercased() {
        case "1", "on", "true", "yes": return .on
        default: return .off
        }
    }
}

/// The reasoning-effort levels defined by chat templates that support them.
/// The Qwen3.8-Flash-Next template accepts `reasoning_effort` while thinking
/// is on and injects an effort-specific instruction into the system block
/// (`xhigh` is its default; `medium` is accepted but injects no text). K2's
/// template accepts `low`, `medium` and `high` (`high` is its default) and
/// raises on `xhigh`. Each family lists the subset it accepts in
/// `ModelReasoningControl`, so this enum is the union, not one template's set.
/// Ornith 1.5 and Qwen 3.6 templates define no effort levels, so those
/// families reject these values at the surface instead of faking them.
/// `ReasoningLevel` is the one-picker view over this and `ModelThinkingMode`.
public enum ModelReasoningEffort: String, Codable, CaseIterable, Sendable {
    case low
    case medium
    case high
    case xhigh

    /// Environment resolution mirroring `ModelThinkingMode.resolved`:
    /// `TINYTITAN_REASONING_EFFORT` selects a level, and unset or unknown values
    /// keep the safe default of nil (the template's own default applies).
    public static func resolved(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ModelReasoningEffort? {
        guard let raw = environment["TINYTITAN_REASONING_EFFORT"]?.lowercased() else {
            return nil
        }
        return ModelReasoningEffort(rawValue: raw)
    }
}
