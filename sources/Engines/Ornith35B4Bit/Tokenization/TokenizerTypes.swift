import Foundation

// The tokenizer-facing enums: its error type and the two reasoning-mode
// switches the server and CLI share.
//
// Split out of `Tokenizer.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion.
public enum GFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate
    case unsupportedForDialect(String)
    /// The `--model`/catalogue path named a directory that is not there, or is
    /// a regular file. Reported before any tokenizer work so the message names
    /// the path instead of surfacing as a missing-template problem.
    case modelDirectoryNotFound(String)

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        case .unsupportedForDialect(let operation):
            return "operation is not supported for this tokenizer's chat dialect: \(operation)"
        case .modelDirectoryNotFound(let path):
            return "model directory not found: \(path)"
        }
    }
}

/// The binary reasoning switch exposed by compatible Qwen/Ornith chat
/// templates. Ornith 1.5 accepts `enable_thinking=true|false`; it does not
/// define low/medium/high effort levels or a thinking-token budget.
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
