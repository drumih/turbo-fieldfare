import TinyTitanCore
import Foundation

/// Which reasoning controls a family's chat template truthfully defines.
/// TinyTitan exposes exactly what the installed template implements: Ornith 1.5
/// and Qwen 3.6 define only the binary `enable_thinking` switch, while the
/// Qwen3.8-Flash-Next template additionally defines `reasoning_effort`
/// low|medium|xhigh (default xhigh) and K2-Horizon defines low|medium|high
/// (default high) while thinking is on. Verified against the pinned upstream
/// `chat_template.jinja` on 2026-08-27; see docs/qwen38-flash-next-port.md and
/// docs/plan-k2-horizon.md. Re-verified 2026-09-11 against every installed
/// template (Qwen 3.6, AgentWorld, Ornith 1.5, Qwen3.8, and the CPU engine's
/// Qwen3.5): ReasoningLevelTemplateTests renders each one.
public typealias ModelReasoningControl = TinyTitanCore.ReasoningControl

public enum ModelReasoningControlError: Error, Equatable, CustomStringConvertible {
    case effortUnsupported(family: ModelFamily, effort: ModelReasoningEffort)
    case effortRequiresThinkingOn(effort: ModelReasoningEffort)
    case thinkingRequired(family: ModelFamily)

    public var description: String {
        switch self {
        case .effortUnsupported(let family, let effort):
            return "reasoning effort \(effort.rawValue) is not supported: the "
                + "\(family.rawValue) chat template does not define that level"
        case .effortRequiresThinkingOn(let effort):
            return "reasoning effort \(effort.rawValue) requires thinking on; "
                + "the chat template ignores effort while thinking is off"
        case .thinkingRequired(let family):
            return "the \(family.rawValue) chat template has no thinking-off "
                + "mode: its generation prompt always opens a think block"
        }
    }
}

extension ModelFamily {
    public var reasoningControl: ModelReasoningControl {
        switch self {
        case .qwen36, .qwen36MTP, .qwen35Dense:
            return .binaryThinking
        case .qwen38flash, .qwen38flashMTP:
            return .thinkingWithEffortLevels(
                defaultEffort: .xhigh, supported: [.low, .medium, .xhigh])
        case .qwen4flash, .qwen4flashMTP:
            // Same four-mode template as the 3.8 preview. Declared separately
            // so a 4.0 template that adds or drops a level changes only here.
            return .thinkingWithEffortLevels(
                defaultEffort: .xhigh, supported: [.low, .medium, .xhigh])
        case .k2Horizon:
            return .effortOnly(
                defaultEffort: .high, supported: [.low, .medium, .high])
        case .laguna:
            // Laguna's template takes `enable_thinking` as a boolean, so the
            // runtime offers the ordinary on/off switch.
            return .binaryThinking
        }
    }

    /// Whether the template can render a turn that does not think. K2 cannot:
    /// its generation prompt always opens a think block, so a configurator
    /// must offer an effort rather than an off switch.
    public var supportsThinkingOff: Bool {
        if case .effortOnly = reasoningControl { return false }
        return true
    }

    /// The effort a template applies for this family under the given
    /// settings: the explicit override, else the template default, else nil
    /// for binary families or while thinking is off.
    public func effectiveReasoningEffort(
        thinkingMode: ModelThinkingMode,
        effort: ModelReasoningEffort?
    ) -> ModelReasoningEffort? {
        switch reasoningControl {
        case .binaryThinking:
            return nil
        case .thinkingWithEffortLevels(let defaultEffort, _),
            .effortOnly(let defaultEffort, _):
            return thinkingMode.isEnabled ? effort ?? defaultEffort : nil
        }
    }

    /// Rejects a reasoning-effort request this family's template does not
    /// define. A nil effort always passes: it means the binary control, or
    /// the template's own default level for effort-aware families.
    public func validateReasoning(
        thinkingMode: ModelThinkingMode,
        effort: ModelReasoningEffort?
    ) throws {
        switch reasoningControl {
        case .binaryThinking:
            if let effort {
                throw ModelReasoningControlError.effortUnsupported(
                    family: self, effort: effort)
            }
        case .thinkingWithEffortLevels(_, let supported):
            guard let effort else { return }
            guard thinkingMode.isEnabled else {
                throw ModelReasoningControlError.effortRequiresThinkingOn(
                    effort: effort)
            }
            guard supported.contains(effort) else {
                throw ModelReasoningControlError.effortUnsupported(
                    family: self, effort: effort)
            }
        case .effortOnly(_, let supported):
            guard thinkingMode.isEnabled else {
                throw ModelReasoningControlError.thinkingRequired(family: self)
            }
            if let effort, !supported.contains(effort) {
                throw ModelReasoningControlError.effortUnsupported(
                    family: self, effort: effort)
            }
        }
    }

    /// Levels this family's chat template actually honours, `.off` first.
    public var supportedReasoningLevels: [ReasoningLevel] {
        reasoningControl.supportedLevels
    }

    /// Runtime settings for a level; throws for a level the family does not
    /// support.
    public func runtimeReasoning(for level: ReasoningLevel) throws
        -> (thinking: ModelThinkingMode, effort: ModelReasoningEffort?)
    {
        try reasoningControl.runtimeReasoning(for: level, family: rawValue)
    }
}

/// One vocabulary of thinking levels for every family, so a configurator can
/// offer a single picker. Each family answers which of these its template
/// renders differently; the others are refused rather than mapped to a
/// neighbour, because a level that does not change the prompt does not
/// change the model's behaviour either.
///
/// That refusal is right for a *configurator*, where a mistyped level should
/// be caught. It is wrong for a *request*: a coding agent that asks for
/// `xhigh` on a model with no effort levels must still get an answer. So the
/// request path maps a level to the nearest one the model renders, through
/// `ReasoningFallback.effectiveLevel` on the server, and `requested(_:)`
/// below is what turns a client's free-text spelling into a level.
public typealias ReasoningLevel = TinyTitanCore.ReasoningLevel
public typealias ReasoningLevelError = TinyTitanCore.ReasoningLevelError

extension ModelReasoningControl {
    /// Shared by the GPU and CPU families so the two cannot disagree on what
    /// a level means.
    func runtimeReasoning(for level: ReasoningLevel, family: String) throws
        -> (thinking: ModelThinkingMode, effort: ModelReasoningEffort?)
    {
        let supported = supportedLevels
        guard supported.contains(level) else {
            throw ReasoningLevelError.unsupported(
                family: family, level: level, supported: supported)
        }
        switch level {
        case .off: return (.off, nil)
        case .on: return (.on, nil)
        default:
            // Every supported level other than off/on is an effort level by
            // construction of `supportedLevels`, and shares its raw value.
            return (.on, ModelReasoningEffort(rawValue: level.rawValue))
        }
    }
}

/// The reasoning a single request asks the model to render at.
///
/// Distinct from the server's `ServerReasoningProfile`, which describes what
/// the model was loaded with. When the two agree a session reuses its own
/// tokenizer; when they differ this is a mid-session switch and the session
/// resolves a tokenizer for it. `off` is a real request: turning thinking
/// off inside a session is the case a coding agent reaches for when a model
/// starts over-thinking a simple turn.
public typealias RequestReasoning = TinyTitanCore.RequestReasoning
