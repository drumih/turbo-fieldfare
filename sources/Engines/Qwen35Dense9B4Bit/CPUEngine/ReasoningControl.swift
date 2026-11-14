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
public enum ModelReasoningControl: Sendable, Equatable {
    case binaryThinking
    /// The levels the template accepts are listed, not inferred from the
    /// enum's cases: Qwen3.8 accepts xhigh but not high, K2 accepts high but
    /// not xhigh, and a family that advertised the other one's set would send
    /// an effort its template raises on.
    case thinkingWithEffortLevels(
        defaultEffort: ModelReasoningEffort,
        supported: [ModelReasoningEffort])
    /// An effort template with no closed form. K2's generation prompt always
    /// opens a think tag and selects it from the effort (think / think_fast /
    /// think_faster), so there is no thinking-off prompt to offer: `.off` is
    /// not in `supported` and validating it is an error.
    case effortOnly(
        defaultEffort: ModelReasoningEffort,
        supported: [ModelReasoningEffort])
}

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
public enum ReasoningLevel: String, CaseIterable, Sendable, Codable {
    case off, on, minimal, low, medium, high, xhigh, max

    public var displayName: String {
        self == .xhigh ? "extra high" : rawValue
    }

    init(effort: ModelReasoningEffort) {
        // Exhaustive on purpose: a new effort case must be given its level
        // here before it can compile.
        switch effort {
        case .low: self = .low
        case .medium: self = .medium
        case .high: self = .high
        case .xhigh: self = .xhigh
        }
    }

    /// Position on the one effort ladder, `off` lowest. Used to choose the
    /// nearest level a model does support.
    var rank: Int {
        switch self {
        case .off: return 0
        case .on, .minimal: return 1
        case .low: return 2
        case .medium: return 3
        case .high: return 4
        case .xhigh: return 5
        case .max: return 6
        }
    }
}

extension ReasoningLevel {
    /// The level a client asked for, as free text.
    ///
    /// Coding agents send vocabulary this project never defined (`ultra`,
    /// `none`, `extra-high`, `x-high`, `thinking`, `auto`). They all mean
    /// something on the ladder, so they are mapped rather than rejected, and
    /// an unknown word is reported instead of failing the request.
    public static func requested(_ raw: String) -> ReasoningLevel? {
        let word = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch word {
        case "": return nil
        case "0", "off", "false", "no", "none", "disabled", "nothink", "no_think":
            return .off
        case "1", "on", "true", "yes", "enabled", "think", "thinking", "auto", "adaptive",
            "default":
            return .on
        case "minimal", "min", "least", "tiny":
            return .minimal
        case "low", "light":
            return .low
        case "medium", "med", "moderate", "normal", "standard", "balanced":
            return .medium
        case "high", "hard":
            return .high
        case "xhigh", "extra-high", "extra_high", "extra high", "x-high", "very-high", "highest":
            return .xhigh
        case "max", "maximum", "ultra", "extreme", "unlimited":
            return .max
        default:
            return ReasoningLevel(rawValue: word)
        }
    }

    /// The template effort this level names, if it names one.
    public var effort: ModelReasoningEffort? {
        ModelReasoningEffort(rawValue: rawValue)
    }
}

public enum ReasoningLevelError: Error, Equatable, CustomStringConvertible {
    case unsupported(
        family: String, level: ReasoningLevel,
        supported: [ReasoningLevel])

    public var description: String {
        switch self {
        case .unsupported(let family, let level, let supported):
            return "thinking level \(level.displayName) is not supported: the "
                + "\(family) chat template renders only "
                + supported.map(\.displayName).joined(separator: ", ")
        }
    }
}

extension ModelReasoningControl {
    /// A binary template renders two prompts: the closed and the open think
    /// block. An effort template renders off plus one prompt per effort it
    /// accepts. Plain `.on` is not a level there: it selects the template's
    /// default effort and renders byte-identically to naming that effort, so
    /// offering both would put two entries on one prompt.
    ///
    /// `ModelReasoningEffort` is the union of the shipped templates' levels;
    /// each control carries the subset its own template accepts, because
    /// Qwen3.8 raises on `high` and K2 raises on `xhigh`.
    /// Public because the server's request path maps a client's level to the
    /// nearest one the served model renders, and names the set it chose from
    /// in its log line.
    public var supportedLevels: [ReasoningLevel] {
        switch self {
        case .binaryThinking:
            return [.off, .on]
        case .thinkingWithEffortLevels(_, let supported):
            return [.off] + supported.map(ReasoningLevel.init(effort:))
        case .effortOnly(_, let supported):
            return supported.map(ReasoningLevel.init(effort:))
        }
    }

    /// What plain "thinking on" loads: the switch itself for a binary
    /// template, the template's default effort for one with levels. The
    /// router maps a server-wide `on` through this so it means the same as
    /// `--thinking on` on a single-model server.
    public var levelWhenOn: ReasoningLevel {
        switch self {
        case .binaryThinking:
            return .on
        case .thinkingWithEffortLevels(let defaultEffort, _),
            .effortOnly(let defaultEffort, _):
            return ReasoningLevel(effort: defaultEffort)
        }
    }

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
public struct RequestReasoning: Sendable, Equatable {
    public let thinkingMode: ModelThinkingMode
    public let effort: ModelReasoningEffort?

    public init(thinkingMode: ModelThinkingMode, effort: ModelReasoningEffort?) {
        self.thinkingMode = thinkingMode
        // Effort is inert while thinking is off, and every template ignores
        // it there; dropping it here keeps "off" equal to "off" whatever
        // effort came alongside it, so a switch to off is a cache hit.
        self.effort = thinkingMode.isEnabled ? effort : nil
    }

    /// Whether a session loaded at `loaded` can serve this request without
    /// resolving a different tokenizer.
    public func matches(_ loaded: RequestReasoning) -> Bool { self == loaded }
}
