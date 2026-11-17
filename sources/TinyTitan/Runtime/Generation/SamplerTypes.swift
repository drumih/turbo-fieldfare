import TinyTitanCore
import Foundation
import Metal

// The sampler's value types: the canonical generation defaults, the knobs a
// caller threads through, and the two path enums the A/B switches select.
//
// Split out of `Sampler.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion.
/// Canonical sampling defaults. The bare constants are the house values; a
/// family whose model card specifies otherwise overrides them in `forFamily`,
/// because shipping a model at settings its authors did not intend is a
/// quality decision disguised as a default.
public enum GenerationDefaults {
    public static let temperature: Float = 0.6
    public static let topK = 20
    public static let topP: Float = 0.95
    public static let presencePenalty: Float = 0
    /// Min-p is zero in every shipped row, which means the filter is off. The
    /// engine does not implement a min-p filter yet, so `validate` refuses a
    /// non-zero value rather than silently sampling as if it were zero.
    public static let minP: Float = 0

    public struct Sampling: Sendable, Equatable {
        public var temperature: Float
        public var topK: Int
        public var topP: Float
        public var minP: Float
        /// OpenAI presence penalty: subtracted from the logit of every id the
        /// history already contains, once per distinct id.
        public var presencePenalty: Float
        public init(
            temperature: Float, topK: Int, topP: Float,
            minP: Float = GenerationDefaults.minP,
            presencePenalty: Float = GenerationDefaults.presencePenalty
        ) {
            self.temperature = temperature
            self.topK = topK
            self.topP = topP
            self.minP = minP
            self.presencePenalty = presencePenalty
        }
    }

    public static let house = Sampling(
        temperature: temperature,
        topK: topK, topP: topP)

    /// Qwen3.8-Flash-Next publishes **two** rows, because the checkpoint is
    /// specified with different sampling inside and outside thinking mode.
    /// Thinking is the card's default; instruct (non-thinking) raises the
    /// penalty on repetition.
    public static let qwen38Thinking = Sampling(
        temperature: 1.0, topK: topK,
        topP: 0.95)
    public static let qwen38Instruct = Sampling(
        temperature: 0.7, topK: topK,
        topP: 0.80, presencePenalty: 1.5)

    /// Qwen 4.0 Flash's two rows.
    ///
    /// The 4.0 card is not published yet, so these are the 3.8 preview's rows
    /// copied as a starting contract (thinking is the default). They are
    /// separate constants on purpose: the 4.0 checkpoint can change its
    /// instruct penalty without touching the 3.8 model's behavior.
    public static let qwen4Thinking = Sampling(
        temperature: 1.0, topK: topK,
        topP: 0.95)
    public static let qwen4Instruct = Sampling(
        temperature: 0.7, topK: topK,
        topP: 0.80, presencePenalty: 1.5)

    /// Defaults for a family, used wherever the caller did not ask for a value.
    /// An explicit request always wins -- this only fills the gap.
    ///
    /// Qwen3.8 keeps the thinking row here for callers that have no mode to
    /// give; a caller that knows the request's thinking mode must use
    /// `forFamily(_:thinking:)`.
    public static func forFamily(_ family: ModelFamily) -> Sampling {
        switch family {
        case .qwen38flash, .qwen38flashMTP:
            return qwen38Thinking
        case .qwen4flash, .qwen4flashMTP:
            return qwen4Thinking
        case .qwen35Dense:
            // Qwen 3.5 is a Qwen 3.6-lineage card: 0.6 / top-p 0.95.
            return Sampling(temperature: 0.6, topK: topK, topP: 0.95)
        case .k2Horizon:
            // The K2 model card asks for 1.0 / 0.95 at high reasoning effort;
            // top-k is the runtime's usual 20 (the card names no top-k).
            return Sampling(temperature: 1.0, topK: topK, topP: 0.95)
        case .laguna:
            // generation_config.json: temperature 1.0, top_p 1.0, top_k 20.
            return Sampling(temperature: 1.0, topK: 20, topP: 1.0)
        default:
            return house
        }
    }

    /// The family's row for the mode the request will actually run in.
    public static func forFamily(_ family: ModelFamily, thinking: Bool) -> Sampling {
        switch family {
        case .qwen38flash, .qwen38flashMTP:
            return thinking ? qwen38Thinking : qwen38Instruct
        case .qwen4flash, .qwen4flashMTP:
            return thinking ? qwen4Thinking : qwen4Instruct
        default:
            return forFamily(family)
        }
    }
}

// MARK: - TT-097: the Core contract a served model advertises

extension ModelReasoningControl {
    /// The Core control the validator, the router and the session speak. The
    /// runtime row is the truth; this is its spelling in the shared contract,
    /// so the family knowledge lives in exactly one place.
    public var engineControl: ReasoningControl {
        switch self {
        case .binaryThinking:
            .binaryThinking
        case .thinkingWithEffortLevels(let defaultEffort, let supported):
            .thinkingWithEffortLevels(defaultEffort: defaultEffort, supported: supported)
        case .effortOnly(let defaultEffort, let supported):
            .effortOnly(defaultEffort: defaultEffort, supported: supported)
        }
    }
}

extension ModelFamily {
    /// The family's reasoning control in the shared contract's spelling.
    public var engineReasoningControl: ReasoningControl {
        reasoningControl.engineControl
    }

    /// The Core sampling profile a served model advertises, derived from the
    /// identity the install manifest carries. The two-row families publish an
    /// instruct row and a thinking row: the instruct row is the base (what a
    /// thinking-off request must sample with) and the thinking row the twin,
    /// because the Core validator resolves the row from the profile it is
    /// handed and has no family table to fall back to. Single-row families
    /// advertise their profile row with no twin.
    ///
    /// `EngineModelMetadata` (the loaded model) and the server catalog (the
    /// model before it loads) both advertise through this one helper, so a
    /// mid-session thinking switch samples the same row wherever the model
    /// was when the request arrived.
    public func engineSamplingProfile(identity: ManifestIdentity) -> SamplingProfile {
        let profile = ModelProfile.resolve(identity: identity).sampling
        let twoRow =
            self == .qwen38flash || self == .qwen38flashMTP
            || self == .qwen4flash || self == .qwen4flashMTP
        let base = twoRow ? GenerationDefaults.forFamily(self, thinking: false) : profile
        let thinking = twoRow ? GenerationDefaults.forFamily(self, thinking: true) : nil
        return SamplingProfile(
            temperature: base.temperature, topK: base.topK,
            topP: base.topP, minP: base.minP,
            presencePenalty: base.presencePenalty,
            thinking: thinking.map {
                ThinkingDefaults(
                    enabledByDefault: true, temperature: $0.temperature,
                    topK: $0.topK, topP: $0.topP, presencePenalty: $0.presencePenalty)
            })
    }
}
