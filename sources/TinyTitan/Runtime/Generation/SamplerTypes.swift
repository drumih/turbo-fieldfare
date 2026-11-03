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

/// Generation knobs threaded from the caller through the `Generator` into the
/// sampler. Pure value type; one per `generate(...)` call.
///
/// Canonical home is here (the sampler is the primary consumer); `Generator`
/// reuses the same type rather than redeclaring it.
public struct GenerationConfig: Sendable {
    public var maxNewTokens: Int = 256
    public var temperature: Float = GenerationDefaults.temperature
    public var topK: Int? = GenerationDefaults.topK
    public var topP: Float? = GenerationDefaults.topP
    /// OpenAI-compatible presence penalty: subtracted once from the logit of
    /// every id the history already contains. Zero is neutral.
    public var presencePenalty: Float = GenerationDefaults.presencePenalty
    /// Min-p filter threshold. Zero means off, which is every shipped row; a
    /// non-zero value is refused by `validate` until the filter is implemented.
    public var minP: Float = GenerationDefaults.minP
    public var repetitionPenalty: Float = 1.0
    public var seed: UInt64?  // nil = nondeterministic
    public var stopStrings: [String] = []
    public var extraStopTokens: Set<Int32> = []
    /// A grammar the sampled tokens must stay inside, when the request asked
    /// for structured output. Nil -- the only value every caller but the
    /// server's JSON modes uses -- means the whole vocabulary is available and
    /// generation is bit-identical to what it was before constraints existed.
    ///
    /// The constraint is stateful and is advanced by the decode loop, once per
    /// generated token; see `JSONConstraint`.
    public var constraint: JSONConstraint?

    public init(
        maxNewTokens: Int = 256,
        temperature: Float = GenerationDefaults.temperature,
        topK: Int? = GenerationDefaults.topK,
        topP: Float? = GenerationDefaults.topP,
        presencePenalty: Float = GenerationDefaults.presencePenalty,
        minP: Float = GenerationDefaults.minP,
        repetitionPenalty: Float = 1.0,
        seed: UInt64? = nil,
        stopStrings: [String] = [],
        extraStopTokens: Set<Int32> = [],
        constraint: JSONConstraint? = nil
    ) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.presencePenalty = presencePenalty
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stopStrings = stopStrings
        self.extraStopTokens = extraStopTokens
        self.constraint = constraint
    }

    public func validate() throws {
        guard maxNewTokens > 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "maxNewTokens must be greater than zero")
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "temperature must be finite and nonnegative")
        }
        if let topK, !(1...256).contains(topK) {
            throw GeneratorError.invalidGenerationConfig(
                "topK must be between 1 and 256")
        }
        if let topP, !topP.isFinite || topP <= 0 || topP > 1 {
            throw GeneratorError.invalidGenerationConfig(
                "topP must be greater than zero and at most one")
        }
        // The only float the sampler divides by. At 0 the penalty pass computes
        // `z / 0` (+infinity), and below 1 it *multiplies* the repeated logit,
        // so a penalty under one rewards repetition -- the opposite of the flag.
        // Every production entry point checks this separately; a library caller
        // that went through `validate` alone did not.
        guard repetitionPenalty.isFinite, repetitionPenalty >= 1 else {
            throw GeneratorError.invalidGenerationConfig(
                "repetitionPenalty must be finite and at least one")
        }
        guard presencePenalty.isFinite else {
            throw GeneratorError.invalidGenerationConfig(
                "presencePenalty must be finite")
        }
        guard minP.isFinite, minP >= 0, minP < 1 else {
            throw GeneratorError.invalidGenerationConfig(
                "minP must be at least zero and below one")
        }
        guard minP == 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "minP is not implemented; every published row uses zero")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw GeneratorError.invalidGenerationConfig(
                "topP below one requires topK; full-vocabulary nucleus sampling is not implemented")
        }
    }

}

/// Which path a `sample(...)` call took.
enum SamplePath: Sendable, Equatable {
    case greedyGPU
    case gpuSampled
    case hostPenalty
}

/// Which Top-K implementation serves a `1...64` sampled request.
///
/// `tiled` is production: a three-stage reduction that keeps the top 64 of
/// every 1,024-entry tile, so the whole vocabulary reaches one final tile in
/// three dispatches. `generic` forces the older single-threadgroup kernel that
/// extracts Top-K in k full vocabulary passes.
///
/// The two are required to agree token-for-token — `SampleTopK64Tests` pins
/// that across k, temperature, and seed — so this exists to measure the
/// difference, not to choose behavior. It is the control arm that made the
/// +30.07% (4-bit) / +11.36% (8-bit) qualification an interleaved same-binary
/// A/B instead of a comparison against a separately-built baseline.
public enum RuntimeSamplerPath: String, Codable, Sendable {
    case tiled
    case generic

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeSamplerPath {
        guard let raw = environment["TINYTITAN_SAMPLER_PATH"] else { return .tiled }
        guard let value = RuntimeSamplerPath(rawValue: raw) else {
            throw GeneratorError.invalidSamplerPath(raw)
        }
        return value
    }
}
