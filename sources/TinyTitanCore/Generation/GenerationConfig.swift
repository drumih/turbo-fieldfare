import Foundation

/// Generation knobs threaded from the caller through the generator into the
/// sampler. Pure value type; one per `generate(...)` call. The default values
/// are the house sampling constants (a family whose model card specifies
/// otherwise overrides them at the call site via its own defaults row).
public struct GenerationConfig: Sendable {
    public var maxNewTokens: Int = 256
    public var temperature: Float = 0.6
    public var topK: Int? = 20
    public var topP: Float? = 0.95
    /// OpenAI-compatible presence penalty: subtracted once from the logit of
    /// every id the history already contains. Zero is neutral.
    public var presencePenalty: Float = 0
    /// Min-p filter threshold. Zero means off, which is every shipped row; a
    /// non-zero value is refused by `validate` until the filter is implemented.
    public var minP: Float = 0
    public var repetitionPenalty: Float = 1.0
    public var seed: UInt64?  // nil = nondeterministic
    public var stopStrings: [String] = []
    public var extraStopTokens: Set<Int32> = []
    /// A grammar the sampled tokens must stay inside, when the request asked
    /// for structured output. Nil means the whole vocabulary is available.
    /// The constraint is stateful and is advanced by the decode loop, once per
    /// generated token; see `JSONConstraint`.
    public var constraint: JSONConstraint?

    public init(
        maxNewTokens: Int = 256,
        temperature: Float = 0.6,
        topK: Int? = 20,
        topP: Float? = 0.95,
        presencePenalty: Float = 0,
        minP: Float = 0,
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
package enum SamplePath: Sendable, Equatable {
    case greedyGPU
    case gpuSampled
    case hostPenalty
}

/// Which Top-K implementation serves a `1...64` sampled request. `tiled` is
/// production; `generic` forces the older single-threadgroup kernel. The two
/// are required to agree token-for-token, so this exists to measure the
/// difference, not to choose behavior.
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
