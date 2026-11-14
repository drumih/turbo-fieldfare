import Foundation

/// Which architectures the CPU engine can serve, and a straight answer when
/// it cannot serve one.
///
/// The engine reads every dimension from the snapshot's own config — layer
/// count, head counts, head widths, the attention interval, the convolution
/// kernel — so a second model in the same family needs no code. What it
/// cannot read from a config is the *shape* of a layer: a Gated DeltaNet
/// with full attention every fourth layer, a gated MLP, and how the output
/// head attaches — tied to the embedding for the 2B and 4B, a separate
/// `lm_head` for the 9B. That is what this enumerates.
///
/// A model outside it is refused by name rather than run wrongly. This
/// project has shipped a converter whose folded norms were silently wrong
/// once; a forward pass that runs on the wrong architecture and produces
/// fluent nonsense is the same failure with a longer feedback loop.
public enum CPUModelFamily: String, Sendable, CaseIterable {
    /// Qwen3.5's dense text model: Gated DeltaNet, full attention every
    /// fourth layer, a fused attention output gate, partial rotary, and a
    /// tied embedding. Qwen3.5-2B is the model this was built against.
    case qwen35Dense = "qwen3_5_dense"
    /// The `model_type` values that map here. The converter writes the
    /// canonical one; the originals carry their own.
    static let aliases: [String: CPUModelFamily] = [
        "qwen3_5_dense": .qwen35Dense,
        "qwen3_5_text": .qwen35Dense,
        "qwen3_5": .qwen35Dense,
    ]

    public static func resolve(modelType: String?) -> CPUModelFamily? {
        guard let modelType else { return nil }
        return aliases[modelType.lowercased()]
    }

    /// Why a snapshot cannot be served here, in words an operator can act on.
    public static func refusal(modelType: String?) -> String {
        let named = modelType.map { "`\($0)`" } ?? "an unnamed architecture"
        return "the CPU engine does not implement \(named). It serves "
            + aliases.keys.sorted().joined(separator: ", ")
            + ". A new family needs its layer shape written, not just a "
            + "config: the dimensions come from the snapshot, the shape does "
            + "not."
    }

    /// Qwen3.5's template defines only the binary `enable_thinking` switch
    /// (and, unlike Qwen 3.6, treats an absent switch as off; the tokenizer
    /// always sets it, so that default never decides anything here).
    var reasoningControl: ModelReasoningControl {
        switch self {
        case .qwen35Dense: return .binaryThinking
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

    /// What plain "thinking on" loads for this family.
    public var levelWhenOn: ReasoningLevel { reasoningControl.levelWhenOn }

    /// The Qwen 3.5 series runs at temperature 0.6 / top-p 0.95; top-k is the
    /// house value. Stated here rather than borrowed from `house`, which
    /// holds the same numbers today, so a house change cannot move it.
    public var samplingDefaults: GenerationDefaults.Sampling {
        switch self {
        case .qwen35Dense:
            return GenerationDefaults.Sampling(
                temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95)
        }
    }
}
