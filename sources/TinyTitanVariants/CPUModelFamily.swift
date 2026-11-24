import Foundation
import TinyTitanCore

/// Which architectures the dense CPU variants can serve, and a straight answer
/// when they cannot serve one (moved out of 6.x under TT-134).
public enum VariantCPUModelFamily: String, Sendable, CaseIterable {
    /// Qwen3.5's dense text model: Gated DeltaNet, full attention every
    /// fourth layer, a fused attention output gate, partial rotary, and a
    /// tied embedding (2B/4B) or separate `lm_head` (9B).
    case qwen35Dense = "qwen3_5_dense"

    static let aliases: [String: VariantCPUModelFamily] = [
        "qwen3_5_dense": .qwen35Dense,
        "qwen3_5_text": .qwen35Dense,
        "qwen3_5": .qwen35Dense,
    ]

    public static func resolve(modelType: String?) -> VariantCPUModelFamily? {
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

    var reasoningControl: ReasoningControl {
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
    /// house value.
    public var samplingDefaults: SamplingProfile {
        switch self {
        case .qwen35Dense:
            return SamplingProfile(
                temperature: 0.6, topK: SamplingProfile.house.topK, topP: 0.95,
                presencePenalty: 0)
        }
    }
}
