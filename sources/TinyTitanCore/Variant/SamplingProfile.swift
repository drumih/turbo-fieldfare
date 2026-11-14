import Foundation

/// A variant's sampling constants (7.0 spec §7), held in the variant target.
/// The fields are the instruct/default row; `thinking` carries the thinking-mode
/// row for checkpoints that publish two (the 6.x thinking/instruct two-row
/// pattern), or nil for a model with a single row.
public struct SamplingProfile: Sendable, Equatable {
    public let temperature: Float
    public let topK: Int
    public let topP: Float
    public let presencePenalty: Float
    public let thinking: ThinkingDefaults?

    public init(
        temperature: Float, topK: Int, topP: Float,
        presencePenalty: Float, thinking: ThinkingDefaults? = nil
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.presencePenalty = presencePenalty
        self.thinking = thinking
    }
}

/// The thinking-mode contract for a checkpoint that publishes a separate
/// thinking row: whether thinking is the card's default, and the sampling row
/// used while thinking. A model with one row leaves this nil.
public struct ThinkingDefaults: Sendable, Equatable {
    public let enabledByDefault: Bool
    public let temperature: Float
    public let topK: Int
    public let topP: Float
    public let presencePenalty: Float

    public init(
        enabledByDefault: Bool,
        temperature: Float = 1.0, topK: Int = 20, topP: Float = 0.95,
        presencePenalty: Float = 0
    ) {
        self.enabledByDefault = enabledByDefault
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.presencePenalty = presencePenalty
    }
}
