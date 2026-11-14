import Foundation

/// The dense variant's sampling defaults, copied from the runtime's
/// GenerationDefaults (the house row the dense family uses). Provenance:
/// sources/TinyTitan/Runtime/Generation/SamplerTypes.swift. The variant owns
/// its sampling contract; only the dense house row is needed here.
public enum GenerationDefaults {
    public static let temperature: Float = 0.6
    public static let topK = 20
    public static let topP: Float = 0.95
    public static let presencePenalty: Float = 0
    public static let minP: Float = 0

    public struct Sampling: Sendable, Equatable {
        public var temperature: Float
        public var topK: Int
        public var topP: Float
        public var minP: Float
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
        temperature: temperature, topK: topK, topP: topP)
}
