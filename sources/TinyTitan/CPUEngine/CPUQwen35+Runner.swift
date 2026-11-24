import Foundation
import TinyTitanCore

/// The Core CPU runner contract (TT-097, extended by TT-134).
extension CPUQwen35: CPUEngineRunner {
    /// The family's own row; `house` only for a snapshot whose architecture
    /// no family claims, which the loader refuses before it gets here.
    public var samplingDefaults: SamplingProfile {
        guard let row = snapshot.family?.samplingDefaults else { return .house }
        return SamplingProfile(
            temperature: row.temperature, topK: row.topK, topP: row.topP,
            minP: row.minP, presencePenalty: row.presencePenalty)
    }

    public func makeResident() -> Int { snapshot.makeResident() }
}
