import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// K2-Horizon's serving defaults: the model card asks for temperature 1.0,
/// top-p 0.95 and high reasoning effort. These are the values a request that
/// names neither inherits, so they are pinned rather than left to the house
/// row.
@Suite("K2 serving defaults")
struct K2SurfaceTests {
    @Test func samplingFollowsTheModelCard() {
        let sampling = GenerationDefaults.forFamily(.k2Horizon)
        #expect(sampling.temperature == 1.0)
        #expect(sampling.topP == 0.95)
        #expect(sampling.topK == GenerationDefaults.topK)
        // Thinking off does not change the card's row.
        #expect(
            GenerationDefaults.forFamily(.k2Horizon, thinking: false) == sampling)
    }

    @Test func reasoningDefaultsToHighEffort() {
        #expect(
            ModelFamily.k2Horizon.effectiveReasoningEffort(
                thinkingMode: .on, effort: nil) == .high)
        #expect(
            ModelFamily.k2Horizon.effectiveReasoningEffort(
                thinkingMode: .on, effort: .low) == .low)
        #expect(
            ModelFamily.k2Horizon.effectiveReasoningEffort(
                thinkingMode: .off, effort: .high) == nil)
        let settings = try? ModelFamily.k2Horizon.runtimeReasoning(for: .high)
        #expect(settings?.thinking == .on)
        #expect(settings?.effort == .high)
        #expect(ModelFamily.k2Horizon.reasoningControl.levelWhenOn == .high)
        #expect(!ModelFamily.k2Horizon.supportsThinkingOff)
        #expect(
            ModelFamily.k2Horizon.supportedReasoningLevels == [.low, .medium, .high])
    }
}
