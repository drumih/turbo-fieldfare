import Testing
import TinyTitan
import TinyTitanCore

/// TT-134: the host reads its house sampling defaults from Core's
/// `SamplingProfile.house`; while the 6.x runtime still carries
/// `GenerationDefaults`, the two must not drift apart.
@Suite struct HouseSamplingDefaultsTests {
    @Test func coreHouseProfileEqualsTheSixXHouseRow() {
        let core = SamplingProfile.house
        let legacy = GenerationDefaults.house
        #expect(core.temperature == legacy.temperature)
        #expect(core.topK == legacy.topK)
        #expect(core.topP == legacy.topP)
        #expect(core.minP == legacy.minP)
        #expect(core.presencePenalty == legacy.presencePenalty)
    }
}
