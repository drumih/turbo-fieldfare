import Testing
@testable import TurboFieldfare

@Suite struct RDAdvicePolicyTests {
    @Test func parsesAdaptiveSeparatelyFromBounded() {
        #expect(RDAdvicePolicyMode.parse("default") == .default)
        #expect(RDAdvicePolicyMode.parse("off") == .off)
        #expect(RDAdvicePolicyMode.parse("bounded") == .bounded)
        #expect(RDAdvicePolicyMode.parse("adaptive") == .adaptive)
    }

    @Test func adaptiveSkipsLargeMissFanout() {
        let state = RDAdviceAdaptivePolicyState()
        #expect(state.shouldSkip(position: 128,
                                 requestedMisses: 16,
                                 estimatedBytes: 16 * 3_358_720,
                                 canOverlapUsefulGPUWork: true))
    }

    @Test func adaptiveAllowsModerateFanoutBeforeSlowCall() {
        let state = RDAdviceAdaptivePolicyState()
        #expect(!state.shouldSkip(position: 128,
                                  requestedMisses: 8,
                                  estimatedBytes: 8 * 3_358_720,
                                  canOverlapUsefulGPUWork: true))
    }

    @Test func adaptiveSkipsRemainderOfCurrentProducedTokenAfterSlowCall() {
        var state = RDAdviceAdaptivePolicyState()
        state.update(after: ExpertIOAdviceResult(requested: 8,
                                                 failed: 0,
                                                 calls: 8,
                                                 bytes: 8 * 3_358_720,
                                                 maxCallNanos: 1_500_000),
                     position: 128)
        #expect(state.shouldSkip(position: 128,
                                 requestedMisses: 8,
                                 estimatedBytes: 8 * 3_358_720,
                                 canOverlapUsefulGPUWork: true))
        #expect(!state.shouldSkip(position: 129,
                                  requestedMisses: 8,
                                  estimatedBytes: 8 * 3_358_720,
                                  canOverlapUsefulGPUWork: true))
    }

    @Test func adaptiveSkipsWhenAdviceCannotOverlapUsefulWork() {
        let state = RDAdviceAdaptivePolicyState()
        #expect(state.shouldSkip(position: 128,
                                 requestedMisses: 8,
                                 estimatedBytes: 8 * 3_358_720,
                                 canOverlapUsefulGPUWork: false))
    }

    @Test func adaptiveSkipsLargeByteFanout() {
        let state = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(missCap: 128,
                                                 byteCap: 32 * 1_048_576,
                                                 slowCallNanos: 1_000_000))
        #expect(state.shouldSkip(position: 128,
                                 requestedMisses: 16,
                                 estimatedBytes: 16 * 3_358_720,
                                 canOverlapUsefulGPUWork: true))
    }
}
