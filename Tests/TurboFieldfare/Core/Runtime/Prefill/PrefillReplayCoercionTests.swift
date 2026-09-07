import Testing
@testable import TurboFieldfare

/// A stored conversation is replayed through prefill whatever the turn
/// preference says.
///
/// With prefill off, a replay ran through the per-token decode loop at decode
/// speed: a 12K-token reopen took longer than the service's restore deadline,
/// which then dropped the connection and forced a reload.
@Suite struct PrefillReplayCoercionTests {
    @Test func prefillOffIsCoercedToChunkedForAReplay() {
        let coerced = PrefillRuntimeConfig.off.coercedForReplay()
        #expect(coerced != nil)
        #expect(coerced?.mode == .chunked)
    }

    @Test func aPrefillThatAlreadyRunsIsLeftAlone() {
        #expect(PrefillRuntimeConfig.defaultChunked.coercedForReplay() == nil)
    }
}
