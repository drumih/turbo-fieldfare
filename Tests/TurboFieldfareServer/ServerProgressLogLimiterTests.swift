import Testing
@testable import TurboFieldfareServerCore

struct ServerProgressLogLimiterTests {
    @Test func prefillLogsAtIntervalsAndCompletion() {
        let limiter = ServerProgressLogLimiter()

        #expect(!limiter.shouldLog(.prefill(done: 128, total: 2_500)))
        #expect(limiter.shouldLog(.prefill(done: 1_024, total: 2_500)))
        #expect(!limiter.shouldLog(.prefill(done: 1_536, total: 2_500)))
        #expect(limiter.shouldLog(.prefill(done: 2_048, total: 2_500)))
        #expect(limiter.shouldLog(.prefill(done: 2_500, total: 2_500)))
        #expect(!limiter.shouldLog(.prefill(done: 2_500, total: 2_500)))
    }

    @Test func decodeLogsFirstTokenAndIntervals() {
        let limiter = ServerProgressLogLimiter()

        #expect(limiter.shouldLog(.decode(completionTokens: 1)))
        #expect(!limiter.shouldLog(.decode(completionTokens: 15)))
        #expect(limiter.shouldLog(.decode(completionTokens: 16)))
        #expect(!limiter.shouldLog(.decode(completionTokens: 31)))
        #expect(limiter.shouldLog(.decode(completionTokens: 32)))
        #expect(!limiter.shouldLog(.decode(completionTokens: 32)))
    }
}
