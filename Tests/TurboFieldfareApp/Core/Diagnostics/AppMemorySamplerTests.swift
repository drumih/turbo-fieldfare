import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppMemorySamplerTests {
    @Test func samplingDoesNotThrowAndTracksPeak() {
        let sampler = AppMemorySampler()
        sampler.resetPeak()
        _ = sampler.sample()
        _ = sampler.sample()
        if let peak = sampler.peakBytes {
            #expect(peak > 0)
        }
    }

    @Test func samplingReportsProcessFootprint() {
        let footprint: UInt64 = 2 * 1_024 * 1_024 * 1_024
        let sampler = AppMemorySampler(processFootprint: { footprint })

        #expect(sampler.sample() == footprint)
        #expect(sampler.peakBytes == footprint)
    }

    @Test func failedSampleDoesNotSetPeak() {
        let sampler = AppMemorySampler(processFootprint: { nil })

        #expect(sampler.sample() == nil)
        #expect(sampler.peakBytes == nil)
    }
}
