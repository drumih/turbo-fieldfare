import Testing

@testable import TurboFieldfareAppCore

@Suite
struct DownloadETAEstimatorTests {
    @Test func warmsUpAndIgnoresReusedBytesForRate() {
        var estimator = DownloadETAEstimator()
        #expect(estimator.update(
            .init(reusedBytes: 500, downloadedThisRunBytes: 0, totalBytes: 1_000),
            timestamp: 0) == .estimating)

        let result = estimator.update(
            .init(
                reusedBytes: 500,
                downloadedThisRunBytes: 100 * 1024 * 1024,
                totalBytes: 500 + 200 * 1024 * 1024),
            timestamp: 10)
        guard case .remaining(let seconds) = result else {
            Issue.record("expected an ETA")
            return
        }
        #expect(seconds > 9 && seconds < 11)
    }

    @Test func smallChangesKeepTheVisibleEstimateStable() {
        var estimator = DownloadETAEstimator()
        let mib = UInt64(1024 * 1024)
        _ = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 0, totalBytes: 600 * mib),
            timestamp: 0)
        let first = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 100 * mib, totalBytes: 600 * mib),
            timestamp: 10)
        let second = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 210 * mib, totalBytes: 600 * mib),
            timestamp: 41)
        #expect(first == second)
    }

    @Test func formatterUsesCoarseMinuteBuckets() {
        #expect(DownloadETAFormatter.remainingString(390) == "About 7 min remaining")
        #expect(DownloadETAFormatter.remainingString(410) == "About 7 min remaining")
    }
}
