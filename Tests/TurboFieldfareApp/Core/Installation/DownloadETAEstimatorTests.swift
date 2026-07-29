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

    @Test func holdsVisibleEstimateWithinUpdateInterval() {
        var estimator = DownloadETAEstimator()
        let mib = UInt64(1024 * 1024)
        _ = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 0, totalBytes: 600 * mib),
            timestamp: 0)
        let first = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 100 * mib, totalBytes: 600 * mib),
            timestamp: 10)
        let second = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 200 * mib, totalBytes: 600 * mib),
            timestamp: 39)
        #expect(first == second)
    }

    @Test func publishesUpdatedEstimateAfterUpdateInterval() {
        var estimator = DownloadETAEstimator()
        let mib = UInt64(1024 * 1024)
        _ = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 0, totalBytes: 600 * mib),
            timestamp: 0)
        let first = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 100 * mib, totalBytes: 600 * mib),
            timestamp: 10)
        let second = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 200 * mib, totalBytes: 600 * mib),
            timestamp: 40)

        #expect(first == .remaining(seconds: 50))
        #expect(second == .remaining(seconds: 80))
    }

    @Test func steadyDownloadAdvancesVisibleMinuteCountdown() {
        var estimator = DownloadETAEstimator()
        let mib = UInt64(1024 * 1024)
        let total = UInt64(3_640) * mib

        _ = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 0, totalBytes: total),
            timestamp: 0)
        let presentations = [
            estimator.update(
                .init(reusedBytes: 0, downloadedThisRunBytes: 40 * mib, totalBytes: total),
                timestamp: 10),
            estimator.update(
                .init(reusedBytes: 0, downloadedThisRunBytes: 280 * mib, totalBytes: total),
                timestamp: 70),
            estimator.update(
                .init(reusedBytes: 0, downloadedThisRunBytes: 520 * mib, totalBytes: total),
                timestamp: 130),
            estimator.update(
                .init(reusedBytes: 0, downloadedThisRunBytes: 760 * mib, totalBytes: total),
                timestamp: 190),
        ]

        #expect(presentations.compactMap(DownloadETAFormatter.string) == [
            "About 15 min remaining",
            "About 14 min remaining",
            "About 13 min remaining",
            "About 12 min remaining",
        ])
    }

    @Test func byteCounterRegressionRestartsWarmup() {
        var estimator = DownloadETAEstimator()
        let mib = UInt64(1024 * 1024)
        let total = UInt64(600) * mib
        _ = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 0, totalBytes: total),
            timestamp: 0)
        #expect(estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 40 * mib, totalBytes: total),
            timestamp: 10) != .estimating)

        #expect(estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 10 * mib, totalBytes: total),
            timestamp: 20) == .estimating)
    }

    @Test func completionHidesETA() {
        var estimator = DownloadETAEstimator()
        let mib = UInt64(1024 * 1024)
        _ = estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 0, totalBytes: 40 * mib),
            timestamp: 0)

        #expect(estimator.update(
            .init(reusedBytes: 0, downloadedThisRunBytes: 40 * mib, totalBytes: 40 * mib),
            timestamp: 10) == .hidden)
    }

    @Test func formatterUsesCoarseMinuteBuckets() {
        #expect(DownloadETAFormatter.remainingString(390) == "About 7 min remaining")
        #expect(DownloadETAFormatter.remainingString(410) == "About 7 min remaining")
    }
}
