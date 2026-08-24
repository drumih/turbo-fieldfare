import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// The companion download is 1.5 GB, so it needs the same answer to "how long
/// is this going to take" as the model download: a percentage and a remaining
/// time, not a bar on its own.
@Suite struct AppVisionInstallProgressTests {
    @MainActor
    private func makeModel(_ events: [AppModelInstallEvent])
        throws -> (AppModel, URL) {
        let directory = try makeCompleteModelInstall("vision-progress")
        let model = AppModel(
            modelDirectory: directory,
            client: MockLifecycleInferenceClient(),
            visionInstaller: MockVisionPackInstallerClient(
                events: events, holdOpen: true))
        return (model, directory)
    }

    /// Generous, because this runs alongside the GPU suites: these assertions
    /// are about what is reported, not about how fast it gets there.
    @MainActor
    private func waitUntil(_ reached: @MainActor () -> Bool) async {
        let deadline = Date().addingTimeInterval(60)
        while !reached(), Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    @MainActor
    @Test func downloadingReportsAFractionAndATimeEstimate() async throws {
        let (model, directory) = try makeModel([
            .checking,
            .copyingPayload(reusedBytes: 0,
                            downloadedThisRunBytes: 200_000_000,
                            totalBytes: 1_600_000_000),
        ])
        defer { try? FileManager.default.removeItem(at: directory) }

        model.installVisionPack()
        await waitUntil { model.visionInstallProgressFraction != nil }

        let fraction = try #require(model.visionInstallProgressFraction,
                                    "the download never reported progress")
        #expect(abs(fraction - 0.125) < 0.001)
        // The estimator needs a warmup before it will commit to a number, but
        // the user must be told that rather than shown nothing.
        #expect(model.visionInstallETAText != nil,
                "the companion download showed no time estimate at all")
        #expect(model.visionInstallETAPresentation != .hidden)
        model.cancelVisionInstall()
    }

    /// Once the bytes stop moving the estimate is meaningless and must go.
    @MainActor
    @Test func theEstimateIsClearedOnceDownloadingEnds() async throws {
        let (model, directory) = try makeModel([
            .copyingPayload(reusedBytes: 0,
                            downloadedThisRunBytes: 200_000_000,
                            totalBytes: 1_600_000_000),
            .finalizing,
        ])
        defer { try? FileManager.default.removeItem(at: directory) }

        model.installVisionPack()
        await waitUntil { model.visionInstallState == .finalizing }
        try #require(model.visionInstallState == .finalizing,
                     "the install never reached finalizing")
        #expect(model.visionInstallETAText == nil)
        #expect(model.visionInstallETAPresentation == .hidden)
        model.cancelVisionInstall()
    }
}
