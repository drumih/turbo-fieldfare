import Foundation
import Testing
@testable import TurboFieldfareRepackCore

/// The installer printed nothing between the command and "Installed" for a
/// 14.6 GB transfer. Both installers already took a progress closure and
/// `main.swift` passed none, so the data existed and never reached anyone.
@Suite struct InstallProgressReporterTests {
    @Test func everyPhaseProducesALine() {
        let cases: [ModelInstallProgress] = [
            .downloadingMetadata,
            .planning(downloadBytes: 1_500_000_000, outputBytes: 1_200_000_000),
            .reservingOutput(bytes: 1_200_000_000),
            .copyingPayload(reusedBytes: 0, downloadedThisRunBytes: 600_000_000,
                            totalBytes: 1_200_000_000),
            .hashingOutput("vision_weights.bin"),
            .finalizing,
        ]
        for progress in cases {
            #expect(InstallProgressReporter.line(for: progress) != nil,
                    "\(progress) reported nothing")
        }
    }

    @Test func theCopyLineCarriesBytesAndPercent() {
        let line = InstallProgressReporter.line(for: .copyingPayload(
            reusedBytes: 0, downloadedThisRunBytes: 600_000_000,
            totalBytes: 1_200_000_000))
        #expect(line == "[install] 572.2 MiB of 1.1 GiB (50%)")
    }

    /// A resumed install has bytes it did not fetch, and saying so is the
    /// difference between "stalled" and "already had half of it".
    @Test func aResumedCopySaysWhatWasAlreadyOnDisk() {
        let line = InstallProgressReporter.line(for: .copyingPayload(
            reusedBytes: 400_000_000, downloadedThisRunBytes: 200_000_000,
            totalBytes: 1_200_000_000))
        #expect(line?.contains("already on disk") == true)
    }

    /// Rate limited within a phase, never across one: a 14.6 GB copy must not
    /// scroll, and a phase change must not be swallowed by the throttle.
    @Test func linesAreThrottledPerPhaseNotGlobally() {
        final class Sink: @unchecked Sendable {
            var lines: [String] = []
        }
        let sink = Sink()
        let reporter = InstallProgressReporter(
            minimumInterval: 2, write: { sink.lines.append($0) })
        reporter.report(.copyingPayload(reusedBytes: 0, downloadedThisRunBytes: 1,
                                        totalBytes: 100), now: 0)
        reporter.report(.copyingPayload(reusedBytes: 0, downloadedThisRunBytes: 2,
                                        totalBytes: 100), now: 0.5)
        #expect(sink.lines.count == 1, "the copy counter was not throttled")
        reporter.report(.finalizing, now: 0.6)
        #expect(sink.lines.count == 2, "a phase change was throttled away")
        reporter.report(.copyingPayload(reusedBytes: 0, downloadedThisRunBytes: 3,
                                        totalBytes: 100), now: 3.0)
        #expect(sink.lines.count == 3)
    }

    @Test func byteFormattingStaysReadable() {
        #expect(InstallProgressReporter.bytes(512) == "512 B")
        #expect(InstallProgressReporter.bytes(1_144_373_248) == "1.1 GiB")
    }
}
