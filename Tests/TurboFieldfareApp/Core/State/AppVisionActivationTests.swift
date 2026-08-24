import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareRepackCore
@testable import TurboFieldfareAppCore

/// Activation reads about 1.5 GB to verify the pack before it goes live. It
/// used to do that as an uncancellable, progress-free, app-wide block.
@Suite struct AppVisionActivationTests {
    @MainActor
    private func model(_ installer: MockVisionPackInstallerClient) -> AppModel {
        let model = AppModel(visionInstaller: installer)
        model.visionInstallState = .readyToActivate(
            FileManager.default.temporaryDirectory)
        return model
    }

    @MainActor
    @Test func activationReportsHowFarItsVerificationHasGot() async throws {
        let installer = MockVisionPackInstallerClient(
            preparedValid: true, activationProgress: [0.25, 0.5, 0.75, 1])
        let model = model(installer)
        model.activateVisionPack()

        var seen: [Double] = []
        // Generous, because this runs alongside the GPU suites: the assertions
        // below are about ordering and cleanup, not about how fast it got there.
        let deadline = Date().addingTimeInterval(60)
        while model.isInstallingVisionPack, Date() < deadline {
            if let fraction = model.visionInstallProgressFraction,
               seen.last != fraction {
                seen.append(fraction)
            }
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        try #require(!model.isInstallingVisionPack, "activation never finished")
        #expect(!seen.isEmpty, "activation showed no progress at all")
        #expect(seen == seen.sorted(), "progress went backwards: \(seen)")
        #expect(model.visionActivationProgress == nil,
                "progress outlived the activation")
    }

    /// The only long phase is the hash, and it runs before anything is renamed,
    /// so abandoning it leaves the prepared pack exactly as it was.
    @MainActor
    @Test func activationCanBeCancelledAndLeavesThePackActivatable() async throws {
        let installer = MockVisionPackInstallerClient(
            preparedValid: true,
            activationProgress: Array(repeating: 0.1, count: 200))
        let model = model(installer)
        model.activateVisionPack()
        #expect(model.canCancelVisionInstall,
                "activation offered no way out")

        try await Task.sleep(nanoseconds: 60_000_000)
        model.cancelVisionInstall()
        let deadline = Date().addingTimeInterval(60)
        while model.isInstallingVisionPack, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(!model.isInstallingVisionPack, "cancelling activation never landed")
        #expect(model.visionActivationProgress == nil)

        // Activating again is allowed: nothing was consumed.
        #expect(installer.activationCount == 1)
        model.visionInstallState = .readyToActivate(
            FileManager.default.temporaryDirectory)
        #expect(model.canActivateVisionPack)
    }

    /// A download that finished and verifies is activatable whatever happened
    /// afterwards; presenting it as "needs attention" hid Activate behind a
    /// Resume that only repeats work already done.
    @MainActor
    @Test func aValidPreparedDownloadStillOffersActivateAfterAFailure() async throws {
        // `.recoverable` is only reachable when a saved download exists on
        // disk, so the fixture has to be real.
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-activate-\(UUID().uuidString)",
                                    isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let text = root.appendingPathComponent("model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: text, withIntermediateDirectories: true)
        let companion = try VisionPackLocation.companionURL(forTextModel: text)
        let partial = companion.deletingLastPathComponent()
            .appendingPathComponent(companion.lastPathComponent + ".partial",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: partial, withIntermediateDirectories: true)

        let installer = MockVisionPackInstallerClient(preparedValid: true)
        let model = AppModel(modelDirectory: text, visionInstaller: installer)
        #expect(model.hasPartialVisionPackDownload)
        model.finishVisionInstallFailure(
            RepackError.configurationInvalid(detail: "network died"),
            generation: 0)

        guard case .readyToActivate = model.visionInstallState else {
            Issue.record("a verifiable download was hidden behind this state")
            return
        }
    }

    /// But a failure *during* verification must not send the user back to
    /// Activate, or the same corrupt pack is offered forever.
    @MainActor
    @Test func averificationFailureDoesNotOfferActivateAgain() async throws {
        let installer = MockVisionPackInstallerClient(
            preparedValid: true,
            activationError: .configurationInvalid(detail: "hash mismatch"))
        let model = model(installer)
        model.activateVisionPack()
        let deadline = Date().addingTimeInterval(60)
        while model.isInstallingVisionPack, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        try #require(!model.isInstallingVisionPack, "activation never finished")

        if case .readyToActivate = model.visionInstallState {
            Issue.record("a pack that failed verification was offered for activation again")
        }
    }
}
