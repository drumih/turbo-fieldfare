import Foundation
import Testing

@testable import TinyTitanRepackCore
import TinyTitanCore

/// `RemoteStreamingRepacker.run()` reads the install state on disk before it
/// issues a single request, and every branch of that read is a decision about
/// someone's half-finished download. Each one is asserted here.
extension RemotePayloadCopyTests {

    /// Build a `<output>.partial` directory.
    private func makePartial(_ paths: RemoteInstallPaths) throws {
        try Posix.mkdirP(paths.partialDirectory)
    }

    /// Build a `<output>.resume.json` checkpoint. The preflight only asks
    /// whether the file exists; it does not read it before deciding.
    private func makeCheckpoint(_ paths: RemoteInstallPaths) throws {
        try Data("{}".utf8).write(to: URL(fileURLWithPath: paths.checkpointFile))
    }

    /// Build the final directory, optionally with the two markers that make it
    /// a verified install rather than a half-written one.
    private func makeFinal(
        _ paths: RemoteInstallPaths,
        manifest: Bool,
        receipt: Bool
    ) throws {
        try Posix.mkdirP(paths.finalDirectory)
        if manifest {
            try Data("{}".utf8).write(
                to: URL(
                    fileURLWithPath: (paths.finalDirectory as NSString)
                        .appendingPathComponent("manifest.json")))
        }
        if receipt {
            try Data("{}".utf8).write(
                to: URL(
                    fileURLWithPath: (paths.finalDirectory as NSString)
                        .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)))
        }
    }

    /// Why the preflight refused, or what it did instead of refusing.
    private func preflightReason(
        output: String,
        resume: Bool = false,
        overwrite: Bool = true
    ) async -> String {
        do {
            _ = try await RemoteStreamingRepacker(
                options: remoteOptions(
                    outputDir: output, session: fakeHFSession(),
                    resume: resume, overwrite: overwrite)
            ).run()
            return "the run completed instead of refusing"
        } catch let error as RepackError {
            return error.description
        } catch {
            return "refused for another reason: \(error)"
        }
    }

    @Test func freshRunIntoAnExistingInstallWithoutOverwriteRefuses() async throws {
        let root = tmpDirForRemote("preflight-overwrite")
        let paths = try RemoteInstallPaths(
            outputDirectory: (root as NSString).appendingPathComponent("model.ssdai"))
        defer { cleanUpRemote([root]) }
        try makeFinal(paths, manifest: true, receipt: true)

        let reason = await preflightReason(output: paths.finalDirectory, overwrite: false)
        #expect(
            reason.contains("output directory already exists"),
            "an install on disk is not overwritten without being told to: \(reason)")
    }

    @Test func resumeAfterTheCrashWindowClearsTheStaleCheckpoint() async throws {
        let root = tmpDirForRemote("preflight-crash")
        let paths = try RemoteInstallPaths(
            outputDirectory: (root as NSString).appendingPathComponent("model.ssdai"))
        defer { cleanUpRemote([root]) }
        // The rename into place succeeded and the checkpoint outlived it.
        try makeFinal(paths, manifest: true, receipt: true)
        try makeCheckpoint(paths)

        let reason = await preflightReason(output: paths.finalDirectory, resume: true)
        #expect(
            reason.contains("no resumable install state exists"),
            "a finished install has nothing to resume, and must not read as corrupt: \(reason)")
        #expect(
            !FileManager.default.fileExists(atPath: paths.checkpointFile),
            "the checkpoint that outlived the rename is dropped once the install is re-verified")
    }

    @Test func resumeIntoAnUnverifiedFinalDirectoryRefusesAndKeepsTheCheckpoint() async throws {
        let root = tmpDirForRemote("preflight-unverified")
        let paths = try RemoteInstallPaths(
            outputDirectory: (root as NSString).appendingPathComponent("model.ssdai"))
        defer { cleanUpRemote([root]) }
        try makeFinal(paths, manifest: true, receipt: false)
        try makeCheckpoint(paths)

        let reason = await preflightReason(output: paths.finalDirectory, resume: true)
        #expect(
            reason.contains("without a complete verified install"),
            "a final directory missing its receipt is not a finished install: \(reason)")
        #expect(
            FileManager.default.fileExists(atPath: paths.checkpointFile),
            "the refusal must not delete the state an operator still needs to read")
    }

    @Test func checkpointWithoutPartialRefusesAsCorrupt() async throws {
        let root = tmpDirForRemote("preflight-orphan-checkpoint")
        let paths = try RemoteInstallPaths(
            outputDirectory: (root as NSString).appendingPathComponent("model.ssdai"))
        defer { cleanUpRemote([root]) }
        try makeCheckpoint(paths)

        let reason = await preflightReason(output: paths.finalDirectory, resume: true)
        #expect(
            reason.contains("must exist together"),
            "a checkpoint with no partial directory behind it is corrupt, not resumable: \(reason)")
    }

    @Test func partialWithoutCheckpointRefusesAsCorrupt() async throws {
        let root = tmpDirForRemote("preflight-orphan-partial")
        let paths = try RemoteInstallPaths(
            outputDirectory: (root as NSString).appendingPathComponent("model.ssdai"))
        defer { cleanUpRemote([root]) }
        try makePartial(paths)

        let reason = await preflightReason(output: paths.finalDirectory)
        #expect(
            reason.contains("must exist together"),
            "a partial directory with no checkpoint has no verified ranges to reuse: \(reason)")
    }

    @Test func freshRunOverASavedDownloadAsksForResumeOrDiscard() async throws {
        let root = tmpDirForRemote("preflight-saved")
        let paths = try RemoteInstallPaths(
            outputDirectory: (root as NSString).appendingPathComponent("model.ssdai"))
        defer { cleanUpRemote([root]) }
        try makePartial(paths)
        try makeCheckpoint(paths)

        let reason = await preflightReason(output: paths.finalDirectory)
        #expect(
            reason.contains("resume or discard"),
            "starting over is the operator's decision, not the default: \(reason)")
        #expect(
            FileManager.default.fileExists(atPath: paths.partialDirectory),
            "the refusal must leave the transferred ranges on disk")
    }

    @Test func resumeWithNothingSavedRefuses() async throws {
        let root = tmpDirForRemote("preflight-nothing")
        let paths = try RemoteInstallPaths(
            outputDirectory: (root as NSString).appendingPathComponent("model.ssdai"))
        defer { cleanUpRemote([root]) }

        let reason = await preflightReason(output: paths.finalDirectory, resume: true)
        #expect(
            reason.contains("no resumable install state exists"),
            "--resume over an empty directory says so instead of starting a fresh download: \(reason)"
        )
    }
}
