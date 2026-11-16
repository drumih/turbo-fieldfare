import Foundation
import Synchronization
import Testing

@testable import TinyTitanRepackCore
@testable import TinyTitanCore

extension RemotePayloadCopyTests {
    @Test func remotePayloadCopyCompletes() async throws {
        let snapshotDir = tmpDirForRemote("snap")
        let remoteOutput = tmpPathForRemote("remote")
        defer { cleanUpRemote([snapshotDir, remoteOutput]) }
        let snapshot = try SyntheticSnapshot.build(
            at: snapshotDir,
            seed: 0x1020_3040_5060_7080)

        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDir,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: true)
        let recorder = InstallProgressRecorder()

        let result = try await RemoteStreamingRepacker(
            options: remoteOptions(
                outputDir: remoteOutput,
                session: fakeHFSession())
        ).run { recorder.append($0) }

        #expect(result.reusedBytes == 0)
        #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)
        for relativePath in [
            "model_weights.bin",
            "packed_experts/layout.json",
            "packed_experts/layer_00.bin",
            "packed_experts/layer_01.bin",
            "manifest.json",
        ] {
            let remote = (remoteOutput as NSString).appendingPathComponent(relativePath)
            #expect(FileManager.default.fileExists(atPath: remote))
        }
        #expect(recorder.values.contains(.finalizing))
        try assertRemoteTokenizerFilesRecorded(
            outputDir: remoteOutput,
            expectsOptionalSpecialTokens: true)
    }

    @Test func cancellationPreservesCommittedRangesForResume() async throws {
        let snapshotDir = tmpDirForRemote("snap-resume")
        let output = tmpPathForRemote("remote-resume")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.build(
            at: snapshotDir,
            seed: 0x56_4738_2910)

        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDir,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: false)
        let seen = Mutex<[UInt64: Int]>([:])
        let task = Task {
            try await RemoteStreamingRepacker(
                options: remoteOptions(outputDir: output, session: fakeHFSession())
            ).run { progress in
                guard case .copyingPayload(_, let downloaded, _) = progress,
                    downloaded > 0
                else { return }
                let count = seen.withLock {
                    $0[downloaded, default: 0] += 1
                    return $0[downloaded] ?? 0
                }
                if count == 3 {
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            }
        }
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }

        let checkpoint = try RemoteInstallCheckpoint.load(from: output + ".resume.json")
        #expect(!checkpoint.completedRanges.isEmpty)

        let result = try await RemoteStreamingRepacker(
            options: remoteOptions(
                outputDir: output,
                session: fakeHFSession(),
                resume: true)
        ).run()
        #expect(result.reusedBytes > 0)
        #expect(result.downloadedThisRunBytes < result.remoteBytesToDownload)
        #expect(
            FileManager.default.fileExists(
                atPath: (output as NSString).appendingPathComponent("manifest.json")))
    }

    /// Cancel a run partway through, so the partial directory and its checkpoint
    /// are left on disk exactly as an interrupted install leaves them.
    private func leaveResumableState(snapshotDir: String, output: String) async throws {
        let seen = Mutex<Int>(0)
        let task = Task {
            try await RemoteStreamingRepacker(
                options: remoteOptions(outputDir: output, session: fakeHFSession())
            ).run { progress in
                guard case .copyingPayload(_, let downloaded, _) = progress,
                    downloaded > 0
                else { return }
                let count = seen.withLock {
                    $0 += 1
                    return $0
                }
                if count == 3 {
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            }
        }
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
    }

    /// Why a resume refused, or what it did instead of refusing.
    private func resumeRefusal(output: String, repoID: String) async -> String {
        do {
            _ = try await RemoteStreamingRepacker(
                options: remoteOptions(
                    outputDir: output, session: fakeHFSession(),
                    resume: true, repoID: repoID)
            ).run()
            return "the resume completed instead of refusing"
        } catch let error as RepackError {
            return error.description
        } catch {
            return "refused for another reason: \(error)"
        }
    }

    @Test func resumingDownloadedRangesFromADifferentRepoRefuses() async throws {
        let snapshotDir = tmpDirForRemote("snap-resume-repo")
        let output = tmpPathForRemote("remote-resume-repo")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.build(
            at: snapshotDir,
            seed: 0x56_4738_2910)
        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDir,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: false)
        try await leaveResumableState(snapshotDir: snapshotDir, output: output)
        let checkpoint = try RemoteInstallCheckpoint.load(from: output + ".resume.json")
        #expect(!checkpoint.completedRanges.isEmpty)

        let reason = await resumeRefusal(
            output: output, repoID: "owner/a-different-model")
        #expect(
            reason.contains("different source"),
            "the saved ranges were copied from another repo: \(reason)")
    }

    @Test func resumingAfterTheSourceIndexChangedRefuses() async throws {
        let snapshotDir = tmpDirForRemote("snap-resume-index")
        let output = tmpPathForRemote("remote-resume-index")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.build(
            at: snapshotDir,
            seed: 0x56_4738_2911)
        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDir,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: false)
        try await leaveResumableState(snapshotDir: snapshotDir, output: output)
        let checkpoint = try RemoteInstallCheckpoint.load(from: output + ".resume.json")
        #expect(!checkpoint.completedRanges.isEmpty)

        let name = "model.safetensors.index.json"
        let original = try #require(FakeHFURLProtocol.files[name])
        let object = try #require(
            JSONSerialization.jsonObject(with: original) as? [String: Any])
        var drifted = object
        drifted["movedBy"] = "a commit the checkpoint does not know"
        FakeHFURLProtocol.files[name] = try JSONSerialization.data(withJSONObject: drifted)

        let reason = await resumeRefusal(output: output, repoID: checkpoint.repoID)
        #expect(
            reason.contains("copy plan changed"),
            "the index the resume reads is not the one the ranges came from: \(reason)")
    }
}
