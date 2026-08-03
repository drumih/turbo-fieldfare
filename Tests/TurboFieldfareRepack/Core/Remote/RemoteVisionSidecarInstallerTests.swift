import Foundation
import Synchronization
import Testing

@testable import TurboFieldfareRepackCore

@Suite(.serialized)
struct RemoteVisionSidecarInstallerTests {
    @Test func augmentsCompletedModelWithoutChangingTextPayload() async throws {
        let snapshotDirectory = tmpDirForRemote("vision-snapshot")
        let output = tmpPathForRemote("vision-model")
        defer { cleanUpVision([snapshotDirectory, output]) }
        let snapshot = try SyntheticSnapshot.build(
            at: snapshotDirectory,
            seed: 0xABCD_1122_3344)

        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDirectory,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: false)
        _ = try await RemoteStreamingRepacker(
            options: remoteOptions(outputDir: output, session: fakeHFSession())
        ).run()

        let rootManifestPath = (output as NSString).appendingPathComponent("manifest.json")
        let rootWeightsPath = (output as NSString).appendingPathComponent("model_weights.bin")
        let manifestBefore = try Data(contentsOf: URL(fileURLWithPath: rootManifestPath))
        let weightsBefore = try Sha256Stream.hashFile(path: rootWeightsPath)
        FakeHFURLProtocol.requestCounts = [:]
        FakeHFURLProtocol.requestedRanges = [:]
        let audit = RepackAudit()

        let result = try await RemoteVisionSidecarInstaller(
            options: visionOptions(
                modelDirectory: output,
                session: fakeHFSession()),
            audit: audit
        ).run()

        #expect(result.entryCount == 3)
        #expect(result.sourceTensorCount == 7)
        #expect(result.reusedBytes == 0)
        #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)
        #expect(audit.wholeFileHeapBuffers == false)
        #expect(audit.largestRemoteTransferBytes <= 512)
        #expect(try Data(contentsOf: URL(fileURLWithPath: rootManifestPath)) == manifestBefore)
        #expect(try Sha256Stream.hashFile(path: rootWeightsPath) == weightsBefore)

        let visionDirectory = (output as NSString).appendingPathComponent("vision")
        for name in ["weights.bin", "manifest.json", "verified-install.json"] {
            #expect(FileManager.default.fileExists(
                atPath: (visionDirectory as NSString).appendingPathComponent(name)))
        }
        #expect(!FileManager.default.fileExists(
            atPath: (output as NSString).appendingPathComponent(".vision.partial")))
        #expect(!FileManager.default.fileExists(
            atPath: (output as NSString).appendingPathComponent(".vision.resume.json")))

        let verified = try VisionSidecarVerifier.run(modelDirectory: output)
        #expect(verified.entryCount == 3)
        #expect(verified.weightsBytesVerified > 0)
    }

    @Test func cancelledVisionInstallResumesCommittedRanges() async throws {
        let snapshotDirectory = tmpDirForRemote("vision-resume-snapshot")
        let output = tmpPathForRemote("vision-resume-model")
        defer { cleanUpVision([snapshotDirectory, output]) }
        let snapshot = try SyntheticSnapshot.build(
            at: snapshotDirectory,
            seed: 0x1122_7788_AABB)

        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDirectory,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: false)
        _ = try await RemoteStreamingRepacker(
            options: remoteOptions(outputDir: output, session: fakeHFSession())
        ).run()

        let seen = Mutex<[UInt64: Int]>([:])
        let task = Task {
            try await RemoteVisionSidecarInstaller(
                options: visionOptions(
                    modelDirectory: output,
                    session: fakeHFSession(),
                    rangeChunkBytes: 128)
            ).run { progress in
                guard case .copyingPayload(_, let downloaded, _) = progress,
                      downloaded > 0 else { return }
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

        let checkpointPath = (output as NSString)
            .appendingPathComponent(".vision.resume.json")
        let checkpoint = try RemoteInstallCheckpoint.load(from: checkpointPath)
        #expect(!checkpoint.completedRanges.isEmpty)

        let result = try await RemoteVisionSidecarInstaller(
            options: visionOptions(
                modelDirectory: output,
                session: fakeHFSession(),
                rangeChunkBytes: 128,
                resume: true)
        ).run()
        #expect(result.reusedBytes > 0)
        #expect(result.downloadedThisRunBytes < result.remoteBytesToDownload)
        _ = try VisionSidecarVerifier.run(modelDirectory: output)
    }

    @Test func discardRemovesOnlyVisionResumeState() throws {
        let output = tmpDirForRemote("vision-discard-model")
        defer { cleanUpVision([output]) }
        let partial = (output as NSString).appendingPathComponent(".vision.partial")
        let checkpoint = (output as NSString).appendingPathComponent(".vision.resume.json")
        let keep = (output as NSString).appendingPathComponent("keep.txt")
        try FileManager.default.createDirectory(
            atPath: partial,
            withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: URL(fileURLWithPath: checkpoint))
        try Data("keep".utf8).write(to: URL(fileURLWithPath: keep))

        try RemoteVisionSidecarInstaller.discardPartial(modelDirectory: output)

        #expect(!FileManager.default.fileExists(atPath: partial))
        #expect(!FileManager.default.fileExists(atPath: checkpoint))
        #expect(FileManager.default.fileExists(atPath: keep))
    }

    @Test func rejectsVisionSnapshotThatDoesNotMatchRootManifest() async throws {
        let snapshotDirectory = tmpDirForRemote("vision-mismatch-snapshot")
        let output = tmpPathForRemote("vision-mismatch-model")
        defer { cleanUpVision([snapshotDirectory, output]) }
        let snapshot = try SyntheticSnapshot.build(at: snapshotDirectory)

        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDirectory,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: false)
        _ = try await RemoteStreamingRepacker(
            options: remoteOptions(outputDir: output, session: fakeHFSession())
        ).run()

        let manifestPath = (output as NSString).appendingPathComponent("manifest.json")
        var manifest = try JSONSerialization.jsonObject(with:
            Data(contentsOf: URL(fileURLWithPath: manifestPath))) as! [String: Any]
        manifest["sourceSnapshotHash"] = "sha256:" + String(repeating: "0", count: 64)
        let changed = try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys])
        try changed.write(to: URL(fileURLWithPath: manifestPath), options: .atomic)

        await #expect(throws: RepackError.self) {
            _ = try await RemoteVisionSidecarInstaller(
                options: visionOptions(
                    modelDirectory: output,
                    session: fakeHFSession())
            ).run()
        }
        #expect(!FileManager.default.fileExists(
            atPath: (output as NSString).appendingPathComponent("vision")))
        #expect(!FileManager.default.fileExists(
            atPath: (output as NSString).appendingPathComponent(".vision.partial")))
    }
}

private func visionOptions(
    modelDirectory: String,
    session: RemoteDownloadSession,
    rangeChunkBytes: Int = 512,
    resume: Bool = false
) -> RemoteVisionSidecarOptions {
    RemoteVisionSidecarOptions(
        repoID: "owner/model",
        revision: "main",
        modelDirectory: modelDirectory,
        requireKnownSource: false,
        rangeChunkBytes: rangeChunkBytes,
        minFreeReserveBytes: 0,
        resume: resume,
        downloadSession: session,
        baseURL: URL(string: "https://hf.test")!,
        retryBaseDelayNs: 0)
}

private func cleanUpVision(_ paths: [String]) {
    for path in paths {
        cleanUpRemote([path])
        try? FileManager.default.removeItem(atPath:
            (path as NSString).appendingPathComponent(".vision.partial"))
        try? FileManager.default.removeItem(atPath:
            (path as NSString).appendingPathComponent(".vision.resume.json"))
    }
}
