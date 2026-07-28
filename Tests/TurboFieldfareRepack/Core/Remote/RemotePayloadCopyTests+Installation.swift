import Foundation
import Synchronization
import Testing

@testable import TurboFieldfareRepackCore

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
    #expect(FileManager.default.fileExists(
      atPath: (output as NSString).appendingPathComponent("manifest.json")))
  }
}
