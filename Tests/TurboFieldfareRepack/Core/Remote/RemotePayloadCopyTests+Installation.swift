import Foundation
import Synchronization
import Testing

@testable import TurboFieldfareRepackCore

extension RemotePayloadCopyTests {
  @Test func remoteInstallProgressIsMonotonicAndComplete() async throws {
    let snapshotDir = tmpDirForRemote("snap-progress")
    let remoteOutput = tmpPathForRemote("remote-progress")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0x10_2938_4756)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    let recorder = InstallProgressRecorder()

    _ = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: remoteOutput, session: fakeHFSession())
    )
    .run { recorder.append($0) }

    let payload = recorder.values.compactMap { event -> (UInt64, UInt64)? in
      guard case .copyingPayload(let downloaded, let total) = event else { return nil }
      return (downloaded, total)
    }
    #expect(payload.count > 2)
    #expect(payload.first?.0 == 0)
    #expect(payload.last?.0 == payload.last?.1)
    #expect(
      zip(payload, payload.dropFirst()).allSatisfy { pair in
        pair.0.0 <= pair.1.0
      })
    #expect(payload.allSatisfy { $0.0 <= $0.1 })
    #expect(
      recorder.values.contains {
        if case .planning = $0 { return true }
        return false
      })
    #expect(
      recorder.values.contains {
        if case .hashingOutput = $0 { return true }
        return false
      })
    #expect(recorder.values.contains(.finalizing))
  }

  @Test func cancellationCleansPartialAndAllowsImmediateRetry() async throws {
    let snapshotDir = tmpDirForRemote("snap-cancel")
    let remoteOutput = tmpPathForRemote("remote-cancel")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0x56_4738_2910)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    let options = remoteOptions(
      outputDir: remoteOutput,
      session: fakeHFSession(),
      retainPartialOnFailure: false)

    let cancelledInstall = Task {
      try await RemoteStreamingRepacker(options: options).run { event in
        if case .copyingPayload(let downloaded, _) = event, downloaded > 0 {
          withUnsafeCurrentTask { $0?.cancel() }
        }
      }
    }
    await #expect(throws: CancellationError.self) {
      _ = try await cancelledInstall.value
    }

    #expect(!FileManager.default.fileExists(atPath: remoteOutput))
    #expect(
      !FileManager.default.fileExists(
        atPath: remoteOutput + ".partial.\(getpid())"))

    _ = try await RemoteStreamingRepacker(
      options: remoteOptions(
        outputDir: remoteOutput,
        session: fakeHFSession(),
        retainPartialOnFailure: false)
    ).run()
    #expect(
      FileManager.default.fileExists(
        atPath: (remoteOutput as NSString).appendingPathComponent("manifest.json")))
  }

  @Test func remoteTokenizerOptionalSpecialTokensMissingStillInstalls() async throws {
    let snapshotDir = tmpDirForRemote("snap-tokenizer-optional")
    let remoteOutput = tmpDirForRemote("remote-tokenizer-optional")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0x6655_4433_2211)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    let session = fakeHFSession()

    _ = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: remoteOutput, session: session)
    ).run()

    try assertRemoteTokenizerFilesRecorded(
      outputDir: remoteOutput,
      expectsOptionalSpecialTokens: false)

    let verification = try VerifiedInstallTool.run(
      options: VerifyInstallOptions(inputGTurbo: remoteOutput))
    #expect(FileManager.default.fileExists(atPath: verification.receiptPath))
    #expect(verification.fileCount > 1)
    #expect(verification.bytesVerified > 0)
    #expect(verification.unexpectedEntries.isEmpty)
  }

  @Test func remoteTokenizerMissingRequiredFileFailsInstall() async throws {
    let snapshotDir = tmpDirForRemote("snap-tokenizer-required")
    let remoteOutput = tmpPathForRemote("remote-tokenizer-required")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0x7766_5544_3322)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: false,
      includeOptionalTokenizer: true)
    let session = fakeHFSession()

    await #expect(throws: RepackError.self) {
      _ = try await RemoteStreamingRepacker(
        options: remoteOptions(outputDir: remoteOutput, session: session)
      ).run()
    }
    #expect(!FileManager.default.fileExists(atPath: remoteOutput))
  }

  @Test func remoteInstallRejectsPerTensorGroupSizeOverride() async throws {
    let snapshotDir = tmpDirForRemote("snap-group-size")
    let remoteOutput = tmpPathForRemote("remote-group-size")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0x65_726F_7570)

    resetFakeHF()
    var files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    let configData = try #require(files["config.json"])
    var config = try #require(
      JSONSerialization.jsonObject(with: configData) as? [String: Any])
    var quantization = try #require(config["quantization"] as? [String: Any])
    let key = "language_model.model.layers.0.router.proj"
    var override = try #require(quantization[key] as? [String: Any])
    override["group_size"] = 32
    quantization[key] = override
    config["quantization"] = quantization
    files["config.json"] = try JSONSerialization.data(withJSONObject: config)
    FakeHFURLProtocol.files = files

    await #expect(throws: RepackError.self) {
      _ = try await RemoteStreamingRepacker(
        options: remoteOptions(outputDir: remoteOutput, session: fakeHFSession())
      ).run()
    }
    #expect(!FileManager.default.fileExists(atPath: remoteOutput))
  }

}
