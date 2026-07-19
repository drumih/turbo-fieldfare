import Foundation
import Synchronization
import Testing

@testable import TurboFieldfareRepackCore

extension RemotePayloadCopyTests {
  @Test func remoteRangeRetriesURLErrorAndCompletes() async throws {
    let snapshotDir = tmpDirForRemote("snap-retry-url")
    let remoteOutput = tmpPathForRemote("remote-retry-url")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0x8877_6655_4433)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    FakeHFURLProtocol.failures["GET:model-00001-of-00001.safetensors"] = [
      .url(.networkConnectionLost),
      .url(.timedOut),
    ]
    let audit = RepackAudit()
    let recorder = InstallProgressRecorder()

    let result = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: remoteOutput, session: fakeHFSession()),
      audit: audit
    ).run { recorder.append($0) }

    #expect(audit.remoteRangeRetries == 2)
    #expect((FakeHFURLProtocol.requestCounts["GET:model-00001-of-00001.safetensors"] ?? 0) >= 3)
    let payload = recorder.values.compactMap { event -> UInt64? in
      guard case .copyingPayload(let downloaded, _) = event else { return nil }
      return downloaded
    }
    #expect(payload.last == result.remoteBytesToDownload)
    #expect(zip(payload, payload.dropFirst()).allSatisfy { $0.0 <= $0.1 })
    try assertNoInternalRemoteDirs(outputDir: remoteOutput)
  }

  @Test func remotePersistentServerErrorStopsAfterConfiguredAttempts() async throws {
    let snapshotDir = tmpDirForRemote("snap-retry-500")
    let remoteOutput = tmpPathForRemote("remote-retry-500")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0x9988_7766_5544)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    FakeHFURLProtocol.failures["HEAD:model.safetensors.index.json"] = [
      .http(500),
      .http(500),
      .http(500),
    ]
    let audit = RepackAudit()

    await #expect(throws: RepackError.self) {
      _ = try await RemoteStreamingRepacker(
        options: remoteOptions(
          outputDir: remoteOutput,
          session: fakeHFSession(),
          rangeRetryAttempts: 3),
        audit: audit
      ).run()
    }

    #expect(FakeHFURLProtocol.requestCounts["HEAD:model.safetensors.index.json"] == 3)
    #expect(audit.remoteRangeRetries == 2)
    #expect(!FileManager.default.fileExists(atPath: remoteOutput))
  }

  @Test func remote404DoesNotRetry() async throws {
    let snapshotDir = tmpDirForRemote("snap-retry-404")
    let remoteOutput = tmpPathForRemote("remote-retry-404")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0xAA99_8877_6655)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    FakeHFURLProtocol.files.removeValue(forKey: "config.json")
    let audit = RepackAudit()

    await #expect(throws: RepackError.self) {
      _ = try await RemoteStreamingRepacker(
        options: remoteOptions(outputDir: remoteOutput, session: fakeHFSession()),
        audit: audit
      ).run()
    }

    #expect(FakeHFURLProtocol.requestCounts["HEAD:config.json"] == 1)
    #expect(audit.remoteRangeRetries == 0)
    #expect(!FileManager.default.fileExists(atPath: remoteOutput))
  }

  @Test func remoteTruncatedRangeRetriesAndCleansTemp() async throws {
    let snapshotDir = tmpDirForRemote("snap-retry-truncated")
    let remoteOutput = tmpPathForRemote("remote-retry-truncated")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snap = try SyntheticSnapshot.build(at: snapshotDir, seed: 0xBBAA_9988_7766)

    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snap,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: false)
    FakeHFURLProtocol.failures["GET:model-00001-of-00001.safetensors"] = [.truncatedBody]
    let audit = RepackAudit()

    _ = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: remoteOutput, session: fakeHFSession()),
      audit: audit
    ).run()

    #expect(audit.remoteRangeRetries == 1)
    try assertNoInternalRemoteDirs(outputDir: remoteOutput)
  }

}
