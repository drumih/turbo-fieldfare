import Foundation
import Synchronization
import Testing

@testable import TurboFieldfareRepackCore

extension RemotePayloadCopyTests {
  @Test func qwenRemoteInstallCompletesWithFamilyManifest() async throws {
    let snapshotDir = tmpDirForRemote("qwen-snap")
    let remoteOutput = tmpPathForRemote("qwen-remote")
    defer { cleanUpRemote([snapshotDir, remoteOutput]) }
    let snapshot = try SyntheticSnapshot.buildQwen(
      at: snapshotDir,
      seed: 0x0102_0304_0506_0708)

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
      "packed_experts/layer_02.bin",
      "packed_experts/layer_03.bin",
      "manifest.json",
    ] {
      let remote = (remoteOutput as NSString).appendingPathComponent(relativePath)
      #expect(FileManager.default.fileExists(atPath: remote))
    }
    #expect(recorder.values.contains(.finalizing))
    try assertRemoteTokenizerFilesRecorded(
      outputDir: remoteOutput,
      expectsOptionalSpecialTokens: true)

    // The manifest carries the qwen36 family extension fields.
    let manifestData = try Data(contentsOf: URL(fileURLWithPath:
      (remoteOutput as NSString).appendingPathComponent("manifest.json")))
    let manifest = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
    let arch = manifest["arch"] as! [String: Any]
    #expect(arch["family"] as? String == "qwen36")
    #expect(arch["attnOutputGate"] as? Bool == true)
    #expect(arch["attentionScale"] as? Double == 0.125)
    #expect(arch["embeddingScaledBySqrtHidden"] as? Bool == false)
    #expect(arch["routerScaled"] as? Bool == false)
    #expect(arch["ffnSandwichNorms"] as? Bool == false)
    #expect(arch["sharedExpertGated"] as? Bool == true)
    #expect(arch["ropeNeoxSubdim"] as? Bool == true)
    #expect(arch["linearNumKHeads"] as? Int == 2)
    #expect(arch["linearNumVHeads"] as? Int == 4)
    #expect(arch["linearKeyHeadDim"] as? Int == 32)
    #expect(arch["linearValueHeadDim"] as? Int == 32)
    #expect(arch["linearConvKernelSize"] as? Int == 4)
    #expect(arch["fullAttentionLayerMask"] as? [Int] == [2, 2, 2, 1])
    #expect(arch["tieWordEmbeddings"] as? Bool == false)
    #expect(arch["hiddenActivation"] as? String == "silu")

    let quant = manifest["quant"] as! [String: [String: Any]]
    #expect(quant["embedding"]?["weightBits"] as? Int == 4)
    #expect(quant["attention"]?["weightBits"] as? Int == 4)
    #expect(quant["router"]?["weightBits"] as? Int == 8)
    #expect(quant["sharedExpert"]?["weightBits"] as? Int == 4)
    #expect(quant["routedExpert"]?["weightBits"] as? Int == 4)

    // The finished install passes post-hoc verification.
    let verify = try VerifiedInstallTool.run(
      options: VerifyInstallOptions(inputGTurbo: remoteOutput))
    #expect(verify.unexpectedEntries.isEmpty)
    #expect(verify.fileCount > 0)
  }

  @Test func qwenCancellationPreservesCommittedRangesForResume() async throws {
    let snapshotDir = tmpDirForRemote("qwen-snap-resume")
    let output = tmpPathForRemote("qwen-remote-resume")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snapshot = try SyntheticSnapshot.buildQwen(
      at: snapshotDir,
      seed: 0x99_1122_3344)

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
    let manifestPath = (output as NSString).appendingPathComponent("manifest.json")
    #expect(FileManager.default.fileExists(atPath: manifestPath))
    let manifest = try JSONSerialization.jsonObject(
      with: Data(contentsOf: URL(fileURLWithPath: manifestPath))) as! [String: Any]
    let arch = manifest["arch"] as! [String: Any]
    #expect(arch["family"] as? String == "qwen36")
  }
}
