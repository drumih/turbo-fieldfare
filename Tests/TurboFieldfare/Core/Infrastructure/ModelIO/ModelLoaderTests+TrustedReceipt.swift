import Foundation
import Metal
import Testing

@testable import TurboFieldfare

extension ModelLoaderTests {
  @Test func trustedReceiptModeRequiresReceipt() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir,
        device: device,
        expecting: .gemma4Toy(),
        integrityPolicy: .sizeCheckTrustedReceipt)
    } throws: { error in
      if case ModelError.trustedReceiptInvalid = error { return true }
      return false
    }
  }

  @Test func trustedReceiptReaderRejectsOversizedMetadataBeforeDecode() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)

    #expect {
      _ = try VerifiedInstallReceiptReader.load(directoryURL: dir, maxBytes: 16)
    } throws: { error in
      if case ModelError.trustedReceiptInvalid(let detail) = error {
        return detail.contains("metadata cap")
      }
      return false
    }
  }

  @Test func trustedReceiptModeSkipsSameSizeLayerShaMismatch() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let layerURL =
      dir
      .appendingPathComponent("packed_experts")
      .appendingPathComponent("layer_00.bin")
    try Self.flipByte(in: layerURL, at: 64)
    let device = try #require(MTLCreateSystemDefaultDevice())

    #expect {
      let defaultModel = try Model.load(
        directoryURL: dir,
        device: device,
        expecting: .gemma4Toy())
      _ = try defaultModel.routedExpert(layer: 0, expert: 0)
    } throws: { error in
      if case ModelError.checksumMismatch = error { return true }
      return false
    }

    let trustedModel = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)
    _ = try trustedModel.routedExpert(layer: 0, expert: 0)
  }

  @Test func trustedReceiptModeRejectsWrongSizedLayerFile() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let layerURL =
      dir
      .appendingPathComponent("packed_experts")
      .appendingPathComponent("layer_00.bin")
    let handle = try FileHandle(forWritingTo: layerURL)
    try handle.truncate(atOffset: 1024)
    try handle.close()

    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir,
        device: device,
        expecting: .gemma4Toy(),
        integrityPolicy: .sizeCheckTrustedReceipt)
    } throws: { error in
      if case ModelError.trustedReceiptInvalid = error { return true }
      return false
    }
  }

  @Test func trustedReceiptModeReportsReceiptValidationTiming() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let device = try #require(MTLCreateSystemDefaultDevice())
    var stats = ModelLoadStats()
    _ = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt,
      loadStats: &stats)
    #expect(stats.manifestSha256Nanos > 0)
    #expect(stats.receiptValidationNanos > 0)
    #expect(stats.eagerSha256Nanos > 0)
  }

  @Test func trustedReceiptModeRejectsExtraReceiptFileEntry() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      var files = root["files"] as! [String: Any]
      files["unexpected.bin"] = ["size": 0, "sha256": String(repeating: "0", count: 64)]
      root["files"] = files
    }
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir,
        device: device,
        expecting: .gemma4Toy(),
        integrityPolicy: .sizeCheckTrustedReceipt)
    } throws: { error in
      if case ModelError.trustedReceiptInvalid = error { return true }
      return false
    }
  }

  @Test func trustedReceiptModeRejectsMissingReceiptFileEntry() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      var files = root["files"] as! [String: Any]
      files.removeValue(forKey: "packed_experts/layer_00.bin")
      root["files"] = files
    }
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir,
        device: device,
        expecting: .gemma4Toy(),
        integrityPolicy: .sizeCheckTrustedReceipt)
    } throws: { error in
      if case ModelError.trustedReceiptInvalid(let detail) = error {
        return detail.contains("file set mismatch")
          && detail.contains("packed_experts/layer_00.bin")
      }
      return false
    }
  }

  @Test func trustedReceiptModeRejectsStaleManifestBindingBeforeManifestTrust() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      root["manifestSha256"] = String(repeating: "0", count: 64)
      var files = root["files"] as! [String: Any]
      files["manifest.json"] = [
        "size": (files["manifest.json"] as! [String: Any])["size"]!,
        "sha256": String(repeating: "0", count: 64),
      ]
      root["files"] = files
    }
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir,
        device: device,
        expecting: .gemma4Toy(),
        integrityPolicy: .sizeCheckTrustedReceipt)
    } throws: { error in
      if case ModelError.trustedReceiptInvalid(let detail) = error {
        return detail.contains("manifest SHA mismatch")
      }
      return false
    }
  }

  @Test func trustedReceiptModeRejectsDifferentModelDirectoryBinding() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      root["modelDirectoryPath"] =
        dir
        .deletingLastPathComponent()
        .appendingPathComponent("other.gturbo")
        .standardizedFileURL
        .path
    }
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir,
        device: device,
        expecting: .gemma4Toy(),
        integrityPolicy: .sizeCheckTrustedReceipt)
    } throws: { error in
      if case ModelError.trustedReceiptInvalid(let detail) = error {
        return detail.contains("model directory mismatch")
      }
      return false
    }
  }

}
