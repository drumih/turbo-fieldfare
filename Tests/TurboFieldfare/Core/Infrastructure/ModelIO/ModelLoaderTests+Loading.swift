import Foundation
import Metal
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareFormat

extension ModelLoaderTests {
  @Test func loadsValidDirectory() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .gemma4Toy())
    let embed = model.embedding
    #expect(embed.length == UInt64(1024 * 64 / 2))
    #expect(embed.shape.0 == 1024 && embed.shape.1 == 64)
    let norm = model.finalNorm
    #expect(norm.length == UInt64(64 * 2))
    // Tied lm_head returns the same view as embedding.
    #expect(model.lmHead.offset == model.embedding.offset)
    #expect(model.lmHead.length == model.embedding.length)
  }

  @Test func loadsLayoutDeclaredNonstandardLayerFilenames() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let packed = dir.appendingPathComponent("packed_experts")
    let layoutURL = packed.appendingPathComponent("layout.json")
    var layout = try JSONSerialization.jsonObject(
      with: Data(contentsOf: layoutURL)) as! [String: Any]
    var layers = layout["layers"] as! [[String: Any]]

    let manifestURL = dir.appendingPathComponent("manifest.json")
    var manifest = try JSONSerialization.jsonObject(
      with: Data(contentsOf: manifestURL)) as! [String: Any]
    var files = manifest["files"] as! [String: [String: Any]]
    for index in layers.indices {
      let oldName = layers[index]["file"] as! String
      let newName = "experts_\(index).bin"
      try FileManager.default.moveItem(
        at: packed.appendingPathComponent(oldName),
        to: packed.appendingPathComponent(newName))
      layers[index]["file"] = newName
      files["packed_experts/\(newName)"] = files.removeValue(
        forKey: "packed_experts/\(oldName)")
    }
    layout["layers"] = layers
    let layoutData = try JSONSerialization.data(
      withJSONObject: layout, options: [.sortedKeys, .withoutEscapingSlashes])
    try layoutData.write(to: layoutURL)
    files["packed_experts/layout.json"]?["size"] = layoutData.count
    files["packed_experts/layout.json"]?["sha256"] =
      try Sha256Verifier.hashFile(at: layoutURL)
    manifest["files"] = files
    try JSONSerialization.data(
      withJSONObject: manifest, options: [.sortedKeys, .withoutEscapingSlashes])
      .write(to: manifestURL)

    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device, expecting: .gemma4Toy())
    let expert = try model.routedExpert(layer: 1, expert: 2)
    let bytes = expert.buffer.contents().advanced(by: Int(expert.offset))
      .assumingMemoryBound(to: UInt8.self)
    #expect(bytes[0] == 1)
    #expect(bytes[1] == 2)
  }

  @Test func residentBytesAreReadableFromBuffer() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .gemma4Toy())
    let norm = model.finalNorm
    let contents = norm.buffer.contents()
    // Norm region was patterned 0xC0 | (i & 0x3F).
    for i in 0..<Int(norm.length) {
      let got = contents.load(fromByteOffset: Int(norm.offset) + i, as: UInt8.self)
      #expect(got == UInt8(0xC0 | (i & 0x3F)), "norm byte \(i)")
    }
  }

  @Test func missingManifestFailsPartialInstall() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try FileManager.default.removeItem(at: dir.appendingPathComponent("manifest.json"))
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir, device: device,
        expecting: .gemma4Toy())
    } throws: { error in
      if case ModelError.partialInstall = error { return true }
      return false
    }
  }

  @Test func mismatchedShaFailsChecksumMismatch() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    // Flip one byte at the very end of the resident region (not inside
    // the index, which the loader reads earlier and would error
    // differently). Manifest sha was computed before this corruption.
    let url = dir.appendingPathComponent("model_weights.bin")
    var data = try Data(contentsOf: url)
    data[data.count - 1] ^= 0xFF
    try data.write(to: url)
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir, device: device,
        expecting: .gemma4Toy())
    } throws: { error in
      if case ModelError.checksumMismatch = error { return true }
      return false
    }
  }

  @Test func malformedZeroPayloadResidentEntryThrowsWithoutTrap() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let weightsURL = dir.appendingPathComponent("model_weights.bin")
    var weights = try Data(contentsOf: weightsURL)
    let firstEntry = GTurboFormatV1.residentHeaderBytes
    weights.replaceSubrange((firstEntry + 8)..<(firstEntry + 24),
                            with: repeatElement(UInt8(0), count: 16))
    try weights.write(to: weightsURL)

    let manifestURL = dir.appendingPathComponent("manifest.json")
    var manifest = try JSONSerialization.jsonObject(
      with: Data(contentsOf: manifestURL)) as! [String: Any]
    var files = manifest["files"] as! [String: [String: Any]]
    files["model_weights.bin"]?["sha256"] = try Sha256Verifier.hashFile(at: weightsURL)
    manifest["files"] = files
    try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
      .write(to: manifestURL)

    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
    } throws: { error in
      if case ModelError.indexCorrupt(let detail) = error {
        return detail.contains("primary payload is empty")
      }
      return false
    }
  }

  @Test func integrityPoliciesExposeIdenticalResidentAndRoutedBytes() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let device = try #require(MTLCreateSystemDefaultDevice())
    let full = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .fullSha256)
    let trusted = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      integrityPolicy: .sizeCheckTrustedReceipt)

    #expect(full.embedding.length == trusted.embedding.length)
    let fullEmbedding = full.embedding.buffer.contents().advanced(by: Int(full.embedding.offset))
    let trustedEmbedding = trusted.embedding.buffer.contents().advanced(
      by: Int(trusted.embedding.offset))
    #expect(memcmp(fullEmbedding, trustedEmbedding, Int(full.embedding.length)) == 0)

    let fullExpert = try full.routedExpert(layer: 0, expert: 0)
    let trustedExpert = try trusted.routedExpert(layer: 0, expert: 0)
    #expect(fullExpert.length == trustedExpert.length)
    let fullExpertBytes = fullExpert.buffer.contents().advanced(by: Int(fullExpert.offset))
    let trustedExpertBytes = trustedExpert.buffer.contents().advanced(by: Int(trustedExpert.offset))
    #expect(memcmp(fullExpertBytes, trustedExpertBytes, Int(fullExpert.length)) == 0)
  }

  @Test func nonPageAlignedExpertStrideFailsAtManifest() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let manifestURL = dir.appendingPathComponent("manifest.json")
    var root =
      try JSONSerialization.jsonObject(
        with: Data(contentsOf: manifestURL)) as! [String: Any]
    root["expertStride"] = 1024
    let data = try JSONSerialization.data(withJSONObject: root)
    try data.write(to: manifestURL)
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir, device: device,
        expecting: .gemma4Toy())
    } throws: { error in
      if case ModelError.expertStrideNotPageAligned = error { return true }
      return false
    }
  }

}
