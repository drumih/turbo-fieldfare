import CryptoKit
import Foundation
import TurboFieldfare
import TurboFieldfareFormat

/// A text model with a valid companion pack installed beside it — the state in
/// which the app may actually accept images.
///
/// Tests that need image input must build this rather than lean on the default
/// model path: a developer machine with a real pack in `scratch/` made those
/// tests pass for a reason that had nothing to do with the test.
func makeVisionReadyModelInstall(_ tag: String) throws -> URL {
    let text = try makeCompleteModelInstall(tag)
    try installVisionCompanion(forTextModel: text)
    return text
}

@discardableResult
func installVisionCompanion(forTextModel text: URL) throws -> URL {
    func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    let textManifestData = try Data(
        contentsOf: text.appendingPathComponent("manifest.json"))
    let textManifestSHA = digest(textManifestData)
    guard let object = try JSONSerialization.jsonObject(with: textManifestData)
            as? [String: Any],
          let sourceSnapshotHash = object["sourceSnapshotHash"] as? String else {
        throw CocoaError(.fileReadCorruptFile)
    }

    let companion = try VisionPackLocation.companionURL(forTextModel: text)
    try FileManager.default.createDirectory(
        at: companion, withIntermediateDirectories: true)

    let weights = Data(repeating: 0, count: 4_096)
    let processor = Data("{}".utf8)
    let files = [
        GTurboVisionFormatV1.weightsFile: GTurboManifestFileV1(
            size: UInt64(weights.count), sha256: digest(weights)),
        GTurboVisionFormatV1.processorFile: GTurboManifestFileV1(
            size: UInt64(processor.count), sha256: digest(processor)),
    ]
    let revision = String(repeating: "a", count: 40)
    let manifest = GTurboVisionManifestV1(
        modelID: "fixture/model", sourceRevision: revision,
        sourceIndexSha256: String(repeating: "b", count: 64),
        processorConfigSha256: digest(processor),
        compatibleTextSourceSnapshotHash: sourceSnapshotHash,
        compatibleTextManifestSha256: textManifestSHA,
        files: files,
        tensors: [.init(name: "vision.weight", executionPosition: 0,
                        offset: 0, size: 2, shape: [1], dtype: .bf16)])
    let manifestData = try GTurboVisionManifestCodec.encode(manifest)
    let manifestSHA = digest(manifestData)
    let receipt = GTurboVisionReceiptV1(
        manifestSha256: manifestSHA,
        companionDirectoryPath: companion.standardizedFileURL.path,
        compatibleTextManifestSha256: textManifestSHA,
        sourceRepoID: "fixture/model", sourceRevision: revision,
        verificationTimestamp: "fixture", toolVersion: "fixture",
        files: files.merging([
            GTurboVisionFormatV1.manifestFile: GTurboManifestFileV1(
                size: UInt64(manifestData.count), sha256: manifestSHA),
        ]) { _, new in new })

    try weights.write(to: companion.appendingPathComponent(
        GTurboVisionFormatV1.weightsFile))
    try processor.write(to: companion.appendingPathComponent(
        GTurboVisionFormatV1.processorFile))
    try manifestData.write(to: companion.appendingPathComponent(
        GTurboVisionFormatV1.manifestFile))
    try GTurboVisionReceiptCodec.encode(receipt).write(
        to: companion.appendingPathComponent(GTurboVisionFormatV1.receiptFile))
    return companion
}
