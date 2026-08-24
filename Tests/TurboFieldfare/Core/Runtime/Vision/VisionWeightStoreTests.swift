import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite struct VisionWeightStoreTests {
    private let zeroSHA = String(repeating: "0", count: 64)

    @Test func companionLocationPreservesTextModelStem() throws {
        let text = URL(fileURLWithPath: "/models/custom-name.gturbo")
        #expect(try VisionPackLocation.companionURL(forTextModel: text).path
                == "/models/custom-name.vision.gturbo")
        #expect(throws: VisionPackError.self) {
            try VisionPackLocation.companionURL(
                forTextModel: URL(fileURLWithPath: "/models/custom-name"))
        }
    }

    @Test func opensTrustedBoundedCompanionWithoutHashingWeights() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let store = try VisionWeightStore.open(
            directoryURL: fixture.root,
            compatibleTextSourceSnapshotHash: "text-snapshot",
            compatibleTextManifestSha256: zeroSHA)
        #expect(store.manifest.modelID == "fixture/model")
        #expect(store.manifest.tensors.map(\.name) == ["vision.weight"])
        let mapped = try store.mapRegion(
            executionPosition: 0,
            device: try #require(MTLCreateSystemDefaultDevice()))
        #expect(mapped.residentSize == 2)
        #expect(mapped.mappedLength == 2)
    }

    @Test func rejectsWrongTextBindingReceiptPathAndUnexpectedEntry() throws {
        let incompatible = try makeFixture()
        defer { try? FileManager.default.removeItem(at: incompatible.root) }
        #expect(throws: VisionPackError.self) {
            try VisionWeightStore.open(
                directoryURL: incompatible.root,
                compatibleTextSourceSnapshotHash: "other",
                compatibleTextManifestSha256: zeroSHA)
        }

        let wrongPath = try makeFixture(receiptDirectoryPath: "/tmp/partial")
        defer { try? FileManager.default.removeItem(at: wrongPath.root) }
        #expect(throws: VisionPackError.self) {
            try VisionWeightStore.open(
                directoryURL: wrongPath.root,
                compatibleTextSourceSnapshotHash: "text-snapshot",
                compatibleTextManifestSha256: zeroSHA)
        }

        let extra = try makeFixture()
        defer { try? FileManager.default.removeItem(at: extra.root) }
        try Data().write(to: extra.root.appendingPathComponent("extra"))
        #expect(throws: VisionPackError.self) {
            try VisionWeightStore.open(
                directoryURL: extra.root,
                compatibleTextSourceSnapshotHash: "text-snapshot",
                compatibleTextManifestSha256: zeroSHA)
        }
    }

    private func makeFixture(receiptDirectoryPath: String? = nil)
        throws -> (root: URL, manifest: GTurboVisionManifestV1) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-weight-store-\(UUID().uuidString).vision.gturbo")
        try FileManager.default.createDirectory(at: root,
                                                withIntermediateDirectories: false)

        let weights = Data(repeating: 0, count: Int(GTurboVisionFormatV1.alignmentBytes))
        let processor = Data("{}".utf8)
        let weightSHA = Sha256Verifier.hashData(weights)
        let processorSHA = Sha256Verifier.hashData(processor)
        let manifest = GTurboVisionManifestV1(
            modelID: "fixture/model", sourceRevision: "fixture-revision",
            sourceIndexSha256: zeroSHA,
            processorConfigSha256: processorSHA,
            compatibleTextSourceSnapshotHash: "text-snapshot",
            compatibleTextManifestSha256: zeroSHA,
            files: [
                GTurboVisionFormatV1.weightsFile: .init(
                    size: UInt64(weights.count), sha256: weightSHA),
                GTurboVisionFormatV1.processorFile: .init(
                    size: UInt64(processor.count), sha256: processorSHA),
            ],
            tensors: [.init(
                name: "vision.weight", executionPosition: 0,
                offset: 0, size: 2, shape: [1], dtype: .bf16)])
        let manifestData = try GTurboVisionManifestCodec.encode(manifest)
        let manifestSHA = Sha256Verifier.hashData(manifestData)
        let receipt = GTurboVisionReceiptV1(
            manifestSha256: manifestSHA,
            companionDirectoryPath: receiptDirectoryPath ?? root.standardizedFileURL.path,
            compatibleTextManifestSha256: zeroSHA,
            sourceRepoID: "fixture/model", sourceRevision: "fixture-revision",
            verificationTimestamp: "2026-08-05T00:00:00Z",
            toolVersion: "fixture",
            files: [
                GTurboVisionFormatV1.manifestFile: .init(
                    size: UInt64(manifestData.count), sha256: manifestSHA),
                GTurboVisionFormatV1.weightsFile: manifest.files[
                    GTurboVisionFormatV1.weightsFile]!,
                GTurboVisionFormatV1.processorFile: manifest.files[
                    GTurboVisionFormatV1.processorFile]!,
            ])

        try weights.write(to: root.appendingPathComponent(GTurboVisionFormatV1.weightsFile))
        try processor.write(to: root.appendingPathComponent(GTurboVisionFormatV1.processorFile))
        try manifestData.write(to: root.appendingPathComponent(GTurboVisionFormatV1.manifestFile))
        try GTurboVisionReceiptCodec.encode(receipt).write(
            to: root.appendingPathComponent(GTurboVisionFormatV1.receiptFile))
        return (root, manifest)
    }
}
