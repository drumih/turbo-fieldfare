import Foundation
import Testing
@testable import TurboFieldfareFormat

private enum VisionFormatFixture {
    static let zeroSHA = String(repeating: "0", count: 64)

    static func manifest(tensors: [GTurboVisionTensorRegionV1]? = nil)
        -> GTurboVisionManifestV1 {
        let defaultTensors = [
            GTurboVisionTensorRegionV1(
                name: "vision_tower.patch_embedder.input_proj.weight",
                executionPosition: 0, offset: 0, size: 512,
                shape: [16, 16], dtype: .bf16),
            GTurboVisionTensorRegionV1(
                name: "embed_vision.embedding_projection.weight",
                executionPosition: 1, offset: GTurboVisionFormatV1.alignmentBytes,
                size: 256, shape: [8, 8], dtype: .u32,
                quantization: GTurboVisionQuantizationV1(
                    weightBits: 4, scheme: "affine", scaleType: "bf16",
                    biasType: "bf16", groupSize: 64)),
        ]
        return GTurboVisionManifestV1(
            modelID: "fixture/model", sourceRevision: "fixture-revision",
            sourceIndexSha256: zeroSHA, processorConfigSha256: zeroSHA,
            compatibleTextSourceSnapshotHash: "fixture-text-snapshot",
            compatibleTextManifestSha256: zeroSHA,
            files: [
                GTurboVisionFormatV1.weightsFile: GTurboManifestFileV1(
                    size: 2 * GTurboVisionFormatV1.alignmentBytes, sha256: zeroSHA),
                GTurboVisionFormatV1.processorFile: GTurboManifestFileV1(
                    size: 2, sha256: zeroSHA),
            ],
            tensors: tensors ?? defaultTensors)
    }

    static func receipt() -> GTurboVisionReceiptV1 {
        GTurboVisionReceiptV1(
            manifestSha256: zeroSHA,
            companionDirectoryPath: "/fixture/model.vision.gturbo",
            compatibleTextManifestSha256: zeroSHA,
            sourceRepoID: "fixture/model", sourceRevision: "fixture-revision",
            verificationTimestamp: "2026-08-05T00:00:00Z",
            toolVersion: "fixture",
            files: [
                GTurboVisionFormatV1.manifestFile: .init(size: 1, sha256: zeroSHA),
                GTurboVisionFormatV1.weightsFile: .init(
                    size: 2 * GTurboVisionFormatV1.alignmentBytes, sha256: zeroSHA),
                GTurboVisionFormatV1.processorFile: .init(size: 2, sha256: zeroSHA),
            ])
    }
}

@Suite struct GTurboVisionFormatCompatibilityTests {
    @Test func manifestAndReceiptRoundTripDeterministically() throws {
        let manifest = VisionFormatFixture.manifest()
        let manifestData = try GTurboVisionManifestCodec.encode(manifest)
        #expect(try GTurboVisionManifestCodec.decode(manifestData) == manifest)
        #expect(try GTurboVisionManifestCodec.encode(manifest) == manifestData)

        let receipt = VisionFormatFixture.receipt()
        let receiptData = try GTurboVisionReceiptCodec.encode(receipt)
        #expect(try GTurboVisionReceiptCodec.decode(receiptData) == receipt)
        #expect(try GTurboVisionReceiptCodec.encode(receipt) == receiptData)
    }

    @Test func rejectsUnknownVersion() {
        let original = VisionFormatFixture.manifest()
        let changed = GTurboVisionManifestV1(
            versionMajor: 2, modelID: original.modelID,
            sourceRevision: original.sourceRevision,
            sourceIndexSha256: original.sourceIndexSha256,
            processorConfigSha256: original.processorConfigSha256,
            compatibleTextSourceSnapshotHash: original.compatibleTextSourceSnapshotHash,
            compatibleTextManifestSha256: original.compatibleTextManifestSha256,
            files: original.files, tensors: original.tensors)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(changed)
        }
    }

    @Test func rejectsReorderedRegions() {
        let original = VisionFormatFixture.manifest().tensors
        let changed = [
            GTurboVisionTensorRegionV1(
                name: original[0].name, executionPosition: 1,
                offset: original[0].offset, size: original[0].size,
                shape: original[0].shape, dtype: original[0].dtype),
            original[1],
        ]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: changed))
        }
    }

    @Test func rejectsMisalignedOverlappingAndOutOfFileRegions() {
        let invalid: [GTurboVisionTensorRegionV1] = [
            .init(name: "a", executionPosition: 0, offset: 1,
                  size: 2, shape: [1], dtype: .bf16),
            .init(name: "b", executionPosition: 1,
                  offset: 2 * GTurboVisionFormatV1.alignmentBytes,
                  size: 2, shape: [1], dtype: .bf16),
        ]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: invalid))
        }
    }

    @Test func rejectsInvalidStorageSizeAndQuantization() {
        let badSize = [GTurboVisionTensorRegionV1(
            name: "a", executionPosition: 0, offset: 0,
            size: 3, shape: [1], dtype: .bf16)]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: badSize))
        }

        let badQuant = [GTurboVisionTensorRegionV1(
            name: "a", executionPosition: 0, offset: 0,
            size: 4, shape: [1], dtype: .u32)]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: badQuant))
        }
    }

    @Test func rejectsUnexpectedPayloadOrReceiptFile() {
        let original = VisionFormatFixture.manifest()
        var files = original.files
        files["extra.bin"] = .init(size: 1, sha256: VisionFormatFixture.zeroSHA)
        let changed = GTurboVisionManifestV1(
            modelID: original.modelID, sourceRevision: original.sourceRevision,
            sourceIndexSha256: original.sourceIndexSha256,
            processorConfigSha256: original.processorConfigSha256,
            compatibleTextSourceSnapshotHash: original.compatibleTextSourceSnapshotHash,
            compatibleTextManifestSha256: original.compatibleTextManifestSha256,
            files: files, tensors: original.tensors)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(changed)
        }

        let receipt = VisionFormatFixture.receipt()
        let missing = GTurboVisionReceiptV1(
            manifestSha256: receipt.manifestSha256,
            companionDirectoryPath: receipt.companionDirectoryPath,
            compatibleTextManifestSha256: receipt.compatibleTextManifestSha256,
            sourceRepoID: receipt.sourceRepoID,
            sourceRevision: receipt.sourceRevision,
            verificationTimestamp: receipt.verificationTimestamp,
            toolVersion: receipt.toolVersion,
            files: [GTurboVisionFormatV1.manifestFile:
                    receipt.files[GTurboVisionFormatV1.manifestFile]!])
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionReceiptCodec.encode(missing)
        }
    }
}
