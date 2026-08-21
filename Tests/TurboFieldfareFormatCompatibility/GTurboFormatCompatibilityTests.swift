import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite struct GTurboFormatCompatibilityTests {
    @Test func productionWritersMatchPreRefactorV1Fixtures() throws {
        let fixture = makeFixture()
        let manifestData = try GTurboJSON.encodeManifest(
            plan: fixture.plan,
            modelID: "fixture/model",
            sourceSnapshotHash: "fixture-snapshot",
            files: fixture.files,
            expertsPerLayer: fixture.config.numExperts,
            numLayers: fixture.config.numLayers,
            expertStride: fixture.expertStride,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 4,
                attention: 4,
                router: 8,
                sharedExpert: 4,
                routedExpert: 4))
        let layoutData = try GTurboJSON.encodeLayout(
            plan: fixture.plan, expertStride: fixture.expertStride)
        let indexData = try ResidentWriter.encodeIndex(plan: fixture.plan.resident)

        let frozenManifest = try frozenFixture("manifest.json")
        let frozenLayout = try frozenFixture("layout.json")
        let frozenIndex = try frozenFixture("resident-index.bin")
        #expect(manifestData == frozenManifest)
        #expect(layoutData == frozenLayout)
        #expect(indexData == frozenIndex)

        let manifest = try ManifestReader.decode(
            data: frozenManifest,
            expecting: fixture.config)
        let layout = try PackedExpertsLayoutReader.decode(
            data: frozenLayout, manifest: manifest)
        #expect(layout.expert(layer: 0, expert: 1).offset == fixture.expertStride)
        let manifestRoot = try #require(
            JSONSerialization.jsonObject(with: frozenManifest) as? [String: Any])
        #expect(manifestRoot["bitWidthOverridesHonored"] as? Int == 120)

        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-index-fixture-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: temp) }
        try frozenIndex.write(to: temp)
        let index = try ResidentIndexReader.load(fileURL: temp)
        #expect(index.header.indexSize == fixture.plan.resident.indexSize)
        #expect(index.entries[fixture.plan.resident.entries[0].name]?.dtype
                == GTurboFormatV1.DType.u32.rawValue)

        let hashes = [
            hash(frozenManifest),
            hash(frozenLayout),
            hash(frozenIndex),
        ]
        #expect(hashes == [
            "f0c4d01ac7f1747416a0274dd26bb585e2e43ae294b02a8daf272e7129d46dce",
            "acf57a355128d1d8afb1d09b75dc1aa7fef98687871b1ff9f738dd511cb6e341",
            "aa705246112c17d4b60422a2705bda53e96553f7ff96e09b00e6fbb5a4ffa594",
        ], "fixture hashes: \(hashes)")
    }

    @Test func sharedConstantsMatchExistingFacades() {
        #expect(GTurboJSON.magic == GTurboFormatV1.magic)
        #expect(GTurboJSON.versionMajor == GTurboFormatV1.versionMajor)
        #expect(GTurboBinary.indexHeaderBytes == GTurboFormatV1.residentHeaderBytes)
        #expect(GTurboBinary.indexEntryBytes == GTurboFormatV1.residentEntryBytes)
    }

    private func makeFixture() -> (
        config: ArchConfig,
        plan: RepackPlan,
        files: [(relativePath: String, info: GTurboJSON.FileEntry)],
        expertStride: UInt64
    ) {
        let config = ArchConfig(
            hiddenSize: 64, intermediateSize: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 1,
            headDim: 16, fullHeadDim: 32, vocabSize: 1024,
            slidingWindow: 128, finalLogitSoftcap: 30,
            ropeTheta: 10_000, fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25, numLayers: 1, numExperts: 2,
            topKExperts: 1, tieWordEmbeddings: true, attentionKEqV: true,
            fullAttentionLayerMask: [0], hiddenActivation: "gelu_pytorch_tanh")
        let arch = ArchInfo(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            moeIntermediateSize: config.moeIntermediateSize,
            numHeads: config.numHeads,
            numKVHeads: config.numKVHeads,
            numFullKVHeads: config.numFullKVHeads,
            headDim: config.headDim,
            fullHeadDim: config.fullHeadDim,
            vocabSize: config.vocabSize,
            slidingWindow: config.slidingWindow,
            finalLogitSoftcap: config.finalLogitSoftcap,
            ropeTheta: config.ropeTheta,
            fullRopeTheta: config.fullRopeTheta,
            partialRotaryFactor: config.partialRotaryFactor,
            numLayers: config.numLayers,
            numExperts: config.numExperts,
            topKExperts: config.topKExperts,
            tieWordEmbeddings: config.tieWordEmbeddings,
            attentionKEqV: config.attentionKEqV,
            fullAttentionLayerMask: config.fullAttentionLayerMask,
            hiddenActivation: config.hiddenActivation)
        let source = SourceTensor(
            name: "fixture.weight", shardPath: "/dev/null", dtype: .u32,
            shape: [1, 1], absoluteOffset: 0, sizeBytes: 16)
        let indexSize = GTurboFormatV1.alignmentBytes
        let residentEntry = ResidentEntry(
            name: "language_model.model.embed_tokens.weight",
            dtype: GTurboFormatV1.DType.u32.rawValue,
            logicalShape4: [1, 1, 0, 0],
            fileOffset: indexSize, sizeBytes: 16,
            scaleOffset: indexSize + 16, scaleSize: 8,
            biasOffset: indexSize + 24, biasSize: 8,
            quantSpec: QuantSpec(bits: 4),
            sourceWeight: source, sourceScales: source, sourceBiases: source)
        let nameBytes = Array(residentEntry.name.utf8)
        let resident = ResidentFilePlan(
            path: "/fixture/model_weights.bin",
            entries: [residentEntry],
            stringTable: nameBytes,
            stringTableOffsets: [0],
            indexSize: indexSize,
            residentSize: 32)
        let expertStride = GTurboFormatV1.alignmentBytes
        let slice = PerExpertTensorSlice(
            role: "gate", component: "weights",
            dtype: GTurboFormatV1.DType.u32.rawValue,
            logicalShape: [8, 8], offsetInExpertBlob: 0,
            sizeInExpertBlob: 32, sourceOffsetPerExpert: 32,
            sourceTensor: source, bitsForWeights: 4)
        let layer = LayerFilePlan(
            layerIndex: 0, path: "/fixture/packed_experts/layer_00.bin",
            expertsPerLayer: 2, expertStride: expertStride,
            subTensors: [slice])
        let plan = RepackPlan(
            arch: arch, baseMode: "affine", baseGroupSize: 64,
            bitsOverrideCount: 120, resident: resident, layers: [layer],
            matchedModelID: nil, excludedMultimodalTensorNames: [])
        let zeroSHA = String(repeating: "0", count: 64)
        let files: [(relativePath: String, info: GTurboJSON.FileEntry)] = [
            ("model_weights.bin", .init(size: resident.totalSize, sha256: zeroSHA)),
            ("packed_experts/layout.json", .init(size: 1, sha256: zeroSHA)),
            ("packed_experts/layer_00.bin", .init(
                size: 2 * expertStride, sha256: zeroSHA)),
        ]
        return (config, plan, files, expertStride)
    }

    private func hash(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    @Test func visionCompanionCodecsMatchV1Fixtures() throws {
        let zeroSHA = String(repeating: "0", count: 64)
        let manifest = GTurboVisionManifestV1(
            modelID: "fixture/model", sourceRevision: "fixture-revision",
            sourceIndexSha256: zeroSHA, processorConfigSha256: zeroSHA,
            compatibleTextSourceSnapshotHash: "fixture-text-snapshot",
            compatibleTextManifestSha256: zeroSHA,
            files: [
                GTurboVisionFormatV1.weightsFile: .init(size: 32_768, sha256: zeroSHA),
                GTurboVisionFormatV1.processorFile: .init(size: 2, sha256: zeroSHA),
            ],
            tensors: [
                .init(
                    name: "vision_tower.patch_embedder.input_proj.weight",
                    executionPosition: 0, offset: 0, size: 512,
                    shape: [16, 16], dtype: .bf16),
                .init(
                    name: "embed_vision.embedding_projection.weight",
                    executionPosition: 1, offset: 16_384, size: 256,
                    shape: [8, 8], dtype: .u32,
                    quantization: .init(
                        weightBits: 4, scheme: "affine", scaleType: "bf16",
                        biasType: "bf16", groupSize: 64)),
            ])
        let receipt = GTurboVisionReceiptV1(
            manifestSha256: zeroSHA,
            companionDirectoryPath: "/fixture/model.vision.gturbo",
            compatibleTextManifestSha256: zeroSHA,
            sourceRepoID: "fixture/model", sourceRevision: "fixture-revision",
            verificationTimestamp: "2026-08-05T00:00:00Z",
            toolVersion: "fixture",
            files: [
                GTurboVisionFormatV1.manifestFile: .init(size: 1, sha256: zeroSHA),
                GTurboVisionFormatV1.weightsFile: .init(size: 32_768, sha256: zeroSHA),
                GTurboVisionFormatV1.processorFile: .init(size: 2, sha256: zeroSHA),
            ])

        let manifestFixture = try frozenFixture("vision-manifest.json")
        let receiptFixture = try frozenFixture("vision-receipt.json")
        #expect(try GTurboVisionManifestCodec.encode(manifest) == manifestFixture)
        #expect(try GTurboVisionReceiptCodec.encode(receipt) == receiptFixture)
        #expect(try GTurboVisionManifestCodec.decode(manifestFixture) == manifest)
        #expect(try GTurboVisionReceiptCodec.decode(receiptFixture) == receipt)
    }

    private func frozenFixture(_ name: String) throws -> Data {
        let url = try #require(Bundle.module.url(
            forResource: name, withExtension: "base64", subdirectory: "Fixtures/v1"))
        let encoded = String(decoding: try Data(contentsOf: url), as: UTF8.self)
        return try #require(Data(base64Encoded: encoded, options: .ignoreUnknownCharacters))
    }
}
