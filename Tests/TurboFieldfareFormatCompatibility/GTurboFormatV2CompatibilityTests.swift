import Foundation
import Testing
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite struct GTurboFormatV2CompatibilityTests {
    @Test func v2WritersMatchFrozenFixtures() throws {
        let manifestData = try GTurboManifestCodecV2.encode(V2GoldenFixture.manifest())
        let layoutData = try GTurboPackedExpertsLayoutCodecV2.encode(V2GoldenFixture.layout())

        let frozenManifest = try frozenFixture("manifest.json")
        let frozenLayout = try frozenFixture("layout.json")
        #expect(manifestData == frozenManifest)
        #expect(layoutData == frozenLayout)

        let manifest = try GTurboManifestCodecV2.decode(frozenManifest)
        #expect(manifest.arch.kdaLayers == [1, 3])
        #expect(manifest.arch.fullAttnLayers == [2])
        #expect(manifest.arch.denseLayers == [1])
        #expect(manifest.quant.routedExpert.scheme
                == GTurboFormatV2.quantSchemeMxfp4E2M1G32E8M0)
        let layout = try GTurboPackedExpertsLayoutCodecV2.decode(frozenLayout)
        #expect(layout.layers.map(\.layer) == [1, 2])
        #expect(layout.layers[0].experts[1].offset == GTurboFormatV2.alignmentBytes)
        #expect(Set(layout.layers[0].experts[0].tensors.keys)
                == Set(GTurboFormatV2.mxfp4ExpertTensorNames))
        try GTurboV2StructuralValidator.crossValidate(manifest: manifest, layout: layout)

        let hashes = [hash(frozenManifest), hash(frozenLayout)]
        #expect(hashes == [
            "e5f5af8c924aec9350320011d8f8cd7f551dc70956d66416ab85cbe4f89bc921",
            "4e65f814b81bb515745b8581edb9d1ed9502e7543611a2fa33cff65a1daf8e24",
        ], "fixture hashes: \(hashes)")
    }

    @Test func v2ReusesV1ResidentIndexWireFormat() throws {
        #expect(GTurboFormatV2.alignmentBytes == GTurboFormatV1.alignmentBytes)
        let header = GTurboResidentIndexHeaderV2(
            indexSize: GTurboFormatV2.alignmentBytes, residentSize: 64, entryCount: 1)
        var bytes = Data(repeating: 0, count: Int(header.indexSize))
        bytes.withUnsafeMutableBytes { raw in
            GTurboResidentIndexCodecV2.writeHeader(into: raw.baseAddress!, header: header)
            let entry = GTurboResidentIndexEntryV2(
                name: "fixture.weight", dtype: GTurboFormatV1.DType.u32.rawValue,
                fileOffset: GTurboFormatV2.alignmentBytes, sizeBytes: 16,
                shape: [1, 1, 0, 0],
                scaleOffset: GTurboFormatV2.alignmentBytes + 16, scaleSize: 8,
                biasOffset: GTurboFormatV2.alignmentBytes + 24, biasSize: 8)
            GTurboResidentIndexCodecV2.writeEntry(
                into: raw.baseAddress!.advanced(by: GTurboFormatV1.residentHeaderBytes),
                entry: entry, nameOffset: 96)
            for (index, byte) in Array("fixture.weight".utf8).enumerated() {
                raw[96 + index] = byte
            }
        }
        let entries = try bytes.withUnsafeBytes {
            try GTurboResidentIndexCodecV2.decodeRegion($0, header: header)
        }
        #expect(entries.map(\.name) == ["fixture.weight"])
    }

    private func hash(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    private func frozenFixture(_ name: String) throws -> Data {
        let url = try #require(Bundle.module.url(
            forResource: name, withExtension: "base64", subdirectory: "Fixtures/v2"))
        let encoded = String(decoding: try Data(contentsOf: url), as: UTF8.self)
        return try #require(Data(base64Encoded: encoded, options: .ignoreUnknownCharacters))
    }
}

enum V2GoldenFixture {
    static let zeroSHA = String(repeating: "0", count: 64)
    static let stride = GTurboFormatV2.alignmentBytes

    static func manifest() -> GTurboManifestV2 {
        GTurboManifestV2(
            flags: [
                "streamingPresent": true,
                "latentMoE": true,
                "kdaLayers": true,
                "attnRes": true,
            ],
            modelID: "fixture/kimi-k3-tiny",
            sourceSnapshotHash: "fixture-snapshot",
            arch: GTurboManifestArchV2(
                hiddenSize: 64, vocabSize: 256, numLayers: 3,
                denseMLPIntermediateSize: 128,
                rmsNormEpsilon: 1e-5, tieWordEmbeddings: false,
                hiddenActivation: "situ_glu", bosTokenID: 250, eosTokenID: 251,
                denseLayers: [1], kdaLayers: [1, 3], fullAttnLayers: [2],
                kda: GTurboManifestKDAV2(
                    numHeads: 2, headDim: 8, convWidth: 4,
                    decayLowRankSize: 4, decayProjectionSize: 16,
                    gateLowerBound: -5, fullRankOutputGate: true),
                mla: GTurboManifestMLAV2(
                    numHeads: 2, qLoraRank: 16, kvLoraRank: 8,
                    qkNopeHeadDim: 8, qkRopeHeadDim: 4, vHeadDim: 8,
                    outputGate: true),
                attnRes: GTurboManifestAttnResV2(blockSize: 12),
                moe: GTurboManifestMoEV2(
                    numExperts: 2, topKExperts: 1,
                    latentBottleneckSize: 32, expertIntermediateSize: 32,
                    numSharedExperts: 1, sharedExpertIntermediateSize: 64,
                    situGLUGateBeta: 4, situGLUUpBeta: 25,
                    routerRenormalize: true, routerCorrectionBias: true)),
            quant: GTurboManifestQuantV2(
                embedding: GTurboManifestQuantSlotV2(
                    weightBits: 8, scheme: GTurboFormatV2.quantSchemeAffine8G64,
                    scaleType: "BF16", biasType: "BF16", groupSize: 64),
                attention: GTurboManifestQuantSlotV2(
                    weightBits: 4, scheme: GTurboFormatV2.quantSchemeAffine4G64,
                    scaleType: "BF16", biasType: "BF16", groupSize: 64),
                router: GTurboManifestQuantSlotV2(
                    weightBits: 32, scheme: GTurboFormatV2.quantSchemeFP32,
                    scaleType: "none", biasType: "none", groupSize: 1),
                sharedExpert: GTurboManifestQuantSlotV2(
                    weightBits: 4, scheme: GTurboFormatV2.quantSchemeAffine4G64,
                    scaleType: "BF16", biasType: "BF16", groupSize: 64),
                latentProjection: GTurboManifestQuantSlotV2(
                    weightBits: 4, scheme: GTurboFormatV2.quantSchemeAffine4G64,
                    scaleType: "BF16", biasType: "BF16", groupSize: 64),
                denseMLP: GTurboManifestQuantSlotV2(
                    weightBits: 4, scheme: GTurboFormatV2.quantSchemeAffine4G64,
                    scaleType: "BF16", biasType: "BF16", groupSize: 64),
                routedExpert: GTurboManifestQuantSlotV2(
                    weightBits: 4, scheme: GTurboFormatV2.quantSchemeMxfp4E2M1G32E8M0,
                    scaleType: "E8M0", biasType: "none", groupSize: 32)),
            files: [
                "model_weights.bin": GTurboManifestFileV2(size: 16_448, sha256: zeroSHA),
                "packed_experts/layout.json": GTurboManifestFileV2(size: 1, sha256: zeroSHA),
                "packed_experts/layer_01.bin": GTurboManifestFileV2(
                    size: 2 * stride, sha256: zeroSHA),
                "packed_experts/layer_02.bin": GTurboManifestFileV2(
                    size: 2 * stride, sha256: zeroSHA),
            ],
            expertsPerLayer: 2,
            numLayers: 3,
            expertStride: stride)
    }

    static func layout() -> GTurboPackedExpertsLayoutV2 {
        func packed(_ offset: UInt64) -> GTurboSubTensorV2 {
            GTurboSubTensorV2(offset: offset, size: 64,
                              dtype: GTurboFormatV2.mxfp4PackedDType,
                              shape: [4, 32], bits: GTurboFormatV2.mxfp4PackedBits)
        }
        func scales(_ offset: UInt64) -> GTurboSubTensorV2 {
            GTurboSubTensorV2(offset: offset, size: 4,
                              dtype: GTurboFormatV2.mxfp4ScaleDType,
                              shape: [4, 1], bits: GTurboFormatV2.mxfp4ScaleBits)
        }
        let tensors: [String: GTurboSubTensorV2] = [
            "w1_packed": packed(0), "w1_scales": scales(64),
            "w2_packed": packed(68), "w2_scales": scales(132),
            "w3_packed": packed(136), "w3_scales": scales(200),
        ]
        return GTurboPackedExpertsLayoutV2(
            expertStride: stride,
            numLayers: 3,
            expertsPerLayer: 2,
            layers: [1, 2].map { layerID in
                GTurboLayerV2(
                    layer: layerID,
                    file: "layer_0\(layerID).bin",
                    experts: (0..<2).map { expert in
                        GTurboExpertV2(
                            expert: expert,
                            physicalRank: nil,
                            offset: UInt64(expert) * stride,
                            size: stride,
                            tensors: tensors)
                    })
            })
    }
}
