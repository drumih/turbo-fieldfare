import Foundation
import Testing
@testable import TurboFieldfareFormat

private enum FormatV2Fixture {
    static let zeroSHA = String(repeating: "0", count: 64)
    static let stride = GTurboFormatV2.alignmentBytes

    static func manifest(sourceSnapshotHash: String? = "snapshot",
                         minor: Int = 0) -> GTurboManifestV2 {
        GTurboManifestV2(
            versionMinor: minor,
            flags: [
                "streamingPresent": true,
                "latentMoE": true,
                "kdaLayers": true,
                "attnRes": true,
            ],
            modelID: "fixture/kimi-k3-tiny",
            sourceSnapshotHash: sourceSnapshotHash,
            arch: arch(),
            quant: quant,
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

    static func arch(denseLayers: [Int] = [1],
                     kdaLayers: [Int] = [1, 3],
                     fullAttnLayers: [Int] = [2]) -> GTurboManifestArchV2 {
        GTurboManifestArchV2(
            hiddenSize: 64, vocabSize: 256, numLayers: 3,
            denseMLPIntermediateSize: 128,
            rmsNormEpsilon: 1e-5, tieWordEmbeddings: false,
            hiddenActivation: "situ_glu", bosTokenID: 250, eosTokenID: 251,
            denseLayers: denseLayers, kdaLayers: kdaLayers,
            fullAttnLayers: fullAttnLayers,
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
                routerRenormalize: true, routerCorrectionBias: true))
    }

    static let affine4 = GTurboManifestQuantSlotV2(
        weightBits: 4, scheme: GTurboFormatV2.quantSchemeAffine4G64,
        scaleType: "BF16", biasType: "BF16", groupSize: 64)

    static let quant = GTurboManifestQuantV2(
        embedding: GTurboManifestQuantSlotV2(
            weightBits: 8, scheme: GTurboFormatV2.quantSchemeAffine8G64,
            scaleType: "BF16", biasType: "BF16", groupSize: 64),
        attention: affine4,
        router: GTurboManifestQuantSlotV2(
            weightBits: 32, scheme: GTurboFormatV2.quantSchemeFP32,
            scaleType: "none", biasType: "none", groupSize: 1),
        sharedExpert: affine4,
        latentProjection: affine4,
        denseMLP: affine4,
        routedExpert: GTurboManifestQuantSlotV2(
            weightBits: 4, scheme: GTurboFormatV2.quantSchemeMxfp4E2M1G32E8M0,
            scaleType: "E8M0", biasType: "none", groupSize: 32))

    // Six subtensors per expert: w1/w2/w3 packed [4, 32] (64 B) + scales
    // [4, 1] (4 B) each, 204 B total inside a 16,384 B fixed-stride blob.
    static func subtensors() -> [String: GTurboSubTensorV2] {
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
        return [
            "w1_packed": packed(0), "w1_scales": scales(64),
            "w2_packed": packed(68), "w2_scales": scales(132),
            "w3_packed": packed(136), "w3_scales": scales(200),
        ]
    }

    static func layout(explicitIDs: Bool = true,
                       explicitRanks: Bool = true) -> GTurboPackedExpertsLayoutV2 {
        GTurboPackedExpertsLayoutV2(
            expertStride: stride,
            numLayers: 3,
            expertsPerLayer: 2,
            layers: [1, 2].map { layerID in
                GTurboLayerV2(
                    layer: layerID,
                    file: "layer_0\(layerID).bin",
                    experts: (0..<2).map { expert in
                        GTurboExpertV2(
                            expert: explicitIDs ? expert : nil,
                            physicalRank: explicitRanks ? expert : nil,
                            offset: UInt64(expert) * stride,
                            size: stride,
                            tensors: subtensors())
                    })
            })
    }
}

@Suite struct QuantizationMXFP4Tests {
    @Test func decodeLUTHoldsExactE2M1Values() {
        #expect(QuantizationMXFP4.e2m1DecodeLUT == [
            0, 0.5, 1, 1.5, 2, 3, 4, 6,
            -0.0, -0.5, -1, -1.5, -2, -3, -4, -6,
        ])
        // Index 8 is negative zero.
        #expect(QuantizationMXFP4.e2m1DecodeLUT[8].sign == .minus)
        #expect(QuantizationMXFP4.e2m1DecodeLUT[8] == 0)
    }

    @Test func nibbleOrderIsLowEvenHighOdd() {
        // 0x2E: element 0 reads the low nibble 0xE (-4), element 1 the high
        // nibble 0x2 (1) — same convention as the repo's int4 packing.
        #expect(QuantizationMXFP4.valueIndex(0, in: 0x2E) == 0x0E)
        #expect(QuantizationMXFP4.valueIndex(1, in: 0x2E) == 0x02)
    }

    @Test func e8m0ScaleDecodesExactPowersOfTwo() {
        #expect(QuantizationMXFP4.decodeScale(127).bitPattern == 0x3F80_0000)  // 1
        #expect(QuantizationMXFP4.decodeScale(128).bitPattern == 0x4000_0000)  // 2
        #expect(QuantizationMXFP4.decodeScale(126).bitPattern == 0x3F00_0000)  // 0.5
        #expect(QuantizationMXFP4.decodeScale(129).bitPattern == 0x4080_0000)  // 4
        #expect(QuantizationMXFP4.decodeScale(0).bitPattern == 0x0040_0000)    // 2^-127
        #expect(QuantizationMXFP4.decodeScale(254).bitPattern == 0x7F00_0000)  // 2^127
    }

    @Test func e8m0Scale255ZeroesTheGroup() {
        #expect(QuantizationMXFP4.decodeScale(255) == 0)
        let packed = [UInt8](repeating: 0x77, count: 16)  // every value 6
        #expect(QuantizationMXFP4.dequantizeMxfp4Group(packed: packed, scale: 255)
                == [Float](repeating: 0, count: 32))
    }

    @Test func dequantizeGroupMatchesHandComputedValues() {
        let packed: [UInt8] = [
            0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        ]
        let expected: [Float] = [
            0, 0.5, 1, 1.5, 2, 3, 4, 6,
            -0, -0.5, -1, -1.5, -2, -3, -4, -6,
            0.5, 0, 1.5, 1, 3, 2, 6, 4,
            -0.5, -0, -1.5, -1, -3, -2, -6, -4,
        ]
        #expect(QuantizationMXFP4.dequantizeMxfp4Group(packed: packed, scale: 127)
                == expected)
        // Scale 128 doubles every value.
        #expect(QuantizationMXFP4.dequantizeMxfp4Group(packed: packed, scale: 128)
                == expected.map { $0 * 2 })
    }

    @Test func matvecMatchesHandComputedRows() {
        // rows = 2, columns = 32 (one group per row), vector all ones.
        // Row 0: nibbles all 1 (0.5) x scale 1   -> 16.
        // Row 1: nibbles all 9 (-0.5) x scale 4  -> -64.
        let packed = [UInt8](repeating: 0x11, count: 16)
            + [UInt8](repeating: 0x99, count: 16)
        let out = QuantizationMXFP4.matvecMxfp4(
            packedWeights: packed, scales: [127, 129],
            rows: 2, columns: 32, vector: [Float](repeating: 1, count: 32))
        #expect(out == [16, -64])

        // Non-uniform vector: values 0...31, nibbles all 2 (1.0) x scale 1.
        let ramp = (0..<32).map(Float.init)
        let rampOut = QuantizationMXFP4.matvecMxfp4(
            packedWeights: [UInt8](repeating: 0x22, count: 16), scales: [127],
            rows: 1, columns: 32, vector: ramp)
        #expect(rampOut == [496])
    }

    @Test func matvecAppliesPerGroupScales() {
        // columns = 64, one row, vector all ones.
        // Group 0: nibbles all 2 (1.0) x scale 1   -> 32.
        // Group 1: nibbles all 3 (1.5) x scale 0.5 -> 24.
        let packed = [UInt8](repeating: 0x22, count: 16)
            + [UInt8](repeating: 0x33, count: 16)
        let out = QuantizationMXFP4.matvecMxfp4(
            packedWeights: packed, scales: [127, 126],
            rows: 1, columns: 64, vector: [Float](repeating: 1, count: 64))
        #expect(out == [56])
    }
}

@Suite struct GTurboManifestV2CodecTests {
    @Test func roundTripPreservesOptionalPresenceAndWriterField() throws {
        let manifest = FormatV2Fixture.manifest()
        let encoded = try GTurboManifestCodecV2.encode(manifest)
        #expect(try GTurboManifestCodecV2.decode(encoded) == manifest)
        let root = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(root["versionMajor"] as? Int == 2)
        #expect(root["expertStride"] as? Int == 16_384)
    }

    @Test func acceptsAdditiveMinorAndUnknownTopLevelKey() throws {
        let data = try GTurboManifestCodecV2.encode(FormatV2Fixture.manifest(minor: 9))
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["futureMetadata"] = ["ignored": true]
        let changed = try JSONSerialization.data(withJSONObject: root)
        #expect(try GTurboManifestCodecV2.decode(changed).versionMinor == 9)
    }

    @Test func acceptsAbsentSourceSnapshotHash() throws {
        let data = try GTurboManifestCodecV2.encode(FormatV2Fixture.manifest())
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root.removeValue(forKey: "sourceSnapshotHash")
        let changed = try JSONSerialization.data(withJSONObject: root)
        #expect(try GTurboManifestCodecV2.decode(changed).sourceSnapshotHash == nil)
    }

    @Test func refuseDefaultsThrowsOnAnyMissingArchField() throws {
        let data = try GTurboManifestCodecV2.encode(FormatV2Fixture.manifest())
        let root = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        let arch = try #require(root["arch"] as? [String: Any])
        for key in arch.keys {
            var mutated = root
            var mutatedArch = arch
            mutatedArch.removeValue(forKey: key)
            mutated["arch"] = mutatedArch
            let changed = try JSONSerialization.data(withJSONObject: mutated)
            #expect(throws: GTurboFormatError.self,
                    "missing arch.\(key) must fail decoding") {
                try GTurboManifestCodecV2.decode(changed)
            }
        }
    }

    @Test(arguments: ["blockSize"]) func refuseDefaultsThrowsOnMissingNestedField(
        _ key: String) throws {
        let data = try GTurboManifestCodecV2.encode(FormatV2Fixture.manifest())
        var root = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        var arch = try #require(root["arch"] as? [String: Any])
        var attnRes = try #require(arch["attnRes"] as? [String: Any])
        attnRes.removeValue(forKey: key)
        arch["attnRes"] = attnRes
        root["arch"] = arch
        let changed = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: GTurboFormatError.self) { try GTurboManifestCodecV2.decode(changed) }
    }

    @Test(arguments: ["quant", "files", "arch"])
    func throwsOnMissingRequiredTopLevelField(_ key: String) throws {
        let data = try GTurboManifestCodecV2.encode(FormatV2Fixture.manifest())
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root.removeValue(forKey: key)
        let changed = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: GTurboFormatError.self) { try GTurboManifestCodecV2.decode(changed) }
    }

    @Test func rejectsUnknownFlag() throws {
        let manifest = FormatV2Fixture.manifest()
        let invalid = GTurboManifestV2(
            flags: manifest.flags.merging(["turboQuantKV": true]) { _, new in new },
            modelID: manifest.modelID,
            sourceSnapshotHash: manifest.sourceSnapshotHash,
            arch: manifest.arch, quant: manifest.quant, files: manifest.files,
            expertsPerLayer: manifest.expertsPerLayer,
            numLayers: manifest.numLayers, expertStride: manifest.expertStride)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodecV2.encode(invalid)
        }
    }

    @Test func rejectsUnalignedExpertStride() throws {
        let manifest = FormatV2Fixture.manifest()
        let invalid = GTurboManifestV2(
            flags: manifest.flags, modelID: manifest.modelID,
            sourceSnapshotHash: manifest.sourceSnapshotHash,
            arch: manifest.arch, quant: manifest.quant, files: manifest.files,
            expertsPerLayer: manifest.expertsPerLayer,
            numLayers: manifest.numLayers, expertStride: 8_192)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodecV2.encode(invalid)
        }
    }

    @Test func rejectsArchDimensionMismatch() throws {
        let manifest = FormatV2Fixture.manifest()
        let invalid = GTurboManifestV2(
            flags: manifest.flags, modelID: manifest.modelID,
            sourceSnapshotHash: manifest.sourceSnapshotHash,
            arch: manifest.arch, quant: manifest.quant, files: manifest.files,
            expertsPerLayer: manifest.expertsPerLayer,
            numLayers: 4, expertStride: manifest.expertStride)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodecV2.encode(invalid)
        }
    }

    @Test func rejectsOverlappingAttentionLists() throws {
        let manifest = FormatV2Fixture.manifest()
        let invalid = GTurboManifestV2(
            flags: manifest.flags, modelID: manifest.modelID,
            sourceSnapshotHash: manifest.sourceSnapshotHash,
            arch: FormatV2Fixture.arch(kdaLayers: [1, 2, 3], fullAttnLayers: [2]),
            quant: manifest.quant, files: manifest.files,
            expertsPerLayer: manifest.expertsPerLayer,
            numLayers: manifest.numLayers, expertStride: manifest.expertStride)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodecV2.encode(invalid)
        }
    }

    @Test func rejectsLayerListNotCoveringEveryLayer() throws {
        let manifest = FormatV2Fixture.manifest()
        let invalid = GTurboManifestV2(
            flags: manifest.flags, modelID: manifest.modelID,
            sourceSnapshotHash: manifest.sourceSnapshotHash,
            arch: FormatV2Fixture.arch(kdaLayers: [1], fullAttnLayers: [2]),
            quant: manifest.quant, files: manifest.files,
            expertsPerLayer: manifest.expertsPerLayer,
            numLayers: manifest.numLayers, expertStride: manifest.expertStride)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodecV2.encode(invalid)
        }
    }

    @Test func rejectsUnsortedLayerList() throws {
        let manifest = FormatV2Fixture.manifest()
        let invalid = GTurboManifestV2(
            flags: manifest.flags, modelID: manifest.modelID,
            sourceSnapshotHash: manifest.sourceSnapshotHash,
            arch: FormatV2Fixture.arch(kdaLayers: [3, 1]),
            quant: manifest.quant, files: manifest.files,
            expertsPerLayer: manifest.expertsPerLayer,
            numLayers: manifest.numLayers, expertStride: manifest.expertStride)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodecV2.encode(invalid)
        }
    }

    @Test func rejectsOutOfRangeLayerID() throws {
        let manifest = FormatV2Fixture.manifest()
        let invalid = GTurboManifestV2(
            flags: manifest.flags, modelID: manifest.modelID,
            sourceSnapshotHash: manifest.sourceSnapshotHash,
            arch: FormatV2Fixture.arch(kdaLayers: [1, 3], fullAttnLayers: [2, 4]),
            quant: manifest.quant, files: manifest.files,
            expertsPerLayer: manifest.expertsPerLayer,
            numLayers: manifest.numLayers, expertStride: manifest.expertStride)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodecV2.encode(invalid)
        }
    }

    @Test func rejectsUnsafeManifestPaths() throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: GTurboManifestCodecV2.encode(FormatV2Fixture.manifest())) as? [String: Any])
        var files = try #require(root["files"] as? [String: Any])
        files["../escape.bin"] = files.removeValue(forKey: "model_weights.bin")
        root["files"] = files
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: GTurboFormatError.self) { try GTurboManifestCodecV2.decode(data) }
    }
}

@Suite struct GTurboPackedExpertsLayoutV2CodecTests {
    @Test func roundTripPreservesIdentityFallback() throws {
        let layout = FormatV2Fixture.layout(explicitIDs: false, explicitRanks: false)
        let encoded = try GTurboPackedExpertsLayoutCodecV2.encode(layout)
        #expect(try GTurboPackedExpertsLayoutCodecV2.decode(encoded) == layout)
    }

    @Test func acceptsCanonicalSubtensorSchema() throws {
        try GTurboV2StructuralValidator.validate(FormatV2Fixture.layout())
    }

    private func layoutWithTensors(
        _ tensors: [String: GTurboSubTensorV2]) -> GTurboPackedExpertsLayoutV2 {
        GTurboPackedExpertsLayoutV2(
            expertStride: FormatV2Fixture.stride, numLayers: 3, expertsPerLayer: 1,
            layers: [GTurboLayerV2(
                layer: 1, file: "layer_01.bin",
                experts: [GTurboExpertV2(
                    expert: 0, physicalRank: 0, offset: 0,
                    size: FormatV2Fixture.stride, tensors: tensors)])])
    }

    @Test func rejectsMissingSubtensor() throws {
        var tensors = FormatV2Fixture.subtensors()
        tensors.removeValue(forKey: "w2_scales")
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.validate(layoutWithTensors(tensors))
        }
    }

    @Test func rejectsExtraSubtensor() throws {
        var tensors = FormatV2Fixture.subtensors()
        tensors["w4_packed"] = tensors["w1_packed"]
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.validate(layoutWithTensors(tensors))
        }
    }

    @Test func rejectsWrongScaleBits() throws {
        var tensors = FormatV2Fixture.subtensors()
        let scales = tensors["w1_scales"]!
        tensors["w1_scales"] = GTurboSubTensorV2(
            offset: scales.offset, size: scales.size, dtype: scales.dtype,
            shape: scales.shape, bits: 4)
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.validate(layoutWithTensors(tensors))
        }
    }

    @Test func rejectsSizeShapeMismatch() throws {
        var tensors = FormatV2Fixture.subtensors()
        let packed = tensors["w1_packed"]!
        tensors["w1_packed"] = GTurboSubTensorV2(
            offset: packed.offset, size: packed.size - 1, dtype: packed.dtype,
            shape: packed.shape, bits: packed.bits)
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.validate(layoutWithTensors(tensors))
        }
    }

    @Test func rejectsPackedScalesShapeMismatch() throws {
        var tensors = FormatV2Fixture.subtensors()
        // 64 values per row need 2 scale bytes per row, not 1.
        tensors["w1_packed"] = GTurboSubTensorV2(
            offset: 0, size: 128, dtype: GTurboFormatV2.mxfp4PackedDType,
            shape: [4, 64], bits: GTurboFormatV2.mxfp4PackedBits)
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.validate(layoutWithTensors(tensors))
        }
    }

    @Test func rejectsTensorOutsideExpertBlob() throws {
        var tensors = FormatV2Fixture.subtensors()
        let scales = tensors["w3_scales"]!
        tensors["w3_scales"] = GTurboSubTensorV2(
            offset: FormatV2Fixture.stride - 2, size: scales.size,
            dtype: scales.dtype, shape: scales.shape, bits: scales.bits)
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.validate(layoutWithTensors(tensors))
        }
    }

    @Test func rejectsOverlappingTensorRanges() throws {
        var tensors = FormatV2Fixture.subtensors()
        let scales = tensors["w1_scales"]!
        tensors["w1_scales"] = GTurboSubTensorV2(
            offset: 62, size: scales.size, dtype: scales.dtype,
            shape: scales.shape, bits: scales.bits)
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.validate(layoutWithTensors(tensors))
        }
    }

    @Test func rejectsOffsetRankMismatch() throws {
        let layout = GTurboPackedExpertsLayoutV2(
            expertStride: FormatV2Fixture.stride, numLayers: 3, expertsPerLayer: 2,
            layers: [GTurboLayerV2(
                layer: 1, file: "layer_01.bin",
                experts: (0..<2).map { expert in
                    GTurboExpertV2(
                        expert: expert, physicalRank: expert,
                        offset: UInt64(1 - expert) * FormatV2Fixture.stride,
                        size: FormatV2Fixture.stride,
                        tensors: FormatV2Fixture.subtensors())
                })])
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.validate(layout)
        }
    }

    @Test func rejectsDenseLayerCoverageMismatch() throws {
        // Layout covering layer 0 (dense in the manifest) must fail.
        let layout = GTurboPackedExpertsLayoutV2(
            expertStride: FormatV2Fixture.stride, numLayers: 3, expertsPerLayer: 2,
            layers: [0, 1, 2].map { layerID in
                GTurboLayerV2(
                    layer: layerID, file: "layer_0\(layerID).bin",
                    experts: (0..<2).map { expert in
                        GTurboExpertV2(
                            expert: expert, physicalRank: expert,
                            offset: UInt64(expert) * FormatV2Fixture.stride,
                            size: FormatV2Fixture.stride,
                            tensors: FormatV2Fixture.subtensors())
                    })
            })
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.crossValidate(
                manifest: FormatV2Fixture.manifest(), layout: layout)
        }
    }
}

@Suite struct GTurboV2StructuralValidatorTests {
    @Test func acceptsMatchingDocuments() throws {
        try GTurboV2StructuralValidator.crossValidate(
            manifest: FormatV2Fixture.manifest(), layout: FormatV2Fixture.layout())
    }

    @Test func rejectsMissingMoELayer() throws {
        let layout = GTurboPackedExpertsLayoutV2(
            expertStride: FormatV2Fixture.stride, numLayers: 3, expertsPerLayer: 2,
            layers: [FormatV2Fixture.layout().layers[0]])
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.crossValidate(
                manifest: FormatV2Fixture.manifest(), layout: layout)
        }
    }

    @Test func rejectsWrongLayerFileSize() throws {
        var sizes = FormatV2Fixture.manifest().files.mapValues(\.size)
        sizes["packed_experts/layer_02.bin"] = FormatV2Fixture.stride
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.crossValidate(
                manifestNumLayers: 3, manifestExpertsPerLayer: 2,
                manifestExpertStride: FormatV2Fixture.stride,
                manifestDenseLayers: [1],
                manifestFileSizes: sizes, layout: FormatV2Fixture.layout())
        }
    }

    @Test func rejectsOutOfRangeDenseLayer() throws {
        #expect(throws: GTurboFormatError.self) {
            try GTurboV2StructuralValidator.crossValidate(
                manifestNumLayers: 3, manifestExpertsPerLayer: 2,
                manifestExpertStride: FormatV2Fixture.stride,
                manifestDenseLayers: [1, 4],
                manifestFileSizes: FormatV2Fixture.manifest().files.mapValues(\.size),
                layout: FormatV2Fixture.layout())
        }
    }
}

@Suite struct KimiK3FormatProfileTests {
    @Test func canonicalLayerListsMatchOfficialConfig() throws {
        #expect(KimiK3FormatProfile.denseLayers == [1])
        #expect(KimiK3FormatProfile.fullAttnLayers == [
            4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,
            52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93,
        ])
        #expect(KimiK3FormatProfile.kdaLayers.count == 69)
        #expect(KimiK3FormatProfile.fullAttnLayers.count == 24)
        let union = Set(KimiK3FormatProfile.denseLayers)
            .union(KimiK3FormatProfile.kdaLayers)
            .union(KimiK3FormatProfile.fullAttnLayers)
        #expect(union == Set(1...93))
    }

    @Test func canonicalProfilePassesManifestValidation() throws {
        let manifest = GTurboManifestV2(
            flags: KimiK3FormatProfile.flags,
            modelID: KimiK3FormatProfile.modelID,
            sourceSnapshotHash: nil,
            arch: KimiK3FormatProfile.arch,
            quant: KimiK3FormatProfile.quant,
            files: [
                "model_weights.bin": GTurboManifestFileV2(
                    size: 16_448, sha256: FormatV2Fixture.zeroSHA),
            ],
            expertsPerLayer: KimiK3FormatProfile.expertsPerLayer,
            numLayers: KimiK3FormatProfile.numLayers,
            expertStride: KimiK3FormatProfile.expertStride)
        try GTurboManifestCodecV2.validate(manifest)
    }

    @Test func canonicalExpertStrideMatchesMXFP4Layout() throws {
        let latent = 3_584, intermediate = 3_072
        let matrixParams = [latent * intermediate, intermediate * latent,
                            latent * intermediate]
        let expertBytes = matrixParams.reduce(0) { $0 + UInt64($1 / 2 + $1 / 32) }
        #expect(expertBytes == 17_547_264)
        #expect(expertBytes == KimiK3FormatProfile.expertStride)
        #expect(expertBytes % GTurboFormatV2.alignmentBytes == 0)
    }

    @Test func canonicalCrossValidationCoversAllMoELayers() throws {
        // Full 92-layer, 896-expert canonical layout, contiguous six-tensor
        // blobs tiling the whole 17,547,264-byte stride.
        let w1: UInt64 = 3_072 * 3_584 / 2, w1s: UInt64 = 3_072 * 3_584 / 32
        let w2: UInt64 = 3_584 * 3_072 / 2, w2s: UInt64 = 3_584 * 3_072 / 32
        let w3 = w1, w3s = w1s
        let tensors: [String: GTurboSubTensorV2] = [
            "w1_packed": GTurboSubTensorV2(
                offset: 0, size: w1, dtype: GTurboFormatV2.mxfp4PackedDType,
                shape: [3_072, 3_584], bits: 4),
            "w1_scales": GTurboSubTensorV2(
                offset: w1, size: w1s, dtype: GTurboFormatV2.mxfp4ScaleDType,
                shape: [3_072, 112], bits: 8),
            "w2_packed": GTurboSubTensorV2(
                offset: w1 + w1s, size: w2, dtype: GTurboFormatV2.mxfp4PackedDType,
                shape: [3_584, 3_072], bits: 4),
            "w2_scales": GTurboSubTensorV2(
                offset: w1 + w1s + w2, size: w2s,
                dtype: GTurboFormatV2.mxfp4ScaleDType,
                shape: [3_584, 96], bits: 8),
            "w3_packed": GTurboSubTensorV2(
                offset: w1 + w1s + w2 + w2s, size: w3,
                dtype: GTurboFormatV2.mxfp4PackedDType,
                shape: [3_072, 3_584], bits: 4),
            "w3_scales": GTurboSubTensorV2(
                offset: w1 + w1s + w2 + w2s + w3, size: w3s,
                dtype: GTurboFormatV2.mxfp4ScaleDType,
                shape: [3_072, 112], bits: 8),
        ]
        #expect(w1 + w1s + w2 + w2s + w3 + w3s == KimiK3FormatProfile.expertStride)
        let moeLayers = (0..<93).filter { !KimiK3FormatProfile.denseLayers.contains($0 + 1) }
        #expect(moeLayers.count == 92)
        let layout = GTurboPackedExpertsLayoutV2(
            expertStride: KimiK3FormatProfile.expertStride,
            numLayers: KimiK3FormatProfile.numLayers,
            expertsPerLayer: KimiK3FormatProfile.expertsPerLayer,
            layers: moeLayers.map { layerID in
                GTurboLayerV2(
                    layer: layerID,
                    file: String(format: "layer_%02d.bin", layerID),
                    experts: (0..<896).map { expert in
                        GTurboExpertV2(
                            expert: expert, physicalRank: nil,
                            offset: UInt64(expert) * KimiK3FormatProfile.expertStride,
                            size: KimiK3FormatProfile.expertStride,
                            tensors: tensors)
                    })
            })
        _ = try GTurboPackedExpertsLayoutCodecV2.encode(layout)
        let layerSize = UInt64(KimiK3FormatProfile.expertsPerLayer)
            * KimiK3FormatProfile.expertStride
        var files: [String: GTurboManifestFileV2] = [
            "model_weights.bin": GTurboManifestFileV2(
                size: 16_448, sha256: FormatV2Fixture.zeroSHA),
        ]
        for layerID in moeLayers {
            files[String(format: "packed_experts/layer_%02d.bin", layerID)] =
                GTurboManifestFileV2(size: layerSize, sha256: FormatV2Fixture.zeroSHA)
        }
        let manifest = GTurboManifestV2(
            flags: KimiK3FormatProfile.flags,
            modelID: KimiK3FormatProfile.modelID,
            sourceSnapshotHash: nil,
            arch: KimiK3FormatProfile.arch,
            quant: KimiK3FormatProfile.quant,
            files: files,
            expertsPerLayer: KimiK3FormatProfile.expertsPerLayer,
            numLayers: KimiK3FormatProfile.numLayers,
            expertStride: KimiK3FormatProfile.expertStride)
        try GTurboV2StructuralValidator.crossValidate(manifest: manifest, layout: layout)
    }
}
