import Foundation
import Testing
@testable import TurboFieldfareFormat

private enum FormatFixture {
    static let zeroSHA = String(repeating: "0", count: 64)

    static func manifest(sourceSnapshotHash: String? = "snapshot",
                         quant: GTurboManifestQuantV1? = FormatFixture.quant,
                         bitWidthOverridesHonored: Int? = 120,
                         minor: Int = 0) -> GTurboManifestV1 {
        GTurboManifestV1(
            versionMinor: minor,
            flags: [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            modelID: "fixture/model",
            sourceSnapshotHash: sourceSnapshotHash,
            arch: GTurboManifestArchV1(
                hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
                numHeads: 4, numKVHeads: 2, numFullKVHeads: 1,
                headDim: 16, fullHeadDim: 32, vocabSize: 1024,
                slidingWindow: 128, finalLogitSoftcap: 30,
                ropeTheta: 10_000, fullRopeTheta: 1_000_000,
                partialRotaryFactor: 0.25, numLayers: 1, numExperts: 2,
                topKExperts: 1, tieWordEmbeddings: true, attentionKEqV: true,
                hiddenActivation: "gelu_pytorch_tanh", fullAttentionLayerMask: [0]),
            quant: quant,
            files: [
                "model_weights.bin": GTurboManifestFileV1(size: 16_448, sha256: zeroSHA),
                "packed_experts/layout.json": GTurboManifestFileV1(size: 1, sha256: zeroSHA),
                "packed_experts/layer_00.bin": GTurboManifestFileV1(
                    size: 2 * GTurboFormatV1.alignmentBytes, sha256: zeroSHA),
            ],
            expertsPerLayer: 2,
            numLayers: 1,
            expertStride: GTurboFormatV1.alignmentBytes,
            bitWidthOverridesHonored: bitWidthOverridesHonored)
    }

    static let quantSlot = GTurboManifestQuantSlotV1(
        weightBits: 4, scheme: "affine", scaleType: "BF16",
        biasType: "BF16", groupSize: 64)

    static let quant = GTurboManifestQuantV1(
        embedding: quantSlot, attention: quantSlot, router: quantSlot,
        sharedExpert: quantSlot, routedExpert: quantSlot)

    static func layout(explicitIDs: Bool = true,
                       explicitRanks: Bool = true) -> GTurboPackedExpertsLayoutV1 {
        let tensor = GTurboSubTensorV1(
            offset: 0, size: 32, dtype: "U32", shape: [8, 8], bits: 4)
        return GTurboPackedExpertsLayoutV1(
            expertStride: GTurboFormatV1.alignmentBytes,
            numLayers: 1,
            expertsPerLayer: 2,
            layers: [GTurboLayerV1(
                layer: 0,
                file: "layer_00.bin",
                experts: (0..<2).map { expert in
                    GTurboExpertV1(
                        expert: explicitIDs ? expert : nil,
                        physicalRank: explicitRanks ? expert : nil,
                        offset: UInt64(expert) * GTurboFormatV1.alignmentBytes,
                        size: GTurboFormatV1.alignmentBytes,
                        tensors: ["gate": tensor])
                })])
    }

    static func residentBytes(nameOffset: UInt32 = 96,
                              nameBytes: [UInt8] = Array("weight".utf8),
                              dtype: UInt8 = GTurboFormatV1.DType.u32.rawValue,
                              reserved: UInt8 = 0,
                              shape: [UInt32] = [1, 1, 0, 0],
                              fileOffset: UInt64 = GTurboFormatV1.alignmentBytes,
                              sizeBytes: UInt64 = 16,
                              scaleOffset: UInt64 = GTurboFormatV1.alignmentBytes + 16,
                              scaleSize: UInt64 = 8,
                              biasOffset: UInt64 = GTurboFormatV1.alignmentBytes + 24,
                              biasSize: UInt64 = 8,
                              header: GTurboResidentIndexHeaderV1 = .init(
                                indexSize: GTurboFormatV1.alignmentBytes,
                                residentSize: 64, entryCount: 1)) -> Data {
        var bytes = Data(repeating: 0, count: Int(header.indexSize))
        bytes.withUnsafeMutableBytes { raw in
            GTurboResidentIndexCodec.writeHeader(into: raw.baseAddress!, header: header)
            guard raw.count >= 96 else { return }
            let entry = GTurboResidentIndexEntryV1(
                name: String(decoding: nameBytes, as: UTF8.self), dtype: dtype,
                fileOffset: fileOffset, sizeBytes: sizeBytes, shape: shape,
                scaleOffset: scaleOffset, scaleSize: scaleSize,
                biasOffset: biasOffset, biasSize: biasSize)
            GTurboResidentIndexCodec.writeEntry(
                into: raw.baseAddress!.advanced(by: GTurboFormatV1.residentHeaderBytes),
                entry: entry, nameOffset: nameOffset)
            raw[GTurboFormatV1.residentHeaderBytes + 7] = reserved
            if Int(nameOffset) + nameBytes.count <= raw.count {
                for (index, byte) in nameBytes.enumerated() {
                    raw[Int(nameOffset) + index] = byte
                }
            }
        }
        return bytes
    }
}

@Suite struct GTurboManifestCodecTests {
    @Test func roundTripPreservesOptionalPresenceAndWriterField() throws {
        let manifest = FormatFixture.manifest()
        let encoded = try GTurboManifestCodec.encode(manifest)
        #expect(try GTurboManifestCodec.decode(encoded) == manifest)
        let root = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(root["bitWidthOverridesHonored"] as? Int == 120)
    }

    @Test func acceptsAdditiveMinorAndUnknownTopLevelKey() throws {
        let data = try GTurboManifestCodec.encode(FormatFixture.manifest(minor: 9))
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["futureMetadata"] = ["ignored": true]
        let changed = try JSONSerialization.data(withJSONObject: root)
        #expect(try GTurboManifestCodec.decode(changed).versionMinor == 9)
    }

    @Test(arguments: ["sourceSnapshotHash", "quant", "bitWidthOverridesHonored"])
    func acceptsAbsentLegacyOptionalField(_ key: String) throws {
        let data = try GTurboManifestCodec.encode(FormatFixture.manifest())
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root.removeValue(forKey: key)
        let changed = try JSONSerialization.data(withJSONObject: root)
        let decoded = try GTurboManifestCodec.decode(changed)
        if key == "sourceSnapshotHash" { #expect(decoded.sourceSnapshotHash == nil) }
        if key == "quant" { #expect(decoded.quant == nil) }
        if key == "bitWidthOverridesHonored" { #expect(decoded.bitWidthOverridesHonored == nil) }
    }

    @Test func rejectsUnsafeManifestPaths() throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: GTurboManifestCodec.encode(FormatFixture.manifest())) as? [String: Any])
        var files = try #require(root["files"] as? [String: Any])
        files["../escape.bin"] = files.removeValue(forKey: "model_weights.bin")
        root["files"] = files
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: GTurboFormatError.self) { try GTurboManifestCodec.decode(data) }
    }

    @Test(arguments: ["manifest.json", "verified-install.json", "tokenizer"])
    func rejectsReservedManifestPaths(_ reserved: String) throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: GTurboManifestCodec.encode(FormatFixture.manifest())) as? [String: Any])
        var files = try #require(root["files"] as? [String: Any])
        files[reserved] = files["model_weights.bin"]
        root["files"] = files
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: GTurboFormatError.self) { try GTurboManifestCodec.decode(data) }
    }

    @Test func rejectsFileDirectoryPrefixCollision() throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: GTurboManifestCodec.encode(FormatFixture.manifest())) as? [String: Any])
        var files = try #require(root["files"] as? [String: Any])
        files["packed_experts"] = files["model_weights.bin"]
        root["files"] = files
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: GTurboFormatError.self) { try GTurboManifestCodec.decode(data) }
    }
}

@Suite struct GTurboPackedExpertsLayoutCodecTests {
    @Test func roundTripPreservesIdentityFallback() throws {
        let layout = FormatFixture.layout(explicitIDs: false, explicitRanks: false)
        let encoded = try GTurboPackedExpertsLayoutCodec.encode(layout)
        #expect(try GTurboPackedExpertsLayoutCodec.decode(encoded) == layout)
    }

    @Test func rejectsMixedExplicitAndPositionalIDs() throws {
        let base = FormatFixture.layout()
        let experts = [
            base.layers[0].experts[0],
            GTurboExpertV1(
                expert: nil, physicalRank: 1,
                offset: GTurboFormatV1.alignmentBytes,
                size: GTurboFormatV1.alignmentBytes,
                tensors: base.layers[0].experts[1].tensors),
        ]
        let layout = GTurboPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [GTurboLayerV1(layer: 0, file: "layer_00.bin", experts: experts)])
        #expect(throws: GTurboFormatError.self) {
            try GTurboPackedExpertsLayoutCodec.encode(layout)
        }
    }

    @Test func rejectsSlashBearingLayerBasename() throws {
        let base = FormatFixture.layout()
        let layout = GTurboPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [GTurboLayerV1(
                layer: 0, file: "nested/layer.bin", experts: base.layers[0].experts)])
        #expect(throws: GTurboFormatError.self) {
            try GTurboPackedExpertsLayoutCodec.encode(layout)
        }
    }

    @Test func rejectsReservedLayoutBasename() throws {
        let base = FormatFixture.layout()
        let layout = GTurboPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [GTurboLayerV1(
                layer: 0, file: "layout.json", experts: base.layers[0].experts)])
        #expect(throws: GTurboFormatError.self) {
            try GTurboPackedExpertsLayoutCodec.encode(layout)
        }
    }

    @Test func crossValidationRequiresExactLayerManifestEntry() throws {
        var sizes = FormatFixture.manifest().files.mapValues(\.size)
        sizes["packed_experts/layer_00.bin"] = nil
        sizes["packed_experts/./layer_00.bin"] = 2 * GTurboFormatV1.alignmentBytes
        #expect(throws: GTurboFormatError.self) {
            try GTurboV1StructuralValidator.crossValidate(
                manifestNumLayers: 1, manifestExpertsPerLayer: 2,
                manifestExpertStride: GTurboFormatV1.alignmentBytes,
                manifestFileSizes: sizes, layout: FormatFixture.layout())
        }
    }
}

@Suite struct GTurboResidentIndexCodecTests {
    @Test func decodesValidRegion() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(header: header)
        let entries = try bytes.withUnsafeBytes {
            try GTurboResidentIndexCodec.decodeRegion($0, header: header)
        }
        #expect(entries.map(\.name) == ["weight"])
    }

    @Test func rejectsOverlappingResidentPayloadRanges() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            scaleOffset: GTurboFormatV1.alignmentBytes + 8,
            header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes {
                try GTurboResidentIndexCodec.decodeRegion($0, header: header)
            }
        }
    }

    @Test func rejectsNameInsideEntryTable() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            nameOffset: UInt32(GTurboFormatV1.residentHeaderBytes), header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes { try GTurboResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsInvalidUTF8Name() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(nameBytes: [0xFF], header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes { try GTurboResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsNonzeroReservedByte() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(reserved: 1, header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes { try GTurboResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsUnknownDType() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(dtype: 255, header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes { try GTurboResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsEntryTableOverflow() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes,
            residentSize: 64, entryCount: UInt64.max)
        let bytes = FormatFixture.residentBytes(header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes { try GTurboResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsPayloadRangeOutsideResidentRegion() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            fileOffset: GTurboFormatV1.alignmentBytes + 56,
            sizeBytes: 16, header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes { try GTurboResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsUnalignedIndexRegion() throws {
        let header = GTurboResidentIndexHeaderV1(indexSize: 128, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            fileOffset: 128, scaleOffset: 144, biasOffset: 152, header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes { try GTurboResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsEmptyPrimaryPayload() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            fileOffset: 0, sizeBytes: 0,
            scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
            header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes { try GTurboResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsNoncanonicalShape() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(shape: [1, 0, 1, 0], header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes { try GTurboResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }

    @Test func rejectsAbsentCompanionWithNonzeroOffset() throws {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)
        let bytes = FormatFixture.residentBytes(
            scaleOffset: GTurboFormatV1.alignmentBytes + 16, scaleSize: 0,
            biasOffset: 0, biasSize: 0, header: header)
        #expect(throws: GTurboFormatError.self) {
            try bytes.withUnsafeBytes { try GTurboResidentIndexCodec.decodeRegion($0, header: header) }
        }
    }
}

@Suite struct GTurboV1StructuralValidatorTests {
    @Test func acceptsMatchingDocuments() throws {
        try GTurboV1StructuralValidator.crossValidate(
            manifest: FormatFixture.manifest(), layout: FormatFixture.layout())
    }

    @Test func rejectsTensorOutsideExpertBlob() throws {
        let base = FormatFixture.layout()
        let invalid = GTurboSubTensorV1(
            offset: base.expertStride - 8, size: 16,
            dtype: "U32", shape: [1], bits: 4)
        let expert = GTurboExpertV1(
            expert: 0, physicalRank: 0, offset: 0, size: base.expertStride,
            tensors: ["gate": invalid])
        let layout = GTurboPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 1,
            layers: [GTurboLayerV1(layer: 0, file: "layer_00.bin", experts: [expert])])
        #expect(throws: GTurboFormatError.self) {
            try GTurboV1StructuralValidator.validate(layout)
        }
    }

    @Test func rejectsDuplicateLayerIDs() throws {
        let base = FormatFixture.layout()
        let layout = GTurboPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 2, expertsPerLayer: 2,
            layers: [
                base.layers[0],
                GTurboLayerV1(
                    layer: 0, file: "other.bin", experts: base.layers[0].experts),
            ])
        #expect(throws: GTurboFormatError.self) {
            try GTurboV1StructuralValidator.validate(layout)
        }
    }

    @Test func rejectsDuplicateLogicalExpertIDs() throws {
        let base = FormatFixture.layout()
        let experts = base.layers[0].experts.enumerated().map { position, expert in
            GTurboExpertV1(
                expert: 0, physicalRank: position,
                offset: UInt64(position) * base.expertStride, size: expert.size,
                tensors: expert.tensors)
        }
        let layout = GTurboPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [GTurboLayerV1(layer: 0, file: "layer_00.bin", experts: experts)])
        #expect(throws: GTurboFormatError.self) {
            try GTurboV1StructuralValidator.validate(layout)
        }
    }

    @Test func rejectsDuplicatePhysicalRanks() throws {
        let base = FormatFixture.layout()
        let experts = base.layers[0].experts.enumerated().map { position, expert in
            GTurboExpertV1(
                expert: position, physicalRank: 0,
                offset: 0, size: expert.size, tensors: expert.tensors)
        }
        let layout = GTurboPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [GTurboLayerV1(layer: 0, file: "layer_00.bin", experts: experts)])
        #expect(throws: GTurboFormatError.self) {
            try GTurboV1StructuralValidator.validate(layout)
        }
    }

    @Test func rejectsOverlappingTensorRanges() throws {
        let base = FormatFixture.layout()
        let tensor = GTurboSubTensorV1(
            offset: 16, size: 32, dtype: "U32", shape: [8, 8], bits: 4)
        let expert = GTurboExpertV1(
            expert: 0, physicalRank: 0, offset: 0, size: base.expertStride,
            tensors: [
                "gate": base.layers[0].experts[0].tensors["gate"]!,
                "up": tensor,
            ])
        let layout = GTurboPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 1,
            layers: [GTurboLayerV1(layer: 0, file: "layer_00.bin", experts: [expert])])
        #expect(throws: GTurboFormatError.self) {
            try GTurboV1StructuralValidator.validate(layout)
        }
    }
}
