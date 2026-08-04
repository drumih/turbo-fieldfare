import Metal
import Testing

@testable import TurboFieldfare

@Suite struct ModelRuntimeSchemaValidationTests {
    @Test func rejectsMissingRequiredResidentTensor() throws {
        let model = try loadToy()
        var entries = model.residentIndex.entries
        entries["language_model.model.embed_tokens.weight"] = nil
        let index = ResidentIndex(header: model.residentIndex.header, entries: entries)

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: index,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsMissingRoutedTensorRole() throws {
        let model = try loadToy()
        let layout = replacingExpert(model.packedExpertsLayout, layer: 0, expert: 0) { expert in
            var tensors = expert.subTensors
            tensors["gate"] = nil
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsNonuniformRoutedTensorMetadata() throws {
        let model = try loadToy()
        let layout = replacingExpert(model.packedExpertsLayout, layer: 0, expert: 1) { expert in
            var tensors = expert.subTensors
            let gate = tensors["gate"]!
            tensors["gate"] = SubTensorEntry(
                offset: gate.offset + 1,
                size: gate.size,
                dtype: gate.dtype,
                shape: gate.shape,
                bits: gate.bits)
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsRoutedTensorRangeOutsideUInt32() throws {
        let model = try loadToy()
        let layout = replacingExpert(model.packedExpertsLayout, layer: 0, expert: 0) { expert in
            var tensors = expert.subTensors
            let gate = tensors["gate"]!
            tensors["gate"] = SubTensorEntry(
                offset: UInt64(UInt32.max),
                size: 2,
                dtype: gate.dtype,
                shape: gate.shape,
                bits: gate.bits)
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsMalformedResidentAffineMetadata() throws {
        let model = try loadToy()
        let name = "language_model.model.embed_tokens.weight"
        let entry = try #require(model.residentIndex.entries[name])
        let invalidEntries = [
            copy(entry, dtype: 1),
            copy(entry, shape: (entry.shape.0, entry.shape.1 + 1, 0, 0)),
            copy(entry, sizeBytes: entry.sizeBytes + 1),
            copy(entry, scaleSize: entry.scaleSize + 2),
            copy(entry, fileOffset: entry.fileOffset + 1),
            copy(entry, scaleOffset: entry.scaleOffset + 1),
            copy(entry, biasSize: entry.biasSize + 2),
            copy(entry, biasOffset: entry.biasOffset + 1),
        ]

        for invalid in invalidEntries {
            var entries = model.residentIndex.entries
            entries[name] = invalid
            let index = ResidentIndex(header: model.residentIndex.header, entries: entries)
            #expect(throws: ModelError.self) {
                try Model.validateRuntimeSchema(
                    residentIndex: index,
                    layout: model.packedExpertsLayout,
                    manifest: model.manifest,
                    config: .gemma4Toy())
            }
        }
    }

    @Test func rejectsMalformedRoutedAffineMetadata() throws {
        let model = try loadToy()
        let gate = try #require(model.packedExpertsLayout.layers[0].experts[0].subTensors["gate"])
        let invalidEntries = [
            SubTensorEntry(offset: gate.offset, size: gate.size, dtype: "BF16",
                           shape: gate.shape, bits: gate.bits),
            SubTensorEntry(offset: gate.offset, size: gate.size, dtype: gate.dtype,
                           shape: [gate.shape[0], gate.shape[1] + 1], bits: gate.bits),
            SubTensorEntry(offset: gate.offset, size: gate.size, dtype: gate.dtype,
                           shape: gate.shape, bits: 8),
            SubTensorEntry(offset: gate.offset, size: gate.size + 1, dtype: gate.dtype,
                           shape: gate.shape, bits: gate.bits),
            SubTensorEntry(offset: gate.offset + 1, size: gate.size, dtype: gate.dtype,
                           shape: gate.shape, bits: gate.bits),
        ]

        for invalid in invalidEntries {
            let layout = replacingRoleForAllExperts(
                model.packedExpertsLayout, layer: 0, role: "gate", with: invalid)
            #expect(throws: ModelError.self) {
                try Model.validateRuntimeSchema(
                    residentIndex: model.residentIndex,
                    layout: layout,
                    manifest: model.manifest,
                    config: .gemma4Toy())
            }
        }
    }

    @Test func rejectsExecutableLoadWithoutQuantMetadata() throws {
        let directory = try ModelLoaderTests.writeToySynthetic(includeQuant: false)
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: ModelError.self) {
            try Model.load(directoryURL: directory, device: device, expecting: .gemma4Toy())
        }
    }

    @Test func rejectsOverflowingDerivedDimensions() throws {
        let model = try loadToy()
        let toy = ArchConfig.gemma4Toy()
        let config = ArchConfig(
            hiddenSize: toy.hiddenSize,
            intermediateSize: toy.intermediateSize,
            moeIntermediateSize: toy.moeIntermediateSize,
            numHeads: Int.max,
            numKVHeads: toy.numKVHeads,
            numFullKVHeads: toy.numFullKVHeads,
            headDim: 2,
            fullHeadDim: toy.fullHeadDim,
            vocabSize: toy.vocabSize,
            slidingWindow: toy.slidingWindow,
            finalLogitSoftcap: toy.finalLogitSoftcap,
            ropeTheta: toy.ropeTheta,
            fullRopeTheta: toy.fullRopeTheta,
            partialRotaryFactor: toy.partialRotaryFactor,
            numLayers: toy.numLayers,
            numExperts: toy.numExperts,
            topKExperts: toy.topKExperts,
            tieWordEmbeddings: toy.tieWordEmbeddings,
            attentionKEqV: toy.attentionKEqV,
            fullAttentionLayerMask: toy.fullAttentionLayerMask,
            hiddenActivation: toy.hiddenActivation)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: config)
        }
    }

    private func loadToy() throws -> Model {
        let directory = try ModelLoaderTests.writeToySynthetic()
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try Model.load(
            directoryURL: directory,
            device: device,
            expecting: .gemma4Toy())
    }

    private func replacingExpert(
        _ layout: PackedExpertsLayout,
        layer layerIndex: Int,
        expert expertIndex: Int,
        transform: (ExpertEntry) -> ExpertEntry
    ) -> PackedExpertsLayout {
        var layers = layout.layers
        let layer = layers[layerIndex]
        var experts = layer.experts
        experts[expertIndex] = transform(experts[expertIndex])
        layers[layerIndex] = LayerLayout(
            layer: layer.layer,
            file: layer.file,
            experts: experts)
        return PackedExpertsLayout(
            expertStride: layout.expertStride,
            numLayers: layout.numLayers,
            expertsPerLayer: layout.expertsPerLayer,
            layers: layers)
    }

    private func replacingRoleForAllExperts(
        _ layout: PackedExpertsLayout,
        layer layerIndex: Int,
        role: String,
        with replacement: SubTensorEntry
    ) -> PackedExpertsLayout {
        var layers = layout.layers
        let layer = layers[layerIndex]
        let experts = layer.experts.map { expert in
            var tensors = expert.subTensors
            tensors[role] = replacement
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }
        layers[layerIndex] = LayerLayout(layer: layer.layer, file: layer.file, experts: experts)
        return PackedExpertsLayout(
            expertStride: layout.expertStride,
            numLayers: layout.numLayers,
            expertsPerLayer: layout.expertsPerLayer,
            layers: layers)
    }

    private func copy(
        _ entry: ResidentIndexEntry,
        dtype: UInt8? = nil,
        fileOffset: UInt64? = nil,
        sizeBytes: UInt64? = nil,
        shape: (UInt32, UInt32, UInt32, UInt32)? = nil,
        scaleOffset: UInt64? = nil,
        scaleSize: UInt64? = nil,
        biasOffset: UInt64? = nil,
        biasSize: UInt64? = nil
    ) -> ResidentIndexEntry {
        ResidentIndexEntry(
            name: entry.name,
            dtype: dtype ?? entry.dtype,
            fileOffset: fileOffset ?? entry.fileOffset,
            sizeBytes: sizeBytes ?? entry.sizeBytes,
            shape: shape ?? entry.shape,
            scaleOffset: scaleOffset ?? entry.scaleOffset,
            scaleSize: scaleSize ?? entry.scaleSize,
            biasOffset: biasOffset ?? entry.biasOffset,
            biasSize: biasSize ?? entry.biasSize)
    }
}
