import Testing
import Foundation
import Metal
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

/// Loader tests for `K3Model`: a tiny but schema-complete `.gturbo` v2 bundle
/// (3 layers: 1 dense KDA, 1 MLA, 1 MoE-KDA; 2 experts/layer) fabricated in a
/// temp dir with the v2 writers, then loaded end-to-end. Asserts the happy
/// path, refuse-defaults schema validation, arch mismatch, checksum policy,
/// and lazy per-layer expert file handles.
@Suite struct K3ModelLoaderTests {

    /// Toy K3 dims. Every affine matrix keeps columns % 64 == 0 (the g64
    /// contract); MXFP4 dims stay multiples of 32.
    static func tinyConfig() -> K3ArchConfig {
        K3ArchConfig(
            hiddenSize: 64, vocabSize: 256, numLayers: 3,
            denseMLPIntermediateSize: 128,
            rmsNormEpsilon: 1e-5, tieWordEmbeddings: false,
            hiddenActivation: "situ_glu", bosTokenID: 250, eosTokenID: 251,
            denseLayers: [1], kdaLayers: [1, 3], fullAttnLayers: [2],
            kdaNumHeads: 2, kdaHeadDim: 32, kdaConvWidth: 4,
            kdaDecayLowRankSize: 64, kdaDecayProjectionSize: 64,
            kdaGateLowerBound: -5.0, kdaFullRankOutputGate: true,
            mlaNumHeads: 2, mlaQLoraRank: 64, mlaKVLoraRank: 64,
            mlaQKNopeHeadDim: 64, mlaQKRopeHeadDim: 64, mlaVHeadDim: 64,
            mlaOutputGate: true,
            attnResBlockSize: 12,
            moeNumExperts: 2, moeTopKExperts: 2,
            moeLatentBottleneckSize: 64, moeExpertIntermediateSize: 64,
            moeNumSharedExperts: 1, moeSharedExpertIntermediateSize: 64,
            situGLUGateBeta: 4.0, situGLUUpBeta: 25.0,
            routerRenormalize: true, routerCorrectionBias: true,
            expertsPerLayer: 2, expertStride: 16_384)
    }

    struct BuiltBundle {
        let url: URL
        /// Serialized embedding weights/scales/biases payload for the
        /// byte-exact registry check.
        let embeddingWeights: [UInt8]
        let embeddingScales: [UInt8]
        let embeddingBiases: [UInt8]
    }

    // MARK: - Bundle fabrication

    /// Deterministic toy weight value (house-style formula).
    static func toyValue(_ row: Int, _ col: Int, salt: Int) -> Float {
        Float((row % 7) - 3) * 0.01 + Float((col % 11) - 5) * 0.002
            + Float(salt % 5) * 0.003
    }

    /// Serialize one schema entry's payload (weights + scales + biases).
    static func payload(for entry: K3Model.SchemaEntry) -> (weights: [UInt8],
                                                            scales: [UInt8],
                                                            biases: [UInt8]) {
        func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        switch entry.kind {
        case .affine(let rows, let columns, let bits):
            let rowValues = (0..<rows).map { row in
                (0..<columns).map { toyValue(row, $0, salt: entry.name.utf8.count) }
            }
            var weights: [UInt8] = []
            var scales: [UInt8] = []
            var biases: [UInt8] = []
            if bits == 4 {
                for row in rowValues.map({ Quantization.quantizeInt4Affine($0) }) {
                    weights.append(contentsOf: row.packed)
                    appendU16(row.scales, to: &scales)
                    appendU16(row.biases, to: &biases)
                }
            } else {
                precondition(bits == 8)
                for row in rowValues.map({ Quantization.quantizeInt8Affine($0) }) {
                    weights.append(contentsOf: row.packed)
                    appendU16(row.scales, to: &scales)
                    appendU16(row.biases, to: &biases)
                }
            }
            return (weights, scales, biases)
        case .bf16Vector(let count):
            var weights: [UInt8] = []
            appendU16([UInt16](repeating: Quantization.bf16Bits(1.0), count: count),
                      to: &weights)
            return (weights, [], [])
        case .fp32Vector(let count):
            var weights: [UInt8] = []
            for i in 0..<count {
                var value = Float(i) * 0.001 - 0.01
                withUnsafeBytes(of: &value) { weights.append(contentsOf: $0) }
            }
            return (weights, [], [])
        case .fp32Matrix(let rows, let columns):
            var weights: [UInt8] = []
            for row in 0..<rows {
                for col in 0..<columns {
                    var value = toyValue(row, col, salt: 1)
                    withUnsafeBytes(of: &value) { weights.append(contentsOf: $0) }
                }
            }
            return (weights, [], [])
        }
    }

    /// Write the tiny bundle. `missingTensor` drops one name from the
    /// resident index (payload bytes remain) to exercise refuse-defaults
    /// schema validation.
    static func writeTinyBundle(missingTensor: String? = nil,
                                layoutTrailingWhitespaceBytes: Int = 0,
                                quant: GTurboManifestQuantV2
                                    = KimiK3FormatProfile.quantInt4) throws -> BuiltBundle {
        let config = tinyConfig()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("k3-gturbo-toy-\(UUID().uuidString)")
        let expertsDir = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: expertsDir,
                                                withIntermediateDirectories: true)

        // 1. Resident payload: every schema tensor, in schema order.
        let schema = K3Model.schemaEntries(config: config, quant: quant)
        struct Payload {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            var weights: [UInt8]
            var scales: [UInt8]
            var biases: [UInt8]
        }
        let payloads: [Payload] = schema.map { entry in
            let bytes = payload(for: entry)
            let dtype: UInt8
            let shape: [UInt32]
            switch entry.kind {
            case .affine(let rows, let columns, _):
                dtype = GTurboFormatV1.DType.u32.rawValue
                shape = [UInt32(rows), UInt32(columns), 0, 0]
            case .bf16Vector(let count):
                dtype = GTurboFormatV1.DType.bf16.rawValue
                shape = [UInt32(count), 0, 0, 0]
            case .fp32Vector(let count):
                dtype = GTurboFormatV1.DType.fp32.rawValue
                shape = [UInt32(count), 0, 0, 0]
            case .fp32Matrix(let rows, let columns):
                dtype = GTurboFormatV1.DType.fp32.rawValue
                shape = [UInt32(rows), UInt32(columns), 0, 0]
            }
            return Payload(name: entry.name, dtype: dtype, shape: shape,
                           weights: bytes.weights, scales: bytes.scales,
                           biases: bytes.biases)
        }

        let indexed = payloads.filter { $0.name != missingTensor }
        let names = indexed.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboFormatV1.residentHeaderBytes
        let entryBytes = GTurboFormatV1.residentEntryBytes
        let stringTableBase = headerBytes + names.count * entryBytes
        var nameOffsets: [UInt32] = []
        var nameCursor = 0
        for name in names {
            nameOffsets.append(UInt32(stringTableBase + nameCursor))
            nameCursor += name.utf8.count
        }
        let alignment = GTurboFormatV2.alignmentBytes
        let rawIndexBytes = UInt64(stringTableBase + stringTable.count)
        let indexSize = ((rawIndexBytes + alignment - 1) / alignment) * alignment

        // 2. Assign payload offsets (16-byte aligned), then serialize.
        var cursor = indexSize
        var entries: [GTurboResidentIndexEntryV2] = []
        var regions: [(offset: UInt64, bytes: [UInt8])] = []
        for payload in payloads {
            cursor = ((cursor + 15) / 16) * 16
            let weightOffset = cursor
            cursor += UInt64(payload.weights.count)
            let scaleOffset = payload.scales.isEmpty ? 0 : cursor
            cursor += UInt64(payload.scales.count)
            let biasOffset = payload.biases.isEmpty ? 0 : cursor
            cursor += UInt64(payload.biases.count)
            regions.append((weightOffset, payload.weights))
            if !payload.scales.isEmpty { regions.append((scaleOffset, payload.scales)) }
            if !payload.biases.isEmpty { regions.append((biasOffset, payload.biases)) }
            if indexed.contains(where: { $0.name == payload.name }) {
                entries.append(GTurboResidentIndexEntryV2(
                    name: payload.name, dtype: payload.dtype,
                    fileOffset: weightOffset, sizeBytes: UInt64(payload.weights.count),
                    shape: payload.shape,
                    scaleOffset: scaleOffset, scaleSize: UInt64(payload.scales.count),
                    biasOffset: biasOffset, biasSize: UInt64(payload.biases.count)))
            }
        }
        let residentSize = cursor - indexSize

        var fileBuf = [UInt8](repeating: 0, count: Int(cursor))
        fileBuf.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboResidentIndexCodecV2.writeHeader(
                into: base,
                header: GTurboResidentIndexHeaderV2(
                    indexSize: indexSize, residentSize: residentSize,
                    entryCount: UInt64(entries.count)))
            for (index, entry) in entries.enumerated() {
                GTurboResidentIndexCodecV2.writeEntry(
                    into: base.advanced(by: headerBytes + index * entryBytes),
                    entry: entry, nameOffset: nameOffsets[index])
            }
            _ = stringTable.withUnsafeBytes { table in
                memcpy(base.advanced(by: stringTableBase), table.baseAddress!,
                       stringTable.count)
            }
            for region in regions {
                _ = region.bytes.withUnsafeBytes { bytes in
                    memcpy(base.advanced(by: Int(region.offset)), bytes.baseAddress!,
                           bytes.count)
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 3. packed_experts: layers 1 and 2 (0-based MoE layers), two experts
        //    each at the canonical 6-subtensor MXFP4 layout.
        let dLatent = UInt32(config.moeLatentBottleneckSize)
        let inter = UInt32(config.moeExpertIntermediateSize)
        let offsets = K3ExpertSubtensorOffsets.canonical(dLatent: dLatent,
                                                         intermediate: inter)
        let blobSize = K3ExpertSubtensorOffsets.canonicalBlobSize(dLatent: dLatent,
                                                                  intermediate: inter)
        let stride = config.expertStride
        precondition(blobSize <= stride)
        let subtensors: [String: GTurboSubTensorV2] = [
            "w1_packed": GTurboSubTensorV2(
                offset: UInt64(offsets.w1PackedOff), size: UInt64(inter) * UInt64(dLatent) / 2,
                dtype: GTurboFormatV2.mxfp4PackedDType, shape: [inter, dLatent],
                bits: GTurboFormatV2.mxfp4PackedBits),
            "w1_scales": GTurboSubTensorV2(
                offset: UInt64(offsets.w1ScalesOff), size: UInt64(inter) * UInt64(dLatent) / 32,
                dtype: GTurboFormatV2.mxfp4ScaleDType, shape: [inter, dLatent / 32],
                bits: GTurboFormatV2.mxfp4ScaleBits),
            "w2_packed": GTurboSubTensorV2(
                offset: UInt64(offsets.w2PackedOff), size: UInt64(dLatent) * UInt64(inter) / 2,
                dtype: GTurboFormatV2.mxfp4PackedDType, shape: [dLatent, inter],
                bits: GTurboFormatV2.mxfp4PackedBits),
            "w2_scales": GTurboSubTensorV2(
                offset: UInt64(offsets.w2ScalesOff), size: UInt64(dLatent) * UInt64(inter) / 32,
                dtype: GTurboFormatV2.mxfp4ScaleDType, shape: [dLatent, inter / 32],
                bits: GTurboFormatV2.mxfp4ScaleBits),
            "w3_packed": GTurboSubTensorV2(
                offset: UInt64(offsets.w3PackedOff), size: UInt64(inter) * UInt64(dLatent) / 2,
                dtype: GTurboFormatV2.mxfp4PackedDType, shape: [inter, dLatent],
                bits: GTurboFormatV2.mxfp4PackedBits),
            "w3_scales": GTurboSubTensorV2(
                offset: UInt64(offsets.w3ScalesOff), size: UInt64(inter) * UInt64(dLatent) / 32,
                dtype: GTurboFormatV2.mxfp4ScaleDType, shape: [inter, dLatent / 32],
                bits: GTurboFormatV2.mxfp4ScaleBits),
        ]
        var layerFiles: [String] = []
        for layer in [1, 2] {
            let basename = String(format: "layer_%02d.bin", layer)
            layerFiles.append(basename)
            var bytes = [UInt8](repeating: 0, count: Int(stride) * 2)
            for expert in 0..<2 {
                let base = expert * Int(stride)
                for i in 0..<Int(blobSize) {
                    bytes[base + i] = UInt8(truncatingIfNeeded:
                        0x30 + layer * 16 + expert * 4 + (i % 4))
                }
            }
            try Data(bytes).write(to: expertsDir.appendingPathComponent(basename))
        }

        let layout = GTurboPackedExpertsLayoutV2(
            expertStride: stride,
            numLayers: config.numLayers,
            expertsPerLayer: config.expertsPerLayer,
            layers: [1, 2].map { layerID in
                GTurboLayerV2(
                    layer: layerID,
                    file: String(format: "layer_%02d.bin", layerID),
                    experts: (0..<2).map { expert in
                        GTurboExpertV2(
                            expert: expert, physicalRank: nil,
                            offset: UInt64(expert) * stride, size: stride,
                            tensors: subtensors)
                    })
            })
        var layoutData = try GTurboPackedExpertsLayoutCodecV2.encode(layout)
        if layoutTrailingWhitespaceBytes > 0 {
            // Whitespace after a JSON document is legal. This lets the
            // fixture exercise the loader's real metadata-size boundary
            // without changing the tiny bundle's structural contract.
            layoutData.append(Data(repeating: 0x20,
                                   count: layoutTrailingWhitespaceBytes))
        }
        try layoutData.write(to: expertsDir.appendingPathComponent("layout.json"))

        // 4. manifest.json with real sizes + hashes.
        var files: [String: GTurboManifestFileV2] = [
            "model_weights.bin": GTurboManifestFileV2(
                size: UInt64(fileBuf.count), sha256: weightsSha),
            "packed_experts/layout.json": GTurboManifestFileV2(
                size: UInt64(layoutData.count),
                sha256: Sha256Verifier.hashData(layoutData)),
        ]
        for basename in layerFiles {
            let sha = try Sha256Verifier.hashFile(
                at: expertsDir.appendingPathComponent(basename))
            files["packed_experts/\(basename)"] = GTurboManifestFileV2(
                size: stride * 2, sha256: sha)
        }
        let manifest = GTurboManifestV2(
            flags: KimiK3FormatProfile.flags,
            modelID: "fixture/kimi-k3-tiny",
            sourceSnapshotHash: nil,
            arch: GTurboManifestArchV2(
                hiddenSize: config.hiddenSize, vocabSize: config.vocabSize,
                numLayers: config.numLayers,
                denseMLPIntermediateSize: config.denseMLPIntermediateSize,
                rmsNormEpsilon: config.rmsNormEpsilon,
                tieWordEmbeddings: config.tieWordEmbeddings,
                hiddenActivation: config.hiddenActivation,
                bosTokenID: config.bosTokenID, eosTokenID: config.eosTokenID,
                denseLayers: config.denseLayers, kdaLayers: config.kdaLayers,
                fullAttnLayers: config.fullAttnLayers,
                kda: GTurboManifestKDAV2(
                    numHeads: config.kdaNumHeads, headDim: config.kdaHeadDim,
                    convWidth: config.kdaConvWidth,
                    decayLowRankSize: config.kdaDecayLowRankSize,
                    decayProjectionSize: config.kdaDecayProjectionSize,
                    gateLowerBound: config.kdaGateLowerBound,
                    fullRankOutputGate: config.kdaFullRankOutputGate),
                mla: GTurboManifestMLAV2(
                    numHeads: config.mlaNumHeads, qLoraRank: config.mlaQLoraRank,
                    kvLoraRank: config.mlaKVLoraRank,
                    qkNopeHeadDim: config.mlaQKNopeHeadDim,
                    qkRopeHeadDim: config.mlaQKRopeHeadDim,
                    vHeadDim: config.mlaVHeadDim,
                    outputGate: config.mlaOutputGate),
                attnRes: GTurboManifestAttnResV2(blockSize: config.attnResBlockSize),
                moe: GTurboManifestMoEV2(
                    numExperts: config.moeNumExperts,
                    topKExperts: config.moeTopKExperts,
                    latentBottleneckSize: config.moeLatentBottleneckSize,
                    expertIntermediateSize: config.moeExpertIntermediateSize,
                    numSharedExperts: config.moeNumSharedExperts,
                    sharedExpertIntermediateSize: config.moeSharedExpertIntermediateSize,
                    situGLUGateBeta: config.situGLUGateBeta,
                    situGLUUpBeta: config.situGLUUpBeta,
                    routerRenormalize: config.routerRenormalize,
                    routerCorrectionBias: config.routerCorrectionBias)),
            quant: quant,
            files: files,
            expertsPerLayer: config.expertsPerLayer,
            numLayers: config.numLayers,
            expertStride: config.expertStride)
        try GTurboManifestCodecV2.encode(manifest)
            .write(to: dir.appendingPathComponent("manifest.json"))

        let embedding = payloads.first {
            $0.name == "language_model.model.embed_tokens.weight"
        }!
        return BuiltBundle(url: dir,
                           embeddingWeights: embedding.weights,
                           embeddingScales: embedding.scales,
                           embeddingBiases: embedding.biases)
    }

    static func loadTiny(_ url: URL,
                         expecting: K3ArchConfig? = nil) throws -> K3Model {
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try K3Model.load(bundleURL: url, device: device,
                                expecting: expecting ?? tinyConfig())
    }

    // MARK: - Tests

    @Test func loadsTinyBundle() throws {
        let bundle = try Self.writeTinyBundle()
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        let model = try Self.loadTiny(bundle.url)
        #expect(model.modelID == "fixture/kimi-k3-tiny")
        #expect(model.config.numLayers == 3)

        // Global tensors.
        #expect(model.embedding.shape.0 == 256 && model.embedding.shape.1 == 64)
        #expect(model.lmHead.shape.0 == 256 && model.lmHead.shape.1 == 64)
        #expect(model.lmHead.offset != model.embedding.offset)  // untied
        #expect(model.finalNorm.length == 64 * 2)
        #expect(model.outputAttnResProj.length == 64 * 2)

        // Layer 0: KDA + dense MLP.
        #expect(model.kdaQProj(layer: 0).shape.0 == 64)
        #expect(model.kdaQProj(layer: 0).shape.1 == 64)
        #expect(model.denseGateProj(layer: 0).shape.0 == 128)
        // Layer 1: MLA + MoE.
        #expect(model.mlaKVBProj(layer: 1).shape.0 == 256)
        #expect(model.mlaKVBProj(layer: 1).shape.1 == 64)
        #expect(model.routerGate(layer: 1).shape.0 == 2)
        #expect(model.routerGate(layer: 1).dtype == GTurboFormatV1.DType.fp32.rawValue)
        #expect(model.routedDownProj(layer: 1).shape.0 == 64)
        #expect(model.sharedGateProj(layer: 1).shape.0 == 64)
        // Layer 2: KDA + MoE.
        #expect(model.kdaOProj(layer: 2).shape.0 == 64)
        #expect(model.kdaOProj(layer: 2).shape.1 == 64)
        #expect(throws: ModelError.self) {
            _ = try model.tensor("language_model.model.layers.2.mlp.gate_proj.weight")
        }

        // Lazy expert layer files.
        #expect(model.openLayerFileCount() == 0)
        let layer1 = try model.expertLayerFile(1)
        #expect(model.openLayerFileCount() == 1)
        #expect(layer1.expertsPerLayer == 2)
        #expect(layer1.expertOffsets == [0, 16_384])
        #expect(layer1.expertStride == 16_384)
        let again = try model.expertLayerFile(1)
        #expect(model.openLayerFileCount() == 1)
        #expect(again.expertOffsets == layer1.expertOffsets)
        close(layer1.fileDescriptor)
        close(again.fileDescriptor)
    }

    @Test func loadsTinyBundleWithInt8Trunk() throws {
        let bundle = try Self.writeTinyBundle(quant: KimiK3FormatProfile.quantInt8)
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        let model = try Self.loadTiny(bundle.url)
        // Embedding is int8 in both profiles; a trunk projection proves the
        // manifest-selected schema retained twice the int4 primary payload.
        let projection = model.kdaQProj(layer: 0)
        #expect(projection.length == UInt64(64 * 64))
        #expect(model.sharedGateProj(layer: 1).length == UInt64(64 * 64))
    }

    @Test func rejectsMixedTrunkQuantProfile() throws {
        let q4 = KimiK3FormatProfile.quantInt4
        let q8 = KimiK3FormatProfile.quantInt8
        let mixed = GTurboManifestQuantV2(
            embedding: q8.embedding,
            attention: q8.attention,
            router: q8.router,
            sharedExpert: q4.sharedExpert,
            latentProjection: q8.latentProjection,
            denseMLP: q8.denseMLP,
            routedExpert: q8.routedExpert)
        let bundle = try Self.writeTinyBundle(quant: mixed)
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        #expect(throws: ModelError.self) {
            _ = try Self.loadTiny(bundle.url)
        }
    }

    @Test func loadsLayoutLargerThanLegacyMetadataCap() throws {
        let legacyCap = 16 * 1024 * 1024
        let bundle = try Self.writeTinyBundle(
            layoutTrailingWhitespaceBytes: legacyCap + 1)
        defer { try? FileManager.default.removeItem(at: bundle.url) }

        let model = try Self.loadTiny(bundle.url)
        #expect(model.config.numLayers == 3)
        #expect(K3Model.packedExpertsLayoutMaxBytes == 256 * 1024 * 1024)
        #expect(K3Model.packedExpertsLayoutMaxBytes >= 134_715_780)
    }

    @Test func tensorViewsReturnCorrectBytes() throws {
        let bundle = try Self.writeTinyBundle()
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        let model = try Self.loadTiny(bundle.url)
        let view = model.embedding
        #expect(view.length == UInt64(bundle.embeddingWeights.count))
        #expect(view.scaleLength == UInt64(bundle.embeddingScales.count))
        #expect(view.biasLength == UInt64(bundle.embeddingBiases.count))

        let base = view.buffer.contents()
        func bytes(_ offset: UInt64, _ count: Int) -> [UInt8] {
            [UInt8](UnsafeRawBufferPointer(
                start: base.advanced(by: Int(offset)), count: count))
        }
        #expect(bytes(view.offset, bundle.embeddingWeights.count)
                == bundle.embeddingWeights)
        #expect(bytes(view.scaleOffset, bundle.embeddingScales.count)
                == bundle.embeddingScales)
        #expect(bytes(view.biasOffset, bundle.embeddingBiases.count)
                == bundle.embeddingBiases)
    }

    @Test func missingTensorThrows() throws {
        let bundle = try Self.writeTinyBundle(
            missingTensor: "language_model.model.norm.weight")
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        #expect {
            _ = try Self.loadTiny(bundle.url)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("language_model.model.norm.weight")
        }
    }

    @Test func archMismatchThrows() throws {
        let bundle = try Self.writeTinyBundle()
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        #expect {
            _ = try Self.loadTiny(bundle.url, expecting: .kimiK3)
        } throws: { error in
            guard case ModelError.archMismatch(let field, _, _) = error else { return false }
            return field == "hiddenSize"
        }
    }

    @Test func residentChecksumMismatchThrows() throws {
        let bundle = try Self.writeTinyBundle()
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        let url = bundle.url.appendingPathComponent("model_weights.bin")
        var data = try Data(contentsOf: url)
        data[data.count - 1] ^= 0xFF
        try data.write(to: url)
        #expect {
            _ = try Self.loadTiny(bundle.url)
        } throws: { error in
            if case ModelError.checksumMismatch = error { return true }
            return false
        }
    }

    @Test func denseLayerHasNoExpertFile() throws {
        let bundle = try Self.writeTinyBundle()
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        let model = try Self.loadTiny(bundle.url)
        #expect {
            _ = try model.expertLayerFile(0)
        } throws: { error in
            if case ModelError.indexCorrupt = error { return true }
            return false
        }
    }

    @Test func layerFileShaVerifiedLazily() throws {
        let bundle = try Self.writeTinyBundle()
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        // Corrupt a layer file AFTER the manifest is written: load must still
        // succeed (layer files are verified lazily), the first touch throws.
        let layerURL = bundle.url
            .appendingPathComponent("packed_experts")
            .appendingPathComponent("layer_01.bin")
        var data = try Data(contentsOf: layerURL)
        data[data.count - 1] ^= 0xFF
        try data.write(to: layerURL)
        let model = try Self.loadTiny(bundle.url)
        #expect(model.openLayerFileCount() == 0)
        #expect {
            _ = try model.expertLayerFile(1)
        } throws: { error in
            if case ModelError.checksumMismatch = error { return true }
            return false
        }
    }
}
