import Testing
import Foundation
import Metal
@testable import TurboFieldfare
@testable import TurboFieldfareRepackCore

@Suite struct ModelLoaderTests {
    static func dummySource(_ name: String) -> SourceTensor {
        SourceTensor(name: name, shardPath: "/dev/null", dtype: .u32,
                     shape: [1024, 64], absoluteOffset: 0, sizeBytes: 0)
    }

    /// Build a minimal valid `model.gturbo/` directory in a temp dir and
    /// return the URL. Uses the toy ArchConfig `gemma4Toy()`: 2 layers,
    /// 8 experts, hidden 64, vocab 1024. Resident contains the embedding
    /// (alias: lmHead), final norm, and the tiny layer-resident tensors needed
    /// to construct `RealForwardRunner` in unit tests.
    static func writeToySynthetic() throws -> URL {
        let toy = ArchConfig.gemma4Toy()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-toy-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        // 1. Resident region: embedding + final norm + runner-init tensors.
        struct ResidentSpec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
        }

        let d = toy.hiddenSize
        let f = toy.intermediateSize
        let embedSize = UInt64(toy.vocabSize * toy.hiddenSize)
        let bf16DBytes = UInt64(d * MemoryLayout<UInt16>.stride)

        func int4AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let groups = (cols + Quantization.groupSize - 1) / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * MemoryLayout<UInt16>.stride)
            return ResidentSpec(name: name,
                                dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols),
                                scaleBytes: auxBytes,
                                biasBytes: auxBytes)
        }

        func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }

        func toyExpertRows(rows: Int, cols: Int, expert: Int, role: Int) -> [[Float]] {
            (0..<rows).map { row in
                (0..<cols).map { col in
                    Float(expert + 1) * 0.001
                        + Float(role + 1) * 0.003
                        + Float((row % 7) - 3) * 0.0004
                        + Float((col % 11) - 5) * 0.0002
                }
            }
        }

        func appendProjection(rows: [[Float]], to bytes: inout [UInt8], component: String) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
            switch component {
            case "packed":
                for row in quantized { bytes.append(contentsOf: row.packed) }
            case "scales":
                for row in quantized { appendU16(row.scales, to: &bytes) }
            case "biases":
                for row in quantized { appendU16(row.biases, to: &bytes) }
            default:
                preconditionFailure("unknown projection component \(component)")
            }
        }

        func toyExpertBlob(expert: Int) -> (bytes: [UInt8], tensors: [String: [String: Any]]) {
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]

            func addProjection(prefix: String, rows: Int, cols: Int, role: Int) {
                let projectionRows = toyExpertRows(rows: rows, cols: cols, expert: expert, role: role)
                let packedOffset = bytes.count
                appendProjection(rows: projectionRows, to: &bytes, component: "packed")
                tensors[prefix] = [
                    "offset": packedOffset, "size": bytes.count - packedOffset,
                    "dtype": "U32", "shape": [rows, cols],
                    "bits": 4,
                ]
                let scalesOffset = bytes.count
                appendProjection(rows: projectionRows, to: &bytes, component: "scales")
                tensors["\(prefix)_scales"] = [
                    "offset": scalesOffset, "size": bytes.count - scalesOffset,
                    "dtype": "BF16", "shape": [rows, cols / Quantization.groupSize],
                ]
                let biasesOffset = bytes.count
                appendProjection(rows: projectionRows, to: &bytes, component: "biases")
                tensors["\(prefix)_biases"] = [
                    "offset": biasesOffset, "size": bytes.count - biasesOffset,
                    "dtype": "BF16", "shape": [rows, cols / Quantization.groupSize],
                ]
            }

            addProjection(prefix: "gate", rows: toy.moeIntermediateSize, cols: d, role: 0)
            addProjection(prefix: "up", rows: toy.moeIntermediateSize, cols: d, role: 1)
            addProjection(prefix: "down", rows: d, cols: toy.moeIntermediateSize, role: 2)
            return (bytes, tensors)
        }

        var specs: [ResidentSpec] = [
            ResidentSpec(name: "language_model.model.embed_tokens.weight",
                         dtype: 0,
                         shape: [UInt32(toy.vocabSize), UInt32(toy.hiddenSize), 0, 0],
                         weightBytes: embedSize,
                         scaleBytes: UInt64(toy.vocabSize * (d / Quantization.groupSize) * MemoryLayout<UInt16>.stride),
                         biasBytes: UInt64(toy.vocabSize * (d / Quantization.groupSize) * MemoryLayout<UInt16>.stride)),
            ResidentSpec(name: "language_model.model.norm.weight",
                         dtype: 1,
                         shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                         weightBytes: bf16DBytes,
                         scaleBytes: 0,
                         biasBytes: 0),
        ]
        for L in 0..<toy.numLayers {
            let isFull = toy.fullAttentionLayerMask[L] != 0
            let headDim = isFull ? toy.fullHeadDim : toy.headDim
            let numKVHeads = isFull ? toy.numFullKVHeads : toy.numKVHeads
            let qDim = toy.numHeads * headDim
            let kvDim = numKVHeads * headDim
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).input_layernorm.weight",
                dtype: 1,
                shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                weightBytes: bf16DBytes,
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).self_attn.q_proj.weight",
                rows: qDim,
                cols: d))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).self_attn.k_proj.weight",
                rows: kvDim,
                cols: d))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).self_attn.v_proj.weight",
                rows: kvDim,
                cols: d))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).self_attn.o_proj.weight",
                rows: d,
                cols: qDim))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).post_attention_layernorm.weight",
                dtype: 1,
                shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                weightBytes: bf16DBytes,
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).pre_feedforward_layernorm.weight",
                dtype: 1,
                shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                weightBytes: bf16DBytes,
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).pre_feedforward_layernorm_2.weight",
                dtype: 1,
                shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                weightBytes: bf16DBytes,
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).self_attn.q_norm.weight",
                dtype: 1,
                shape: [UInt32(headDim), 0, 0, 0],
                weightBytes: UInt64(headDim * MemoryLayout<UInt16>.stride),
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).self_attn.k_norm.weight",
                dtype: 1,
                shape: [UInt32(headDim), 0, 0, 0],
                weightBytes: UInt64(headDim * MemoryLayout<UInt16>.stride),
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).mlp.gate_proj.weight",
                rows: f,
                cols: d))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).mlp.up_proj.weight",
                rows: f,
                cols: d))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).mlp.down_proj.weight",
                rows: d,
                cols: f))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).post_feedforward_layernorm_1.weight",
                dtype: 1,
                shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                weightBytes: bf16DBytes,
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).post_feedforward_layernorm_2.weight",
                dtype: 1,
                shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                weightBytes: bf16DBytes,
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).post_feedforward_layernorm.weight",
                dtype: 1,
                shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                weightBytes: bf16DBytes,
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).layer_scalar",
                dtype: 1,
                shape: [1, 0, 0, 0],
                weightBytes: UInt64(MemoryLayout<UInt16>.stride),
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).router.scale",
                dtype: 1,
                shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                weightBytes: bf16DBytes,
                scaleBytes: 0,
                biasBytes: 0))
            specs.append(int4AffineSpec(
                "language_model.model.layers.\(L).router.proj.weight",
                rows: toy.numExperts,
                cols: d))
            specs.append(ResidentSpec(
                name: "language_model.model.layers.\(L).router.per_expert_scale",
                dtype: 1,
                shape: [UInt32(toy.numExperts), 0, 0, 0],
                weightBytes: UInt64(toy.numExperts * MemoryLayout<UInt16>.stride),
                scaleBytes: 0,
                biasBytes: 0))
        }

        let names = specs.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboBinary.indexHeaderBytes
        let entryBytes  = GTurboBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameAbsOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameAbsOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        let indexBytes = UInt64(stringTableBase + stringTable.count)

        var entries: [ResidentEntry] = []
        entries.reserveCapacity(specs.count)
        var payloadCursor = indexBytes
        for spec in specs {
            let weightOffset = payloadCursor
            let scaleOffset = spec.scaleBytes > 0 ? weightOffset + spec.weightBytes : 0
            let biasOffset = spec.biasBytes > 0 ? scaleOffset + spec.scaleBytes : 0
            entries.append(ResidentEntry(
                name: spec.name,
                dtype: spec.dtype,
                logicalShape4: spec.shape,
                fileOffset: weightOffset,
                sizeBytes: spec.weightBytes,
                scaleOffset: scaleOffset,
                scaleSize: spec.scaleBytes,
                biasOffset: biasOffset,
                biasSize: spec.biasBytes,
                quantSpec: nil,
                sourceWeight: Self.dummySource(spec.name),
                sourceScales: nil,
                sourceBiases: nil))
            payloadCursor += spec.weightBytes + spec.scaleBytes + spec.biasBytes
        }
        let residentSize = payloadCursor - indexBytes

        let totalBytes = Int(indexBytes + residentSize)
        var fileBuf = [UInt8](repeating: 0, count: totalBytes)
        fileBuf.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: indexBytes,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            for (i, e) in entries.enumerated() {
                let dst = base.advanced(by: entriesBase + i * entryBytes)
                GTurboBinary.writeIndexEntry(into: dst, entry: e,
                                             nameOffset: nameAbsOffsets[i])
            }
            _ = stringTable.withUnsafeBytes { sb in
                memcpy(base.advanced(by: stringTableBase), sb.baseAddress!, stringTable.count)
            }
            // Recognizable resident payload pattern in the norm region only;
            // other payload bytes stay zero except router.scale unit BF16s.
            let normStart = Int(entries[1].fileOffset)
            for i in 0..<Int(entries[1].sizeBytes) {
                base.advanced(by: normStart + i)
                    .assumingMemoryBound(to: UInt8.self)[0] = UInt8(0xC0 | (i & 0x3F))
            }
            for entry in entries where entry.dtype == 0 {
                memset(base.advanced(by: Int(entry.fileOffset)), 0x11, Int(entry.sizeBytes))
                if entry.scaleSize > 0 {
                    let scales = base.advanced(by: Int(entry.scaleOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    for i in 0..<(Int(entry.scaleSize) / MemoryLayout<UInt16>.stride) {
                        scales[i] = Quantization.bf16Bits(0.01)
                    }
                }
            }
            for entry in entries where entry.dtype == 1 && entry.name != "language_model.model.norm.weight" {
                let dst = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<(Int(entry.sizeBytes) / MemoryLayout<UInt16>.stride) {
                    dst[i] = Quantization.bf16Bits(1.0)
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 2. packed_experts: 2 layer files, each `expertsPerLayer * expertStride`
        // bytes. expertStride must be a multiple of getpagesize() (16 KB).
        let expertStride: UInt64 = 16384
        let layerBytes = Int(expertStride) * toy.numExperts
        for L in 0..<toy.numLayers {
            var payload = Data(count: layerBytes)
            for E in 0..<toy.numExperts {
                let blob = toyExpertBlob(expert: E).bytes
                let baseB = E * Int(expertStride)
                precondition(blob.count <= Int(expertStride),
                             "toy expert blob exceeds stride")
                for (i, byte) in blob.enumerated() {
                    payload[baseB + i] = byte
                }
                // Tag bytes outside the projection region so the round-trip
                // test can identify which blob is being read.
                payload[baseB + 0] = UInt8(L)
                payload[baseB + 1] = UInt8(E)
                payload[baseB + 2] = 0xC1
                payload[baseB + 3] = 0xC2
            }
            let url = exp.appendingPathComponent(String(format: "layer_%02d.bin", L))
            try payload.write(to: url)
        }
        var layerShaByName: [String: String] = [:]
        for L in 0..<toy.numLayers {
            let basename = String(format: "layer_%02d.bin", L)
            let url = exp.appendingPathComponent(basename)
            layerShaByName["packed_experts/\(basename)"] = try Sha256Verifier.hashFile(at: url)
        }

        // 3. layout.json
        var layersArr: [[String: Any]] = []
        for L in 0..<toy.numLayers {
            var experts: [[String: Any]] = []
            for E in 0..<toy.numExperts {
                let blob = toyExpertBlob(expert: E)
                experts.append([
                    "expert": E,
                    "offset": UInt64(E) * expertStride,
                    "size":   expertStride,
                    "tensors": blob.tensors,
                ])
            }
            layersArr.append([
                "layer": L,
                "file": String(format: "layer_%02d.bin", L),
                "experts": experts,
            ])
        }
        let layoutRoot: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": toy.numLayers,
            "expertsPerLayer": toy.numExperts,
            "layers": layersArr,
        ]
        let layoutData = try JSONSerialization.data(
            withJSONObject: layoutRoot, options: [.sortedKeys])
        let layoutURL = exp.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)

        // 4. manifest.json
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": Int(totalBytes), "sha256": weightsSha],
            "packed_experts/layout.json": ["size": layoutData.count, "sha256": layoutSha],
        ]
        for (rel, sha) in layerShaByName {
            files[rel] = ["size": layerBytes, "sha256": sha]
        }

        let archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize, "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads, "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim, "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize, "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta, "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers, "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
        ]
        let manifestRoot: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": expertStride,
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    static func writeVerifiedInstallReceipt(directoryURL dir: URL) throws {
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: .gemma4Toy())
        let manifestURL = dir.appendingPathComponent("manifest.json")
        let manifestSha = try Sha256Verifier.hashFile(at: manifestURL)
        let manifestSize = try FileManager.default
            .attributesOfItem(atPath: manifestURL.path)[.size] as! NSNumber
        var receiptFiles = manifest.files.mapValues {
            VerifiedInstallReceipt.FileEntry(size: $0.size, sha256: $0.sha256)
        }
        receiptFiles["manifest.json"] = VerifiedInstallReceipt.FileEntry(
            size: manifestSize.uint64Value,
            sha256: manifestSha)
        let receipt = VerifiedInstallReceipt(
            manifestSha256: manifestSha,
            modelDirectoryPath: dir.standardizedFileURL.path,
            sourceRepoID: "toy",
            sourceRevision: "test",
            verificationTimestamp: "2026-07-01T00:00:00Z",
            toolVersion: "test",
            files: receiptFiles)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(receipt)
        try data.write(to: dir.appendingPathComponent(VerifiedInstallReceiptReader.fileName))
    }

    static func mutateReceipt(directoryURL dir: URL,
                                      transform: (inout [String: Any]) throws -> Void) throws {
        let receiptURL = dir.appendingPathComponent(VerifiedInstallReceiptReader.fileName)
        var root = try JSONSerialization.jsonObject(
            with: Data(contentsOf: receiptURL)) as! [String: Any]
        try transform(&root)
        let data = try JSONSerialization.data(withJSONObject: root,
                                              options: [.sortedKeys, .withoutEscapingSlashes])
        try data.write(to: receiptURL)
    }

    static func flipByte(in url: URL, at offset: UInt64) throws {
        let handle = try FileHandle(forUpdating: url)
        try handle.seek(toOffset: offset)
        let byte = try #require(try handle.read(upToCount: 1)?.first)
        try handle.seek(toOffset: offset)
        try handle.write(contentsOf: [byte ^ 0xFF])
        try handle.close()
    }

    // MARK: - Positive

    // MARK: - Negative

    // MARK: - Lazy fd / SHA

}
