import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanRepackCore
@testable import TinyTitanCore

@Suite struct ModelLoaderTests {
    static func dummySource(_ name: String) -> SourceTensor {
        SourceTensor(
            name: name, shardPath: "/dev/null", dtype: .u32,
            shape: [1024, 64], absoluteOffset: 0, sizeBytes: 0)
    }

    /// Build a minimal valid `model.ssdai/` directory in a temp dir and
    /// return the URL. Uses the toy ArchConfig `qwenToy()`: 4 layers
    /// (alternating gated-DeltaNet linear and full attention), 8 experts,
    /// hidden 64, vocab 1024, untied lm_head. Resident contains the embedding,
    /// the separate lm_head, the final norm, and the Qwen layer-resident
    /// tensors (router, gated shared expert, full-attention or linear_attn
    /// bundle per layer) needed to construct `RealForwardRunner` in unit
    /// tests. No auxiliary sandwich/scale tensors, matching the real model.
    static func writeToySynthetic() throws -> URL {
        let toy = ArchConfig.qwenToy()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-toy-\(UUID().uuidString)")
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
        let embedSize = UInt64(toy.vocabSize * toy.hiddenSize)
        let bf16DBytes = UInt64(d * MemoryLayout<UInt16>.stride)

        func int4AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let groups = (cols + Quantization.groupSize - 1) / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * MemoryLayout<UInt16>.stride)
            // int4 packs 2 values per byte
            let packedWeightBytes = UInt64(rows * cols / 2)
            return ResidentSpec(
                name: name,
                dtype: 0,
                shape: [UInt32(rows), UInt32(cols), 0, 0],
                weightBytes: packedWeightBytes,
                scaleBytes: auxBytes,
                biasBytes: auxBytes)
        }

        func int8AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            // int8: 1 value per byte, no packing
            let weightBytes = UInt64(rows * cols)
            // int8 affine still has scale/bias aux metadata
            let groups = (cols + Quantization.groupSize - 1) / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * MemoryLayout<UInt16>.stride)
            return ResidentSpec(
                name: name,
                dtype: 0,
                shape: [UInt32(rows), UInt32(cols), 0, 0],
                weightBytes: weightBytes,
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
                let projectionRows = toyExpertRows(
                    rows: rows, cols: cols, expert: expert, role: role)
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
            // Embedding is int4 affine: weight data is packed 2 values per byte.
            ResidentSpec(
                name: "language_model.model.embed_tokens.weight",
                dtype: 0,
                shape: [UInt32(toy.vocabSize), UInt32(toy.hiddenSize), 0, 0],
                weightBytes: embedSize / 2,
                scaleBytes: UInt64(
                    toy.vocabSize * (d / Quantization.groupSize) * MemoryLayout<UInt16>.stride),
                biasBytes: UInt64(
                    toy.vocabSize * (d / Quantization.groupSize) * MemoryLayout<UInt16>.stride)),
            // Qwen carries a separate untied lm_head.
            ResidentSpec(
                name: "language_model.lm_head.weight",
                dtype: 0,
                shape: [UInt32(toy.vocabSize), UInt32(toy.hiddenSize), 0, 0],
                weightBytes: embedSize / 2,
                scaleBytes: UInt64(
                    toy.vocabSize * (d / Quantization.groupSize) * MemoryLayout<UInt16>.stride),
                biasBytes: UInt64(
                    toy.vocabSize * (d / Quantization.groupSize) * MemoryLayout<UInt16>.stride)),
            ResidentSpec(
                name: "language_model.model.norm.weight",
                dtype: 1,
                shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                weightBytes: bf16DBytes,
                scaleBytes: 0,
                biasBytes: 0),
        ]
        for L in 0..<toy.numLayers {
            let prefix = "language_model.model.layers.\(L)"
            specs.append(
                ResidentSpec(
                    name: "\(prefix).input_layernorm.weight",
                    dtype: 1,
                    shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                    weightBytes: bf16DBytes,
                    scaleBytes: 0,
                    biasBytes: 0))
            specs.append(
                ResidentSpec(
                    name: "\(prefix).post_attention_layernorm.weight",
                    dtype: 1,
                    shape: [UInt32(toy.hiddenSize), 0, 0, 0],
                    weightBytes: bf16DBytes,
                    scaleBytes: 0,
                    biasBytes: 0))
            // Router (8-bit, matching quant.router) + the sigmoid-gated shared
            // expert gate (also at the router width, 8-bit on the target).
            specs.append(
                int8AffineSpec(
                    "\(prefix).mlp.gate.weight",
                    rows: toy.numExperts,
                    cols: d))
            specs.append(
                int8AffineSpec(
                    "\(prefix).mlp.shared_expert_gate.weight",
                    rows: 1,
                    cols: d))
            specs.append(
                int4AffineSpec(
                    "\(prefix).mlp.shared_expert.gate_proj.weight",
                    rows: toy.intermediateSize,
                    cols: d))
            specs.append(
                int4AffineSpec(
                    "\(prefix).mlp.shared_expert.up_proj.weight",
                    rows: toy.intermediateSize,
                    cols: d))
            specs.append(
                int4AffineSpec(
                    "\(prefix).mlp.shared_expert.down_proj.weight",
                    rows: d,
                    cols: toy.intermediateSize))
            if toy.layerIsLinear(L) {
                // Gated-DeltaNet layers carry only the linear_attn bundle.
                let la = toy.linearAttention
                specs.append(
                    int4AffineSpec(
                        "\(prefix).linear_attn.in_proj_qkv.weight",
                        rows: la.qkvDim,
                        cols: d))
                specs.append(
                    int4AffineSpec(
                        "\(prefix).linear_attn.in_proj_z.weight",
                        rows: la.valueDim,
                        cols: d))
                specs.append(
                    int4AffineSpec(
                        "\(prefix).linear_attn.in_proj_a.weight",
                        rows: la.numVHeads,
                        cols: d))
                specs.append(
                    int4AffineSpec(
                        "\(prefix).linear_attn.in_proj_b.weight",
                        rows: la.numVHeads,
                        cols: d))
                specs.append(
                    int4AffineSpec(
                        "\(prefix).linear_attn.out_proj.weight",
                        rows: d,
                        cols: la.valueDim))
                specs.append(
                    ResidentSpec(
                        name: "\(prefix).linear_attn.conv1d.weight",
                        dtype: 1,
                        shape: [UInt32(la.qkvDim), UInt32(la.convKernelSize), 1, 0],
                        weightBytes: UInt64(
                            la.qkvDim * la.convKernelSize * MemoryLayout<UInt16>.stride),
                        scaleBytes: 0,
                        biasBytes: 0))
                specs.append(
                    ResidentSpec(
                        name: "\(prefix).linear_attn.A_log",
                        dtype: 1,
                        shape: [UInt32(la.numVHeads), 0, 0, 0],
                        weightBytes: UInt64(la.numVHeads * MemoryLayout<UInt16>.stride),
                        scaleBytes: 0,
                        biasBytes: 0))
                specs.append(
                    ResidentSpec(
                        name: "\(prefix).linear_attn.dt_bias",
                        dtype: 1,
                        shape: [UInt32(la.numVHeads), 0, 0, 0],
                        weightBytes: UInt64(la.numVHeads * MemoryLayout<UInt16>.stride),
                        scaleBytes: 0,
                        biasBytes: 0))
                specs.append(
                    ResidentSpec(
                        name: "\(prefix).linear_attn.norm.weight",
                        dtype: 1,
                        shape: [UInt32(la.valueHeadDim), 0, 0, 0],
                        weightBytes: UInt64(la.valueHeadDim * MemoryLayout<UInt16>.stride),
                        scaleBytes: 0,
                        biasBytes: 0))
            } else {
                // Full-attention layer: gate-packed q_proj (2x rows) and
                // per-head q/k norms.
                let queryDim = 2 * toy.numHeads * toy.fullHeadDim
                let kvDim = toy.numFullKVHeads * toy.fullHeadDim
                specs.append(
                    ResidentSpec(
                        name: "\(prefix).self_attn.q_norm.weight",
                        dtype: 1,
                        shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                        weightBytes: UInt64(toy.fullHeadDim * MemoryLayout<UInt16>.stride),
                        scaleBytes: 0,
                        biasBytes: 0))
                specs.append(
                    ResidentSpec(
                        name: "\(prefix).self_attn.k_norm.weight",
                        dtype: 1,
                        shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                        weightBytes: UInt64(toy.fullHeadDim * MemoryLayout<UInt16>.stride),
                        scaleBytes: 0,
                        biasBytes: 0))
                specs.append(
                    int4AffineSpec(
                        "\(prefix).self_attn.q_proj.weight",
                        rows: queryDim,
                        cols: d))
                specs.append(
                    int4AffineSpec(
                        "\(prefix).self_attn.k_proj.weight",
                        rows: kvDim,
                        cols: d))
                specs.append(
                    int4AffineSpec(
                        "\(prefix).self_attn.v_proj.weight",
                        rows: kvDim,
                        cols: d))
                specs.append(
                    int4AffineSpec(
                        "\(prefix).self_attn.o_proj.weight",
                        rows: d,
                        cols: toy.numHeads * toy.fullHeadDim))
            }
        }

        let names = specs.map(\.name)
        let stringTable = Data(names.joined().utf8)
        let headerBytes = SSDAIBinary.indexHeaderBytes
        let entryBytes = SSDAIBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameAbsOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameAbsOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        let indexBytes = UInt64(stringTableBase + stringTable.count)
        // Pad the index to 16 KB alignment (SSDAI v1 format requirement).
        let alignmentBytes: UInt64 = 16_384
        let alignedIndexBytes =
            ((indexBytes + alignmentBytes - 1) / alignmentBytes)
            * alignmentBytes

        var entries: [ResidentEntry] = []
        entries.reserveCapacity(specs.count)
        var payloadCursor = alignedIndexBytes
        for spec in specs {
            let weightOffset = payloadCursor
            let scaleOffset = spec.scaleBytes > 0 ? weightOffset + spec.weightBytes : 0
            let biasOffset = spec.biasBytes > 0 ? scaleOffset + spec.scaleBytes : 0
            entries.append(
                ResidentEntry(
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
        let residentSize = payloadCursor - alignedIndexBytes

        let totalBytes = Int(alignedIndexBytes + residentSize)
        var fileBuf = [UInt8](repeating: 0, count: totalBytes)
        try fileBuf.withUnsafeMutableBytes { raw in
            let base = try #require(raw.baseAddress)
            SSDAIBinary.writeIndexHeader(
                into: base,
                indexSize: alignedIndexBytes,
                residentSize: residentSize,
                entryCount: UInt64(entries.count))
            for (i, e) in entries.enumerated() {
                let dst = base.advanced(by: entriesBase + i * entryBytes)
                SSDAIBinary.writeIndexEntry(
                    into: dst, entry: e,
                    nameOffset: nameAbsOffsets[i])
            }
            stringTable.withUnsafeBytes { sb in
                // An empty table has nothing to copy; the source pointer is only
                // meaningful when there is a byte to read.
                if let source = sb.baseAddress {
                    memcpy(base.advanced(by: stringTableBase), source, stringTable.count)
                }
            }
            // Recognizable resident payload pattern in the final-norm region
            // only; other payload bytes stay zero except quantized scale
            // regions.
            let normEntry = try #require(
                entries.first {
                    $0.name == "language_model.model.norm.weight"
                })
            let normStart = Int(normEntry.fileOffset)
            for i in 0..<Int(normEntry.sizeBytes) {
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
            for entry in entries
            where entry.dtype == 1 && entry.name != "language_model.model.norm.weight" {
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
                precondition(
                    blob.count <= Int(expertStride),
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
                    "size": expertStride,
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
        let quantSlotInt4: [String: Any] = [
            "weightBits": 4, "scheme": "affine",
            "scaleType": "bf16", "biasType": "bf16",
            "groupSize": 64,
        ]
        let quantSlotInt8: [String: Any] = [
            "weightBits": 8, "scheme": "affine",
            "scaleType": "bf16", "biasType": "bf16",
            "groupSize": 64,
        ]
        let quant: [String: Any] = [
            "embedding": quantSlotInt4,
            "attention": quantSlotInt4,
            "router": quantSlotInt8,
            "sharedExpert": quantSlotInt4,
            "routedExpert": quantSlotInt4,
        ]
        let manifestRoot: [String: Any] = [
            "magic": "SSDAI",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "toy",
            "arch": archDict,
            "quant": quant,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": expertStride,
        ]
        let manifestData = try JSONSerialization.data(
            withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    static func writeVerifiedInstallReceipt(directoryURL dir: URL) throws {
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: .qwenToy())
        let manifestURL = dir.appendingPathComponent("manifest.json")
        let manifestSha = try Sha256Verifier.hashFile(at: manifestURL)
        let manifestSize = try #require(
            try FileManager.default
                .attributesOfItem(atPath: manifestURL.path)[.size] as? NSNumber)
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

    static func mutateReceipt(
        directoryURL dir: URL,
        transform: (inout [String: Any]) throws -> Void
    ) throws {
        let receiptURL = dir.appendingPathComponent(VerifiedInstallReceiptReader.fileName)
        var root = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: receiptURL)) as? [String: Any])
        try transform(&root)
        let data = try JSONSerialization.data(
            withJSONObject: root,
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
