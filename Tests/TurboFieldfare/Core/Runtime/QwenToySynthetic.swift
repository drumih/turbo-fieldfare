import Foundation
@testable import TurboFieldfare
@testable import TurboFieldfareRepackCore

/// Synthetic Qwen 3.6 toy fixture: a tiny runnable `.gturbo/` directory with
/// the qwen36 tensor-name contract (linear_attn.* on mask-2 layers,
/// self_attn.* with packed [query ; gate] q_proj on mask-1 layers, mlp.gate
/// router, gated shared expert, untied lm_head, NO Gemma sandwich tensors).
/// Mirrors `ModelLoaderTests.writeToySynthetic` for the Gemma toy.
enum QwenToySynthetic {

    /// Build the toy directory in a temp dir and return its URL.
    static func write() throws -> URL {
        let toy = ArchConfig.qwen36Toy()
        let la = toy.linearAttention
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-qwen-toy-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        struct ResidentSpec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
        }

        let d = toy.hiddenSize
        let u16 = MemoryLayout<UInt16>.stride

        func int4AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let groups = cols / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(name: name,
                                dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols / 2),
                                scaleBytes: auxBytes,
                                biasBytes: auxBytes)
        }

        func int8AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let groups = cols / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(name: name,
                                dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols),
                                scaleBytes: auxBytes,
                                biasBytes: auxBytes)
        }

        func bf16Spec(_ name: String, shape: [UInt32], count: Int) -> ResidentSpec {
            ResidentSpec(name: name,
                         dtype: 1,
                         shape: shape,
                         weightBytes: UInt64(count * u16),
                         scaleBytes: 0,
                         biasBytes: 0)
        }

        // 1. Resident specs.
        var specs: [ResidentSpec] = [
            int4AffineSpec("language_model.model.embed_tokens.weight",
                           rows: toy.vocabSize, cols: d),
            int4AffineSpec("language_model.lm_head.weight",
                           rows: toy.vocabSize, cols: d),
            bf16Spec("language_model.model.norm.weight",
                     shape: [UInt32(d), 0, 0, 0], count: d),
        ]
        for L in 0..<toy.numLayers {
            let prefix = "language_model.model.layers.\(L)"
            specs.append(bf16Spec("\(prefix).input_layernorm.weight",
                                  shape: [UInt32(d), 0, 0, 0], count: d))
            specs.append(bf16Spec("\(prefix).post_attention_layernorm.weight",
                                  shape: [UInt32(d), 0, 0, 0], count: d))
            specs.append(int8AffineSpec("\(prefix).mlp.gate.weight",
                                        rows: toy.numExperts, cols: d))
            specs.append(int8AffineSpec("\(prefix).mlp.shared_expert_gate.weight",
                                        rows: 1, cols: d))
            specs.append(int8AffineSpec("\(prefix).mlp.shared_expert.gate_proj.weight",
                                        rows: toy.intermediateSize, cols: d))
            specs.append(int8AffineSpec("\(prefix).mlp.shared_expert.up_proj.weight",
                                        rows: toy.intermediateSize, cols: d))
            specs.append(int8AffineSpec("\(prefix).mlp.shared_expert.down_proj.weight",
                                        rows: d, cols: toy.intermediateSize))
            if toy.layerIsLinear(L) {
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_qkv.weight",
                                            rows: la.qkvDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_z.weight",
                                            rows: la.valueDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_a.weight",
                                            rows: la.numVHeads, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_b.weight",
                                            rows: la.numVHeads, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.out_proj.weight",
                                            rows: d, cols: la.valueDim))
                specs.append(bf16Spec("\(prefix).linear_attn.conv1d.weight",
                                      shape: [UInt32(la.qkvDim), UInt32(la.convKernelSize), 1, 0],
                                      count: la.qkvDim * la.convKernelSize))
                specs.append(bf16Spec("\(prefix).linear_attn.A_log",
                                      shape: [UInt32(la.numVHeads), 0, 0, 0],
                                      count: la.numVHeads))
                specs.append(bf16Spec("\(prefix).linear_attn.dt_bias",
                                      shape: [UInt32(la.numVHeads), 0, 0, 0],
                                      count: la.numVHeads))
                specs.append(bf16Spec("\(prefix).linear_attn.norm.weight",
                                      shape: [UInt32(la.valueHeadDim), 0, 0, 0],
                                      count: la.valueHeadDim))
            } else {
                let qDim = toy.numHeads * toy.fullHeadDim
                let kvDim = toy.numFullKVHeads * toy.fullHeadDim
                specs.append(int4AffineSpec("\(prefix).self_attn.q_proj.weight",
                                            rows: 2 * qDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).self_attn.k_proj.weight",
                                            rows: kvDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).self_attn.v_proj.weight",
                                            rows: kvDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).self_attn.o_proj.weight",
                                            rows: d, cols: qDim))
                specs.append(bf16Spec("\(prefix).self_attn.q_norm.weight",
                                      shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                                      count: toy.fullHeadDim))
                specs.append(bf16Spec("\(prefix).self_attn.k_norm.weight",
                                      shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                                      count: toy.fullHeadDim))
            }
        }

        // 2. Serialize the resident index + payload.
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
                sourceWeight: ModelLoaderTests.dummySource(spec.name),
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
            // Quantized tensors: weight bytes 0x11 (nibbles/bytes of small
            // positive codes), scales 0.01, biases zero. BF16 tensors: 1.0.
            for entry in entries where entry.dtype == 0 {
                memset(base.advanced(by: Int(entry.fileOffset)), 0x11, Int(entry.sizeBytes))
                if entry.scaleSize > 0 {
                    let scales = base.advanced(by: Int(entry.scaleOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    for i in 0..<(Int(entry.scaleSize) / u16) {
                        scales[i] = Quantization.bf16Bits(0.01)
                    }
                }
            }
            for entry in entries where entry.dtype == 1 {
                let dst = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<(Int(entry.sizeBytes) / u16) {
                    dst[i] = Quantization.bf16Bits(1.0)
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 3. Packed experts: int4 gate/up/down per expert, page-aligned stride.
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

        func toyExpertBlob(expert: Int) -> (bytes: [UInt8], tensors: [String: [String: Any]]) {
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]

            func addProjection(prefix: String, rows: Int, cols: Int, role: Int) {
                let projectionRows = toyExpertRows(rows: rows, cols: cols,
                                                   expert: expert, role: role)
                let quantized = projectionRows.map { Quantization.quantizeInt4Affine($0) }
                let packedOffset = bytes.count
                for row in quantized { bytes.append(contentsOf: row.packed) }
                tensors[prefix] = [
                    "offset": packedOffset, "size": bytes.count - packedOffset,
                    "dtype": "U32", "shape": [rows, cols],
                    "bits": 4,
                ]
                let scalesOffset = bytes.count
                for row in quantized { appendU16(row.scales, to: &bytes) }
                tensors["\(prefix)_scales"] = [
                    "offset": scalesOffset, "size": bytes.count - scalesOffset,
                    "dtype": "BF16", "shape": [rows, cols / Quantization.groupSize],
                ]
                let biasesOffset = bytes.count
                for row in quantized { appendU16(row.biases, to: &bytes) }
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

        // 4. layout.json
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

        // 5. manifest.json (arch v2 with the qwen36 family fields)
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
            "family": toy.family.rawValue,
            "attnOutputGate": toy.attnOutputGate,
            "attentionScale": toy.attentionScale,
            "embeddingScaledBySqrtHidden": toy.embeddingScaledBySqrtHidden,
            "routerScaled": toy.routerScaled,
            "ffnSandwichNorms": toy.ffnSandwichNorms,
            "sharedExpertGated": toy.sharedExpertGated,
            "ropeNeoxSubdim": toy.ropeNeoxSubdim,
            "linearNumKHeads": la.numKHeads,
            "linearNumVHeads": la.numVHeads,
            "linearKeyHeadDim": la.keyHeadDim,
            "linearValueHeadDim": la.valueHeadDim,
            "linearConvKernelSize": la.convKernelSize,
        ]
        let manifestRoot: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "qwen-toy",
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
}

extension ArchConfig {
    /// Tiny Qwen 3.6 baseline: 4 layers alternating gated-DeltaNet linear
    /// (mask 2) and full attention (mask 1), 8 experts (top-8 to satisfy the
    /// decode kernels' fixed top-k), gated shared expert, untied lm_head,
    /// no sandwich norms. Numbers are intentionally toy but respect every
    /// kernel divisibility constraint (D % 64, keyHeadDim % 32, even
    /// rotaryDim).
    static func qwen36Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 128,
            moeIntermediateSize: 128,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 32,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 4,
            numExperts: 8,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [2, 1, 2, 1],
            hiddenActivation: "silu",
            family: .qwen36,
            attnOutputGate: true,
            attentionScale: 0.125,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearAttention: LinearAttentionConfig(
                numKHeads: 2, numVHeads: 4,
                keyHeadDim: 32, valueHeadDim: 32,
                convKernelSize: 4)
        )
    }
}
