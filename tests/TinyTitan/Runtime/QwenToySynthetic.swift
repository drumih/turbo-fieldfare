import Foundation
import Testing
import TinyTitanFormat

@testable import TinyTitan
@testable import TinyTitanRepackCore

/// Synthetic Qwen 3.6 toy fixture: a tiny runnable `.ssdai/` directory with
/// the qwen36 tensor-name contract (linear_attn.* on mask-2 layers,
/// self_attn.* with gate-packed [query ; gate] q_proj on mask-1 layers,
/// mlp.gate router, gated shared expert, untied lm_head, no auxiliary
/// sandwich/scale tensors). Mirrors `ModelLoaderTests.writeToySynthetic` for
/// the Qwen toy config.
enum QwenToySynthetic {

    /// Build the toy directory in a temp dir and return its URL.
    ///
    /// `variedLinearAttention` fills each quantized `linear_attn` tensor with
    /// its own deterministic byte pattern instead of the shared 0x11, so the
    /// GDN projections differ by channel and `a` differs from `b`; a test
    /// that must tell those rows apart needs it.
    ///
    /// `variedHead` does the same for the lm_head (packed values, scales and
    /// biases) and the final norm, so logits differ by vocabulary row; with
    /// the constant pattern every logit is equal.
    ///
    /// `bf16Rest` mirrors the Qwen3.8 BF16Rest install: routed experts stay
    /// affine, and every dense matrix -- embedding, lm_head, router, scalar
    /// gate, shared expert, attention and GDN projections -- is kept at bf16
    /// with no scales or biases, filled with small varied values
    /// (`fillBF16Projections`). The manifest's slots still say `weightBits`
    /// and its `promoted` list names every bf16 matrix but the lm_head, as the
    /// real install's does, so the runtime has to read the head's dtype.
    static func write(
        weightBits: Int = 4, qsa: Bool = false, variedLinearAttention: Bool = false,
        variedHead: Bool = false, bf16Rest: Bool = false
    ) throws -> URL {
        precondition([4, 8].contains(weightBits))
        let toy = qsa ? ArchConfig.qwenToyQSA() : ArchConfig.qwenToy()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-qwen-toy-\(UUID().uuidString)")
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

        func affineSpec(
            _ name: String, rows: Int, cols: Int,
            bits: Int = weightBits
        ) -> ResidentSpec {
            if bf16Rest { return bf16Matrix(name, rows: rows, cols: cols) }
            // Runtime only supports 4 or 8-bit affine.
            let usedBits = bits <= 4 ? 4 : 8
            let groups = cols / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(
                name: name,
                dtype: 0,
                shape: [UInt32(rows), UInt32(cols), 0, 0],
                weightBytes: UInt64(rows * cols * usedBits / 8),
                scaleBytes: auxBytes,
                biasBytes: auxBytes)
        }

        func int8AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            if bf16Rest { return bf16Matrix(name, rows: rows, cols: cols) }
            let groups = cols / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(
                name: name,
                dtype: 0,
                shape: [UInt32(rows), UInt32(cols), 0, 0],
                weightBytes: UInt64(rows * cols),
                scaleBytes: auxBytes,
                biasBytes: auxBytes)
        }

        func bf16Spec(_ name: String, shape: [UInt32], count: Int) -> ResidentSpec {
            ResidentSpec(
                name: name,
                dtype: 1,
                shape: shape,
                weightBytes: UInt64(count * u16),
                scaleBytes: 0,
                biasBytes: 0)
        }

        func bf16Matrix(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            bf16Spec(name, shape: [UInt32(rows), UInt32(cols), 0, 0], count: rows * cols)
        }

        // 1. Resident specs.
        var specs: [ResidentSpec] = [
            affineSpec(
                "language_model.model.embed_tokens.weight",
                rows: toy.vocabSize, cols: d),
            affineSpec(
                "language_model.lm_head.weight",
                rows: toy.vocabSize, cols: d),
            bf16Spec(
                "language_model.model.norm.weight",
                shape: [UInt32(d), 0, 0, 0], count: d),
        ]
        for L in 0..<toy.numLayers {
            let prefix = "language_model.model.layers.\(L)"
            specs.append(
                bf16Spec(
                    "\(prefix).input_layernorm.weight",
                    shape: [UInt32(d), 0, 0, 0], count: d))
            specs.append(
                bf16Spec(
                    "\(prefix).post_attention_layernorm.weight",
                    shape: [UInt32(d), 0, 0, 0], count: d))
            // Router (8-bit, matching the target's quant.router) + the
            // sigmoid-gated shared expert at the sharedExpert slot width.
            specs.append(
                int8AffineSpec(
                    "\(prefix).mlp.gate.weight",
                    rows: toy.numExperts, cols: d))
            // The shared-expert scalar gate is quantized at the router width
            // (8-bit on the target, like mlp.gate) — not the sharedExpert slot.
            specs.append(
                int8AffineSpec(
                    "\(prefix).mlp.shared_expert_gate.weight",
                    rows: 1, cols: d))
            specs.append(
                affineSpec(
                    "\(prefix).mlp.shared_expert.gate_proj.weight",
                    rows: toy.intermediateSize, cols: d))
            specs.append(
                affineSpec(
                    "\(prefix).mlp.shared_expert.up_proj.weight",
                    rows: toy.intermediateSize, cols: d))
            specs.append(
                affineSpec(
                    "\(prefix).mlp.shared_expert.down_proj.weight",
                    rows: d, cols: toy.intermediateSize))
            if toy.layerIsLinear(L) {
                // Gated-DeltaNet layers carry only the linear_attn bundle —
                // no self_attn projections and no q/k norms.
                let la = toy.linearAttention
                specs.append(
                    affineSpec(
                        "\(prefix).linear_attn.in_proj_qkv.weight",
                        rows: la.qkvDim, cols: d))
                specs.append(
                    affineSpec(
                        "\(prefix).linear_attn.in_proj_z.weight",
                        rows: la.valueDim, cols: d))
                specs.append(
                    affineSpec(
                        "\(prefix).linear_attn.in_proj_a.weight",
                        rows: la.numVHeads, cols: d))
                specs.append(
                    affineSpec(
                        "\(prefix).linear_attn.in_proj_b.weight",
                        rows: la.numVHeads, cols: d))
                specs.append(
                    affineSpec(
                        "\(prefix).linear_attn.out_proj.weight",
                        rows: d, cols: la.valueDim))
                specs.append(
                    bf16Spec(
                        "\(prefix).linear_attn.conv1d.weight",
                        shape: [UInt32(la.qkvDim), UInt32(la.convKernelSize), 1, 0],
                        count: la.qkvDim * la.convKernelSize))
                specs.append(
                    bf16Spec(
                        "\(prefix).linear_attn.A_log",
                        shape: [UInt32(la.numVHeads), 0, 0, 0],
                        count: la.numVHeads))
                specs.append(
                    bf16Spec(
                        "\(prefix).linear_attn.dt_bias",
                        shape: [UInt32(la.numVHeads), 0, 0, 0],
                        count: la.numVHeads))
                specs.append(
                    bf16Spec(
                        "\(prefix).linear_attn.norm.weight",
                        shape: [UInt32(la.valueHeadDim), 0, 0, 0],
                        count: la.valueHeadDim))
            } else {
                // Full-attention layer: gate-packed q_proj (2x rows),
                // per-head q/k norms, separate k/v projections.
                let queryDim = 2 * toy.numHeads * toy.fullHeadDim
                let kvDim = toy.numFullKVHeads * toy.fullHeadDim
                specs.append(
                    bf16Spec(
                        "\(prefix).self_attn.q_norm.weight",
                        shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                        count: toy.fullHeadDim))
                specs.append(
                    bf16Spec(
                        "\(prefix).self_attn.k_norm.weight",
                        shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                        count: toy.fullHeadDim))
                specs.append(
                    affineSpec(
                        "\(prefix).self_attn.q_proj.weight",
                        rows: queryDim, cols: d))
                specs.append(
                    affineSpec(
                        "\(prefix).self_attn.k_proj.weight",
                        rows: kvDim, cols: d))
                specs.append(
                    affineSpec(
                        "\(prefix).self_attn.v_proj.weight",
                        rows: kvDim, cols: d))
                if qsa {
                    let indexRows =
                        toy.sparseIndexer.numHeads
                        * toy.sparseIndexer.headDim
                    let indexPrefix =
                        "model.language_model.layers.\(L).self_attn.indexer"
                    specs.append(
                        affineSpec(
                            "\(indexPrefix).index_q_proj.weight",
                            rows: indexRows, cols: d))
                    specs.append(
                        affineSpec(
                            "\(indexPrefix).index_k_proj.weight",
                            rows: indexRows, cols: d))
                    specs.append(
                        bf16Spec(
                            "\(indexPrefix).q_layernorm",
                            shape: [UInt32(toy.sparseIndexer.headDim), 0, 0, 0],
                            count: toy.sparseIndexer.headDim))
                    specs.append(
                        bf16Spec(
                            "\(indexPrefix).k_layernorm",
                            shape: [UInt32(toy.sparseIndexer.headDim), 0, 0, 0],
                            count: toy.sparseIndexer.headDim))
                }
                specs.append(
                    affineSpec(
                        "\(prefix).self_attn.o_proj.weight",
                        rows: d, cols: toy.numHeads * toy.fullHeadDim))
            }
        }

        // 2. Serialize the resident index + payload.
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
        // Align index to 16KB for the SSDAI v1 format validator.
        let alignedIndexBytes =
            ((indexBytes + SSDAIFormatV1.alignmentBytes - 1) & ~(SSDAIFormatV1.alignmentBytes - 1))

        var entries: [ResidentEntry] = []
        entries.reserveCapacity(specs.count)
        var payloadCursor = alignedIndexBytes
        // Align all weight offsets to 4 bytes (UInt32 alignment) for the affine quant kernels.
        let align: UInt64 = UInt64(MemoryLayout<UInt32>.alignment)
        func alignedCursor(_ cursor: UInt64) -> UInt64 {
            ((cursor + align - 1) & ~(align - 1))
        }
        for spec in specs {
            let weightOffset = alignedCursor(payloadCursor)
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
                    sourceWeight: ModelLoaderTests.dummySource(spec.name),
                    sourceScales: nil,
                    sourceBiases: nil))
            // Advance past this tensor's data, aligned to 4 bytes.
            let tensorSize = spec.weightBytes + spec.scaleBytes + spec.biasBytes
            payloadCursor = weightOffset + tensorSize
        }
        let residentSize = alignedCursor(payloadCursor) - alignedIndexBytes

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
            // Quantized tensors: weight bytes 0x11 (nibbles/bytes of small
            // positive codes), scales 0.01, biases zero. BF16 tensors: 1.0.
            for (index, entry) in entries.enumerated() where entry.dtype == 0 {
                let weights = base.advanced(by: Int(entry.fileOffset))
                memset(weights, 0x11, Int(entry.sizeBytes))
                if variedLinearAttention, entry.name.contains(".linear_attn.") {
                    let bytes = weights.assumingMemoryBound(to: UInt8.self)
                    for i in 0..<Int(entry.sizeBytes) {
                        bytes[i] = UInt8(truncatingIfNeeded: (i &* 37) &+ (index &* 101) &+ 7)
                    }
                }
                if entry.scaleSize > 0 {
                    let scales = base.advanced(by: Int(entry.scaleOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    for i in 0..<(Int(entry.scaleSize) / u16) {
                        scales[i] = Quantization.bf16Bits(0.01)
                    }
                }
                if variedHead, entry.name == "language_model.lm_head.weight" {
                    variedHeadFill(entry, base: base)
                }
            }
            for entry in entries where entry.dtype == 1 {
                let dst = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                let varied = variedHead && entry.name == "language_model.model.norm.weight"
                for i in 0..<(Int(entry.sizeBytes) / u16) {
                    dst[i] = Quantization.bf16Bits(varied ? 0.75 + 0.04 * Float(i % 13) : 1.0)
                }
            }
            if bf16Rest { fillBF16Projections(entries, base: base) }
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
                let projectionRows = toyExpertRows(
                    rows: rows, cols: cols,
                    expert: expert, role: role)
                let quantized = projectionRows.map { Quantization.quantizeInt4Affine($0) }
                let packedOffset = bytes.count
                // Use 4-bit for bits <= 4, 8-bit for bits > 4 (runtime only supports 4/8).
                let usedBits = weightBits <= 4 ? 4 : 8
                if usedBits == 4 {
                    for row in quantized { bytes.append(contentsOf: row.packed) }
                } else {
                    bytes += [UInt8](
                        repeating: 0x11,
                        count: rows * cols * usedBits / 8)
                }
                tensors[prefix] = [
                    "offset": packedOffset, "size": bytes.count - packedOffset,
                    "dtype": "U32", "shape": [rows, cols],
                    "bits": usedBits,
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

        let sampleExpertBytes = toyExpertBlob(expert: 0).bytes.count
        let expertStride = UInt64(((sampleExpertBytes + 16_383) / 16_384) * 16_384)
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

        // 5. manifest.json (arch v2 with the qwen36 family fields)
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": Int(totalBytes), "sha256": weightsSha],
            "packed_experts/layout.json": ["size": layoutData.count, "sha256": layoutSha],
        ]
        for (rel, sha) in layerShaByName {
            files[rel] = ["size": layerBytes, "sha256": sha]
        }

        var archDict: [String: Any] = [
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
        if qsa {
            archDict["indexerNumHeads"] = toy.sparseIndexer.numHeads
            archDict["indexerNumKVHeads"] = toy.sparseIndexer.numKVHeads
            archDict["indexerHeadDim"] = toy.sparseIndexer.headDim
            archDict["indexerBudget"] = toy.sparseIndexer.budget
            archDict["indexerCompressRatio"] = toy.sparseIndexer.compressRatio
        }
        var quant: [String: Any] = [
            "embedding": quantSlot(weightBits),
            "attention": quantSlot(weightBits),
            "router": quantSlot(8),
            "sharedExpert": quantSlot(weightBits),
            "routedExpert": quantSlot(weightBits),
        ]
        if bf16Rest { quant["promoted"] = promotedStems(entries) }
        let manifestRoot: [String: Any] = [
            "magic": "SSDAI",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": bf16Rest
                ? "qwen-toy-\(weightBits)bit-bf16rest"
                : qsa ? "qwen-toy-qsa-\(weightBits)bit" : "qwen-toy-\(weightBits)bit",
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

    private static func quantSlot(_ bits: Int) -> [String: Any] {
        // Runtime only supports 4 or 8-bit affine. Use 4 for any bits <= 4,
        // 8 for bits > 4.
        let usedBits = bits <= 4 ? 4 : 8
        return [
            "weightBits": usedBits, "scheme": "affine", "scaleType": "bf16",
            "biasType": "bf16", "groupSize": Quantization.groupSize,
        ]
    }

    // MARK: - Native MTP sidecar

    /// Build the MTP sidecar toy: the companion `.ssdai/` directory for the
    /// Qwen 3.6 native multi-token-prediction draft. Mirrors the REAL
    /// installed sidecar schema (see `Model.validateRuntimeSchema` for
    /// `.qwen36MTP`): one full-attention layer, gate-packed q_proj
    /// (2 * numHeads * fullHeadDim rows), k/v_proj
    /// (numFullKVHeads * fullHeadDim rows), o_proj
    /// (hiddenSize x numHeads * fullHeadDim), per-head q/k norms, the router
    /// (`mlp.gate`, quant.router width) and sigmoid-gated shared expert, the
    /// layer norms, `model.norm.weight`, and the MTP adapter tensors
    /// (`fc.weight` hiddenSize x 2*hiddenSize, `pre_fc_norm_embedding.weight`,
    /// `pre_fc_norm_hidden.weight`). It deliberately carries NO
    /// embedding/lm_head — the runtime shares the target's.
    static func writeMTP(weightBits: Int = 4) throws -> URL {
        precondition([4, 8].contains(weightBits))
        let toy = ArchConfig.qwenToyMTP()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-qwen-mtp-toy-\(UUID().uuidString)")
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

        func affineSpec(
            _ name: String, rows: Int, cols: Int,
            bits: Int = weightBits
        ) -> ResidentSpec {
            // Runtime only supports 4 or 8-bit affine.
            let usedBits = bits <= 4 ? 4 : 8
            let groups = cols / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(
                name: name,
                dtype: 0,
                shape: [UInt32(rows), UInt32(cols), 0, 0],
                weightBytes: UInt64(rows * cols * usedBits / 8),
                scaleBytes: auxBytes,
                biasBytes: auxBytes)
        }

        func bf16Spec(_ name: String, shape: [UInt32], count: Int) -> ResidentSpec {
            ResidentSpec(
                name: name,
                dtype: 1,
                shape: shape,
                weightBytes: UInt64(count * u16),
                scaleBytes: 0,
                biasBytes: 0)
        }

        // 1. Resident specs (MTP adapter + one full-attention layer; no
        // embedding/lm_head — those are shared from the target).
        var specs: [ResidentSpec] = [
            affineSpec("fc.weight", rows: d, cols: 2 * d),
            bf16Spec(
                "pre_fc_norm_embedding.weight",
                shape: [UInt32(d), 0, 0, 0], count: d),
            bf16Spec(
                "pre_fc_norm_hidden.weight",
                shape: [UInt32(d), 0, 0, 0], count: d),
            bf16Spec(
                "language_model.model.norm.weight",
                shape: [UInt32(d), 0, 0, 0], count: d),
        ]
        let prefix = "language_model.model.layers.0"
        specs.append(
            bf16Spec(
                "\(prefix).input_layernorm.weight",
                shape: [UInt32(d), 0, 0, 0], count: d))
        specs.append(
            bf16Spec(
                "\(prefix).post_attention_layernorm.weight",
                shape: [UInt32(d), 0, 0, 0], count: d))
        // Router at the quant.router width (4-bit for the MTP sidecar).
        specs.append(
            affineSpec(
                "\(prefix).mlp.gate.weight",
                rows: toy.numExperts, cols: d, bits: 4))
        specs.append(
            affineSpec(
                "\(prefix).mlp.shared_expert_gate.weight",
                rows: 1, cols: d))
        specs.append(
            affineSpec(
                "\(prefix).mlp.shared_expert.gate_proj.weight",
                rows: toy.intermediateSize, cols: d))
        specs.append(
            affineSpec(
                "\(prefix).mlp.shared_expert.up_proj.weight",
                rows: toy.intermediateSize, cols: d))
        specs.append(
            affineSpec(
                "\(prefix).mlp.shared_expert.down_proj.weight",
                rows: d, cols: toy.intermediateSize))
        // Full-attention layer: gate-packed q_proj, separate k/v, o_proj.
        let queryDim = 2 * toy.numHeads * toy.fullHeadDim
        let kvDim = toy.numFullKVHeads * toy.fullHeadDim
        specs.append(
            bf16Spec(
                "\(prefix).self_attn.q_norm.weight",
                shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                count: toy.fullHeadDim))
        specs.append(
            bf16Spec(
                "\(prefix).self_attn.k_norm.weight",
                shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                count: toy.fullHeadDim))
        specs.append(
            affineSpec(
                "\(prefix).self_attn.q_proj.weight",
                rows: queryDim, cols: d))
        specs.append(
            affineSpec(
                "\(prefix).self_attn.k_proj.weight",
                rows: kvDim, cols: d))
        specs.append(
            affineSpec(
                "\(prefix).self_attn.v_proj.weight",
                rows: kvDim, cols: d))
        specs.append(
            affineSpec(
                "\(prefix).self_attn.o_proj.weight",
                rows: d, cols: toy.numHeads * toy.fullHeadDim))

        // 2. Serialize the resident index + payload (same layout as write()).
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
        let alignedIndexBytes =
            ((indexBytes + SSDAIFormatV1.alignmentBytes - 1) & ~(SSDAIFormatV1.alignmentBytes - 1))

        var entries: [ResidentEntry] = []
        entries.reserveCapacity(specs.count)
        var payloadCursor = alignedIndexBytes
        let align: UInt64 = UInt64(MemoryLayout<UInt32>.alignment)
        func alignedCursor(_ cursor: UInt64) -> UInt64 {
            ((cursor + align - 1) & ~(align - 1))
        }
        for spec in specs {
            let weightOffset = alignedCursor(payloadCursor)
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
                    sourceWeight: ModelLoaderTests.dummySource(spec.name),
                    sourceScales: nil,
                    sourceBiases: nil))
            let tensorSize = spec.weightBytes + spec.scaleBytes + spec.biasBytes
            payloadCursor = weightOffset + tensorSize
        }
        let residentSize = alignedCursor(payloadCursor) - alignedIndexBytes
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

        // 3. Packed experts: one layer, int4 gate/up/down per expert.
        func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }

        func mtpExpertRows(rows: Int, cols: Int, expert: Int, role: Int) -> [[Float]] {
            (0..<rows).map { row in
                (0..<cols).map { col in
                    Float(expert + 1) * 0.001
                        + Float(role + 1) * 0.003
                        + Float((row % 7) - 3) * 0.0004
                        + Float((col % 11) - 5) * 0.0002
                }
            }
        }

        func mtpExpertBlob(expert: Int) -> (bytes: [UInt8], tensors: [String: [String: Any]]) {
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]

            func addProjection(prefix: String, rows: Int, cols: Int, role: Int) {
                let projectionRows = mtpExpertRows(
                    rows: rows, cols: cols,
                    expert: expert, role: role)
                let quantized = projectionRows.map { Quantization.quantizeInt4Affine($0) }
                let packedOffset = bytes.count
                let usedBits = weightBits <= 4 ? 4 : 8
                if usedBits == 4 {
                    for row in quantized { bytes.append(contentsOf: row.packed) }
                } else {
                    bytes += [UInt8](
                        repeating: 0x11,
                        count: rows * cols * usedBits / 8)
                }
                tensors[prefix] = [
                    "offset": packedOffset, "size": bytes.count - packedOffset,
                    "dtype": "U32", "shape": [rows, cols],
                    "bits": usedBits,
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

        let sampleExpertBytes = mtpExpertBlob(expert: 0).bytes.count
        let expertStride = UInt64(((sampleExpertBytes + 16_383) / 16_384) * 16_384)
        let layerBytes = Int(expertStride) * toy.numExperts
        var payload = Data(count: layerBytes)
        for E in 0..<toy.numExperts {
            let blob = mtpExpertBlob(expert: E).bytes
            let baseB = E * Int(expertStride)
            precondition(
                blob.count <= Int(expertStride),
                "MTP toy expert blob exceeds stride")
            for (i, byte) in blob.enumerated() {
                payload[baseB + i] = byte
            }
        }
        let layerURL = exp.appendingPathComponent("layer_00.bin")
        try payload.write(to: layerURL)
        let layerSha = try Sha256Verifier.hashFile(at: layerURL)

        // 4. layout.json
        var expertsArr: [[String: Any]] = []
        for E in 0..<toy.numExperts {
            let blob = mtpExpertBlob(expert: E)
            expertsArr.append([
                "expert": E,
                "offset": UInt64(E) * expertStride,
                "size": expertStride,
                "tensors": blob.tensors,
            ])
        }
        let layoutRoot: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": toy.numLayers,
            "expertsPerLayer": toy.numExperts,
            "layers": [
                [
                    "layer": 0,
                    "file": "layer_00.bin",
                    "experts": expertsArr,
                ]
            ],
        ]
        let layoutData = try JSONSerialization.data(
            withJSONObject: layoutRoot, options: [.sortedKeys])
        let layoutURL = exp.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)

        // 5. manifest.json (qwen36_mtp arch, 4-bit router)
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": Int(totalBytes), "sha256": weightsSha],
            "packed_experts/layout.json": ["size": layoutData.count, "sha256": layoutSha],
            "packed_experts/layer_00.bin": ["size": layerBytes, "sha256": layerSha],
        ]
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
            "magic": "SSDAI",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "qwen-mtp-toy-\(weightBits)bit",
            "arch": archDict,
            "quant": [
                "embedding": quantSlot(weightBits),
                "attention": quantSlot(weightBits),
                "router": quantSlot(4),
                "sharedExpert": quantSlot(weightBits),
                "routedExpert": quantSlot(weightBits),
            ],
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

    /// Deterministic, row-varying lm_head bytes, scales and biases.
    private static func variedHeadFill(_ entry: ResidentEntry, base: UnsafeMutableRawPointer) {
        let bytes = base.advanced(by: Int(entry.fileOffset)).assumingMemoryBound(to: UInt8.self)
        for i in 0..<Int(entry.sizeBytes) {
            bytes[i] = UInt8(truncatingIfNeeded: (i &* 131) ^ (i >> 7) &+ 13)
        }
        let scales = base.advanced(by: Int(entry.scaleOffset)).assumingMemoryBound(to: UInt16.self)
        for i in 0..<(Int(entry.scaleSize) / 2) {
            scales[i] = Quantization.bf16Bits(0.01 + 0.0007 * Float((i % 17) - 8))
        }
        let biases = base.advanced(by: Int(entry.biasOffset)).assumingMemoryBound(to: UInt16.self)
        for i in 0..<(Int(entry.biasSize) / 2) {
            biases[i] = Quantization.bf16Bits(0.02 * Float((i % 11) - 5))
        }
    }

    /// The bf16 matrices of a `bf16Rest` toy: two-dimensional bf16 tensors
    /// (norms are one-dimensional, the conv kernel three), each with its own
    /// deterministic pattern in [-0.05, 0.05] so rows, tokens and projections
    /// all differ.
    static func isBF16Matrix(_ entry: ResidentEntry) -> Bool {
        entry.dtype == 1 && entry.logicalShape4[1] > 0 && entry.logicalShape4[2] == 0
    }

    /// The value `fillBF16Projections` writes at element `i` of entry `index`.
    static func bf16RestValue(_ i: Int, entry index: Int) -> Float {
        0.05 * Float(((i &* 37 &+ index &* 101 &+ 7) % 23) - 11) / 11
    }

    private static func fillBF16Projections(
        _ entries: [ResidentEntry], base: UnsafeMutableRawPointer
    ) {
        for (index, entry) in entries.enumerated() where isBF16Matrix(entry) {
            let dst = base.advanced(by: Int(entry.fileOffset)).assumingMemoryBound(to: UInt16.self)
            for i in 0..<(Int(entry.sizeBytes) / 2) {
                dst[i] = Quantization.bf16Bits(bf16RestValue(i, entry: index))
            }
        }
    }

    /// The manifest's `promoted` stems: every bf16 matrix but the lm_head,
    /// which the real BF16Rest install leaves out of the list too.
    private static func promotedStems(_ entries: [ResidentEntry]) -> [String] {
        entries.filter { isBF16Matrix($0) && !$0.name.hasSuffix("lm_head.weight") }
            .map { String($0.name.dropLast(".weight".count)) }
    }
}

extension ArchConfig {
    /// The MTP sidecar's tiny architecture: a single full-attention layer
    /// (mask [1]) with the same dimensions as the qwen36 target toy so the
    /// sidecar can share the target's embedding/lm_head
    /// (`Model.sharingTargetWeights` requires matching hiddenSize/vocabSize).
    static func qwenToyMTP() -> ArchConfig {
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
            slidingWindow: 1024,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 1,
            numExperts: 8,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [1],
            hiddenActivation: "silu",
            family: .qwen36MTP,
            attnOutputGate: true,
            attentionScale: 0.125,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearAttention: .none)
    }
}

extension ArchConfig {
    /// The qwen toy with QSA enabled: two indexer heads of 32 dims, a 2-cell
    /// budget over 4-cell blocks. Past three visible keys this selects, which
    /// is what lets the streaming acceptance tests cross the window.
    static func qwenToyQSA() -> ArchConfig {
        let base = ArchConfig.qwenToy()
        return ArchConfig(
            hiddenSize: base.hiddenSize,
            intermediateSize: base.intermediateSize,
            moeIntermediateSize: base.moeIntermediateSize,
            numHeads: base.numHeads,
            numKVHeads: base.numKVHeads,
            numFullKVHeads: base.numFullKVHeads,
            headDim: base.headDim,
            fullHeadDim: base.fullHeadDim,
            vocabSize: base.vocabSize,
            slidingWindow: base.slidingWindow,
            finalLogitSoftcap: base.finalLogitSoftcap,
            ropeTheta: base.ropeTheta,
            fullRopeTheta: base.fullRopeTheta,
            partialRotaryFactor: base.partialRotaryFactor,
            numLayers: base.numLayers,
            numExperts: base.numExperts,
            topKExperts: base.topKExperts,
            tieWordEmbeddings: base.tieWordEmbeddings,
            attentionKEqV: base.attentionKEqV,
            fullAttentionLayerMask: base.fullAttentionLayerMask,
            hiddenActivation: base.hiddenActivation,
            family: base.family,
            attnOutputGate: base.attnOutputGate,
            attentionScale: base.attentionScale,
            embeddingScaledBySqrtHidden: base.embeddingScaledBySqrtHidden,
            routerScaled: base.routerScaled,
            ffnSandwichNorms: base.ffnSandwichNorms,
            sharedExpertGated: base.sharedExpertGated,
            ropeNeoxSubdim: base.ropeNeoxSubdim,
            linearAttention: base.linearAttention,
            hyperConnections: base.hyperConnections,
            sparseIndexer: SparseIndexerConfig(
                numHeads: 2, numKVHeads: 1, headDim: 32,
                budget: 2, compressRatio: 4),
            ple: base.ple,
            routerNormTopK: base.routerNormTopK,
            quantGroupSize: base.quantGroupSize)
    }
}
