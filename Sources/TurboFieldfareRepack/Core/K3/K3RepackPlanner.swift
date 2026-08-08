import Foundation
import TurboFieldfareFormat

/// Trunk quantization profile for the K3 repack. Both profiles are accepted
/// by the runtime; embedding remains affine8-g64 and routed experts remain
/// native MXFP4 in either case.
public enum K3TrunkQuant: String, Sendable, CaseIterable {
    case int4
    case int8

    var bits: Int {
        switch self { case .int4: 4; case .int8: 8 }
    }
}

// MARK: - Plan data types

/// What produces the resident bytes of one trunk tensor.
enum K3ResidentTransform: Sendable, Equatable {
    /// Source bytes copy verbatim into the resident file (phase-1 range).
    case verbatim
    /// BF16 source is staged in phase 1 and affine-quantized in phase 2.
    case affine(bits: Int, stagingOffset: UInt64)
    /// BF16 source is staged in phase 1 and widened to FP32 in phase 2
    /// (router gate.weight, q_a/kv_a layernorms).
    case widenFP32(stagingOffset: UInt64)
    /// FP32 source is staged; phase 2 asserts a zero tail and keeps the
    /// `keepCount` prefix (A_log is head-padded in the source checkpoint).
    case truncateFP32(stagingOffset: UInt64, sourceCount: Int, keepCount: Int)
    /// No source bytes: the ftruncate'd resident file already reads as zeros
    /// (conv1d biases exist in the v2 schema but are intentionally unused).
    case synthesizeZeros
}

struct K3ResidentTensorPlan: Sendable {
    /// The v1-wire resident index entry (the v2 resident index reuses the v1
    /// codec), with final file offsets.
    let entry: ResidentEntry
    let transform: K3ResidentTransform
    /// Logical matrix dims; `columns` is the element count for vectors.
    let rows: Int
    let columns: Int
    /// Source coordinates (nil only for `.synthesizeZeros`).
    let source: K3SourceTensor?
}

/// One of the six canonical MXFP4 subtensors inside a packed expert blob
/// (v2 schema order: w1p, w1s, w2p, w2s, w3p, w3s).
struct K3ExpertSubtensorPlan: Sendable {
    let name: String             // "w1_packed" | "w1_scales" | ... | "w3_scales"
    let stem: String             // "w1" | "w2" | "w3"
    let isScales: Bool
    let offsetInBlob: UInt64
    let sizeInBlob: UInt64
    /// Logical shape recorded in layout.json: [rows, cols] for packed
    /// (cols = unpacked width), [rows, cols/32 groups] for scales.
    let logicalShape: [UInt32]
    let bits: Int                // 4 for packed, 8 for scales
}

struct K3LayerFilePlan: Sendable {
    let layerIndex: Int          // 0-based; MoE layers only
    let path: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    /// Six subtensors, canonical order.
    let subtensors: [K3ExpertSubtensorPlan]
    /// `sources[subtensorIndex][expert]` — per-expert source coordinates.
    let sources: [[K3SourceTensor]]
    var fileSize: UInt64 { UInt64(expertsPerLayer) * expertStride }
}

struct K3RepackPlan: Sendable {
    let arch: K3ArchInfo
    let trunkQuant: K3TrunkQuant
    let manifestArch: GTurboManifestArchV2
    let resident: ResidentFilePlan
    /// Same order as `resident.entries`.
    let tensors: [K3ResidentTensorPlan]
    let stagingPath: String
    let stagingSize: UInt64
    let layers: [K3LayerFilePlan]
    let excludedTensorNames: [String]

    /// Final bundle payload bytes (resident + packed experts), excluding the
    /// transient staging file.
    var outputBytes: UInt64 {
        resident.totalSize + layers.reduce(UInt64(0)) { $0 + $1.fileSize }
    }
}

// MARK: - Planner

enum K3RepackPlanner {

    /// Quant slot of an affine trunk tensor (which manifest slot its bits
    /// come from). Embedding pins int8; the other slots follow --trunk-quant.
    enum QuantSlot {
        case embedding, attention, sharedExpert, latentProjection, denseMLP

        func bits(trunkQuant: K3TrunkQuant) -> Int {
            switch self {
            case .embedding: 8
            case .attention, .sharedExpert, .latentProjection, .denseMLP:
                trunkQuant.bits
            }
        }
    }

    enum ResidentKind {
        case affine(slot: QuantSlot, rows: Int, columns: Int)
        case bf16Vector(Int)                        // verbatim; source [n] or [1, n]
        case fp32Vector(Int)                        // verbatim F32
        case fp32Matrix(rows: Int, columns: Int)    // verbatim F32; source [r, c] or [r, 1, c]
        case fp32FromBF16Vector(Int)                // staged widen
        case fp32FromBF16Matrix(rows: Int, columns: Int)  // staged widen
        case fp32ALog(heads: Int)                   // staged truncate
        case fp32Zeros(Int)                         // synthesized (conv1d.bias)
    }

    /// Dims probed from tensor shapes (cross-checked against the config where
    /// the config has a say) before the schema table is generated.
    struct ProbedDims {
        let kdaDecayLowRankSize: Int
        let latentBottleneckSize: Int
        let expertIntermediateSize: Int
        let sharedExpertIntermediateSize: Int
    }

    static func plan(arch: K3ArchInfo,
                     shardHeaders: [K3Safetensors.Header],
                     outputDir: String,
                     trunkQuant: K3TrunkQuant) throws -> K3RepackPlan {
        var registry: [String: K3SourceTensor] = [:]
        registry.reserveCapacity(shardHeaders.reduce(0) { $0 + $1.tensors.count })
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        let probed = try probeDims(arch: arch, registry: registry)
        let schema = schemaEntries(arch: arch, probed: probed)
        let schemaByName = Dictionary(schema.map { ($0.name, $0) },
                                      uniquingKeysWith: { _, _ in fatalError("duplicate schema name") })

        // MARK: Classification walk — every source tensor lands in exactly
        // one bucket or the plan fails.
        struct ExpertKey: Hashable { let layer: Int; let stem: String; let isScales: Bool }
        var trunkSources: [String: K3SourceTensor] = [:]
        var expertSources: [ExpertKey: [K3SourceTensor?]] = [:]
        var excluded: [String] = []
        for (name, tensor) in registry {
            if isExcludedNonLM(name) || isExcludedMTP(name) {
                excluded.append(name)
                continue
            }
            guard name.hasPrefix("language_model.") else {
                throw RepackError.unknownTensorPrefix(name: name)
            }
            if schemaByName[name] != nil {
                trunkSources[name] = tensor
                continue
            }
            if let expert = parseExpertTensorName(name) {
                guard arch.isMoE(layer0: expert.layer) else {
                    throw RepackError.configurationInvalid(detail:
                        "dense layer \(expert.layer) must not carry routed experts: \(name)")
                }
                let key = ExpertKey(layer: expert.layer, stem: expert.stem,
                                    isScales: expert.isScales)
                var slots = expertSources[key] ?? [K3SourceTensor?](repeating: nil,
                                                                     count: arch.moeNumExperts)
                guard slots[expert.expert] == nil else {
                    throw RepackError.configurationInvalid(detail: "duplicate tensor \(name)")
                }
                try validateExpertTensor(tensor, layer: expert.layer, stem: expert.stem,
                                         isScales: expert.isScales, arch: arch, probed: probed)
                slots[expert.expert] = tensor
                expertSources[key] = slots
                continue
            }
            throw RepackError.unknownTensorPrefix(name: name)
        }
        excluded.sort()

        // Every schema tensor must be present (`.fp32Zeros` entries have no
        // source by construction).
        for entry in schema {
            if case .fp32Zeros = entry.kind { continue }
            guard trunkSources[entry.name] != nil else {
                throw RepackError.missingTensor(name: entry.name)
            }
        }
        // Every MoE layer needs the full 6 x numExperts expert inventory.
        for layer in arch.moeLayers0.sorted() {
            for stem in GTurboFormatV2.mxfp4ExpertMatrixStems {
                for isScales in [false, true] {
                    let key = ExpertKey(layer: layer, stem: stem, isScales: isScales)
                    guard let slots = expertSources[key],
                          slots.allSatisfy({ $0 != nil }) else {
                        throw RepackError.missingTensor(name:
                            "language_model.model.layers.\(layer).block_sparse_moe.experts.*."
                                + "\(stem).weight_\(isScales ? "scale" : "packed")")
                    }
                }
            }
        }

        // MARK: Resident file (payloads 16-byte aligned; gaps read as zero).
        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let stagingPath = (outputDir as NSString).appendingPathComponent("trunk-staging.bin")
        var stringTable: [UInt8] = []
        var stringOffsets: [UInt32] = []
        stringTable.reserveCapacity(schema.reduce(0) { $0 + $1.name.utf8.count })
        for entry in schema {
            stringOffsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: entry.name.utf8)
        }
        let rawIndex = UInt64(GTurboBinary.indexHeaderBytes
            + schema.count * GTurboBinary.indexEntryBytes + stringTable.count)
        let indexSize = roundUp(rawIndex, GTurboFormatV2.alignmentBytes)

        var fileCursor = indexSize
        var stagingCursor: UInt64 = 0
        var entries: [ResidentEntry] = []
        var tensors: [K3ResidentTensorPlan] = []
        entries.reserveCapacity(schema.count)
        tensors.reserveCapacity(schema.count)
        for item in schema {
            let source = trunkSources[item.name]
            let planned = try planResidentTensor(item, source: source, arch: arch,
                                                 trunkQuant: trunkQuant,
                                                 fileCursor: &fileCursor,
                                                 stagingCursor: &stagingCursor)
            entries.append(planned.entry)
            tensors.append(planned)
        }
        let resident = ResidentFilePlan(path: residentPath,
                                        entries: entries,
                                        stringTable: stringTable,
                                        stringTableOffsets: stringOffsets,
                                        indexSize: indexSize,
                                        residentSize: fileCursor - indexSize)

        // MARK: Packed expert layer files (MoE layers only).
        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        let latent = UInt64(probed.latentBottleneckSize)
        let inter = UInt64(probed.expertIntermediateSize)
        let subtensors = canonicalSubtensors(latent: latent, inter: inter)
        let blobSize = subtensors.reduce(UInt64(0)) { $0 + $1.sizeInBlob }
        let expertStride = roundUp(blobSize, GTurboFormatV2.alignmentBytes)
        // The v2 format contract: the expert stride is a whole number of
        // 16 KB pages (17,547,264 bytes = 1,071 pages for canonical dims).
        guard expertStride % GTurboFormatV2.alignmentBytes == 0, blobSize > 0,
              blobSize <= expertStride else {
            throw RepackError.configurationInvalid(detail:
                "expert blob \(blobSize) does not fit a 16 KB-aligned stride")
        }
        var layerPlans: [K3LayerFilePlan] = []
        for layer in arch.moeLayers0.sorted() {
            var sources: [[K3SourceTensor]] = []
            sources.reserveCapacity(subtensors.count)
            for sub in subtensors {
                let key = ExpertKey(layer: layer, stem: sub.stem, isScales: sub.isScales)
                // Completeness proven above; force-unwrap is a planner
                // invariant, not user input.
                sources.append(expertSources[key]!.map { $0! })
            }
            layerPlans.append(K3LayerFilePlan(
                layerIndex: layer,
                path: (layersDir as NSString)
                    .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin"),
                expertsPerLayer: arch.moeNumExperts,
                expertStride: expertStride,
                subtensors: subtensors,
                sources: sources))
        }

        let manifestArch = GTurboManifestArchV2(
            hiddenSize: arch.hiddenSize, vocabSize: arch.vocabSize,
            numLayers: arch.numLayers,
            denseMLPIntermediateSize: arch.denseMLPIntermediateSize,
            rmsNormEpsilon: arch.rmsNormEpsilon,
            tieWordEmbeddings: arch.tieWordEmbeddings,
            hiddenActivation: arch.hiddenActivation,
            bosTokenID: arch.bosTokenID, eosTokenID: arch.eosTokenID,
            denseLayers: arch.denseLayers, kdaLayers: arch.kdaLayers,
            fullAttnLayers: arch.fullAttnLayers,
            kda: GTurboManifestKDAV2(
                numHeads: arch.kdaNumHeads, headDim: arch.kdaHeadDim,
                convWidth: arch.kdaConvWidth,
                decayLowRankSize: probed.kdaDecayLowRankSize,
                decayProjectionSize: arch.kdaDecayProjectionSize,
                gateLowerBound: arch.kdaGateLowerBound,
                fullRankOutputGate: arch.kdaFullRankOutputGate),
            mla: GTurboManifestMLAV2(
                numHeads: arch.mlaNumHeads, qLoraRank: arch.mlaQLoraRank,
                kvLoraRank: arch.mlaKVLoraRank,
                qkNopeHeadDim: arch.mlaQKNopeHeadDim,
                qkRopeHeadDim: arch.mlaQKRopeHeadDim,
                vHeadDim: arch.mlaVHeadDim,
                outputGate: arch.mlaOutputGate),
            attnRes: GTurboManifestAttnResV2(blockSize: arch.attnResBlockSize),
            moe: GTurboManifestMoEV2(
                numExperts: arch.moeNumExperts, topKExperts: arch.moeTopKExperts,
                latentBottleneckSize: probed.latentBottleneckSize,
                expertIntermediateSize: probed.expertIntermediateSize,
                numSharedExperts: arch.moeNumSharedExperts,
                sharedExpertIntermediateSize: probed.sharedExpertIntermediateSize,
                situGLUGateBeta: arch.situGLUGateBeta,
                situGLUUpBeta: arch.situGLUUpBeta,
                routerRenormalize: arch.routerRenormalize,
                routerCorrectionBias: arch.routerCorrectionBias))

        return K3RepackPlan(arch: arch, trunkQuant: trunkQuant,
                            manifestArch: manifestArch,
                            resident: resident, tensors: tensors,
                            stagingPath: stagingPath, stagingSize: stagingCursor,
                            layers: layerPlans,
                            excludedTensorNames: excluded)
    }

    // MARK: - Exclusion + expert-name classification

    /// Vision tower / projector tensors never enter the text bundle.
    static func isExcludedNonLM(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") || name.hasPrefix("mm_projector.")
    }

    /// MTP / nextn tensors are excluded when a checkpoint carries them (the
    /// pinned K3 revision has `num_nextn_predict_layers = 0`, i.e. none).
    static func isExcludedMTP(_ name: String) -> Bool {
        let lowered = name.lowercased()
        return lowered.contains("nextn") || lowered.contains(".mtp.")
            || lowered.hasPrefix("mtp.")
    }

    /// Strictly parse
    /// `language_model.model.layers.<N>.block_sparse_moe.experts.<E>.<w1|w2|w3>.weight_<packed|scale>`.
    static func parseExpertTensorName(_ name: String)
        -> (layer: Int, expert: Int, stem: String, isScales: Bool)? {
        let parts = name.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 9,
              parts[0] == "language_model", parts[1] == "model", parts[2] == "layers",
              let layer = Int(parts[3]), layer >= 0,
              parts[4] == "block_sparse_moe", parts[5] == "experts",
              let expert = Int(parts[6]), expert >= 0,
              GTurboFormatV2.mxfp4ExpertMatrixStems.contains(String(parts[7])) else {
            return nil
        }
        let isScales: Bool
        switch parts[8] {
        case "weight_packed": isScales = false
        case "weight_scale":  isScales = true
        default: return nil
        }
        return (layer, expert, String(parts[7]), isScales)
    }

    // MARK: - Dimension probing

    /// Derive the dims the config cannot express (or expresses only
    /// indirectly) from tensor shapes, cross-checking against the config
    /// wherever an official invariant exists. Any disagreement throws.
    private static func probeDims(arch: K3ArchInfo,
                                  registry: [String: K3SourceTensor]) throws -> ProbedDims {
        func tensor(_ name: String) throws -> K3SourceTensor {
            guard let t = registry[name] else { throw RepackError.missingTensor(name: name) }
            return t
        }
        guard let firstKDA = arch.kdaLayers0.min() else {
            throw RepackError.configurationInvalid(detail: "no KDA layers in the schedule")
        }
        guard let firstMoE = arch.moeLayers0.min() else {
            throw RepackError.configurationInvalid(detail: "no MoE layers in the schedule")
        }

        let embed = try tensor("language_model.model.embed_tokens.weight")
        guard embed.dtype == .bf16,
              embed.shape == [UInt64(arch.vocabSize), UInt64(arch.hiddenSize)] else {
            throw RepackError.shapeMismatch(name: embed.name,
                detail: "expected BF16 [\(arch.vocabSize), \(arch.hiddenSize)], "
                    + "got \(embed.dtype) \(embed.shape)")
        }

        let kdaPrefix = "language_model.model.layers.\(firstKDA).self_attn"
        let fA = try tensor("\(kdaPrefix).f_a_proj.weight")
        guard fA.dtype == .bf16, fA.shape.count == 2,
              fA.shape[1] == UInt64(arch.hiddenSize) else {
            throw RepackError.shapeMismatch(name: fA.name,
                detail: "expected BF16 [r, \(arch.hiddenSize)], got \(fA.dtype) \(fA.shape)")
        }
        let decayLowRank = Int(fA.shape[0])
        let fB = try tensor("\(kdaPrefix).f_b_proj.weight")
        guard fB.dtype == .bf16,
              fB.shape == [UInt64(arch.kdaDecayProjectionSize), UInt64(decayLowRank)] else {
            throw RepackError.shapeMismatch(name: fB.name,
                detail: "expected BF16 [\(arch.kdaDecayProjectionSize), \(decayLowRank)], "
                    + "got \(fB.dtype) \(fB.shape)")
        }

        let moePrefix = "language_model.model.layers.\(firstMoE).block_sparse_moe"
        let routedDown = try tensor("\(moePrefix).routed_expert_down_proj.weight")
        guard routedDown.dtype == .bf16, routedDown.shape.count == 2,
              routedDown.shape[1] == UInt64(arch.hiddenSize),
              routedDown.shape[0] == UInt64(arch.moeLatentBottleneckSize) else {
            throw RepackError.shapeMismatch(name: routedDown.name,
                detail: "expected BF16 [\(arch.moeLatentBottleneckSize), \(arch.hiddenSize)], "
                    + "got \(routedDown.dtype) \(routedDown.shape)")
        }
        let latent = Int(routedDown.shape[0])

        let w1Packed = try tensor("\(moePrefix).experts.0.w1.weight_packed")
        guard w1Packed.dtype == .u8, w1Packed.shape.count == 2,
              w1Packed.shape[0] == UInt64(arch.moeExpertIntermediateSize),
              w1Packed.shape[1] * 2 == UInt64(latent) else {
            throw RepackError.shapeMismatch(name: w1Packed.name,
                detail: "expected U8 [\(arch.moeExpertIntermediateSize), \(latent / 2)], "
                    + "got \(w1Packed.dtype) \(w1Packed.shape)")
        }
        let inter = Int(w1Packed.shape[0])
        guard latent % 32 == 0, inter % 32 == 0 else {
            throw RepackError.shapeMismatch(name: w1Packed.name,
                detail: "MXFP4 dims must be multiples of 32, got latent \(latent) inter \(inter)")
        }

        let sharedGate = try tensor("\(moePrefix).shared_experts.gate_proj.weight")
        guard sharedGate.dtype == .bf16, sharedGate.shape.count == 2,
              sharedGate.shape[1] == UInt64(arch.hiddenSize),
              sharedGate.shape[0] == UInt64(arch.moeSharedExpertIntermediateSize) else {
            throw RepackError.shapeMismatch(name: sharedGate.name,
                detail: "expected BF16 [\(arch.moeSharedExpertIntermediateSize), "
                    + "\(arch.hiddenSize)], got \(sharedGate.dtype) \(sharedGate.shape)")
        }

        return ProbedDims(kdaDecayLowRankSize: decayLowRank,
                          latentBottleneckSize: latent,
                          expertIntermediateSize: inter,
                          sharedExpertIntermediateSize: Int(sharedGate.shape[0]))
    }

    // MARK: - Schema table

    private struct SchemaItem {
        let name: String
        let kind: ResidentKind
    }

    /// The full K3 resident schema (docs/K3_DATAFLOW.md §"Checkpoint tensor
    /// names"), mirroring the runtime's `K3Model.schemaEntries` ordering.
    private static func schemaEntries(arch c: K3ArchInfo, probed p: ProbedDims) -> [SchemaItem] {
        var entries: [SchemaItem] = []
        let hidden = c.hiddenSize
        entries.append(SchemaItem(
            name: "language_model.model.embed_tokens.weight",
            kind: .affine(slot: .embedding, rows: c.vocabSize, columns: hidden)))
        entries.append(SchemaItem(
            name: "language_model.lm_head.weight",
            kind: .affine(slot: .embedding, rows: c.vocabSize, columns: hidden)))
        entries.append(SchemaItem(name: "language_model.model.norm.weight",
                                  kind: .bf16Vector(hidden)))
        entries.append(SchemaItem(name: "language_model.model.output_attn_res_proj.weight",
                                  kind: .bf16Vector(hidden)))
        entries.append(SchemaItem(name: "language_model.model.output_attn_res_norm.weight",
                                  kind: .bf16Vector(hidden)))

        let kdaP = c.kdaProjectionSize
        for layer in 0..<c.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            for suffix in ["input_layernorm.weight",
                           "post_attention_layernorm.weight",
                           "self_attention_res_proj.weight",
                           "self_attention_res_norm.weight",
                           "mlp_res_proj.weight",
                           "mlp_res_norm.weight"] {
                entries.append(SchemaItem(name: "\(prefix).\(suffix)",
                                          kind: .bf16Vector(hidden)))
            }
            if c.isKDA(layer0: layer) {
                let attn = "\(prefix).self_attn"
                for stem in ["q_proj.weight", "k_proj.weight", "v_proj.weight",
                             "g_proj.weight"] {
                    entries.append(SchemaItem(
                        name: "\(attn).\(stem)",
                        kind: .affine(slot: .attention, rows: kdaP, columns: hidden)))
                }
                entries.append(SchemaItem(
                    name: "\(attn).o_proj.weight",
                    kind: .affine(slot: .attention, rows: hidden, columns: kdaP)))
                entries.append(SchemaItem(
                    name: "\(attn).f_a_proj.weight",
                    kind: .affine(slot: .attention, rows: p.kdaDecayLowRankSize,
                                  columns: hidden)))
                entries.append(SchemaItem(
                    name: "\(attn).f_b_proj.weight",
                    kind: .affine(slot: .attention, rows: kdaP,
                                  columns: p.kdaDecayLowRankSize)))
                entries.append(SchemaItem(
                    name: "\(attn).b_proj.weight",
                    kind: .affine(slot: .attention, rows: c.kdaNumHeads, columns: hidden)))
                for qkv in ["q", "k", "v"] {
                    entries.append(SchemaItem(
                        name: "\(attn).\(qkv)_conv1d.weight",
                        kind: .fp32Matrix(rows: kdaP, columns: c.kdaConvWidth)))
                    entries.append(SchemaItem(
                        name: "\(attn).\(qkv)_conv1d.bias",
                        kind: .fp32Zeros(kdaP)))
                }
                entries.append(SchemaItem(name: "\(attn).A_log",
                                          kind: .fp32ALog(heads: c.kdaNumHeads)))
                entries.append(SchemaItem(name: "\(attn).dt_bias",
                                          kind: .fp32Vector(kdaP)))
                entries.append(SchemaItem(name: "\(attn).o_norm.weight",
                                          kind: .fp32Vector(c.kdaHeadDim)))
            } else {
                let attn = "\(prefix).self_attn"
                let qElements = c.mlaNumHeads * (c.mlaQKNopeHeadDim + c.mlaQKRopeHeadDim)
                let cacheRow = c.mlaKVLoraRank + c.mlaQKRopeHeadDim
                let kvbRows = c.mlaNumHeads * (c.mlaQKNopeHeadDim + c.mlaVHeadDim)
                let outElements = c.mlaNumHeads * c.mlaVHeadDim
                entries.append(SchemaItem(
                    name: "\(attn).q_a_proj.weight",
                    kind: .affine(slot: .attention, rows: c.mlaQLoraRank, columns: hidden)))
                entries.append(SchemaItem(
                    name: "\(attn).q_a_layernorm.weight",
                    kind: .fp32FromBF16Vector(c.mlaQLoraRank)))
                entries.append(SchemaItem(
                    name: "\(attn).q_b_proj.weight",
                    kind: .affine(slot: .attention, rows: qElements, columns: c.mlaQLoraRank)))
                entries.append(SchemaItem(
                    name: "\(attn).kv_a_proj_with_mqa.weight",
                    kind: .affine(slot: .attention, rows: cacheRow, columns: hidden)))
                entries.append(SchemaItem(
                    name: "\(attn).kv_a_layernorm.weight",
                    kind: .fp32FromBF16Vector(c.mlaKVLoraRank)))
                entries.append(SchemaItem(
                    name: "\(attn).kv_b_proj.weight",
                    kind: .affine(slot: .attention, rows: kvbRows, columns: c.mlaKVLoraRank)))
                entries.append(SchemaItem(
                    name: "\(attn).g_proj.weight",
                    kind: .affine(slot: .attention, rows: outElements, columns: hidden)))
                entries.append(SchemaItem(
                    name: "\(attn).o_proj.weight",
                    kind: .affine(slot: .attention, rows: hidden, columns: outElements)))
            }
            if c.isMoE(layer0: layer) {
                let moe = "\(prefix).block_sparse_moe"
                entries.append(SchemaItem(
                    name: "\(moe).gate.weight",
                    kind: .fp32FromBF16Matrix(rows: c.moeNumExperts, columns: hidden)))
                entries.append(SchemaItem(
                    name: "\(moe).gate.e_score_correction_bias",
                    kind: .fp32Vector(c.moeNumExperts)))
                entries.append(SchemaItem(
                    name: "\(moe).routed_expert_down_proj.weight",
                    kind: .affine(slot: .latentProjection,
                                  rows: p.latentBottleneckSize, columns: hidden)))
                entries.append(SchemaItem(
                    name: "\(moe).routed_expert_up_proj.weight",
                    kind: .affine(slot: .latentProjection,
                                  rows: hidden, columns: p.latentBottleneckSize)))
                entries.append(SchemaItem(
                    name: "\(moe).routed_expert_norm.weight",
                    kind: .bf16Vector(p.latentBottleneckSize)))
                entries.append(SchemaItem(
                    name: "\(moe).shared_experts.gate_proj.weight",
                    kind: .affine(slot: .sharedExpert,
                                  rows: p.sharedExpertIntermediateSize, columns: hidden)))
                entries.append(SchemaItem(
                    name: "\(moe).shared_experts.up_proj.weight",
                    kind: .affine(slot: .sharedExpert,
                                  rows: p.sharedExpertIntermediateSize, columns: hidden)))
                entries.append(SchemaItem(
                    name: "\(moe).shared_experts.down_proj.weight",
                    kind: .affine(slot: .sharedExpert,
                                  rows: hidden, columns: p.sharedExpertIntermediateSize)))
            } else {
                let mlp = "\(prefix).mlp"
                entries.append(SchemaItem(
                    name: "\(mlp).gate_proj.weight",
                    kind: .affine(slot: .denseMLP,
                                  rows: c.denseMLPIntermediateSize, columns: hidden)))
                entries.append(SchemaItem(
                    name: "\(mlp).up_proj.weight",
                    kind: .affine(slot: .denseMLP,
                                  rows: c.denseMLPIntermediateSize, columns: hidden)))
                entries.append(SchemaItem(
                    name: "\(mlp).down_proj.weight",
                    kind: .affine(slot: .denseMLP,
                                  rows: hidden, columns: c.denseMLPIntermediateSize)))
            }
        }
        return entries
    }

    // MARK: - Resident tensor planning

    private static func planResidentTensor(_ item: SchemaItem,
                                           source: K3SourceTensor?,
                                           arch: K3ArchInfo,
                                           trunkQuant: K3TrunkQuant,
                                           fileCursor: inout UInt64,
                                           stagingCursor: inout UInt64) throws
        -> K3ResidentTensorPlan {
        func take(_ size: UInt64, from cursor: inout UInt64) -> UInt64 {
            cursor = roundUp(cursor, 16)
            let offset = cursor
            cursor += size
            return offset
        }
        func makeEntry(name: String, dtype: UInt8, shape4: [UInt32],
                       weightOffset: UInt64, weightSize: UInt64,
                       scaleOffset: UInt64 = 0, scaleSize: UInt64 = 0,
                       biasOffset: UInt64 = 0, biasSize: UInt64 = 0,
                       quantSpec: QuantSpec? = nil) -> ResidentEntry {
            ResidentEntry(
                name: name, dtype: dtype, logicalShape4: shape4,
                fileOffset: weightOffset, sizeBytes: weightSize,
                scaleOffset: scaleOffset, scaleSize: scaleSize,
                biasOffset: biasOffset, biasSize: biasSize,
                quantSpec: quantSpec,
                // The range planner reads sources from the K3 plan, never
                // from this field; a placeholder keeps the shared v1 entry
                // shape without pretending zeros have an origin.
                sourceWeight: source.map {
                    SourceTensor(name: $0.name, shardPath: $0.shardPath,
                                 dtype: $0.dtype == .bf16 ? .bf16 : .fp32,
                                 shape: $0.shape, absoluteOffset: $0.absoluteOffset,
                                 sizeBytes: $0.sizeBytes)
                } ?? SourceTensor(name: name, shardPath: "", dtype: .fp32,
                                  shape: shape4.map(UInt64.init), absoluteOffset: 0,
                                  sizeBytes: 0),
                sourceScales: nil, sourceBiases: nil)
        }
        func requireSource(_ kind: K3SourceTensor.Dtype, _ name: String) throws
            -> K3SourceTensor {
            guard let source else { throw RepackError.missingTensor(name: name) }
            guard source.dtype == kind else {
                throw RepackError.dtypeMismatch(name: name,
                    detail: "expected \(kind), got \(source.dtype)")
            }
            return source
        }
        func requireVectorShape(_ source: K3SourceTensor, _ count: Int,
                                _ name: String) throws {
            let one = [UInt64(count)]
            let two = [1, UInt64(count)]
            guard source.shape == one || source.shape == two else {
                throw RepackError.shapeMismatch(name: name,
                    detail: "expected [\(count)] or [1, \(count)], got \(source.shape)")
            }
        }

        switch item.kind {
        case .affine(let slot, let rows, let columns):
            let source = try requireSource(.bf16, item.name)
            guard columns % K3TrunkQuantizer.groupSize == 0 else {
                throw RepackError.shapeMismatch(name: item.name,
                    detail: "columns \(columns) not a multiple of the affine group size")
            }
            guard source.shape == [UInt64(rows), UInt64(columns)] else {
                throw RepackError.shapeMismatch(name: item.name,
                    detail: "expected BF16 [\(rows), \(columns)], got \(source.shape)")
            }
            let bits = slot.bits(trunkQuant: trunkQuant)
            let weightSize = UInt64(rows) * UInt64(columns) * UInt64(bits) / 8
            let auxSize = UInt64(rows) * UInt64(columns / K3TrunkQuantizer.groupSize) * 2
            let wOff = take(weightSize, from: &fileCursor)
            let sOff = take(auxSize, from: &fileCursor)
            let bOff = take(auxSize, from: &fileCursor)
            let stagingOffset = take(source.sizeBytes, from: &stagingCursor)
            let entry = makeEntry(name: item.name,
                                  dtype: GTurboFormatV1.DType.u32.rawValue,
                                  shape4: [UInt32(rows), UInt32(columns), 0, 0],
                                  weightOffset: wOff, weightSize: weightSize,
                                  scaleOffset: sOff, scaleSize: auxSize,
                                  biasOffset: bOff, biasSize: auxSize,
                                  quantSpec: QuantSpec(bits: bits))
            return K3ResidentTensorPlan(
                entry: entry,
                transform: .affine(bits: bits, stagingOffset: stagingOffset),
                rows: rows, columns: columns, source: source)

        case .bf16Vector(let count):
            let source = try requireSource(.bf16, item.name)
            try requireVectorShape(source, count, item.name)
            let size = UInt64(count) * 2
            let off = take(size, from: &fileCursor)
            return K3ResidentTensorPlan(
                entry: makeEntry(name: item.name,
                                 dtype: GTurboFormatV1.DType.bf16.rawValue,
                                 shape4: [UInt32(count), 0, 0, 0],
                                 weightOffset: off, weightSize: size),
                transform: .verbatim, rows: 1, columns: count, source: source)

        case .fp32Vector(let count):
            let source = try requireSource(.fp32, item.name)
            try requireVectorShape(source, count, item.name)
            let size = UInt64(count) * 4
            let off = take(size, from: &fileCursor)
            return K3ResidentTensorPlan(
                entry: makeEntry(name: item.name,
                                 dtype: GTurboFormatV1.DType.fp32.rawValue,
                                 shape4: [UInt32(count), 0, 0, 0],
                                 weightOffset: off, weightSize: size),
                transform: .verbatim, rows: 1, columns: count, source: source)

        case .fp32Matrix(let rows, let columns):
            let source = try requireSource(.fp32, item.name)
            let flat = [UInt64(rows), UInt64(columns)]
            let ranked = [UInt64(rows), 1, UInt64(columns)]
            guard source.shape == flat || source.shape == ranked else {
                throw RepackError.shapeMismatch(name: item.name,
                    detail: "expected F32 [\(rows), \(columns)] or [\(rows), 1, \(columns)], "
                        + "got \(source.shape)")
            }
            let size = UInt64(rows) * UInt64(columns) * 4
            let off = take(size, from: &fileCursor)
            return K3ResidentTensorPlan(
                entry: makeEntry(name: item.name,
                                 dtype: GTurboFormatV1.DType.fp32.rawValue,
                                 shape4: [UInt32(rows), UInt32(columns), 0, 0],
                                 weightOffset: off, weightSize: size),
                transform: .verbatim, rows: rows, columns: columns, source: source)

        case .fp32FromBF16Vector(let count):
            let source = try requireSource(.bf16, item.name)
            try requireVectorShape(source, count, item.name)
            let size = UInt64(count) * 4
            let off = take(size, from: &fileCursor)
            let stagingOffset = take(source.sizeBytes, from: &stagingCursor)
            return K3ResidentTensorPlan(
                entry: makeEntry(name: item.name,
                                 dtype: GTurboFormatV1.DType.fp32.rawValue,
                                 shape4: [UInt32(count), 0, 0, 0],
                                 weightOffset: off, weightSize: size),
                transform: .widenFP32(stagingOffset: stagingOffset),
                rows: 1, columns: count, source: source)

        case .fp32FromBF16Matrix(let rows, let columns):
            let source = try requireSource(.bf16, item.name)
            guard source.shape == [UInt64(rows), UInt64(columns)] else {
                throw RepackError.shapeMismatch(name: item.name,
                    detail: "expected BF16 [\(rows), \(columns)], got \(source.shape)")
            }
            let size = UInt64(rows) * UInt64(columns) * 4
            let off = take(size, from: &fileCursor)
            let stagingOffset = take(source.sizeBytes, from: &stagingCursor)
            return K3ResidentTensorPlan(
                entry: makeEntry(name: item.name,
                                 dtype: GTurboFormatV1.DType.fp32.rawValue,
                                 shape4: [UInt32(rows), UInt32(columns), 0, 0],
                                 weightOffset: off, weightSize: size),
                transform: .widenFP32(stagingOffset: stagingOffset),
                rows: rows, columns: columns, source: source)

        case .fp32ALog(let heads):
            let source = try requireSource(.fp32, item.name)
            // The source checkpoint stores A_log head-padded (canonical:
            // [headDim] = [128] with 96 trained values and a zero tail). A
            // bare [heads] tensor is accepted too; anything else refuses.
            let count = source.elementCount
            guard source.shape.count == 1,
                  count == UInt64(heads) || count == UInt64(arch.kdaHeadDim) else {
                throw RepackError.shapeMismatch(name: item.name,
                    detail: "expected F32 [\(heads)] or head-padded [\(arch.kdaHeadDim)], "
                        + "got \(source.shape)")
            }
            let size = UInt64(heads) * 4
            let off = take(size, from: &fileCursor)
            let stagingOffset = take(source.sizeBytes, from: &stagingCursor)
            return K3ResidentTensorPlan(
                entry: makeEntry(name: item.name,
                                 dtype: GTurboFormatV1.DType.fp32.rawValue,
                                 shape4: [UInt32(heads), 0, 0, 0],
                                 weightOffset: off, weightSize: size),
                transform: .truncateFP32(stagingOffset: stagingOffset,
                                         sourceCount: Int(count), keepCount: heads),
                rows: 1, columns: heads, source: source)

        case .fp32Zeros(let count):
            let size = UInt64(count) * 4
            let off = take(size, from: &fileCursor)
            return K3ResidentTensorPlan(
                entry: makeEntry(name: item.name,
                                 dtype: GTurboFormatV1.DType.fp32.rawValue,
                                 shape4: [UInt32(count), 0, 0, 0],
                                 weightOffset: off, weightSize: size),
                transform: .synthesizeZeros, rows: 1, columns: count, source: nil)
        }
    }

    // MARK: - Expert blob layout

    /// The six canonical subtensors with cumulative blob offsets, in v2
    /// schema order (w1p, w1s, w2p, w2s, w3p, w3s).
    static func canonicalSubtensors(latent: UInt64, inter: UInt64)
        -> [K3ExpertSubtensorPlan] {
        precondition(latent > 0 && latent % 32 == 0 && inter > 0 && inter % 32 == 0)
        var out: [K3ExpertSubtensorPlan] = []
        var cursor: UInt64 = 0
        for stem in GTurboFormatV2.mxfp4ExpertMatrixStems {
            // w1/w3 map latent -> intermediate; w2 maps intermediate -> latent.
            let rows = stem == "w2" ? latent : inter
            let columns = stem == "w2" ? inter : latent
            let packedSize = rows * columns / 2
            let scaleSize = rows * columns / 32
            out.append(K3ExpertSubtensorPlan(
                name: "\(stem)_packed", stem: stem, isScales: false,
                offsetInBlob: cursor, sizeInBlob: packedSize,
                logicalShape: [UInt32(rows), UInt32(columns)],
                bits: GTurboFormatV2.mxfp4PackedBits))
            cursor += packedSize
            out.append(K3ExpertSubtensorPlan(
                name: "\(stem)_scales", stem: stem, isScales: true,
                offsetInBlob: cursor, sizeInBlob: scaleSize,
                logicalShape: [UInt32(rows), UInt32(columns / 32)],
                bits: GTurboFormatV2.mxfp4ScaleBits))
            cursor += scaleSize
        }
        return out
    }

    private static func validateExpertTensor(_ tensor: K3SourceTensor,
                                             layer: Int, stem: String, isScales: Bool,
                                             arch: K3ArchInfo, probed: ProbedDims) throws {
        guard tensor.dtype == .u8 else {
            throw RepackError.dtypeMismatch(name: tensor.name,
                detail: "expected U8 MXFP4 payload, got \(tensor.dtype)")
        }
        let latent = UInt64(probed.latentBottleneckSize)
        let inter = UInt64(probed.expertIntermediateSize)
        let rows = stem == "w2" ? latent : inter
        let columns = stem == "w2" ? inter : latent
        let expected: [UInt64] = isScales
            ? [rows, columns / 32]
            : [rows, columns / 2]
        guard tensor.shape == expected else {
            throw RepackError.shapeMismatch(name: tensor.name,
                detail: "expected U8 \(expected), got \(tensor.shape)")
        }
    }

    // MARK: - Helpers

    static func roundUp(_ value: UInt64, _ alignment: UInt64) -> UInt64 {
        ((value + alignment - 1) / alignment) * alignment
    }
}

// MARK: - Range plan

/// Builds the phase-1 byte-range plan: verbatim trunk tensors copy straight
/// to their resident offsets, transform inputs stage into `trunk-staging.bin`,
/// and every expert subtensor of every expert lands at its canonical blob
/// offset in the per-layer packed files. Coalescing, fingerprinting, and
/// destination validation reuse the Gemma `RangeCopyPlanner` statics
/// unchanged, so the checkpoint/resume machinery sees the same plan shape.
enum K3RangeCopyPlanner {
    static func plan(repackPlan: K3RepackPlan,
                     rangeChunkBytes: Int,
                     includeExpertPayloads: Bool = true) throws -> RangeCopyPlan {
        var copies: [RangeCopy] = []
        copies.reserveCapacity(repackPlan.tensors.count
            + repackPlan.layers.reduce(0) { $0 + $1.expertsPerLayer * 6 })

        for tensor in repackPlan.tensors {
            switch tensor.transform {
            case .verbatim:
                let source = tensor.source!  // non-nil for every non-zeros transform
                precondition(source.sizeBytes == tensor.entry.sizeBytes,
                             "verbatim source/entry size drift for \(tensor.entry.name)")
                copies.append(RangeCopy(shardID: source.shardPath,
                                        sourceOffset: source.absoluteOffset,
                                        size: source.sizeBytes,
                                        destinationPath: repackPlan.resident.path,
                                        destinationOffset: tensor.entry.fileOffset))
            case .affine(_, let stagingOffset),
                 .widenFP32(let stagingOffset),
                 .truncateFP32(let stagingOffset, _, _):
                let source = tensor.source!
                copies.append(RangeCopy(shardID: source.shardPath,
                                        sourceOffset: source.absoluteOffset,
                                        size: source.sizeBytes,
                                        destinationPath: repackPlan.stagingPath,
                                        destinationOffset: stagingOffset))
            case .synthesizeZeros:
                continue
            }
        }

        if includeExpertPayloads {
            for layer in repackPlan.layers {
                for (subtensorIndex, subtensor) in layer.subtensors.enumerated() {
                    let sources = layer.sources[subtensorIndex]
                    for expert in 0..<layer.expertsPerLayer {
                        let source = sources[expert]
                        precondition(source.sizeBytes == subtensor.sizeInBlob,
                                     "expert source size drift for \(source.name)")
                        copies.append(RangeCopy(
                            shardID: source.shardPath,
                            sourceOffset: source.absoluteOffset,
                            size: subtensor.sizeInBlob,
                            destinationPath: layer.path,
                            destinationOffset: UInt64(expert) * layer.expertStride
                                + subtensor.offsetInBlob))
                    }
                }
            }
        }

        let outputRoot = (repackPlan.resident.path as NSString).deletingLastPathComponent
        try RangeCopyPlanner.validateDestinationIntervals(copies, outputRoot: outputRoot)
        let coalesced = try RangeCopyPlanner.coalesce(copies: copies,
                                                      rangeChunkBytes: rangeChunkBytes)
        let indexData = try ResidentWriter.encodeIndex(plan: repackPlan.resident)
        var indexStream = Sha256Stream()
        indexData.withUnsafeBytes { indexStream.update($0) }
        let indexSha = indexStream.finalizeHexString()

        var expectedOutputs = [
            RemoteExpectedOutput(relativePath: "model_weights.bin",
                                 size: repackPlan.resident.totalSize),
            RemoteExpectedOutput(
                relativePath: (repackPlan.stagingPath as NSString).lastPathComponent,
                size: repackPlan.stagingSize),
        ]
        expectedOutputs.append(contentsOf: repackPlan.layers.map {
            RemoteExpectedOutput(
                relativePath: "packed_experts/" + ($0.path as NSString).lastPathComponent,
                size: $0.fileSize)
        })
        expectedOutputs.sort { $0.relativePath < $1.relativePath }

        let fingerprint = try RangeCopyPlanner.canonicalFingerprint(
            copies: coalesced,
            outputRoot: outputRoot,
            rangeChunkBytes: rangeChunkBytes,
            layoutMode: includeExpertPayloads ? "identity" : "apfs-clone-verified-experts",
            layoutOrderSha256: nil,
            residentIndexSha256: indexSha,
            expectedOutputs: expectedOutputs)
        let downloaded = coalesced.reduce(UInt64(0)) { $0 + $1.size }
        let copied = copies.reduce(UInt64(0)) { $0 + $1.size }
        return RangeCopyPlan(scalarCopies: copies,
                             coalescedCopies: coalesced,
                             remoteBytesToDownload: downloaded,
                             remoteGapBytesDownloaded: downloaded - copied,
                             canonicalFingerprint: fingerprint,
                             residentIndexSha256: indexSha,
                             expectedOutputs: expectedOutputs)
    }
}

// MARK: - v2 manifest / layout encoding

enum K3GTurboJSON {
    static func encodeLayout(plan: K3RepackPlan) throws -> Data {
        let layers: [GTurboLayerV2] = plan.layers.map { lp in
            var subtensors: [String: GTurboSubTensorV2] = [:]
            subtensors.reserveCapacity(lp.subtensors.count)
            for sub in lp.subtensors {
                subtensors[sub.name] = GTurboSubTensorV2(
                    offset: sub.offsetInBlob,
                    size: sub.sizeInBlob,
                    dtype: sub.isScales
                        ? GTurboFormatV2.mxfp4ScaleDType
                        : GTurboFormatV2.mxfp4PackedDType,
                    shape: sub.logicalShape,
                    bits: sub.bits)
            }
            return GTurboLayerV2(
                layer: lp.layerIndex,
                file: (lp.path as NSString).lastPathComponent,
                experts: (0..<lp.expertsPerLayer).map { expert in
                    GTurboExpertV2(expert: expert, physicalRank: nil,
                                   offset: UInt64(expert) * lp.expertStride,
                                   size: lp.expertStride,
                                   tensors: subtensors)
                })
        }
        let stride = plan.layers.first?.expertStride ?? 0
        return try GTurboPackedExpertsLayoutCodecV2.encode(
            GTurboPackedExpertsLayoutV2(
                expertStride: stride,
                numLayers: plan.arch.numLayers,
                expertsPerLayer: plan.layers.first?.expertsPerLayer ?? 0,
                layers: layers))
    }

    static func encodeManifest(plan: K3RepackPlan,
                               modelID: String,
                               sourceSnapshotHash: String,
                               files: [(relativePath: String, size: UInt64, sha256: String)])
        throws -> Data {
        let quant = plan.trunkQuant == .int4
            ? KimiK3FormatProfile.quantInt4
            : KimiK3FormatProfile.quantInt8
        var wireFiles: [String: GTurboManifestFileV2] = [:]
        wireFiles.reserveCapacity(files.count)
        for file in files {
            guard wireFiles.updateValue(
                GTurboManifestFileV2(size: file.size, sha256: file.sha256),
                forKey: file.relativePath) == nil else {
                throw RepackError.configurationInvalid(
                    detail: "duplicate manifest file entry \(file.relativePath)")
            }
        }
        return try GTurboManifestCodecV2.encode(GTurboManifestV2(
            flags: KimiK3FormatProfile.flags,
            modelID: modelID,
            sourceSnapshotHash: sourceSnapshotHash,
            arch: plan.manifestArch,
            quant: quant,
            files: wireFiles,
            expertsPerLayer: plan.layers.first?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: plan.layers.first?.expertStride ?? 0))
    }
}
