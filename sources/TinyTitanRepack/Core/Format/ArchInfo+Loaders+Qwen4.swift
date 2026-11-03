import Foundation

/// Qwen 4.0 Flash config and draft-head loaders.
///
/// Split from `ArchInfo+Loaders.swift` under the 500-line-per-file rule. The
/// bodies are copied from the 3.8 loaders rather than shared, so a 4.0
/// checkpoint can change its geometry without moving the 3.8 port.
extension ArchInfo {

    // MARK: - Qwen 4.0 Flash (`model_type: qwen4_omni_moe`)

    /// The multimodal `Qwen4OmniForConditionalGeneration` checkpoint, read as
    /// text-only. The tensor contract is copied from the 3.8 preview and is a
    /// separate loader so the two can diverge. Every field is read from the
    /// config; nothing is inferred. The vision and audio towers are never
    /// repacked, and their tensors are excluded by the planner's
    /// `isMultimodalTensorName`.
    static func loadQwen4Flash(
        configPath: String,
        tc: [String: Any],
        root: [String: Any]
    ) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func iOpt(_ k: String, _ fallback: Int) -> Int {
            (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue ?? fallback
        }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "linear_attention": mask.append(2)
            case "full_attention": mask.append(1)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        guard
            let theta = (rope["rope_theta"] as? Double)
                ?? (rope["rope_theta"] as? NSNumber)?.doubleValue
        else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.rope_theta")
        }
        guard
            let prf = (rope["partial_rotary_factor"] as? Double)
                ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue
                ?? (tc["partial_rotary_factor"] as? Double)
        else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing partial_rotary_factor")
        }
        // `ple_layer_ids` is 1-based in the config; the runtime indexes layers
        // from 0. Converting here keeps the off-by-one in one place.
        let pleIDs = ((tc["ple_layer_ids"] as? [Int]) ?? []).map { $0 - 1 }
        for id in pleIDs where id < 0 || id >= mask.count {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "ple_layer_ids entry out of range after 1-based "
                    + "conversion: \(id + 1)")
        }
        // The quantized checkpoint declares its own affine group size; TinyTitan
        // must repack at whatever the source used, never at a default.
        let quant =
            (root["quantization"] as? [String: Any])
            ?? (root["quantization_config"] as? [String: Any]) ?? [:]
        let group =
            (quant["group_size"] as? Int)
            ?? (quant["group_size"] as? NSNumber)?.intValue ?? 64
        let headDim = try i("head_dim")

        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("shared_expert_intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: (tc["hidden_act"] as? String) ?? "silu",
            family: .qwen4flash,
            // `output_gate_type: sigmoid` is this family's spelling of the
            // attention output gate qwen36 declares as `attn_output_gate`.
            attnOutputGate: (tc["output_gate_type"] as? String) == "sigmoid",
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: try i("linear_num_key_heads"),
            linearNumVHeads: try i("linear_num_value_heads"),
            linearKeyHeadDim: try i("linear_key_head_dim"),
            linearValueHeadDim: try i("linear_value_head_dim"),
            linearConvKernelSize: try i("linear_conv_kernel_dim"),
            hcCount: try i("hc_count"),
            hcLowRank: try i("hc_lowrank"),
            indexerNumHeads: try i("indexer_n_heads"),
            indexerNumKVHeads: try i("indexer_kv_heads"),
            indexerHeadDim: try i("indexer_head_dim"),
            indexerBudget: try i("indexer_budget"),
            indexerCompressRatio: try i("indexer_compress_ratio"),
            pleLayerIndices: pleIDs,
            pleEmbedDim: iOpt("ple_embed_dim", try i("hidden_size")),
            pleConvKernelSize: try i("ple_conv_kernel_size"),
            pleNgramSize: try i("ngram_size"),
            pleVocabSizeBase: try i("ngram_vocab_size_base"),
            pleHeadsPerNgram: try i("heads_per_ngram"),
            pleVocabDivisor: iOpt("make_ngram_vocab_size_divisible_by", 128),
            routerNormTopK: (tc["norm_topk_prob"] as? Bool) ?? true,
            quantGroupSize: group)
        try crossCheckQwen4Flash(arch, configPath: configPath)
        return arch
    }

    /// The Qwen 4.0 Flash MTP draft, derived from the target's own config.
    ///
    /// Copied from the 3.8 draft derivation: one full-attention layer with its
    /// own 512-expert set, hyper-connection gates and indexer, plus the two
    /// fusion projections. It has no linear-attention layers and no n-gram
    /// block, and it reports no embedding or head of its own -- both are the
    /// target's, shared rather than copied.
    static func qwen4FlashMTP(
        from base: ArchInfo,
        configPath: String
    ) throws -> ArchInfo {
        guard base.family == .qwen4flash else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "MTP draft requires a Qwen 4.0 Flash target")
        }
        return ArchInfo(
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
            numLayers: 1,
            numExperts: base.numExperts,
            topKExperts: base.topKExperts,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [1],
            hiddenActivation: base.hiddenActivation,
            family: .qwen4flashMTP,
            attnOutputGate: base.attnOutputGate,
            attentionScale: base.attentionScale,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            // No DeltaNet bundle and no n-gram block in the draft.
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0,
            hcCount: base.hcCount,
            hcLowRank: base.hcLowRank,
            indexerNumHeads: base.indexerNumHeads,
            indexerNumKVHeads: base.indexerNumKVHeads,
            indexerHeadDim: base.indexerHeadDim,
            indexerBudget: base.indexerBudget,
            indexerCompressRatio: base.indexerCompressRatio,
            pleLayerIndices: [], pleEmbedDim: 0,
            pleConvKernelSize: 0, pleNgramSize: 0,
            pleVocabSizeBase: 0, pleHeadsPerNgram: 0,
            pleVocabDivisor: 0,
            routerNormTopK: base.routerNormTopK,
            quantGroupSize: base.quantGroupSize)
    }

    /// Derives a draft head from whichever Flash target was loaded.
    ///
    /// One entry point so the local and remote importers do not each need a
    /// family switch, and so a new Qwen4-lineage target adds its draft here
    /// once rather than at every call site.
    static func draftHead(
        from base: ArchInfo,
        configPath: String
    ) throws -> ArchInfo {
        switch base.family {
        case .qwen38flash:
            return try qwen38FlashNextMTP(from: base, configPath: configPath)
        case .qwen4flash:
            return try qwen4FlashMTP(from: base, configPath: configPath)
        default:
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "a draft head is not defined for family "
                    + "\(base.family.rawValue)")
        }
    }
}
