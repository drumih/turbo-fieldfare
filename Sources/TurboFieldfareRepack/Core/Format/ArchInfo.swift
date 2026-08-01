import Foundation

/// Model family discriminator, mirrored into `manifest.json -> arch.family`
/// for non-Gemma families (Gemma manifests omit it — the format's original
/// architecture). Raw values match the runtime's `ModelFamily`.
enum RepackModelFamily: String, Sendable, Equatable {
    case gemma4 = "gemma4"
    case qwen36 = "qwen36"
}

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`, 2 if `linear_attention`.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String

    // Family-dependent extensions. Defaults describe Gemma 4 so the Gemma
    // path (and its manifest output) is unchanged.
    let family: RepackModelFamily
    let attnOutputGate: Bool
    let attentionScale: Double
    let embeddingScaledBySqrtHidden: Bool
    let routerScaled: Bool
    let ffnSandwichNorms: Bool
    let sharedExpertGated: Bool
    let ropeNeoxSubdim: Bool
    let linearNumKHeads: Int
    let linearNumVHeads: Int
    let linearKeyHeadDim: Int
    let linearValueHeadDim: Int
    let linearConvKernelSize: Int

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        guard let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        if (root["model_type"] as? String) == "qwen3_5_moe" {
            return try loadQwen36(configPath: configPath, tc: tc)
        }
        return try loadGemma4(configPath: configPath, tc: tc)
    }

    // MARK: - Gemma 4

    private static func loadGemma4(configPath: String,
                                   tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = (ropeFull["partial_rotary_factor"] as? Double)
            ?? (ropeFull["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let fullTheta = (ropeFull["rope_theta"] as? Double)
            ?? (ropeFull["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let swaTheta = (ropeSWA["rope_theta"] as? Double)
            ?? (ropeSWA["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000.0
        let kEqV = (tc["attention_k_eq_v"] as? Bool) ?? false
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let act = (tc["hidden_activation"] as? String) ?? "gelu_pytorch_tanh"
        return ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_global_key_value_heads"),
            headDim: try i("head_dim"),
            fullHeadDim: try i("global_head_dim"),
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: try d("final_logit_softcapping"),
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .gemma4,
            attnOutputGate: false,
            attentionScale: 1.0,
            embeddingScaledBySqrtHidden: true,
            routerScaled: true,
            ffnSandwichNorms: true,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0)
    }

    // MARK: - Qwen 3.6 MoE (`model_type == "qwen3_5_moe"`)

    private static func loadQwen36(configPath: String,
                                   tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "linear_attention": mask.append(2)
            case "full_attention":   mask.append(1)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        guard let theta = (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.rope_theta")
        }
        guard let prf = (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.partial_rotary_factor")
        }
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let gate = (tc["attn_output_gate"] as? Bool) ?? false
        let act = (tc["hidden_act"] as? String) ?? "silu"
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
            tieWordEmbeddings: tie,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .qwen36,
            attnOutputGate: gate,
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
            linearConvKernelSize: try i("linear_conv_kernel_dim"))
        try crossCheckProductionQwen36(arch, configPath: configPath)
        return arch
    }

    /// Production Qwen3.6-35B-A3B baseline (mirrors the runtime's
    /// `ArchConfig.qwen36_35B_A3B`; the repack target has no dependency on the
    /// runtime module). A config that matches the production shape
    /// (hidden 2048, 40 layers) must agree on every field; toy/synthetic
    /// configs are exempt.
    private static func crossCheckProductionQwen36(_ a: ArchInfo,
                                                   configPath: String) throws {
        guard a.hiddenSize == 2048, a.numLayers == 40 else { return }
        var expectedMask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { expectedMask[i] = 1 }
        let expected = ArchInfo(
            hiddenSize: 2048,
            intermediateSize: 512,
            moeIntermediateSize: 512,
            numHeads: 16,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 256,
            fullHeadDim: 256,
            vocabSize: 248_320,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 40,
            numExperts: 256,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .qwen36,
            attnOutputGate: true,
            attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 16,
            linearNumVHeads: 32,
            linearKeyHeadDim: 128,
            linearValueHeadDim: 128,
            linearConvKernelSize: 4)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen3_5_moe config does not match the pinned "
                    + "Qwen3.6-35B-A3B architecture baseline")
        }
    }
}
