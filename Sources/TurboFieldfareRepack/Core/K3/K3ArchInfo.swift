import Foundation

/// Architecture facts parsed from the K3 source `config.json` (the
/// `text_config` wrapper of `moonshotai/Kimi-K3`). Refuse-defaults: every key
/// the v2 manifest needs is required — a missing key throws instead of
/// falling back to a default (a defaulted layer list would silently turn MLA
/// layers into KDA).
///
/// Layer lists (`linear_attn_config.kda_layers` / `full_attn_layers`) are
/// 1-based in the source config and are copied through unchanged — that is
/// exactly the v2 manifest's convention (`GTurboManifestArchV2`).
/// `denseLayers` is derived from `first_k_dense_replace` (1-based count).
///
/// Two v2 arch fields are *not* config keys and are derived here from other
/// required keys, then cross-asserted against tensor shapes by the planner:
/// - `kdaDecayProjectionSize` = `linear_attn_config.num_heads * head_dim`
///   (the f_b_proj row count), and
/// - `moeSharedExpertIntermediateSize` = `num_shared_experts *
///   moe_intermediate_size` (the shared_experts.gate_proj row count).
/// `kdaDecayLowRankSize` has no config expression at all (it equals
/// `head_dim` on the canonical checkpoint but the tiny test fixtures decouple
/// it), so the planner derives it from the f_a_proj row count.
struct K3ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let vocabSize: Int
    let numLayers: Int
    let denseMLPIntermediateSize: Int
    let rmsNormEpsilon: Double
    let tieWordEmbeddings: Bool
    let hiddenActivation: String
    let bosTokenID: Int
    let eosTokenID: Int
    /// 1-based, as recorded on the wire. `denseLayers` overlaps the attention
    /// lists; `kdaLayers`/`fullAttnLayers` are disjoint and cover every layer.
    let denseLayers: [Int]
    let kdaLayers: [Int]
    let fullAttnLayers: [Int]

    // KDA (Kimi Delta Attention)
    let kdaNumHeads: Int
    let kdaHeadDim: Int
    let kdaConvWidth: Int
    let kdaGateLowerBound: Double
    let kdaFullRankOutputGate: Bool

    // MLA (gated multi-head latent attention)
    let mlaNumHeads: Int
    let mlaQLoraRank: Int
    let mlaKVLoraRank: Int
    let mlaQKNopeHeadDim: Int
    let mlaQKRopeHeadDim: Int
    let mlaVHeadDim: Int
    let mlaOutputGate: Bool

    // AttnRes
    let attnResBlockSize: Int

    // LatentMoE
    let moeNumExperts: Int
    let moeTopKExperts: Int
    let moeLatentBottleneckSize: Int
    let moeExpertIntermediateSize: Int
    let moeNumSharedExperts: Int
    let moeSharedExpertIntermediateSize: Int
    let situGLUGateBeta: Double
    let situGLUUpBeta: Double
    let routerRenormalize: Bool
    /// No config key carries this; the noaux_tc router needs the correction
    /// bias, so it is pinned here and the planner asserts the
    /// `e_score_correction_bias` tensors actually exist per MoE layer.
    let routerCorrectionBias: Bool

    /// KDA projection width (`num_heads * head_dim` = 12,288 canonical).
    var kdaProjectionSize: Int { kdaNumHeads * kdaHeadDim }
    /// f_b_proj row count; asserted against tensor shapes by the planner.
    var kdaDecayProjectionSize: Int { kdaNumHeads * kdaHeadDim }

    // MARK: - Derived layer schedules (0-based)

    var mlaLayers0: Set<Int> { Set(fullAttnLayers.map { $0 - 1 }) }
    var kdaLayers0: Set<Int> { Set(kdaLayers.map { $0 - 1 }) }
    var denseLayers0: Set<Int> { Set(denseLayers.map { $0 - 1 }) }
    var moeLayers0: Set<Int> { Set(0..<numLayers).subtracting(denseLayers0) }

    func isKDA(layer0: Int) -> Bool { kdaLayers0.contains(layer0) }
    func isMoE(layer0: Int) -> Bool { moeLayers0.contains(layer0) }

    static func load(configPath: String) throws -> K3ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        guard let la = tc["linear_attn_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath,
                                                detail: "no text_config.linear_attn_config")
        }
        func i(_ dict: [String: Any], _ k: String) throws -> Int {
            guard let n = (dict[k] as? Int) ?? (dict[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ dict: [String: Any], _ k: String) throws -> Double {
            guard let n = (dict[k] as? Double) ?? (dict[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func b(_ dict: [String: Any], _ k: String) throws -> Bool {
            guard let n = (dict[k] as? Bool) ?? (dict[k] as? NSNumber)?.boolValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func layerList(_ k: String) throws -> [Int] {
            guard let raw = la[k] as? [Any] else {
                throw RepackError.configJsonInvalid(path: configPath,
                                                    detail: "missing linear_attn_config.\(k)")
            }
            return try raw.map { e in
                guard let n = (e as? Int) ?? (e as? NSNumber)?.intValue else {
                    throw RepackError.configJsonInvalid(
                        path: configPath,
                        detail: "linear_attn_config.\(k) has a non-integer entry")
                }
                return n
            }
        }

        let numLayers = try i(tc, "num_hidden_layers")
        let firstDense = try i(tc, "first_k_dense_replace")
        guard firstDense >= 0, firstDense <= numLayers else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "first_k_dense_replace \(firstDense) out of range for \(numLayers) layers")
        }
        let kdaLayers = try layerList("kda_layers")
        let fullAttnLayers = try layerList("full_attn_layers")
        let denseLayers = firstDense == 0 ? [] : Array(1...firstDense)
        // Validate the 1-based schedule now so a bad source config fails at
        // metadata time with a clear error instead of at manifest encoding.
        for (name, list) in [("kda_layers", kdaLayers), ("full_attn_layers", fullAttnLayers)] {
            guard list.allSatisfy({ $0 >= 1 && $0 <= numLayers }),
                  Set(list).count == list.count, list == list.sorted() else {
                throw RepackError.configJsonInvalid(
                    path: configPath,
                    detail: "linear_attn_config.\(name) must be sorted unique 1-based layer ids")
            }
        }
        guard Set(kdaLayers).isDisjoint(with: Set(fullAttnLayers)),
              Set(kdaLayers).union(fullAttnLayers).count == numLayers else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "kda_layers/full_attn_layers must be disjoint and cover 1...\(numLayers)")
        }

        // The source writes hidden_act "situ" for the SiTU-GLU pair; the v2
        // manifest records the runtime's spelling.
        let act = try { () throws -> String in
            guard let raw = tc["hidden_act"] as? String else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing hidden_act")
            }
            guard raw == "situ" else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unsupported hidden_act \(raw)")
            }
            return "situ_glu"
        }()

        let numShared = try i(tc, "num_shared_experts")
        let moeIntermediate = try i(tc, "moe_intermediate_size")

        return K3ArchInfo(
            hiddenSize: try i(tc, "hidden_size"),
            vocabSize: try i(tc, "vocab_size"),
            numLayers: numLayers,
            denseMLPIntermediateSize: try i(tc, "intermediate_size"),
            rmsNormEpsilon: try d(tc, "rms_norm_eps"),
            tieWordEmbeddings: try b(tc, "tie_word_embeddings"),
            hiddenActivation: act,
            bosTokenID: try i(tc, "bos_token_id"),
            eosTokenID: try i(tc, "eos_token_id"),
            denseLayers: denseLayers,
            kdaLayers: kdaLayers,
            fullAttnLayers: fullAttnLayers,
            kdaNumHeads: try i(la, "num_heads"),
            kdaHeadDim: try i(la, "head_dim"),
            kdaConvWidth: try i(la, "short_conv_kernel_size"),
            kdaGateLowerBound: try d(la, "gate_lower_bound"),
            kdaFullRankOutputGate: try b(la, "use_full_rank_gate"),
            mlaNumHeads: try i(tc, "num_attention_heads"),
            mlaQLoraRank: try i(tc, "q_lora_rank"),
            mlaKVLoraRank: try i(tc, "kv_lora_rank"),
            mlaQKNopeHeadDim: try i(tc, "qk_nope_head_dim"),
            mlaQKRopeHeadDim: try i(tc, "qk_rope_head_dim"),
            mlaVHeadDim: try i(tc, "v_head_dim"),
            mlaOutputGate: try b(tc, "mla_use_output_gate"),
            attnResBlockSize: try i(tc, "attn_res_block_size"),
            moeNumExperts: try i(tc, "num_experts"),
            moeTopKExperts: try i(tc, "num_experts_per_token"),
            moeLatentBottleneckSize: try i(tc, "routed_expert_hidden_size"),
            moeExpertIntermediateSize: moeIntermediate,
            moeNumSharedExperts: numShared,
            moeSharedExpertIntermediateSize: numShared * moeIntermediate,
            situGLUGateBeta: try d(tc, "activation_situ_beta"),
            situGLUUpBeta: try d(tc, "activation_situ_linear_beta"),
            routerRenormalize: try b(tc, "moe_renormalize"),
            routerCorrectionBias: true)
    }
}
