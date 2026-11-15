import Foundation
import TinyTitanCore

// Laguna-S-2.1's ArchInfo loader, split out of ArchInfo+Loaders.swift under
// the 500-line-per-file rule.

extension ArchInfo {
    /// Poolside Laguna-S-2.1 (`model_type == "laguna"`): the 118B-A8B sparse
    /// model. 48 layers, 256 routed experts at top-10 plus one shared expert,
    /// GQA with 8 KV heads, per-head softplus gating and two RoPE dialects.
    ///
    /// The load-bearing difference from the Qwen families is the layer mask:
    /// the first of every four layers is global attention and the other three
    /// are sliding-window (512). The repacker only needs the mask to write the
    /// manifest; the per-layer head counts (48 global, 72 sliding) are the
    /// runtime's checked Laguna preset.
    static func loadLaguna(
        configPath: String,
        tc: [String: Any]
    ) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(k)")
            }
            return n
        }
        let layers = try i("num_hidden_layers")
        let mask = try lagunaLayerMask(
            configPath: configPath, tc: tc, layers: layers)
        let ropes = try lagunaRopes(configPath: configPath, tc: tc)
        let fullTheta = ropes.fullTheta
        let slidingTheta = ropes.slidingTheta
        let partial = ropes.partial
        let slidingWindow = try i("sliding_window")
        let denseLayers = (tc["mlp_only_layers"] as? [Int]) ?? []
        let headDim = try i("head_dim")
        let gate = (tc["gating"] as? String) ?? "per-element"
        guard gate == "per-head" else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "gating \(gate) is not the per-head form this runtime implements")
        }
        let perLayer = (tc["num_attention_heads_per_layer"] as? [Int]) ?? []
        guard perLayer.isEmpty || perLayer.count == layers else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "num_attention_heads_per_layer does not cover \(layers) layers")
        }
        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: slidingWindow,
            finalLogitSoftcap: 0,
            ropeTheta: slidingTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: partial,
            numLayers: layers,
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: (tc["hidden_act"] as? String) ?? "silu",
            family: .laguna,
            attnOutputGate: true,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0,
            routerNormTopK: (tc["norm_topk_prob"] as? Bool) ?? false,
            quantGroupSize: 64,
            movaNumExperts: 0,
            movaTopKExperts: 0,
            normGroups: 1,
            attentionGate: "softplus",
            routerScaling: try d("moe_routed_scaling_factor"),
            routerHasBias: true,
            movaRouterScaling: 1,
            movaRouterHasBias: false,
            movaActivation: "silu",
            denseLayerIndices: denseLayers)
        try crossCheckLaguna(arch, configPath: configPath)
        return arch
    }

    /// The 0/1 attention mask from `layer_types`.
    private static func lagunaLayerMask(
        configPath: String, tc: [String: Any], layers: Int
    ) throws -> [UInt8] {
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing layer_types")
        }
        guard layerTypes.count == layers else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "layer_types has \(layerTypes.count) entries, "
                    + "num_hidden_layers is \(layers)")
        }
        return try layerTypes.map { kind in
            switch kind {
            case "full_attention": return 1
            case "sliding_attention": return 0
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(kind)\"")
            }
        }
    }

    /// The two RoPE dialects: the global YaRN theta, the sliding plain theta
    /// and the global partial-rotary fraction (sliding is always full).
    private static func lagunaRopes(
        configPath: String, tc: [String: Any]
    ) throws -> (fullTheta: Double, slidingTheta: Double, partial: Double) {
        guard let ropes = tc["rope_parameters"] as? [String: Any],
            let full = ropes["full_attention"] as? [String: Any],
            let sliding = ropes["sliding_attention"] as? [String: Any]
        else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "missing rope_parameters.full_attention/sliding_attention")
        }
        func number(_ dict: [String: Any], _ key: String, _ where_: String) throws -> Double {
            guard let n = (dict[key] as? Double) ?? (dict[key] as? NSNumber)?.doubleValue
            else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(where_).\(key)")
            }
            return n
        }
        return (
            try number(full, "rope_theta", "full_attention"),
            try number(sliding, "rope_theta", "sliding_attention"),
            try number(full, "partial_rotary_factor", "full_attention")
        )
    }

    /// A config that claims the production geometry must agree on every field
    /// the runtime's `ArchConfig.lagunaS21` preset states. Toy fixtures with
    /// another hidden/layer count are exempt, exactly like the Qwen checks.
    static func crossCheckLaguna(_ a: ArchInfo, configPath: String) throws {
        guard a.hiddenSize == 3072, a.numLayers == 48 else { return }
        var problems: [String] = []
        func want(_ ok: Bool, _ what: String) { if !ok { problems.append(what) } }
        want(a.numExperts == 256, "num_experts 256")
        want(a.topKExperts == 10, "num_experts_per_tok 10")
        want(a.moeIntermediateSize == 1024, "moe_intermediate_size 1024")
        want(a.intermediateSize == 12_288, "intermediate_size 12288")
        want(a.vocabSize == 100_352, "vocab_size 100352")
        want(a.numHeads == 48, "num_attention_heads 48")
        want(a.numKVHeads == 8, "num_key_value_heads 8")
        want(a.headDim == 128, "head_dim 128")
        want(a.slidingWindow == 512, "sliding_window 512")
        want(a.ropeTheta == 10_000, "sliding rope_theta 10000")
        want(a.fullRopeTheta == 500_000, "full rope_theta 500000")
        want(a.partialRotaryFactor == 0.5, "full partial_rotary_factor 0.5")
        want(a.denseLayerIndices == [0], "mlp_only_layers [0]")
        want(a.fullAttentionLayerMask.filter { $0 == 1 }.count == 12, "12 global layers")
        want(a.routerScaling == 2.5, "moe_routed_scaling_factor 2.5")
        want(a.routerNormTopK, "norm_topk_prob true")
        guard problems.isEmpty else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "laguna config does not match the supported Laguna-S-2.1 "
                    + "contract: expected " + problems.joined(separator: ", "))
        }
    }
}
