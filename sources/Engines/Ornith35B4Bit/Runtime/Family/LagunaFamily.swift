import Foundation

/// Laguna-S-2.1's tensor-name contract.
///
/// The checkpoint spells the language model at `model.layers.N.*`, the
/// embedding at `model.embed_tokens` and the untied head at `lm_head` -- the
/// same namespace K2-Horizon uses. The differences that matter:
///
/// - the attention output gate is `self_attn.g_proj` and emits one value per
///   *head*, not per channel;
/// - every layer carries per-head `self_attn.q_norm` / `k_norm` weights;
/// - the routed-expert router is `mlp.gate.weight` and its selection-only
///   bias keeps the checkpoint's spelling `mlp.experts.e_score_correction_bias`;
/// - the shared expert is singular (`mlp.shared_expert`), not K2's plural.
package enum LagunaTensors {
    static let embedding = "model.embed_tokens.weight"
    static let lmHead = "lm_head.weight"
    static let finalNorm = "model.norm.weight"
    /// `generation_config.json`: EOS is `<|EOS|>` id 2 and `<|im_end|>` 24.
    package static let eosTokens: Set<Int> = [2, 24]

    static func name(_ layer: Int, _ suffix: String) -> String {
        "model.layers.\(layer).\(suffix)"
    }

    static func inputNorm(_ layer: Int) -> String {
        name(layer, "input_layernorm.weight")
    }
    static func postAttnNorm(_ layer: Int) -> String {
        name(layer, "post_attention_layernorm.weight")
    }
    static func qProj(_ layer: Int) -> String {
        name(layer, "self_attn.q_proj.weight")
    }
    static func kProj(_ layer: Int) -> String {
        name(layer, "self_attn.k_proj.weight")
    }
    static func vProj(_ layer: Int) -> String {
        name(layer, "self_attn.v_proj.weight")
    }
    static func oProj(_ layer: Int) -> String {
        name(layer, "self_attn.o_proj.weight")
    }
    /// One softplus gate per head: `hidden -> heads`.
    static func attnGate(_ layer: Int) -> String {
        name(layer, "self_attn.g_proj.weight")
    }
    static func qNorm(_ layer: Int) -> String {
        name(layer, "self_attn.q_norm.weight")
    }
    static func kNorm(_ layer: Int) -> String {
        name(layer, "self_attn.k_norm.weight")
    }

    static func moeRouter(_ layer: Int) -> String {
        name(layer, "mlp.gate.weight")
    }
    /// The aux-loss-free selection bias; bf16, one value per routed expert.
    static func moeRouterBias(_ layer: Int) -> String {
        name(layer, "mlp.experts.e_score_correction_bias")
    }
    static func sharedGate(_ layer: Int) -> String {
        name(layer, "mlp.shared_expert.gate_proj.weight")
    }
    static func sharedUp(_ layer: Int) -> String {
        name(layer, "mlp.shared_expert.up_proj.weight")
    }
    static func sharedDown(_ layer: Int) -> String {
        name(layer, "mlp.shared_expert.down_proj.weight")
    }
    static func denseGate(_ layer: Int) -> String {
        name(layer, "mlp.gate_proj.weight")
    }
    static func denseUp(_ layer: Int) -> String {
        name(layer, "mlp.up_proj.weight")
    }
    static func denseDown(_ layer: Int) -> String {
        name(layer, "mlp.down_proj.weight")
    }
}

extension TensorSchema {
    static let laguna = TensorSchema(
        embedding: LagunaTensors.embedding,
        lmHead: LagunaTensors.lmHead,
        finalNorm: LagunaTensors.finalNorm,
        qProj: { LagunaTensors.qProj($0) },
        kProj: { LagunaTensors.kProj($0) },
        vProj: { LagunaTensors.vProj($0) },
        oProj: { LagunaTensors.oProj($0) },
        router: { LagunaTensors.moeRouter($0) },
        sharedExpertGate: { LagunaTensors.sharedGate($0) },
        sharedExpertUp: { LagunaTensors.sharedUp($0) },
        sharedExpertDown: { LagunaTensors.sharedDown($0) },
        // The shared expert is a plain MLP: there is no scalar gate to read.
        sharedExpertScalarGate: { _ in "laguna.has_no_shared_expert_gate" },
        inputNorm: { LagunaTensors.inputNorm($0) },
        postAttnNorm: { LagunaTensors.postAttnNorm($0) },
        qNorm: { LagunaTensors.qNorm($0) },
        kNorm: { LagunaTensors.kNorm($0) },
        gdnQKV: { _ in "laguna.has_no_gdn" },
        gdnZ: { _ in "laguna.has_no_gdn" },
        gdnA: { _ in "laguna.has_no_gdn" },
        gdnB: { _ in "laguna.has_no_gdn" },
        gdnOut: { _ in "laguna.has_no_gdn" },
        gdnConv: { _ in "laguna.has_no_gdn" },
        gdnALog: { _ in "laguna.has_no_gdn" },
        gdnDtBias: { _ in "laguna.has_no_gdn" },
        gdnNorm: { _ in "laguna.has_no_gdn" })
}
