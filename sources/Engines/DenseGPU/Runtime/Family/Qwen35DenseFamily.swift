import Foundation

/// Qwen 3.5 **dense** family knowledge (2B, 4B, 9B): the tensor-name contract of
/// a repacked dense `.ssdai`.
///
/// Structurally this is the Qwen 3.6 layer stack -- Gated-DeltaNet linear
/// attention, full attention every fourth layer, partial RoPE, an attention
/// output gate -- with a plain SwiGLU FFN where that family has a routed
/// mixture. The FFN is therefore spelled through the *shared-expert* roles:
/// that is the stage the runner already has for it (`silu(gate(x)) * up(x)`
/// through `down`), the family declares `sharedExpertGated: false` so the
/// scalar gate is never read, and `numExperts == 0` is what tells the runner to
/// skip the routed half. No kernel is family-specific here.
///
/// The two roles this family does not have are pointed at names that cannot
/// exist. That is deliberate: a router read here would be a bug, and a lookup
/// failure names it, where a plausible-looking name would silently feed the FFN
/// gate in as router logits.
extension TensorSchema {
    private static func denseLayer(_ layer: Int, _ suffix: String) -> String {
        "language_model.model.layers.\(layer).\(suffix)"
    }

    static let qwen35Dense = TensorSchema(
        embedding: "language_model.model.embed_tokens.weight",
        // The 2B and 4B tie the embedding and never read this. The 9B does not:
        // it ships a real head at `language_model.lm_head.weight` (the manifest's
        // per-tensor quant key drops the `.weight`, which is the stem, not the
        // tensor name).
        lmHead: "language_model.lm_head.weight",
        finalNorm: "language_model.model.norm.weight",
        qProj: { denseLayer($0, "self_attn.q_proj.weight") },
        kProj: { denseLayer($0, "self_attn.k_proj.weight") },
        vProj: { denseLayer($0, "self_attn.v_proj.weight") },
        oProj: { denseLayer($0, "self_attn.o_proj.weight") },
        router: { _ in "qwen3_5_dense.has_no_router" },
        sharedExpertGate: { denseLayer($0, "mlp.gate_proj.weight") },
        sharedExpertUp: { denseLayer($0, "mlp.up_proj.weight") },
        sharedExpertDown: { denseLayer($0, "mlp.down_proj.weight") },
        sharedExpertScalarGate: { _ in "qwen3_5_dense.has_no_shared_expert_gate" },
        inputNorm: { denseLayer($0, "input_layernorm.weight") },
        postAttnNorm: { denseLayer($0, "post_attention_layernorm.weight") },
        qNorm: { denseLayer($0, "self_attn.q_norm.weight") },
        kNorm: { denseLayer($0, "self_attn.k_norm.weight") },
        gdnQKV: { denseLayer($0, "linear_attn.in_proj_qkv.weight") },
        gdnZ: { denseLayer($0, "linear_attn.in_proj_z.weight") },
        gdnA: { denseLayer($0, "linear_attn.in_proj_a.weight") },
        gdnB: { denseLayer($0, "linear_attn.in_proj_b.weight") },
        gdnOut: { denseLayer($0, "linear_attn.out_proj.weight") },
        gdnConv: { denseLayer($0, "linear_attn.conv1d.weight") },
        gdnALog: { denseLayer($0, "linear_attn.A_log") },
        gdnDtBias: { denseLayer($0, "linear_attn.dt_bias") },
        gdnNorm: { denseLayer($0, "linear_attn.norm.weight") })
}
