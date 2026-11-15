import Foundation

/// Qwen3.5-MoE 35B-A3B family knowledge (Qwen 3.6 and Ornith 1.5): the
/// tensor-name contract of the repacked `.ssdai`. The MTP sidecar shares
/// the layer naming but stores its adapter tensors at the archive root.
extension TensorSchema {
    private static func qwenLayer(_ layer: Int, _ suffix: String) -> String {
        "language_model.model.layers.\(layer).\(suffix)"
    }

    static let qwen36 = TensorSchema(
        embedding: "language_model.model.embed_tokens.weight",
        lmHead: "language_model.lm_head.weight",
        finalNorm: "language_model.model.norm.weight",
        qProj: { qwenLayer($0, "self_attn.q_proj.weight") },
        kProj: { qwenLayer($0, "self_attn.k_proj.weight") },
        vProj: { qwenLayer($0, "self_attn.v_proj.weight") },
        oProj: { qwenLayer($0, "self_attn.o_proj.weight") },
        router: { qwenLayer($0, "mlp.gate.weight") },
        sharedExpertGate: { qwenLayer($0, "mlp.shared_expert.gate_proj.weight") },
        sharedExpertUp: { qwenLayer($0, "mlp.shared_expert.up_proj.weight") },
        sharedExpertDown: { qwenLayer($0, "mlp.shared_expert.down_proj.weight") },
        sharedExpertScalarGate: { qwenLayer($0, "mlp.shared_expert_gate.weight") },
        inputNorm: { qwenLayer($0, "input_layernorm.weight") },
        postAttnNorm: { qwenLayer($0, "post_attention_layernorm.weight") },
        qNorm: { qwenLayer($0, "self_attn.q_norm.weight") },
        kNorm: { qwenLayer($0, "self_attn.k_norm.weight") },
        gdnQKV: { qwenLayer($0, "linear_attn.in_proj_qkv.weight") },
        gdnZ: { qwenLayer($0, "linear_attn.in_proj_z.weight") },
        gdnA: { qwenLayer($0, "linear_attn.in_proj_a.weight") },
        gdnB: { qwenLayer($0, "linear_attn.in_proj_b.weight") },
        gdnOut: { qwenLayer($0, "linear_attn.out_proj.weight") },
        gdnConv: { qwenLayer($0, "linear_attn.conv1d.weight") },
        gdnALog: { qwenLayer($0, "linear_attn.A_log") },
        gdnDtBias: { qwenLayer($0, "linear_attn.dt_bias") },
        gdnNorm: { qwenLayer($0, "linear_attn.norm.weight") })

    static let qwen36MTP = TensorSchema.qwen36
}
