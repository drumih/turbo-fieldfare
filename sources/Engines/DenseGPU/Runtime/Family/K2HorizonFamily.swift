import Foundation

/// K2-Horizon's tensor-name contract.
///
/// The checkpoint is written without the `language_model.` prefix that the
/// Qwen repacks carry: the attention, the routed-experts router and the shared
/// expert live at `model.layers.N.*`, the embedding at `model.embed_tokens` and
/// the untied head at `lm_head`. The shared expert is `shared_experts` (plural)
/// and K2 has no per-head q/k norms and no Gated-DeltaNet, so those roles point
/// at names that cannot exist -- a lookup failure is the diagnosis, where a
/// plausible name would silently feed the wrong tensor.
///
/// Sparse layers replace `v_proj` with MoVA: a `v_router` over 64 value
/// experts, whose weights live at `self_attn.v_experts.N`. The attention gate is
/// a projection of its own (`self_attn.gate_proj`), not the packed [query ;
/// gate] half Qwen writes into `q_proj`.
enum K2HorizonTensors {
    static let embedding = "model.embed_tokens.weight"
    static let lmHead = "lm_head.weight"
    static let finalNorm = "model.norm.weight"

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
    /// The softplus attention gate's projection, `hidden -> heads * headDim`.
    static func attnGate(_ layer: Int) -> String {
        name(layer, "self_attn.gate_proj.weight")
    }

    /// The 100-expert router and its selection-only bias, both bf16.
    static func moeRouter(_ layer: Int) -> String {
        name(layer, "mlp.gate.weight")
    }
    static func moeRouterBias(_ layer: Int) -> String {
        name(layer, "mlp.gate.bias")
    }
    /// The single shared SwiGLU expert, `moeIntermediateSize` wide.
    static func sharedGate(_ layer: Int) -> String {
        name(layer, "mlp.shared_experts.gate_proj.weight")
    }
    static func sharedUp(_ layer: Int) -> String {
        name(layer, "mlp.shared_experts.up_proj.weight")
    }
    static func sharedDown(_ layer: Int) -> String {
        name(layer, "mlp.shared_experts.down_proj.weight")
    }
    /// The dense layers 0-2's SwiGLU MLP, `intermediateSize` wide.
    static func denseGate(_ layer: Int) -> String {
        name(layer, "mlp.gate_proj.weight")
    }
    static func denseUp(_ layer: Int) -> String {
        name(layer, "mlp.up_proj.weight")
    }
    static func denseDown(_ layer: Int) -> String {
        name(layer, "mlp.down_proj.weight")
    }

    /// MoVA's value router over the sparse layer's 64 value experts.
    static func vRouter(_ layer: Int) -> String {
        name(layer, "self_attn.v_router.weight")
    }
    static func vRouterBias(_ layer: Int) -> String {
        name(layer, "self_attn.v_router.bias")
    }
    static func vExpert(_ layer: Int, _ expert: Int) -> String {
        name(layer, "self_attn.v_experts.\(expert).weight")
    }
}

extension TensorSchema {
    static let k2Horizon = TensorSchema(
        embedding: K2HorizonTensors.embedding,
        lmHead: K2HorizonTensors.lmHead,
        finalNorm: K2HorizonTensors.finalNorm,
        qProj: { K2HorizonTensors.qProj($0) },
        kProj: { K2HorizonTensors.kProj($0) },
        // Only the three dense layers carry a v_proj; sparse layers use MoVA.
        vProj: { K2HorizonTensors.vProj($0) },
        oProj: { K2HorizonTensors.oProj($0) },
        router: { K2HorizonTensors.moeRouter($0) },
        sharedExpertGate: { K2HorizonTensors.sharedGate($0) },
        sharedExpertUp: { K2HorizonTensors.sharedUp($0) },
        sharedExpertDown: { K2HorizonTensors.sharedDown($0) },
        // K2's shared expert has no sigmoid scalar gate; the role cannot exist.
        sharedExpertScalarGate: { _ in "k2_horizon.has_no_shared_expert_gate" },
        inputNorm: { K2HorizonTensors.inputNorm($0) },
        postAttnNorm: { K2HorizonTensors.postAttnNorm($0) },
        qNorm: { _ in "k2_horizon.has_no_q_norm" },
        kNorm: { _ in "k2_horizon.has_no_k_norm" },
        gdnQKV: { _ in "k2_horizon.has_no_gdn" },
        gdnZ: { _ in "k2_horizon.has_no_gdn" },
        gdnA: { _ in "k2_horizon.has_no_gdn" },
        gdnB: { _ in "k2_horizon.has_no_gdn" },
        gdnOut: { _ in "k2_horizon.has_no_gdn" },
        gdnConv: { _ in "k2_horizon.has_no_gdn" },
        gdnALog: { _ in "k2_horizon.has_no_gdn" },
        gdnDtBias: { _ in "k2_horizon.has_no_gdn" },
        gdnNorm: { _ in "k2_horizon.has_no_gdn" })
}
