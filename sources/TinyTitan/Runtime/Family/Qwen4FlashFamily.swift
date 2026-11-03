import Foundation

/// Qwen 4.0 Flash (`Qwen4-Omni-120B-Think`) family knowledge: the tensor-name
/// contract of the repacked `.ssdai`.
///
/// This is a deliberate copy of `Qwen38FlashFamily.swift`, kept separate rather
/// than shared. The two models descend from the same Qwen4 preview text
/// architecture, so every name here is expected to agree with the 3.8 path
/// today; the copy exists so a 4.0 checkpoint can rename, add or drop a tensor
/// without a change to the 3.8 contract, and so the `ArchInfo` loader and the
/// runtime schema each have a 4.0 branch to edit on their own. When the weights
/// land, the checked names come from that checkpoint's index, not from this
/// comment.
///
/// The three deviations from the Qwen3.5-MoE families are the same ones the
/// 3.8 path documents, and each would be a silent mis-load if assumed:
///
/// 1. The backbone prefix is `model.language_model.`, the mirror image of
///    qwen36's `language_model.model.`.
/// 2. `lm_head` sits at the archive root, not under the language model.
/// 3. **There are no layer norms.** `input_layernorm` and
///    `post_attention_layernorm` do not exist; the norm that runs before each
///    block is `hc_norm` inside that block's hyper-connection, and the final
///    norm is the model-level mixer's. The norm roles below therefore point at
///    hyper-connection tensors -- they are the same thing by another name, not
///    a substitution.
extension TensorSchema {
    /// The draft head's layer body is the target's, verbatim: the repack
    /// rewrites `mtp.layers.0.*` to the same prefix the target uses, so one
    /// schema serves both. Only the fusion projections and their norms are the
    /// draft's own, and those sit at the archive root rather than in the schema.
    static var qwen4flashMTP: TensorSchema { .qwen4flash }

    private static func qwen4FlashLayer(_ layer: Int, _ suffix: String) -> String {
        "model.language_model.layers.\(layer).\(suffix)"
    }

    static let qwen4flash = TensorSchema(
        embedding: "model.language_model.embed_tokens.weight",
        lmHead: "lm_head.weight",
        // The residual stream is collapsed by the model-level mixer; its
        // `hc_norm` is what a plain transformer calls `model.norm`.
        finalNorm: "model.language_model.hyper_connection_mixer.hc_norm",
        qProj: { qwen4FlashLayer($0, "self_attn.q_proj.weight") },
        kProj: { qwen4FlashLayer($0, "self_attn.k_proj.weight") },
        vProj: { qwen4FlashLayer($0, "self_attn.v_proj.weight") },
        oProj: { qwen4FlashLayer($0, "self_attn.o_proj.weight") },
        router: { qwen4FlashLayer($0, "mlp.gate.weight") },
        sharedExpertGate: { qwen4FlashLayer($0, "mlp.shared_expert.gate_proj.weight") },
        sharedExpertUp: { qwen4FlashLayer($0, "mlp.shared_expert.up_proj.weight") },
        sharedExpertDown: { qwen4FlashLayer($0, "mlp.shared_expert.down_proj.weight") },
        sharedExpertScalarGate: { qwen4FlashLayer($0, "mlp.shared_expert_gate.weight") },
        inputNorm: { qwen4FlashLayer($0, "attn_hyper_connection.hc_norm") },
        postAttnNorm: { qwen4FlashLayer($0, "mlp_hyper_connection.hc_norm") },
        // No `.weight` on the attention norms in this family, unlike qwen36.
        qNorm: { qwen4FlashLayer($0, "self_attn.q_norm") },
        kNorm: { qwen4FlashLayer($0, "self_attn.k_norm") },
        gdnQKV: { qwen4FlashLayer($0, "linear_attn.in_proj_qkv.weight") },
        gdnZ: { qwen4FlashLayer($0, "linear_attn.in_proj_z.weight") },
        gdnA: { qwen4FlashLayer($0, "linear_attn.in_proj_a.weight") },
        gdnB: { qwen4FlashLayer($0, "linear_attn.in_proj_b.weight") },
        gdnOut: { qwen4FlashLayer($0, "linear_attn.out_proj.weight") },
        gdnConv: { qwen4FlashLayer($0, "linear_attn.conv1d.weight") },
        gdnALog: { qwen4FlashLayer($0, "linear_attn.A_log") },
        gdnDtBias: { qwen4FlashLayer($0, "linear_attn.dt_bias") },
        gdnNorm: { qwen4FlashLayer($0, "linear_attn.norm.weight") })
}

/// Roles this family has and the Qwen3.5-MoE families do not. Kept separate
/// from `TensorSchema` so the shared type does not grow optional fields for
/// one family's subsystems; `Model` reaches for these only on this family.
///
/// The Qwen4 declarations are independent of `Qwen38FlashTensors` even where
/// the strings currently agree: a rename on either side must be a two-line,
/// two-family decision rather than one edit that silently moves both.
public enum Qwen4FlashTensors: FlashTensorSet {
    private static func layer(_ l: Int, _ suffix: String) -> String {
        "model.language_model.layers.\(l).\(suffix)"
    }

    // MARK: Hyper-connections (Gated Residual)

    /// Low-rank read mixer, 10240 -> 320 -> 10240, per sublayer.
    public static func attnMixDown(_ l: Int) -> String {
        layer(l, "attn_hyper_connection.input_mix_weight_down.weight")
    }
    public static func attnMixUp(_ l: Int) -> String {
        layer(l, "attn_hyper_connection.input_mix_weight_up.weight")
    }
    /// Per-stream write weights: the block output is injected back into all
    /// four residual streams through these.
    public static func attnInject(_ l: Int) -> String {
        layer(l, "attn_hyper_connection.block_inject_weight.weight")
    }
    public static func mlpMixDown(_ l: Int) -> String {
        layer(l, "mlp_hyper_connection.input_mix_weight_down.weight")
    }
    public static func mlpMixUp(_ l: Int) -> String {
        layer(l, "mlp_hyper_connection.input_mix_weight_up.weight")
    }
    public static func mlpInject(_ l: Int) -> String {
        layer(l, "mlp_hyper_connection.block_inject_weight.weight")
    }
    /// Model-level collapse of the four streams back to one 2560 vector.
    public static let mixerDown =
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"
    public static let mixerUp =
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"

    // MARK: QSA indexer (full-attention layers only)

    public static func indexerQProj(_ l: Int) -> String {
        layer(l, "self_attn.indexer.index_q_proj.weight")
    }
    public static func indexerKProj(_ l: Int) -> String {
        layer(l, "self_attn.indexer.index_k_proj.weight")
    }
    /// The reference's loader adds 1.0 to a zero-centred gamma; the MLX
    /// checkpoint this family installs from has already done so, so these are
    /// used as stored. (Checked on the 3.8 artifact: they centre on 0.96, not
    /// on 0, and the 4.0 path inherits that contract until its own index says
    /// otherwise.)
    public static func indexerQNorm(_ l: Int) -> String {
        layer(l, "self_attn.indexer.q_layernorm")
    }
    public static func indexerKNorm(_ l: Int) -> String {
        layer(l, "self_attn.indexer.k_layernorm")
    }

    // MARK: Per-head attention norms

    public static func qNorm(_ l: Int) -> String { layer(l, "self_attn.q_norm") }
    public static func kNorm(_ l: Int) -> String { layer(l, "self_attn.k_norm") }

    // MARK: Gated DeltaNet (linear-attention layers only)

    public static func gdnQKV(_ l: Int) -> String {
        layer(l, "linear_attn.in_proj_qkv.weight")
    }
    public static func gdnZ(_ l: Int) -> String {
        layer(l, "linear_attn.in_proj_z.weight")
    }
    public static func gdnA(_ l: Int) -> String {
        layer(l, "linear_attn.in_proj_a.weight")
    }
    public static func gdnB(_ l: Int) -> String {
        layer(l, "linear_attn.in_proj_b.weight")
    }
    public static func gdnConv(_ l: Int) -> String {
        layer(l, "linear_attn.conv1d.weight")
    }
    public static func gdnALog(_ l: Int) -> String { layer(l, "linear_attn.A_log") }
    public static func gdnDtBias(_ l: Int) -> String { layer(l, "linear_attn.dt_bias") }
    public static func gdnNorm(_ l: Int) -> String { layer(l, "linear_attn.norm.weight") }
    public static func gdnOut(_ l: Int) -> String {
        layer(l, "linear_attn.out_proj.weight")
    }

    // MARK: PLE n-gram block (one layer only)

    /// Dilated depthwise convolution over the gated n-gram value.
    public static func pleConv(_ l: Int) -> String { layer(l, "ple.conv1d") }
    /// Projects the gathered n-gram embedding to the full residual width.
    public static func pleKeyProj(_ l: Int) -> String {
        layer(l, "ple.key_proj.weight")
    }
    /// Projects the same embedding to one stream's width.
    public static func pleValueProj(_ l: Int) -> String {
        layer(l, "ple.value_proj.weight")
    }
    public static func pleNormKey(_ l: Int) -> String { layer(l, "ple.norm_key") }
    public static func pleNormQuery(_ l: Int) -> String { layer(l, "ple.norm_query") }
    public static func pleNormConv(_ l: Int) -> String { layer(l, "ple.norm_conv") }
    /// The table and its hash constants are passthrough files in the install
    /// root rather than tensors, because the runtime gathers rows off storage.
    public static let ngramTableFile = "ngram_table.bin"
    public static let pleConstantsFile = "ple_constants.json"
}
