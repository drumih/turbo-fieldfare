import Foundation
import TinyTitanCore

/// The checkpoint tensor behind each role the decode graph reads (TT-125).
///
/// The graph was written against the GGUF's names and layouts. Every mapping
/// below was measured against the Q4 install, which holds the same weights
/// (`docs/plan-inkling2-7x-engine.md`, "TT-125: the NVFP4 install against the Q4
/// graph"):
/// - identity, except `attn_rel_bias` (the checkpoint's `rel_logits_proj.proj`
///   transposed) and the K/V convolution kernels of the dynamic-convolution
///   layers (`static_w` transposed, taps reversed);
/// - `attn_res_query` of stored layer L is the checkpoint's `attn_queries.(L-1)`.
///
/// `converted` tensors are the ones the graph reads as F32 (norms, clips,
/// convolutions, residual queries, router, controllers); they are expanded from
/// BF16 at load. Everything else is bound in place as BF16.
enum InklingNVFP4TensorMap {
    enum Transform: Equatable {
        case identity
        /// [a][b] stored, [b][a] read.
        case transpose
        /// [taps][channels] stored, [channels][taps] read with taps reversed.
        case transposeReversed
    }

    struct Source: Equatable {
        let name: String
        let transform: Transform
        /// Expanded to F32 at load (the graph reads this role as F32).
        let converted: Bool
    }

    /// The GGUF-named roles the graph reads as F32.
    static let convertedSuffixes: Set<String> = [
        "attn_norm.weight", "ffn_norm.weight", "attn_q_clip.weight", "attn_rel_clip.weight",
        "attn_rel_bias.weight", "attn_k_conv.weight", "attn_v_conv.weight",
        "attn_post_conv.weight", "ffn_post_conv.weight", "attn_res_query.weight",
        "ffn_res_query.weight", "ffn_gate_inp.weight", "exp_probs_b.bias",
        "attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight",
    ]

    static let globals: [String: (String, Bool)] = [
        "token_embd.weight": ("model.llm.embed.weight", false),
        "token_embd_norm.weight": ("model.llm.embed_norm.weight", true),
        "output.weight": ("model.llm.unembed.weight", false),
        "output_norm.weight": ("model.llm.norm.weight", true),
        "output_res_query.weight": ("model.llm.attn_res.output_query", true),
        "loop_res_query.weight": ("model.llm.middle_cycle_cross_pass_queries", true),
        "loop_kv_res_query.weight": ("model.llm.middle_cycle_cross_pass_kv_queries", true),
    ]

    static let layerNames: [String: String] = [
        "attn_gate.weight": "attn.gate_attn.weight", "attn_k.weight": "attn.wk_dv.weight",
        "attn_v.weight": "attn.wv_dv.weight", "attn_q.weight": "attn.wq_du.weight",
        "attn_output.weight": "attn.wo_ud.weight", "attn_rel.weight": "attn.wr_du.weight",
        "attn_q_clip.weight": "attn.q_l2_clip.scale",
        "attn_rel_clip.weight": "attn.rel_l2_clip.scale",
        "attn_sink_proj.weight": "attn.attn_sink_proj.weight",
        "attn_norm.weight": "attn_norm.weight", "attn_post_conv.weight": "attn_sconv.weight",
        "ffn_norm.weight": "mlp_norm.weight", "ffn_post_conv.weight": "mlp_sconv.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight", "exp_probs_b.bias": "mlp.gate.bias",
        "attn_k_conv_proj.weight": "attn.k_sconv.weight_proj",
        "attn_v_conv_proj.weight": "attn.v_sconv.weight_proj",
        "attn_q_inp_norm.weight": "q_input_norm.weight",
        "attn_k_inp_norm.weight": "k_input_norm.weight",
        "attn_v_inp_norm.weight": "v_input_norm.weight",
    ]

    /// The checkpoint source of the GGUF-named role `name`, or nil when the
    /// checkpoint has none (the role does not exist in this layer).
    static func source(for name: String, exists: (String) -> Bool) -> Source? {
        if let (global, converted) = globals[name] {
            return exists(global)
                ? Source(name: global, transform: .identity, converted: converted)
                : nil
        }
        guard name.hasPrefix("blk."), let dot = name.dropFirst(4).firstIndex(of: "."),
            let layer = Int(name[name.index(name.startIndex, offsetBy: 4)..<dot])
        else { return nil }
        let rest = String(name[name.index(after: dot)...])
        let prefix = "model.llm.layers.\(layer)."
        let converted = convertedSuffixes.contains(rest) || rest.hasPrefix("loop_")
        func found(_ checkpoint: String, _ transform: Transform = .identity) -> Source? {
            exists(checkpoint)
                ? Source(name: checkpoint, transform: transform, converted: converted) : nil
        }
        if let mapped = layerNames[rest] { return found(prefix + mapped) }
        switch rest {
        case "attn_rel_bias.weight":
            return found(prefix + "attn.rel_logits_proj.proj", .transpose)
        case "ffn_res_query.weight":
            return found("model.llm.attn_res.mlp_queries.\(layer)")
        case "attn_res_query.weight":
            return layer > 0 ? found("model.llm.attn_res.attn_queries.\(layer - 1)") : nil
        case "attn_k_conv.weight", "attn_v_conv.weight":
            let kv = rest.hasPrefix("attn_k") ? "k" : "v"
            let dynamic = prefix + "attn.\(kv)_sconv.static_w"
            if exists(dynamic) { return found(dynamic, .transposeReversed) }
            return found(prefix + "attn.\(kv)_sconv.weight")
        default:
            break
        }
        // Loop controllers: loop_<module>_<down|forget|gain>.weight.
        let parts = rest.split(separator: "_")
        if rest.hasPrefix("loop_"), parts.count == 3,
            let kind = parts.last?.split(separator: ".").first
        {
            return found(prefix + "controllers.\(parts[1]).\(kind)_weight")
        }
        return nil
    }

    /// `values` ([rows][columns] as stored) in the layout the graph reads.
    static func apply(_ transform: Transform, to values: [Float], rows: Int, columns: Int)
        -> [Float]
    {
        switch transform {
        case .identity:
            return values
        case .transpose:
            var out = [Float](repeating: 0, count: rows * columns)
            for r in 0..<rows {
                for c in 0..<columns { out[c * rows + r] = values[r * columns + c] }
            }
            return out
        case .transposeReversed:
            // Stored [taps][channels]; read [channels][taps], taps reversed.
            var out = [Float](repeating: 0, count: rows * columns)
            for tap in 0..<rows {
                for channel in 0..<columns {
                    out[channel * rows + (rows - 1 - tap)] = values[tap * columns + channel]
                }
            }
            return out
        }
    }
}
