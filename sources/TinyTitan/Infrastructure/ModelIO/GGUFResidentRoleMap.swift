import Foundation

package enum GGUFResidentRoleError: Error, Equatable, CustomStringConvertible {
    case unknownTensor(String)
    case shapeMismatch(name: String, expected: [UInt64], actual: [UInt64])

    package var description: String {
        switch self {
        case .unknownTensor(let name):
            return "no runtime role is known for GGUF tensor \(name)"
        case .shapeMismatch(let name, let expected, let actual):
            return "\(name) is \(actual), the role expects \(expected)"
        }
    }
}

/// Maps the GGUF's resident (non-expert) tensor names to the runtime's roles,
/// and checks every one against the shape the architecture implies.
///
/// This is a table, not a composition: the file already stores what the runtime
/// reads, including the `[query ; gate]` fusion inside `attn_q` on the twelve
/// full-attention layers. Two names are traps and the table says so:
/// `attn_gate` on a GDN layer is the linear attention's z projection (not an
/// attention gate), and `attn_qkv` is the GDN qkv where full-attention layers
/// carry `attn_q`/`attn_k`/`attn_v` instead.
///
/// A role whose declared shape disagrees with the architecture is refused here,
/// at conversion time, rather than discovered as fluent nonsense after a
/// 100 GB install.
package enum GGUFResidentRoleMap {
    private struct Rule: Sendable {
        let role: String
        /// Expected GGUF dims `[columns, rows]`, or nil when the architecture
        /// config does not carry the value (the GDN state geometry lives only
        /// in the file's own metadata).
        let shape: (@Sendable (ArchConfig) -> [UInt64])?
    }

    private static let perLayer: [String: Rule] = [
        "attn_qkv.weight": Rule(role: "gdnQKV") { config in
            let linear = config.linearAttention
            return [
                UInt64(config.hiddenSize),
                UInt64(
                    2 * linear.numKHeads * linear.keyHeadDim
                        + linear.numVHeads * linear.valueHeadDim),
            ]
        },
        "attn_gate.weight": Rule(role: "gdnZ") { config in
            let linear = config.linearAttention
            return [
                UInt64(config.hiddenSize),
                UInt64(linear.numVHeads * linear.valueHeadDim),
            ]
        },
        "ssm_out.weight": Rule(role: "gdnOut") { config in
            let linear = config.linearAttention
            return [
                UInt64(linear.numVHeads * linear.valueHeadDim),
                UInt64(config.hiddenSize),
            ]
        },
        "ssm_conv1d.weight": Rule(role: "gdnConv") { config in
            let linear = config.linearAttention
            return [
                UInt64(linear.convKernelSize),
                UInt64(
                    2 * linear.numKHeads * linear.keyHeadDim
                        + linear.numVHeads * linear.valueHeadDim),
            ]
        },
        "ssm_alpha.weight": Rule(role: "gdnA", shape: nil),
        "ssm_beta.weight": Rule(role: "gdnB", shape: nil),
        "ssm_a": Rule(role: "gdnALog", shape: nil),
        "ssm_dt.bias": Rule(role: "gdnDtBias", shape: nil),
        "ssm_norm.weight": Rule(role: "gdnNorm") { config in
            [UInt64(config.linearAttention.valueHeadDim)]
        },
        "attn_q.weight": Rule(role: "qProj") { config in
            [UInt64(config.hiddenSize), UInt64(2 * config.numHeads * config.headDim)]
        },
        "attn_k.weight": Rule(role: "kProj") { config in
            [UInt64(config.hiddenSize), UInt64(config.numKVHeads * config.headDim)]
        },
        "attn_v.weight": Rule(role: "vProj") { config in
            [UInt64(config.hiddenSize), UInt64(config.numKVHeads * config.headDim)]
        },
        "attn_output.weight": Rule(role: "oProj") { config in
            [UInt64(config.numHeads * config.headDim), UInt64(config.hiddenSize)]
        },
        "attn_q_norm.weight": Rule(role: "qNorm") { config in [UInt64(config.headDim)] },
        "attn_k_norm.weight": Rule(role: "kNorm") { config in [UInt64(config.headDim)] },
        "ffn_gate_inp.weight": Rule(role: "router") { config in
            [UInt64(config.hiddenSize), UInt64(config.numExperts)]
        },
        "ffn_gate_inp_shexp.weight": Rule(role: "sharedExpertScalarGate") { config in
            [UInt64(config.hiddenSize)]
        },
        "ffn_gate_shexp.weight": Rule(role: "sharedExpertGate") { config in
            [UInt64(config.hiddenSize), UInt64(config.moeIntermediateSize)]
        },
        "ffn_up_shexp.weight": Rule(role: "sharedExpertUp") { config in
            [UInt64(config.hiddenSize), UInt64(config.moeIntermediateSize)]
        },
        "ffn_down_shexp.weight": Rule(role: "sharedExpertDown") { config in
            [UInt64(config.moeIntermediateSize), UInt64(config.hiddenSize)]
        },
        "hc_attn_norm.weight": Rule(role: "inputNorm") { config in
            [UInt64(config.hyperConnections.count * config.hiddenSize)]
        },
        "hc_ffn_norm.weight": Rule(role: "postAttnNorm") { config in
            [UInt64(config.hyperConnections.count * config.hiddenSize)]
        },
        "hc_attn_up.weight": Rule(role: "hcAttnUp") { config in
            [
                UInt64(config.hyperConnections.lowRank),
                UInt64(config.hyperConnections.count * config.hiddenSize),
            ]
        },
        "hc_attn_down.weight": Rule(role: "hcAttnDown") { config in
            [
                UInt64(config.hyperConnections.count * config.hiddenSize),
                UInt64(config.hyperConnections.lowRank),
            ]
        },
        "hc_attn_inject.weight": Rule(role: "hcAttnInject") { config in
            [
                UInt64(config.hyperConnections.count * config.hiddenSize),
                UInt64(config.hyperConnections.count),
            ]
        },
        "hc_ffn_up.weight": Rule(role: "hcFFNUp") { config in
            [
                UInt64(config.hyperConnections.lowRank),
                UInt64(config.hyperConnections.count * config.hiddenSize),
            ]
        },
        "hc_ffn_down.weight": Rule(role: "hcFFNDown") { config in
            [
                UInt64(config.hyperConnections.count * config.hiddenSize),
                UInt64(config.hyperConnections.lowRank),
            ]
        },
        "hc_ffn_inject.weight": Rule(role: "hcFFNInject") { config in
            [
                UInt64(config.hyperConnections.count * config.hiddenSize),
                UInt64(config.hyperConnections.count),
            ]
        },
        "indexer.q_proj.weight": Rule(role: "indexerQ") { config in
            [
                UInt64(config.hiddenSize),
                UInt64(config.sparseIndexer.numHeads * config.sparseIndexer.headDim),
            ]
        },
        "indexer.k_proj.weight": Rule(role: "indexerK") { config in
            [UInt64(config.hiddenSize), UInt64(config.sparseIndexer.headDim)]
        },
        "indexer.q_norm.weight": Rule(role: "indexerQNorm") { config in
            [UInt64(config.sparseIndexer.headDim)]
        },
        "indexer.k_norm.weight": Rule(role: "indexerKNorm") { config in
            [UInt64(config.sparseIndexer.headDim)]
        },
        "ple_key.weight": Rule(role: "pleKey") { config in
            [
                UInt64(config.hiddenSize),
                UInt64(config.hyperConnections.count * config.hiddenSize),
            ]
        },
        "ple_value.weight": Rule(role: "pleValue") { config in
            [UInt64(config.hiddenSize), UInt64(config.hiddenSize)]
        },
        "ple_conv1d.weight": Rule(role: "pleConv") { config in
            [
                UInt64(config.ple.convKernelSize),
                UInt64(config.hyperConnections.count * config.hiddenSize),
            ]
        },
        "ple_norm_conv.weight": Rule(role: "pleNormConv") { config in
            [UInt64(config.hyperConnections.count * config.hiddenSize)]
        },
        "ple_norm_key.weight": Rule(role: "pleNormKey") { config in
            [UInt64(config.hyperConnections.count * config.hiddenSize)]
        },
        "ple_norm_query.weight": Rule(role: "pleNormQuery") { config in
            [UInt64(config.hyperConnections.count * config.hiddenSize)]
        },
    ]

    private static let global: [String: Rule] = [
        "token_embd.weight": Rule(role: "embedding") { config in
            [UInt64(config.hiddenSize), UInt64(config.vocabSize)]
        },
        "output.weight": Rule(role: "lmHead") { config in
            [UInt64(config.hiddenSize), UInt64(config.vocabSize)]
        },
        "output_hc_norm.weight": Rule(role: "finalNorm") { config in
            [UInt64(config.hyperConnections.count * config.hiddenSize)]
        },
        "output_hc_up.weight": Rule(role: "outputHCUp") { config in
            [
                UInt64(config.hyperConnections.lowRank),
                UInt64(config.hyperConnections.count * config.hiddenSize),
            ]
        },
        "output_hc_down.weight": Rule(role: "outputHCDown") { config in
            [
                UInt64(config.hyperConnections.count * config.hiddenSize),
                UInt64(config.hyperConnections.lowRank),
            ]
        },
        "per_layer_token_embd.weight": Rule(role: "pleTable") { config in
            [UInt64(config.ple.headDim), 0]
        },
    ]

    /// Every resident tensor maps to one role, and every shape the architecture
    /// determines is checked. Returns the count per role.
    package static func validate(
        model: GGUFModel, arch: ArchConfig
    ) throws -> [String: Int] {
        var counts: [String: Int] = [:]
        for (name, location) in model.tensors {
            if name.hasSuffix("_exps.weight") { continue }
            let suffix: String
            if let dot = name.firstIndex(of: "."), name.hasPrefix("blk.") {
                let rest = name[name.index(after: dot)...]
                guard let second = rest.firstIndex(of: ".") else {
                    throw GGUFResidentRoleError.unknownTensor(name)
                }
                suffix = String(rest[rest.index(after: second)...])
            } else {
                suffix = name
            }
            guard let rule = perLayer[suffix] ?? global[suffix] else {
                throw GGUFResidentRoleError.unknownTensor(name)
            }
            if let shape = rule.shape {
                let expected = shape(arch)
                let dims = Array(location.tensor.dimensions.prefix(2))
                // A zero in the expectation is a dimension the checkpoint owns
                // (the PLE table's row count); compare only what is known.
                let comparable = zip(expected, dims).filter { $0.0 != 0 }
                guard comparable.allSatisfy({ $0.0 == $0.1 }) else {
                    throw GGUFResidentRoleError.shapeMismatch(
                        name: name, expected: expected, actual: dims)
                }
            }
            counts[rule.role, default: 0] += 1
        }
        return counts
    }
}
