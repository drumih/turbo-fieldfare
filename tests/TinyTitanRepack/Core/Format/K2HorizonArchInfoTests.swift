import Foundation
import Testing

@testable import TinyTitanRepackCore
@testable import TinyTitanCore

/// `ArchInfo` parsing for `model_type: k2_horizon`, checked against the pinned
/// checkpoint's config values (IFM/K2-Horizon-MoVA-36B-A4B at e9ce7176).
@Suite("K2-Horizon ArchInfo")
struct K2HorizonArchInfoTests {
    private static func configJSON(overrides: [String: Any] = [:]) throws -> Data {
        var text: [String: Any] = [
            "hidden_size": 2_560,
            "intermediate_size": 6_144,
            "moe_intermediate_size": 768,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rope_head_dim": 128,
            "vocab_size": 250_624,
            "num_experts": 100,
            "num_experts_per_tok": 8,
            "mova_num_experts": 64,
            "mova_num_experts_per_tok": 4,
            "num_shared_experts": 1,
            "layernorm_num_groups": 2,
            "attention_gate_func": "softplus",
            "router_score_func": "sigmoid",
            "router_scaling_factor": 2.5,
            "moe_gate_bias": true,
            "norm_topk_prob": true,
            "mlp_only_layers": [0, 1, 2],
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "rope_parameters": ["rope_theta": 10_000_000.0],
        ]
        for (key, value) in overrides { text[key] = value }
        let root: [String: Any] = [
            "model_type": "k2_horizon",
            "text_config": text,
            "quantization": ["group_size": 64, "bits": 4, "mode": "affine"],
        ]
        return try JSONSerialization.data(withJSONObject: root)
    }

    private static func load(_ data: Data) throws -> ArchInfo {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("k2-horizon-\(UUID().uuidString).json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        return try ArchInfo.load(configPath: url.path)
    }

    @Test("Parses the pinned checkpoint's text geometry")
    func parsesGeometry() throws {
        let arch = try Self.load(Self.configJSON())
        #expect(arch.family == .k2Horizon)
        #expect(arch.hiddenSize == 2_560)
        #expect(arch.intermediateSize == 6_144)
        #expect(arch.moeIntermediateSize == 768)
        #expect(arch.numLayers == 48)
        #expect(arch.numExperts == 100)
        #expect(arch.topKExperts == 8)
        #expect(arch.numHeads == 32)
        #expect(arch.numKVHeads == 8)
        #expect(arch.headDim == 128)
        #expect(arch.vocabSize == 250_624)
        #expect(!arch.tieWordEmbeddings)
        #expect(arch.ropeTheta == 10_000_000)
        #expect(arch.partialRotaryFactor == 1)
        #expect(arch.fullAttentionLayerMask == [UInt8](repeating: 1, count: 48))
    }

    @Test("Reads the MoVA and grouped-norm extensions")
    func parsesExtensions() throws {
        let arch = try Self.load(Self.configJSON())
        #expect(arch.movaNumExperts == 64)
        #expect(arch.movaTopKExperts == 4)
        #expect(arch.normGroups == 2)
        #expect(arch.attentionGate == "softplus")
        #expect(arch.attnOutputGate)
        #expect(arch.routerScaling == 2.5)
        #expect(arch.routerHasBias)
        #expect(arch.movaRouterScaling == 2.5)
        #expect(arch.movaRouterHasBias)
        #expect(arch.routerNormTopK)
        #expect(arch.denseLayerIndices == [0, 1, 2])
    }

    @Test("Refuses a rope width outside the head")
    func refusesBadRope() throws {
        #expect(throws: (any Error).self) {
            try Self.load(Self.configJSON(overrides: ["rope_head_dim": 256]))
        }
    }
}
