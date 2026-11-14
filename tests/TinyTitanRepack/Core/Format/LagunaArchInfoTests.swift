import Foundation
import Testing

@testable import TinyTitanRepackCore
import TinyTitanCore

/// `ArchInfo` parsing for `model_type: laguna`, checked against the pinned
/// checkpoint's config values (poolside/Laguna-S-2.1 at 0f573140).
@Suite("Laguna ArchInfo")
struct LagunaArchInfoTests {
    private static func configJSON(overrides: [String: Any] = [:]) throws -> Data {
        var text: [String: Any] = [
            "model_type": "laguna",
            "hidden_size": 3_072,
            "intermediate_size": 12_288,
            "moe_intermediate_size": 1_024,
            "shared_expert_intermediate_size": 1_024,
            "num_hidden_layers": 48,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 100_352,
            "num_experts": 256,
            "num_experts_per_tok": 10,
            "norm_topk_prob": true,
            "moe_routed_scaling_factor": 2.5,
            "mlp_only_layers": [0],
            "decoder_sparse_step": 1,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "gating": "per-head",
            "sliding_window": 512,
            "layer_types": (0..<48).map { $0 % 4 == 0 ? "full_attention" : "sliding_attention" },
            "num_attention_heads_per_layer": (0..<48).map { $0 % 4 == 0 ? 48 : 72 },
            "rope_parameters": [
                "full_attention": [
                    "rope_theta": 500_000.0,
                    "rope_type": "yarn",
                    "factor": 128.0,
                    "original_max_position_embeddings": 8_192,
                    "beta_slow": 1.0,
                    "beta_fast": 32.0,
                    "attention_factor": 1.4852030263919618,
                    "partial_rotary_factor": 0.5,
                ],
                "sliding_attention": [
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                    "partial_rotary_factor": 1.0,
                ],
            ],
        ]
        for (key, value) in overrides { text[key] = value }
        return try JSONSerialization.data(withJSONObject: text)
    }

    private static func load(_ data: Data) throws -> ArchInfo {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("laguna-\(UUID().uuidString).json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        return try ArchInfo.load(configPath: url.path)
    }

    @Test("Parses the pinned checkpoint's geometry")
    func parsesGeometry() throws {
        let arch = try Self.load(Self.configJSON())
        #expect(arch.family == .laguna)
        #expect(arch.hiddenSize == 3_072)
        #expect(arch.intermediateSize == 12_288)
        #expect(arch.moeIntermediateSize == 1_024)
        #expect(arch.numLayers == 48)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 10)
        #expect(arch.numHeads == 48)
        #expect(arch.numKVHeads == 8)
        #expect(arch.headDim == 128)
        #expect(arch.vocabSize == 100_352)
        #expect(!arch.tieWordEmbeddings)
        #expect(arch.slidingWindow == 512)
        #expect(arch.ropeTheta == 10_000)
        #expect(arch.fullRopeTheta == 500_000)
        #expect(arch.partialRotaryFactor == 0.5)
        // 12 global layers, every fourth starting at 0.
        #expect(arch.fullAttentionLayerMask == (0..<48).map { $0 % 4 == 0 ? 1 : 0 })
        #expect(arch.denseLayerIndices == [0])
        #expect(arch.routerNormTopK)
        #expect(arch.routerHasBias)
        #expect(arch.routerScaling == 2.5)
        #expect(arch.attentionGate == "softplus")
        #expect(arch.sharedExpertGated == false)
    }

    @Test("A config outside the supported contract is refused by field")
    func refusesWrongContract() throws {
        let parsed = try JSONSerialization.jsonObject(with: Self.configJSON())
        var text = try #require(parsed as? [String: Any])
        text["moe_routed_scaling_factor"] = 1.0
        let data = try JSONSerialization.data(withJSONObject: text)
        var message = ""
        do {
            _ = try Self.load(data)
        } catch {
            message = String(describing: error)
        }
        #expect(message.contains("moe_routed_scaling_factor 2.5"), "\(message)")
    }
}
