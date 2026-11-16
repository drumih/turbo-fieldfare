import Foundation
import Testing

@testable import TinyTitanRepackCore
@testable import TinyTitanCore

/// `ArchInfo` parsing for Ornith-1.5-122B (`model_type: qwen3_5_moe` at hidden
/// 3072 / 48 layers). The 3072/48 contract is checked field by field, and a
/// config that claims that shape but differs in one expert width is refused
/// rather than repacked into a payload the runtime would run with the wrong
/// geometry.
@Suite("Ornith 122B ArchInfo")
struct Ornith122ArchInfoTests {
    private static func configJSON(overrides: [String: Any] = [:]) throws -> Data {
        var layerTypes: [String] = []
        for i in 0..<48 {
            layerTypes.append((i + 1) % 4 == 0 ? "full_attention" : "linear_attention")
        }
        var text: [String: Any] = [
            "hidden_size": 3072, "num_hidden_layers": 48,
            "num_attention_heads": 32, "num_key_value_heads": 2,
            "head_dim": 256, "vocab_size": 248_320,
            "num_experts": 256, "num_experts_per_tok": 8,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 1024,
            "linear_num_key_heads": 16, "linear_num_value_heads": 64,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "attn_output_gate": true,
            "hidden_act": "silu", "tie_word_embeddings": false,
            "layer_types": layerTypes,
            "rope_parameters": [
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
            ],
        ]
        for (k, v) in overrides { text[k] = v }
        let root: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "text_config": text,
            "quantization": ["group_size": 64, "bits": 4, "mode": "affine"],
        ]
        return try JSONSerialization.data(withJSONObject: root)
    }

    private static func load(_ data: Data) throws -> ArchInfo {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ornith122-\(UUID().uuidString).json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        return try ArchInfo.load(configPath: url.path)
    }

    @Test("Parses the 122B geometry")
    func parsesGeometry() throws {
        let a = try Self.load(Self.configJSON())
        #expect(a.family == .qwen36)
        #expect(a.hiddenSize == 3072)
        #expect(a.numLayers == 48)
        #expect(a.numExperts == 256)
        #expect(a.topKExperts == 8)
        #expect(a.intermediateSize == 1024)
        #expect(a.moeIntermediateSize == 1024)
        #expect(a.numHeads == 32)
        #expect(a.numKVHeads == 2)
        #expect(a.headDim == 256)
        #expect(a.linearNumKHeads == 16)
        #expect(a.linearNumVHeads == 64)
        #expect(a.linearValueHeadDim == 128)
        #expect(a.fullAttentionLayerMask.filter { $0 == 1 }.count == 12)
        #expect(a.attnOutputGate)
        #expect(a.sharedExpertGated)
    }

    @Test("Refuses a 3072/48 config with the wrong expert width")
    func refusesWrongGeometry() throws {
        let data = try Self.configJSON(overrides: ["moe_intermediate_size": 1536])
        #expect(throws: (any Error).self) {
            _ = try Self.load(data)
        }
    }
}
