import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2ConfigTests {
    private func json() throws -> Data {
        let object: [String: Any] = [
            "text_config": [
                "hidden_size": 2304,
                "num_hidden_layers": 24,
                "vocab_size": 201_024,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "n_routed_experts": 128,
                "num_experts_per_tok": 8,
                "intermediate_size": 1152,
                "model_max_length": 1_048_576,
                "sliding_window_size": 1024,
                "local_layer_ids": Array(0..<20),
                "rel_extent": 1024,
                "d_rel": 16,
                "sconv_kernel_size": 4,
                "n_layers_per_attn_res_block": 6,
                "loop_config": ["core_start": 6, "core_end": 18, "num_passes": 3],
            ],
            "vision_config": [
                "num_hidden_layers": 23, "hidden_size": 1024, "native_patch_size": 20,
            ],
            "mtp_config": [
                "num_hidden_layers": 3, "hidden_size": 2304, "block_size": 8,
            ],
        ]
        return try JSONSerialization.data(withJSONObject: object)
    }

    @Test func readsTheTextShape() throws {
        let config = try Inkling2Config.parse(try json())
        #expect(config.text.layerCount == 24)
        #expect(config.text.experts == 128)
        #expect(config.text.loop.passes == 3)
        #expect(config.text.localLayers.count == 20)
        #expect(config.vision?.layerCount == 23)
        #expect(config.mtp?.layerCount == 3)
    }

    @Test func refusesTheWrongLayerCount() throws {
        var object = try #require(JSONSerialization.jsonObject(with: try json()) as? [String: Any])
        var text = try #require(object["text_config"] as? [String: Any])
        text["num_hidden_layers"] = 23
        object["text_config"] = text
        let broken = try JSONSerialization.data(withJSONObject: object)
        #expect(throws: Inkling2Config.Error.self) {
            try Inkling2Config.parse(broken)
        }
    }
}
