import Foundation
import Testing

@testable import TinyTitanRepackCore

/// `ArchInfo` parsing for the Qwen 4.0 Flash family (`qwen4-omni-120b-think`).
///
/// The checkpoint is not published yet, so the fixture below is the 3.8
/// preview's verified geometry under the 4.0 model_type. What is pinned here is
/// the 4.0 *path*: it must dispatch to its own family, read every field from the
/// config, gate the production shape, and never route `qwen4_exp` into it.
@Suite("Qwen4Flash ArchInfo")
struct Qwen4FlashArchInfoTests {
    private static func configJSON(
        modelType: String = "qwen4_omni_moe",
        overrides: [String: Any] = [:],
        rootExtras: [String: Any] = [:]
    ) throws -> Data {
        var layerTypes: [String] = []
        for i in 0..<48 {
            layerTypes.append((i + 1) % 4 == 0 ? "full_attention" : "linear_attention")
        }
        var text: [String: Any] = [
            "hidden_size": 2560, "num_hidden_layers": 48,
            "num_attention_heads": 24, "num_key_value_heads": 2,
            "head_dim": 256, "vocab_size": 248_320,
            "num_experts": 512, "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "hc_count": 4, "hc_lowrank": 320,
            "indexer_n_heads": 4, "indexer_kv_heads": 1,
            "indexer_head_dim": 128, "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "ple_layer_ids": [2], "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4, "ngram_size": 3,
            "ngram_vocab_size_base": 20_000_000, "heads_per_ngram": 8,
            "make_ngram_vocab_size_divisible_by": 128,
            "hidden_act": "silu", "output_gate_type": "sigmoid",
            "tie_word_embeddings": false, "layer_types": layerTypes,
            "rope_parameters": [
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
            ],
        ]
        for (k, v) in overrides { text[k] = v }
        var root: [String: Any] = [
            "model_type": modelType,
            "text_config": text,
            "quantization": ["group_size": 64, "bits": 4, "mode": "affine"],
        ]
        for (k, v) in rootExtras { root[k] = v }
        return try JSONSerialization.data(withJSONObject: root)
    }

    private static func load(_ data: Data) throws -> ArchInfo {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4flash-\(UUID().uuidString).json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        return try ArchInfo.load(configPath: url.path)
    }

    @Test("Parses the expected geometry into its own family")
    func parsesGeometry() throws {
        let a = try Self.load(Self.configJSON())
        #expect(a.family == .qwen4flash)
        #expect(a.hiddenSize == 2560)
        #expect(a.numLayers == 48)
        #expect(a.numExperts == 512)
        #expect(a.topKExperts == 10)
        #expect(a.moeIntermediateSize == 640)
        #expect(a.numHeads == 24)
        #expect(a.numKVHeads == 2)
        #expect(a.headDim == 256)
        #expect(a.linearNumKHeads == 16)
        #expect(a.linearNumVHeads == 48)
        #expect(a.vocabSize == 248_320)
        #expect(a.attnOutputGate)
        #expect(a.routerNormTopK)
        #expect(a.hcCount == 4)
        #expect(a.hcLowRank == 320)
        #expect(a.indexerBudget == 2048)
        #expect(a.pleLayerIndices == [1])
    }

    @Test("Every expected 4.0 model_type lands on qwen4flash")
    func acceptsTheExpectedSpellings() throws {
        for spelling in [
            "qwen4_omni_moe", "qwen4_omni", "qwen4_flash",
            "qwen4_omni_moe_text", "qwen4_omni_text", "qwen4_flash_text",
        ] {
            let a = try Self.load(Self.configJSON(modelType: spelling))
            #expect(a.family == .qwen4flash, "\(spelling)")
        }
    }

    @Test("The 3.8 preview keeps its own family")
    func qwen4ExpStaysOnQwen38() throws {
        let a = try Self.load(Self.configJSON(modelType: "qwen4_exp"))
        #expect(a.family == .qwen38flash)
        #expect(a.family != .qwen4flash)
    }

    @Test("A config claiming the 4.0 production shape must match it")
    func contractRejectsMismatch() {
        #expect(throws: (any Error).self) {
            _ = try Self.load(
                Self.configJSON(overrides: ["num_experts_per_tok": 8]))
        }
    }

    @Test("A different width is a different shape, not a contract failure")
    func smallerWidthIsNotTheProductionContract() throws {
        // The cross-check only gates the production shape; a toy or a
        // differently-sized 4.0 checkpoint must still parse, because the
        // published 120B may not be the preview's 125B geometry.
        let a = try Self.load(
            Self.configJSON(overrides: [
                "hidden_size": 1024, "num_hidden_layers": 8,
                "num_experts": 32, "num_experts_per_tok": 4,
                "moe_intermediate_size": 256,
                "shared_expert_intermediate_size": 256,
                "num_attention_heads": 8, "head_dim": 128,
                "hc_count": 2, "hc_lowrank": 64,
                "indexer_n_heads": 2, "indexer_head_dim": 64,
                "ple_embed_dim": 1024,
            ]))
        #expect(a.family == .qwen4flash)
        #expect(a.hiddenSize == 1024)
        #expect(a.numLayers == 8)
    }

    @Test("An out-of-range ple_layer_ids entry is rejected")
    func rejectsBadPLEIndex() {
        #expect(throws: (any Error).self) {
            _ = try Self.load(Self.configJSON(overrides: ["ple_layer_ids": [0]]))
        }
    }

    @Test("A draft head derives from a 4.0 target and keeps its family")
    func draftHeadDerives() throws {
        let base = try Self.load(Self.configJSON())
        let draft = try ArchInfo.qwen4FlashMTP(
            from: base,
            configPath: "/tmp/qwen4-config.json")
        #expect(draft.family == .qwen4flashMTP)
        #expect(draft.numLayers == 1)
        #expect(draft.fullAttentionLayerMask == [1])
        #expect(draft.pleLayerIndices.isEmpty)
        #expect(draft.hcCount == 4)
        #expect(draft.vocabSize == base.vocabSize)
    }

    @Test("The draft entry point refuses a non-Flash target")
    func draftHeadRefusesOtherFamilies() throws {
        let base = try Self.load(Self.configJSON(modelType: "qwen4_exp"))
        #expect(throws: (any Error).self) {
            _ = try ArchInfo.qwen4FlashMTP(
                from: base, configPath: "/tmp/qwen4-config.json")
        }
    }
}
