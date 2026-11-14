import Foundation
import Testing

@testable import TinyTitanRepackCore
import TinyTitanCore

/// `ArchInfo` parsing for `model_type: qwen4_exp`, checked against the real
/// pinned checkpoint's config values (RockTalk/Qwen3.8-Flash-Next-MLX-4bit at
/// 478474da). Every expectation here was read from that config, not assumed.
@Suite("Qwen4Exp ArchInfo")
struct Qwen4ExpArchInfoTests {
    /// The load-bearing subset of the real config. Layer types alternate
    /// 3 linear : 1 full, which is what `full_attention_interval: 4` means.
    private static func configJSON(overrides: [String: Any] = [:]) throws -> Data {
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
        let root: [String: Any] = [
            "model_type": "qwen4_exp",
            "text_config": text,
            "quantization": ["group_size": 64, "bits": 4, "mode": "affine"],
        ]
        return try JSONSerialization.data(withJSONObject: root)
    }

    private static func load(_ data: Data) throws -> ArchInfo {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4exp-\(UUID().uuidString).json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        return try ArchInfo.load(configPath: url.path)
    }

    @Test("Parses the pinned checkpoint's geometry")
    func parsesGeometry() throws {
        let a = try Self.load(Self.configJSON())
        #expect(a.family == .qwen38flash)
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
    }

    @Test("Reads the source group size rather than assuming 64")
    func readsGroupSize() throws {
        let a = try Self.load(Self.configJSON())
        #expect(a.quantGroupSize == 64)
    }

    @Test("Layer mask is 36 gated-DeltaNet and 12 full-attention")
    func layerMask() throws {
        let a = try Self.load(Self.configJSON())
        #expect(a.fullAttentionLayerMask.count == 48)
        #expect(a.fullAttentionLayerMask.filter { $0 == 1 }.count == 12)
        #expect(a.fullAttentionLayerMask.filter { $0 == 2 }.count == 36)
        // Index 3 is the first full-attention layer (config is 3 linear : 1 full).
        #expect(a.fullAttentionLayerMask[3] == 1)
        #expect(a.fullAttentionLayerMask[0] == 2)
    }

    @Test("ple_layer_ids is converted from 1-based to 0-based")
    func pleLayerIndicesAreZeroBased() throws {
        let a = try Self.load(Self.configJSON())
        #expect(a.pleLayerIndices == [1])
        #expect(a.pleNgramSize == 3)
        #expect(a.pleHeadsPerNgram == 8)
        #expect(a.pleVocabSizeBase == 20_000_000)
    }

    @Test("Hyper-connection and indexer geometry is carried through")
    func newSubsystems() throws {
        let a = try Self.load(Self.configJSON())
        #expect(a.hcCount == 4)
        #expect(a.hcLowRank == 320)
        #expect(a.indexerNumHeads == 4)
        #expect(a.indexerNumKVHeads == 1)
        #expect(a.indexerHeadDim == 128)
        #expect(a.indexerBudget == 2048)
        #expect(a.indexerCompressRatio == 4)
    }

    @Test("A config claiming the production shape must match the contract")
    func contractRejectsMismatch() {
        // A different model wearing the same model_type: top-8 instead of 10.
        #expect(throws: (any Error).self) {
            _ = try Self.load(
                Self.configJSON(
                    overrides: ["num_experts_per_tok": 8]))
        }
    }

    @Test("An out-of-range ple_layer_ids entry is rejected, not clamped")
    func rejectsBadPLEIndex() {
        #expect(throws: (any Error).self) {
            _ = try Self.load(Self.configJSON(overrides: ["ple_layer_ids": [0]]))
        }
    }
}

/// The pinned checkpoint's own `config.json` on this host, or `nil` when no copy
/// is present.
///
/// A gate, not a `guard ... else { return }` inside the test body: an early
/// return there is reported as a *passed* test even though it asserted
/// nothing, so the run would claim coverage it did not take. `.enabled(if:)`
/// records the skip instead — the same shape `LibraryContractTests` and the
/// MPP kernel suite use.
private func pinnedQwen38Config() -> String? {
    guard
        let path = ProcessInfo.processInfo.environment["TINYTITAN_QWEN38_CONFIG"],
        FileManager.default.fileExists(atPath: path)
    else { return nil }
    return path
}

/// Parses the real pinned checkpoint config, if a copy is present. Skipped
/// (and *recorded* as skipped) when absent so the suite stays hermetic; run with
/// the file to confirm the loader against the actual artifact rather than a
/// reconstruction.
@Suite("Qwen4Exp real config", .enabled(if: pinnedQwen38Config() != nil))
struct Qwen4ExpRealConfigTests {
    @Test("The pinned checkpoint's own config.json parses and matches")
    func realConfig() throws {
        let path = try #require(pinnedQwen38Config())
        let a = try ArchInfo.load(configPath: path)
        #expect(a.family == .qwen38flash)
        #expect(a.numLayers == 48)
        #expect(a.numExperts == 512)
        #expect(a.topKExperts == 10)
        #expect(a.quantGroupSize == 64)
        #expect(a.pleLayerIndices == [1])
        #expect(a.fullAttentionLayerMask.filter { $0 == 1 }.count == 12)
    }
}
