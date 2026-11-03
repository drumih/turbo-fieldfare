import Foundation
import Testing

@testable import TinyTitanRepackCore

/// The `manifest.json` writer's two contract rules: which architectures carry
/// the extension-geometry block, and what happens when two `quant` keys want
/// the same slot.
///
/// Both were silent before. The block was gated on one family name, so the
/// Qwen3.8-Flash-Next draft — which runs on the target's hyper-connections and
/// indexer — was written with no geometry for the reader to cross-check. And a
/// per-tensor width whose stem landed on a slot name was dropped, which is the
/// exact failure the per-tensor entries exist to fix: the reader falls back to
/// the slot, the strides still divide, and the model answers fluently and
/// wrongly.
@Suite("SSDAIJSON manifest writer")
struct SSDAIJSONTests {

    /// The pinned Qwen3.8-Flash-Next geometry, as `ArchInfo.load` reads it.
    private static func flashNextArch(modelType: String = "qwen4_exp") throws -> ArchInfo {
        let text: [String: Any] = [
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
            "tie_word_embeddings": false,
            "layer_types": (0..<48).map {
                ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention"
            },
            "rope_parameters": [
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
            ],
        ]
        let root: [String: Any] = [
            "model_type": modelType,
            "text_config": text,
            "quantization": ["group_size": 64, "bits": 4, "mode": "affine"],
        ]
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdaijson-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: root).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        return try ArchInfo.load(configPath: url.path)
    }

    /// A plan with no payload: `encodeManifest` reads the arch, the base quant
    /// mode and the resident entries, and nothing else.
    private static func plan(arch: ArchInfo, entries: [ResidentEntry] = []) -> RepackPlan {
        RepackPlan(
            arch: arch, baseMode: "affine", baseGroupSize: 64,
            resident: ResidentFilePlan(
                path: "model_weights.bin", entries: entries,
                stringTable: [], stringTableOffsets: [],
                indexSize: 16_384, residentSize: 0),
            layers: [], matchedModelID: nil,
            excludedMultimodalTensorNames: [], passthroughFiles: [])
    }

    private static func manifest(
        _ plan: RepackPlan,
        bitWidths: SSDAIJSON.QuantBitWidths = SSDAIJSON.QuantBitWidths(
            embedding: 4, attention: 4, router: 8, sharedExpert: 8, routedExpert: 4)
    ) throws -> [String: Any] {
        let data = try SSDAIJSON.encodeManifest(
            plan: plan, modelID: "synthetic", sourceSnapshotHash: "sha256:0",
            files: [], expertsPerLayer: 0, numLayers: plan.arch.numLayers,
            expertStride: 16_384, bitWidths: bitWidths)
        return try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func archDict(_ manifest: [String: Any]) throws -> [String: Any] {
        try #require(manifest["arch"] as? [String: Any])
    }

    // MARK: - The extension-geometry block

    @Test("A Qwen 4.0 Flash manifest declares its own family and geometry")
    func qwen4TargetCarriesItsOwnFamily() throws {
        let arch = try Self.flashNextArch(modelType: "qwen4_omni_moe")
        #expect(arch.family == .qwen4flash)
        let manifest = try Self.manifest(Self.plan(arch: arch))
        let encoded = try Self.archDict(manifest)
        #expect(encoded["family"] as? String == "qwen4flash")
        #expect(encoded["hcCount"] as? Int == 4)
        #expect(encoded["indexerBudget"] as? Int == 2048)
        #expect(encoded["pleLayerIndices"] as? [Int] == [1])
    }

    @Test("A Qwen 4.0 draft is written as its own family with no n-gram block")
    func qwen4DraftCarriesItsOwnFamily() throws {
        let base = try Self.flashNextArch(modelType: "qwen4_omni_moe")
        let draft = try ArchInfo.qwen4FlashMTP(from: base, configPath: "config.json")
        let encoded = try Self.archDict(Self.manifest(Self.plan(arch: draft)))
        #expect(encoded["family"] as? String == "qwen4flash_mtp")
        #expect(encoded["hcCount"] as? Int == 4)
        #expect(encoded["pleLayerIndices"] as? [Int] == [])
    }

    @Test("A Qwen3.8-Flash-Next manifest carries the extension geometry")
    func targetCarriesExtensionGeometry() throws {
        let manifest = try Self.manifest(Self.plan(arch: Self.flashNextArch()))
        let arch = try Self.archDict(manifest)
        #expect(arch["hcCount"] as? Int == 4)
        #expect(arch["indexerBudget"] as? Int == 2048)
        #expect(arch["pleLayerIndices"] as? [Int] == [1])
    }

    /// The finding: the block used to be written for `.qwen38flash` by name, so
    /// the draft that shares its target's hyper-connections and indexer carried
    /// none, and the reader validates only what it is given.
    @Test("The MTP draft carries the hyper-connection and indexer geometry it runs on")
    func draftCarriesExtensionGeometry() throws {
        let base = try Self.flashNextArch()
        let draft = try ArchInfo.qwen38FlashNextMTP(from: base, configPath: "config.json")
        #expect(draft.family == .qwen38flashMTP)
        #expect(draft.declaresExtensionGeometry)
        let arch = try Self.archDict(Self.manifest(Self.plan(arch: draft)))
        #expect(arch["hcCount"] as? Int == 4)
        #expect(arch["hcLowRank"] as? Int == 320)
        #expect(arch["indexerBudget"] as? Int == 2048)
        // The draft has no n-gram block, so it declares no n-gram geometry: the
        // runtime's own draft contract is `ple: .none`, and inheriting the
        // target's dimensions would make the reader refuse a correct install.
        #expect(arch["pleLayerIndices"] as? [Int] == [])
        #expect(arch["pleEmbedDim"] as? Int == 0)
        #expect(arch["pleNgramSize"] as? Int == 0)
    }

    @Test("An architecture without the geometry writes no block, so its bytes do not move")
    func plainFamilyWritesNoExtensionGeometry() throws {
        let manifest = try Self.manifest(Self.plan(arch: Self.qwen36Arch()))
        let arch = try Self.archDict(manifest)
        for key in [
            "hcCount", "hcLowRank", "indexerBudget", "pleLayerIndices",
            "pleEmbedDim", "routerNormTopK", "quantGroupSize",
        ] {
            #expect(arch[key] == nil)
        }
        // The family fields it does carry are unaffected.
        #expect(arch["family"] as? String == "qwen36")
        #expect(arch["linearNumKHeads"] as? Int == 16)
    }

    private static func qwen36Arch() -> ArchInfo {
        ArchInfo(
            hiddenSize: 2048, intermediateSize: 512, moeIntermediateSize: 512,
            numHeads: 16, numKVHeads: 2, numFullKVHeads: 2,
            headDim: 256, fullHeadDim: 256, vocabSize: 248_320, slidingWindow: 0,
            finalLogitSoftcap: 0.0, ropeTheta: 10_000_000.0, fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25, numLayers: 4, numExperts: 256, topKExperts: 8,
            tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: [2, 2, 2, 1], hiddenActivation: "silu",
            family: .qwen36, attnOutputGate: true, attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false, routerScaled: false,
            ffnSandwichNorms: false, sharedExpertGated: true, ropeNeoxSubdim: true,
            linearNumKHeads: 16, linearNumVHeads: 32,
            linearKeyHeadDim: 128, linearValueHeadDim: 128, linearConvKernelSize: 4)
    }

    @Test("A written manifest is version 1.1, the minor that requires the block")
    func writtenVersion() throws {
        let manifest = try Self.manifest(Self.plan(arch: Self.qwen36Arch()))
        #expect(manifest["versionMajor"] as? Int == 1)
        #expect(manifest["versionMinor"] as? Int == 1)
    }

    // MARK: - Per-tensor widths against the slots

    @Test("A stem that lands on a slot name is refused, not dropped")
    func slotCollisionThrows() {
        // The key is the whole dotted name minus `.weight`, so the collision is
        // a tensor literally called `attention.weight`. Implausible for a real
        // checkpoint, and the reason the guard may not choose silently.
        #expect(throws: RepackError.self) {
            _ = try SSDAIJSON.perTensorWidths(
                quantized: [(name: "attention.weight", bits: 8)],
                slotNames: ["embedding", "attention", "router", "sharedExpert", "routedExpert"])
        }
    }

    @Test("Two tensors of different widths behind one stem are refused")
    func conflictingDuplicateStemThrows() {
        #expect(throws: RepackError.self) {
            _ = try SSDAIJSON.perTensorWidths(
                quantized: [
                    (name: "model.layers.0.k_proj.weight", bits: 8),
                    (name: "model.layers.0.k_proj", bits: 4),
                ],
                slotNames: ["attention"])
        }
    }

    @Test("The same width behind one stem is written once")
    func agreeingDuplicateStemIsIdempotent() throws {
        let widths = try SSDAIJSON.perTensorWidths(
            quantized: [
                (name: "model.layers.0.k_proj.weight", bits: 8),
                (name: "model.layers.0.k_proj", bits: 8),
            ],
            slotNames: ["attention"])
        #expect(widths.map(\.stem) == ["model.layers.0.k_proj"])
        #expect(widths.map(\.bits) == [8])
    }

    @Test("An ordinary resident tensor stems to its manifest key")
    func stemsDropTheWeightSuffix() throws {
        let widths = try SSDAIJSON.perTensorWidths(
            quantized: [
                (name: "language_model.model.layers.11.self_attn.k_proj.weight", bits: 8),
                (name: "lm_head", bits: 4),
            ],
            slotNames: ["attention"])
        #expect(
            widths.map(\.stem) == [
                "language_model.model.layers.11.self_attn.k_proj", "lm_head",
            ])
    }

    /// The guard has to be reachable from the writer, not only from the helper:
    /// a plan whose tensor stems onto a slot must fail the install rather than
    /// emit a manifest that quietly loses a width.
    @Test("encodeManifest fails on a resident tensor that would overwrite a slot")
    func encodeManifestRefusesSlotCollision() {
        #expect(throws: RepackError.self) {
            _ = try Self.manifest(
                Self.plan(
                    arch: Self.qwen36Arch(),
                    entries: [Self.entry(name: "attention.weight", bits: 8)]))
        }
    }

    /// K2's config as `ArchInfo.loadK2Horizon` reads it.
    private static func k2Arch() throws -> ArchInfo {
        let text: [String: Any] = [
            "hidden_size": 2_560, "intermediate_size": 6_144,
            "moe_intermediate_size": 768, "num_hidden_layers": 48,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "head_dim": 128, "rope_head_dim": 128, "vocab_size": 250_624,
            "num_experts": 100, "num_experts_per_tok": 8,
            "mova_num_experts": 64, "mova_num_experts_per_tok": 4,
            "layernorm_num_groups": 2, "attention_gate_func": "softplus",
            "router_scaling_factor": 2.5, "moe_gate_bias": true,
            "norm_topk_prob": true, "mlp_only_layers": [0, 1, 2],
            "tie_word_embeddings": false, "hidden_act": "silu",
            "rope_parameters": ["rope_theta": 10_000_000.0],
        ]
        let root: [String: Any] = [
            "model_type": "k2_horizon", "text_config": text,
        ]
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdaijson-k2-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: root).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        return try ArchInfo.load(configPath: url.path)
    }

    /// The K2 router normalizes its selected sigmoid scores. The writer used to
    /// emit every K2 field except this one, and a reader that defaults it to
    /// false (the Qwen value) would take the unnormalized scores as the
    /// mixture weights.
    @Test("A K2 manifest writes the router normalization and the gate function")
    func k2ManifestCarriesItsLayerShape() throws {
        let manifest = try Self.manifest(Self.plan(arch: Self.k2Arch()))
        let arch = try Self.archDict(manifest)
        #expect(arch["routerNormTopK"] as? Bool == true)
        #expect(arch["attentionGate"] as? String == "softplus")
        #expect(arch["normGroups"] as? Int == 2)
        #expect(arch["denseLayerIndices"] as? [Int] == [0, 1, 2])
        #expect(arch["movaNumExperts"] as? Int == 64)
    }

    /// A quantized resident entry with no payload behind it: `quantObject`
    /// reads the name and the width, and nothing else.
    private static func entry(name: String, bits: Int) -> ResidentEntry {
        let source = SourceTensor(
            name: name, shardPath: "model.safetensors", dtype: .u32,
            shape: [4, 4], absoluteOffset: 0, sizeBytes: 16)
        return ResidentEntry(
            name: name, dtype: 0, logicalShape4: [4, 4, 0, 0],
            fileOffset: 0, sizeBytes: 16,
            scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
            quantSpec: QuantSpec(bits: bits),
            sourceWeight: source, sourceScales: nil, sourceBiases: nil)
    }
}
