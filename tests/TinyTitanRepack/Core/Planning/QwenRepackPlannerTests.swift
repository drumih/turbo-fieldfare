import Darwin
import Foundation
import Testing

@testable import TinyTitanRepackCore
@testable import TinyTitanCore

@Suite
struct QwenRepackPlannerTests {

    @Test func qwenArchInfoLoadsFromSyntheticConfig() throws {
        let snapshotDir = temporaryRoot("qwen-arch")
        defer { try? FileManager.default.removeItem(atPath: snapshotDir) }
        _ = try SyntheticSnapshot.buildQwen(at: snapshotDir)

        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))

        #expect(arch.family == .qwen36)
        #expect(arch.hiddenSize == 128)
        #expect(arch.numLayers == 4)
        #expect(arch.fullAttentionLayerMask == [2, 2, 2, 1])
        #expect(arch.numExperts == 2)
        #expect(arch.topKExperts == 2)
        #expect(arch.moeIntermediateSize == 64)
        #expect(arch.intermediateSize == 64)
        #expect(arch.tieWordEmbeddings == false)
        #expect(arch.attentionKEqV == false)
        #expect(arch.hiddenActivation == "silu")
        #expect(arch.ropeTheta == 10_000_000.0)
        #expect(arch.fullRopeTheta == 10_000_000.0)
        #expect(arch.partialRotaryFactor == 0.25)
        #expect(arch.finalLogitSoftcap == 0.0)
        #expect(arch.attnOutputGate == true)
        #expect(arch.attentionScale == 0.125)  // 64^-0.5
        #expect(arch.embeddingScaledBySqrtHidden == false)
        #expect(arch.routerScaled == false)
        #expect(arch.ffnSandwichNorms == false)
        #expect(arch.sharedExpertGated == true)
        #expect(arch.ropeNeoxSubdim == true)
        #expect(arch.linearNumKHeads == 2)
        #expect(arch.linearNumVHeads == 4)
        #expect(arch.linearKeyHeadDim == 32)
        #expect(arch.linearValueHeadDim == 32)
        #expect(arch.linearConvKernelSize == 4)
    }

    @Test func productionQwenConfigParsesAndCrossChecks() throws {
        let root = temporaryRoot("qwen-prod")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { _ in })

        let arch = try ArchInfo.load(configPath: configPath)
        #expect(arch.family == .qwen36)
        #expect(arch.hiddenSize == 2048)
        #expect(arch.numLayers == 40)
        #expect(arch.vocabSize == 248_320)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 8)
        #expect(arch.attentionScale == 0.0625)  // 256^-0.5
        #expect(arch.linearNumKHeads == 16)
        #expect(arch.linearNumVHeads == 32)
        #expect(arch.linearKeyHeadDim == 128)
        #expect(arch.linearValueHeadDim == 128)
        #expect(arch.linearConvKernelSize == 4)
        #expect(arch.fullAttentionLayerMask.count == 40)
        #expect(arch.fullAttentionLayerMask.filter { $0 == 1 }.count == 10)
        for (i, v) in arch.fullAttentionLayerMask.enumerated() {
            #expect(v == ((i + 1) % 4 == 0 ? 1 : 2))
        }
    }

    @Test func productionOrnithConfigParsesThroughSharedArchitecture() throws {
        let root = temporaryRoot("ornith-prod")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(
            to: configPath,
            mutate: { tc in
                tc["pad_token_id"] = 248_044
                tc["router_aux_loss_coef"] = 0.0
                tc["use_cache"] = false
                tc["rope_parameters"] = [
                    "rope_theta": 10_000_000.0,
                    "type": "default",
                    "partial_rotary_factor": 0.25,
                ]
            })

        let arch = try ArchInfo.load(configPath: configPath)
        #expect(arch.family == .qwen36)
        #expect(arch.hiddenSize == 2_048)
        #expect(arch.numLayers == 40)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 8)
        #expect(arch.fullAttentionLayerMask.filter { $0 == 1 }.count == 10)
    }

    @Test func productionQwenConfigMismatchIsRejected() throws {
        let root = temporaryRoot("qwen-prod-bad")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(
            to: configPath,
            mutate: { tc in
                tc["num_attention_heads"] = 8
            })

        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: configPath)
        }
    }

    @Test func qwenSourceFingerprintIsKnown() {
        #expect(
            SourceFingerprint.modelID(
                forIndexSha256:
                    "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea")
                == "qwen3.6-35b-a3b-4bit")
        // 6-bit was withdrawn, so its fingerprint no longer resolves. Asserting
        // nil keeps the removal honest: a source quietly reappearing would fail
        // here rather than shipping an unsupported quantization.
        #expect(
            SourceFingerprint.modelID(
                forIndexSha256:
                    "eaea194dfb961e6a5215dcc6e4dd42d0df6efe8d8686161f2dd00634e0ef43fb")
                == nil)
        #expect(
            SourceFingerprint.modelID(
                forIndexSha256:
                    "3db12edeebeb65cab9a6eeb63cd74be4e0c74139a75f672701290b98230501cf")
                == "qwen3.6-35b-a3b-8bit")
        #expect(
            SourceFingerprint.modelID(
                forIndexSha256:
                    "00e220ddb21ceeb6290a3a1161f97339c553f3d27fc4319900a96edb5cfae74c")
                == "qwen3.6-35b-a3b-mtp-4bit")
        #expect(
            SourceFingerprint.modelID(
                forIndexSha256:
                    "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36")
                == "ornith-1.5-35b-a3b-4bit")
        #expect(
            SourceFingerprint.modelID(
                forIndexSha256:
                    "83c641a791aa957df7d280eef1b0c8faf7a2ec9b19dd3355fb13abae8ae0ed15")
                == "ornith-1.5-35b-a3b-8bit")
    }

    @Test func ornithSourcesArePinnedAndSixBitRemainsUnsupported() {
        #expect(SupportedModelSource.default == .ornith15_8bit)
        #expect(SupportedModelSource.named("ornith15") == .ornith15)
        #expect(SupportedModelSource.named("ornith15-8bit") == .ornith15_8bit)
        #expect(SupportedModelSource.named("ornith15-6bit") == nil)
        #expect(
            SupportedModelSource.ornith15.repoID
                == "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit")
        #expect(
            SupportedModelSource.ornith15_8bit.repoID
                == "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit")
        #expect(
            Set(SupportedModelSource.all.map(\.name)).count
                == SupportedModelSource.all.count)
        #expect(
            Set(SupportedModelSource.all.map(\.modelID)).count
                == SupportedModelSource.all.count)
        // Two sources may share an index digest: Qwen3.8-Flash-Next ships its
        // MTP draft inside the target's own repository, so the pair differ by
        // which namespace they claim rather than by where they come from.
        // What must stay unique is the pairing of digest and namespace, which
        // is what decides what an install actually contains.
        let pins = SupportedModelSource.all.map {
            "\($0.sourceIndexSHA256)|\($0.installsDraftHead)"
        }
        #expect(Set(pins).count == SupportedModelSource.all.count)
        #expect(
            SupportedModelSource.qwen38flashMTP.repoID
                == SupportedModelSource.qwen38flash.repoID)
        #expect(SupportedModelSource.qwen38flashMTP.installsDraftHead)
        #expect(!SupportedModelSource.qwen38flash.installsDraftHead)
    }

    @Test func qwen38DraftClassificationClaimsOnlyTheMTPNamespace() {
        let family = RepackModelFamily.qwen38flashMTP
        // Only `mtp.*` belongs to this install; the target's own namespace is
        // a different install from the same repository.
        #expect(
            RepackPlanner.classify(
                "model.language_model.layers.0.mlp.gate.weight",
                numLayers: 1, family: family)
                == .excludedSidecar)
        #expect(
            RepackPlanner.classify(
                "mtp.fc_hidden.weight",
                numLayers: 1, family: family) == .lmResident)
        #expect(
            RepackPlanner.classify(
                "mtp.pre_fc_norm_hidden",
                numLayers: 1, family: family) == .lmResident)
        // The draft's own 512 experts are streamed like any other layer's.
        #expect(
            RepackPlanner.classify(
                "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
                numLayers: 1, family: family)
                == .routedExpert(role: "gate", layer: 0))
    }

    @Test func mtpClassificationKeepsOnlyAdapterAndOneDecoderLayer() {
        let family = RepackModelFamily.qwen36MTP
        #expect(
            RepackPlanner.classify(
                "fc.weight", numLayers: 1,
                family: family) == .lmResident)
        #expect(
            RepackPlanner.classify(
                "pre_fc_norm_hidden.weight", numLayers: 1,
                family: family) == .lmResident)
        #expect(
            RepackPlanner.classify(
                "norm.weight", numLayers: 1,
                family: family) == .lmResident)
        #expect(
            RepackPlanner.classify(
                "layers.0.mlp.switch_mlp.gate_proj.weight",
                numLayers: 1, family: family)
                == .routedExpert(role: "gate", layer: 0))
        #expect(
            RepackPlanner.classify(
                "embed_tokens.weight", numLayers: 1,
                family: family) == .unknown)
        #expect(
            RepackPlanner.classify(
                "lm_head.weight", numLayers: 1,
                family: family) == .unknown)
    }

    @Test func qwenClassificationBucketsNames() {
        let f = RepackModelFamily.qwen36
        #expect(
            RepackPlanner.classify(
                "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight",
                numLayers: 4, family: f)
                == .routedExpert(role: "gate", layer: 1))
        #expect(
            RepackPlanner.classify(
                "language_model.model.layers.2.mlp.switch_mlp.down_proj.weight",
                numLayers: 4, family: f)
                == .routedExpert(role: "down", layer: 2))
        // The shared expert and its gate stay resident.
        #expect(
            RepackPlanner.classify(
                "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
                numLayers: 4, family: f) == .lmResident)
        #expect(
            RepackPlanner.classify(
                "language_model.model.layers.0.mlp.shared_expert_gate.weight",
                numLayers: 4, family: f) == .lmResident)
        // Untied head and the DeltaNet bundle are resident.
        #expect(
            RepackPlanner.classify(
                "language_model.lm_head.weight", numLayers: 4, family: f) == .lmResident)
        #expect(
            RepackPlanner.classify(
                "language_model.model.layers.0.linear_attn.conv1d.weight",
                numLayers: 4, family: f) == .lmResident)
        // Vision is excluded; unknown prefixes stay unknown.
        #expect(
            RepackPlanner.classify(
                "vision_tower.blocks.0.norm1.weight", numLayers: 4, family: f)
                == .excludedMultimodal)
        #expect(
            RepackPlanner.classify(
                "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                numLayers: 4, family: f) == .unknown)
    }

    @Test func qwenPlanOrdersResidentsAndSlicesExperts() throws {
        let snapshotDir = temporaryRoot("qwen-plan")
        let outputDir = temporaryRoot("qwen-plan-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildQwen(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)

        let plan = try RepackPlanner.plan(
            meta: metadata,
            arch: arch,
            shardHeaders: [header],
            outputDir: outputDir)

        let names = plan.resident.entries.map(\.name)
        // Embedding first; final norm then the untied lm_head last.
        #expect(names.first == "language_model.model.embed_tokens.weight")
        #expect(names.last == "language_model.lm_head.weight")
        #expect(names.dropLast().last == "language_model.model.norm.weight")

        // Layer 0 is a linear-attention layer: DeltaNet bundle, then router,
        // shared-expert gate + MLP, then the two layer norms.
        let expectedLayer0 = [
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.conv1d.weight",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            "mlp.gate.weight",
            "mlp.shared_expert_gate.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ].map { "language_model.model.layers.0." + $0 }
        let layer0 = names.filter { $0.contains(".layers.0.") }
        #expect(layer0 == expectedLayer0)

        // Layer 3 is the full-attention layer.
        let expectedLayer3Prefix = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ].map { "language_model.model.layers.3." + $0 }
        let layer3 = names.filter { $0.contains(".layers.3.") }
        #expect(Array(layer3.prefix(6)) == expectedLayer3Prefix)

        // conv1d/A_log/dt_bias/norm stay unquantized BF16 without companions.
        for suffix in [
            "linear_attn.conv1d.weight", "linear_attn.A_log",
            "linear_attn.dt_bias", "linear_attn.norm.weight",
        ] {
            let entry = try #require(
                plan.resident.entries.first {
                    $0.name == "language_model.model.layers.0." + suffix
                })
            #expect(entry.dtype == 1)
            #expect(entry.quantSpec == nil)
            #expect(entry.sourceScales == nil)
            #expect(entry.sourceBiases == nil)
        }
        // The fused qkv projection is a quantized U32 entry with companions.
        let qkv = try #require(
            plan.resident.entries.first {
                $0.name == "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"
            })
        #expect(qkv.dtype == 0)
        #expect(qkv.quantSpec?.bits == 4)
        #expect(qkv.sourceScales != nil)
        #expect(qkv.sourceBiases != nil)

        // Every layer slices two experts into the fixed 9-slice blob.
        #expect(plan.layers.count == 4)
        for lp in plan.layers {
            #expect(lp.expertsPerLayer == 2)
            #expect(lp.subTensors.count == 9)
            #expect(lp.expertStride % 16_384 == 0)
            let order = lp.subTensors.map { "\($0.role).\($0.component)" }
            #expect(
                order == [
                    "gate.weights", "gate.scales", "gate.biases",
                    "up.weights", "up.scales", "up.biases",
                    "down.weights", "down.scales", "down.biases",
                ])
        }

        // Vision-tower tensors are dropped, not planned.
        #expect(
            plan.excludedMultimodalTensorNames.sorted() == [
                "vision_tower.blocks.0.norm1.weight",
                "vision_tower.patch_embed.proj.weight",
            ])
        #expect(!names.contains { $0.hasPrefix("vision_tower.") })
    }

    /// The manifest writer always emits the Qwen family-extension fields
    /// (family, attnOutputGate, attentionScale, linear-attention dims, ...).
    /// Assert they are present with the values derived from the synthetic
    /// snapshot's config.json — the same contract RemoteQwenInstallTests
    /// checks on the real installed manifest.
    @Test func qwenManifestEmitsFamilyExtensionFields() throws {
        let snapshotDir = temporaryRoot("qwen-manifest")
        let outputDir = temporaryRoot("qwen-manifest-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.build(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        let arch = try ArchInfo.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseHeader(path: snapshot.shardPath)
        let plan = try RepackPlanner.plan(
            meta: metadata, arch: arch, shardHeaders: [header], outputDir: outputDir)

        let data = try SSDAIJSON.encodeManifest(
            plan: plan,
            modelID: "unknown/snapshot",
            sourceSnapshotHash: "sha256:0",
            files: [],
            expertsPerLayer: 2,
            numLayers: arch.numLayers,
            expertStride: 16_384,
            bitWidths: SSDAIJSON.QuantBitWidths(
                embedding: 4, attention: 4, router: 8,
                sharedExpert: 8, routedExpert: 4))
        let obj = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        // The rename's other half: reads accept the legacy magic, writes emit
        // the current one. Pinned here so the on-disk name cannot drift back.
        #expect(obj["magic"] as? String == "SSDAI")
        let archDict = try #require(obj["arch"] as? [String: Any])
        // Family extension fields are always present for the Qwen families
        // and must round-trip the values `ArchInfo.load` derived.
        #expect(archDict["family"] as? String == "qwen36")
        #expect(archDict["attnOutputGate"] as? Bool == true)
        #expect(archDict["attentionScale"] as? Double == 0.125)
        #expect(archDict["embeddingScaledBySqrtHidden"] as? Bool == false)
        #expect(archDict["routerScaled"] as? Bool == false)
        #expect(archDict["ffnSandwichNorms"] as? Bool == false)
        #expect(archDict["sharedExpertGated"] as? Bool == true)
        #expect(archDict["ropeNeoxSubdim"] as? Bool == true)
        #expect(archDict["linearNumKHeads"] as? Int == 2)
        #expect(archDict["linearNumVHeads"] as? Int == 4)
        #expect(archDict["linearKeyHeadDim"] as? Int == 32)
        #expect(archDict["linearValueHeadDim"] as? Int == 32)
        #expect(archDict["linearConvKernelSize"] as? Int == 4)
        #expect(archDict["fullAttentionLayerMask"] as? [Int] == [2, 2, 2, 1])
    }

    // MARK: - Helpers

    private func writeProductionConfig(
        to path: String,
        mutate: (inout [String: Any]) -> Void
    ) throws {
        var layerTypes: [String] = []
        for i in 0..<40 {
            layerTypes.append((i + 1) % 4 == 0 ? "full_attention" : "linear_attention")
        }
        var tc: [String: Any] = [
            "hidden_size": 2048,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248_320,
            "num_hidden_layers": 40,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "layer_types": layerTypes,
            "rope_parameters": [
                "rope_theta": 10_000_000.0,
                "rope_type": "default",
                "partial_rotary_factor": 0.25,
            ],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "attn_output_gate": true,
            "tie_word_embeddings": false,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
        ]
        mutate(&tc)
        let config: [String: Any] = [
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "quantization": ["bits": 4, "group_size": 64, "mode": "affine"],
            "text_config": tc,
        ]
        let data = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("tinytitan-qwen-plan-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }

    private func parseHeader(path: String) throws -> Safetensors.Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var headerSize: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSize) { buffer in
            let base = try #require(buffer.baseAddress)
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: base,
                count: 8,
                offset: 0)
        }
        headerSize = UInt64(littleEndian: headerSize)
        var headerData = Data(count: Int(headerSize))
        try headerData.withUnsafeMutableBytes { buffer in
            let base = try #require(buffer.baseAddress)
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: base,
                count: buffer.count,
                offset: 8)
        }
        return try Safetensors.parseHeaderBytes(
            path: path,
            fileSize: try Posix.fileSize(fd: fd, path: path),
            headerBytes: headerData)
    }
}
