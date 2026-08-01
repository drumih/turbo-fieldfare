import Darwin
import Foundation
import Testing
@testable import TurboFieldfareRepackCore

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
        #expect(arch.attentionScale == 0.125)   // 64^-0.5
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
        #expect(arch.attentionScale == 0.0625) // 256^-0.5
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

    @Test func productionQwenConfigMismatchIsRejected() throws {
        let root = temporaryRoot("qwen-prod-bad")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let configPath = (root as NSString).appendingPathComponent("config.json")
        try writeProductionConfig(to: configPath, mutate: { tc in
            tc["num_attention_heads"] = 8
        })

        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: configPath)
        }
    }

    @Test func qwenSourceFingerprintIsKnown() {
        #expect(SourceFingerprint.modelID(forIndexSha256:
            "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea")
            == "qwen3.6-35b-a3b-4bit")
        #expect(SourceFingerprint.modelID(forIndexSha256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13")
            == "mlx-community/gemma-4-26b-a4b-it-4bit")
    }

    @Test func qwenClassificationBucketsNames() {
        let f = RepackModelFamily.qwen36
        #expect(RepackPlanner.classify(
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight",
            numLayers: 4, family: f)
            == .routedExpert(role: "gate", layer: 1))
        #expect(RepackPlanner.classify(
            "language_model.model.layers.2.mlp.switch_mlp.down_proj.weight",
            numLayers: 4, family: f)
            == .routedExpert(role: "down", layer: 2))
        // The shared expert and its gate stay resident.
        #expect(RepackPlanner.classify(
            "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
            numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "language_model.model.layers.0.mlp.shared_expert_gate.weight",
            numLayers: 4, family: f) == .lmResident)
        // Untied head and the DeltaNet bundle are resident.
        #expect(RepackPlanner.classify(
            "language_model.lm_head.weight", numLayers: 4, family: f) == .lmResident)
        #expect(RepackPlanner.classify(
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            numLayers: 4, family: f) == .lmResident)
        // Vision is excluded; unknown prefixes stay unknown.
        #expect(RepackPlanner.classify(
            "vision_tower.blocks.0.norm1.weight", numLayers: 4, family: f)
            == .excludedMultimodal)
        #expect(RepackPlanner.classify(
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
        for suffix in ["linear_attn.conv1d.weight", "linear_attn.A_log",
                       "linear_attn.dt_bias", "linear_attn.norm.weight"] {
            let entry = try #require(plan.resident.entries.first {
                $0.name == "language_model.model.layers.0." + suffix
            })
            #expect(entry.dtype == 1)
            #expect(entry.quantSpec == nil)
            #expect(entry.sourceScales == nil)
            #expect(entry.sourceBiases == nil)
        }
        // The fused qkv projection is a quantized U32 entry with companions.
        let qkv = try #require(plan.resident.entries.first {
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
            #expect(order == [
                "gate.weights", "gate.scales", "gate.biases",
                "up.weights", "up.scales", "up.biases",
                "down.weights", "down.scales", "down.biases",
            ])
        }

        // Vision-tower tensors are dropped, not planned.
        #expect(plan.excludedMultimodalTensorNames.sorted() == [
            "vision_tower.blocks.0.norm1.weight",
            "vision_tower.patch_embed.proj.weight",
        ])
        #expect(!names.contains { $0.hasPrefix("vision_tower.") })
    }

    @Test func gemmaManifestOmitsFamilyFields() throws {
        let snapshotDir = temporaryRoot("gemma-manifest")
        let outputDir = temporaryRoot("gemma-manifest-out")
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

        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "unknown/snapshot",
            sourceSnapshotHash: "sha256:0",
            files: [],
            expertsPerLayer: 2,
            numLayers: arch.numLayers,
            expertStride: 16_384,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 4, attention: 4, router: 8,
                sharedExpert: 8, routedExpert: 4))
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let archDict = obj["arch"] as! [String: Any]
        for key in ["family", "attnOutputGate", "attentionScale",
                    "embeddingScaledBySqrtHidden", "routerScaled",
                    "ffnSandwichNorms", "sharedExpertGated", "ropeNeoxSubdim",
                    "linearNumKHeads", "linearNumVHeads", "linearKeyHeadDim",
                    "linearValueHeadDim", "linearConvKernelSize"] {
            #expect(archDict[key] == nil, "gemma manifest must omit \(key)")
        }
    }

    // MARK: - Helpers

    private func writeProductionConfig(
        to path: String,
        mutate: (inout [String: Any]) -> Void) throws {
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
                "partial_rotary_factor": 0.25
            ],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "attn_output_gate": true,
            "tie_word_embeddings": false,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        ]
        mutate(&tc)
        let config: [String: Any] = [
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "quantization": ["bits": 4, "group_size": 64, "mode": "affine"],
            "text_config": tc
        ]
        let data = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-qwen-plan-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }

    private func parseHeader(path: String) throws -> Safetensors.Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var headerSize: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSize) {
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: $0.baseAddress!,
                count: 8,
                offset: 0)
        }
        headerSize = UInt64(littleEndian: headerSize)
        var headerData = Data(count: Int(headerSize))
        try headerData.withUnsafeMutableBytes {
            try Posix.preadAll(
                fd: fd,
                path: path,
                buf: $0.baseAddress!,
                count: $0.count,
                offset: 8)
        }
        return try Safetensors.parseHeaderBytes(
            path: path,
            fileSize: try Posix.fileSize(fd: fd, path: path),
            headerBytes: headerData)
    }
}
