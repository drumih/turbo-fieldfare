import Foundation
import Testing
import TinyTitanFormat

@testable import TinyTitanRepackCore

/// The Inkling 2 checkpoint install plan (TT-121 phase 1). The config and name
/// tests are model-free; the plan tests read the real checkpoint's headers
/// (`TINYTITAN_INKLING2_CHECKPOINT` names its directory) and copy no bytes.
@Suite struct Inkling2CheckpointPlannerTests {
    private static var checkpoint: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_CHECKPOINT"]
    }

    /// The checkpoint's `text_config`, trimmed to the keys the reader uses.
    static let config = """
        {"model_type": "inkling2_loop", "text_config": {
          "hidden_size": 2304, "intermediate_size": 1152, "num_hidden_layers": 24,
          "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
          "vocab_size": 201024, "unpadded_vocab_size": 200058, "sliding_window_size": 1024,
          "final_logit_softcapping": 30.0, "n_routed_experts": 128, "num_experts_per_tok": 8,
          "n_shared_experts": 0,
          "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
          "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64,
                          "is_varloop": false},
          "n_layers_per_attn_res_block": 6, "d_rel": 16, "rel_extent": 1024,
          "rel_l2_clip_threshold": 8.0, "sconv_kernel_size": 4, "rms_norm_eps": 1e-06,
          "qk_l2_clip_threshold": 22.627416997969522, "log_scaling_n_floor": 8192,
          "log_scaling_alpha": 0.1, "route_scale": 8.0, "mlp_gate_softcap": 4.0,
          "mlp_up_softcap": 25.0, "model_max_length": 1048576}}
        """

    /// The checkpoint's config describes the same model the GGUF's metadata
    /// does: the layer split, the loop and the constants the variant reads.
    @Test func theConfigReadsAsTheGGUFDescribesTheModel() throws {
        let arch = try Inkling2CheckpointArch.archInfo(
            config: Data(Self.config.utf8), tiedEmbeddings: false)
        #expect(arch.family == .inkling2)
        #expect(arch.numLayers == 24)
        #expect(arch.numExperts == 128 && arch.topKExperts == 8)
        #expect(arch.vocabSize == 201_024)
        #expect(arch.quantGroupSize == 16)
        let global = arch.fullAttentionLayerMask.indices.filter {
            arch.fullAttentionLayerMask[$0] == 1
        }
        #expect(global == [5, 11, 17, 23])
        let geometry = try #require(arch.variantGeometry)
        #expect(geometry.scalars["unrolledLayers"] == 48)
        #expect(geometry.scalars["situBeta"] == 4)
        #expect(geometry.scalars["situLinearBeta"] == 25)
        #expect(geometry.scalars["expertWeightsScale"] == 8)
        #expect(geometry.scalars["vocabulary"] == 200_058)
    }

    @Test func aVariableLoopOrASharedExpertIsRefused() {
        for (from, to) in [
            ("\"is_varloop\": false", "\"is_varloop\": true"),
            ("\"n_shared_experts\": 0", "\"n_shared_experts\": 1"),
        ] {
            let config = Self.config.replacingOccurrences(of: from, with: to)
            #expect(throws: RepackError.self) {
                _ = try Inkling2CheckpointArch.archInfo(
                    config: Data(config.utf8), tiedEmbeddings: false)
            }
        }
    }

    @Test func expertTensorNamesMapToBlobKeys() {
        func key(_ name: String) -> String? {
            Inkling2CheckpointPlanner.expertKey(name).map { "\($0.layer) \($0.key)" }
        }
        #expect(key("model.llm.layers.3.mlp.experts.w13_weight") == "3 gate_up")
        #expect(key("model.llm.layers.20.mlp.experts.w2_weight_scale") == "20 down_scale")
        #expect(key("model.llm.layers.0.mlp.experts.w13_weight_amax") == "0 gate_up_amax")
        // The per-layer denominator and the router stay resident.
        #expect(key("model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator") == nil)
        #expect(key("model.llm.layers.0.mlp.gate.weight") == nil)
    }

    @Test(.enabled(if: checkpoint != nil)) func thePlanMatchesTheModel() throws {
        let source = try RemoteStreamingRepacker.loadInkling2Checkpoint(
            directory: try #require(Self.checkpoint))
        let plan = try Inkling2CheckpointPlanner.plan(source: source, outputDir: "/tmp/ink-plan")
        // Layers 0-19: NVFP4 values (half a byte each), E4M3 scales per 16, and
        // two F32 amax values; page-rounded.
        let values: UInt64 = 2_654_208 + 1_327_104  // 2304 x 2304 and 2304 x 1152, half a byte
        let scales: UInt64 = 331_776 + 165_888  // one byte per 16 values
        let nvfp4 = values + scales
        #expect(
            plan.layers[0].expertStride
                == Inkling2GGUFPlanner.roundUp(nvfp4 + 64 + 4, to: 16_384))
        #expect(plan.layers[0].subTensors.map(\.dtype) == [6, 6, 200, 200, 3, 3])
        #expect(plan.layers[20].subTensors.map(\.dtype) == [7, 7, 201, 201])
        #expect(plan.layers[20].subTensors.first?.bitsForWeights == 8)
        #expect(plan.manifestExpertStride == plan.layers[20].expertStride)
        #expect(plan.excludedMultimodalTensorNames.count == 422)
        #expect(plan.resident.entries.contains { $0.name == "mtp.markov_head.markov_w1.weight" })
        #expect(!plan.resident.entries.contains { $0.name.hasPrefix("model.visual.") })
    }

    /// Every source byte but the vision tower is copied exactly once.
    @Test(.enabled(if: checkpoint != nil)) func theCopyPlanAndLayoutAreValid() throws {
        let source = try RemoteStreamingRepacker.loadInkling2Checkpoint(
            directory: try #require(Self.checkpoint))
        let plan = try Inkling2CheckpointPlanner.plan(source: source, outputDir: "/tmp/ink-plan")
        let copies = try RangeCopyPlanner.plan(repackPlan: plan, rangeChunkBytes: 64 << 20)
        let sourceBytes = source.shards.flatMap(\.header.tensors)
            .filter { !$0.name.hasPrefix(Inkling2CheckpointPlanner.visualPrefix) }
            .reduce(UInt64(0)) { $0 + $1.sizeBytes }
        #expect(copies.scalarCopies.reduce(UInt64(0)) { $0 + $1.size } == sourceBytes)
        let layout = try SSDAIJSON.encodeLayout(plan: plan, expertStride: plan.manifestExpertStride)
        _ = try SSDAIPackedExpertsLayoutCodec.decode(layout)
    }
}
