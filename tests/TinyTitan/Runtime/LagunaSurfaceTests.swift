import Foundation
import Testing

@testable import TinyTitan
import TinyTitanCore

/// The Laguna preset and its surface rules: the geometry a manifest is checked
/// against, the sampling defaults from `generation_config.json`, and the
/// per-layer head counts the probe sizes its buffers from.
@Suite("Laguna surface")
struct LagunaSurfaceTests {
    @Test func presetStatesTheCheckpointGeometry() {
        let config = ArchConfig.lagunaS21
        #expect(config.family == .laguna)
        #expect(config.hiddenSize == 3_072)
        #expect(config.numLayers == 48)
        #expect(config.numExperts == 256)
        #expect(config.topKExperts == 10)
        #expect(config.denseLayerIndices == [0])
        #expect(config.routerNormTopK)
        #expect(config.routerHasBias)
        #expect(config.routerScaling == 2.5)
        #expect(config.attentionGate == .softplus)
        #expect(config.sharedExpertGated == false)
        #expect(config.laguna.enabled)
        #expect(config.laguna.qkNorm)
        #expect(config.laguna.gatePerHead)
        #expect(config.laguna.headsPerLayer.count == 48)
        #expect(config.laguna.headsPerLayer[0] == 48)
        #expect(config.laguna.headsPerLayer[1] == 72)
        #expect(config.fullAttentionLayerMask.filter { $0 == 1 }.count == 12)
    }

    @Test func tokenizerNamesMatchTheCheckpointSpelling() {
        #expect(LagunaTensors.attnGate(3) == "model.layers.3.self_attn.g_proj.weight")
        #expect(
            LagunaTensors.moeRouterBias(3)
                == "model.layers.3.mlp.experts.e_score_correction_bias")
        #expect(
            LagunaTensors.sharedGate(3)
                == "model.layers.3.mlp.shared_expert.gate_proj.weight")
        #expect(LagunaTensors.qNorm(3) == "model.layers.3.self_attn.q_norm.weight")
        #expect(LagunaTensors.eosTokens == [2, 24])
    }

    @Test func samplingFollowsTheGenerationConfig() {
        let sampling = GenerationDefaults.forFamily(.laguna)
        #expect(sampling.temperature == 1.0)
        #expect(sampling.topK == 20)
        #expect(sampling.topP == 1.0)
    }

    @Test func reasoningIsABooleanThinkingSwitch() throws {
        guard
            case .binaryThinking = ModelFamily.laguna.reasoningControl
        else {
            Issue.record("Laguna should use the template's boolean enable_thinking")
            return
        }
        #expect(ModelFamily.laguna.supportsThinkingOff)
    }
}
