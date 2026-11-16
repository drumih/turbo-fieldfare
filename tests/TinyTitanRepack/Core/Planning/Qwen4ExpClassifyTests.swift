import Foundation
import Testing

@testable import TinyTitanRepackCore
@testable import TinyTitanCore

/// Classification over the complete tensor-name list of the pinned
/// Qwen3.8-Flash-Next checkpoint (all 3,164 names from its
/// `model.safetensors.index.json`). A synthetic sample would only prove the
/// classifier handles what I thought to write down; the real list is what
/// catches the family member nobody anticipated.
@Suite("Qwen4Exp tensor classification")
struct Qwen4ExpClassifyTests {
    static let layers = 48

    static func realNames() throws -> [String] {
        let url = try #require(
            Bundle.module.url(
                forResource: "qwen38_tensor_names", withExtension: "txt",
                subdirectory: "Support")
                ?? Bundle.module.url(
                    forResource: "qwen38_tensor_names",
                    withExtension: "txt"))
        return try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n").map(String.init)
    }

    @Test("Every real tensor name classifies; none is unknown")
    func noUnknownTensors() throws {
        let names = try Self.realNames()
        #expect(names.count == 3164)
        var unknown: [String] = []
        for n in names {
            if case .unknown = RepackPlanner.classify(
                n, numLayers: Self.layers, family: .qwen38flash)
            {
                unknown.append(n)
            }
        }
        #expect(unknown.isEmpty, "unclassified: \(unknown.prefix(5))")
    }

    @Test("Routed experts are exactly the switch_mlp tensors, 3 roles x 48 layers")
    func routedExpertCoverage() throws {
        var byLayer: [Int: Set<String>] = [:]
        for n in try Self.realNames() {
            if case .routedExpert(let role, let layer) = RepackPlanner.classify(
                n, numLayers: Self.layers, family: .qwen38flash)
            {
                #expect(n.contains(".mlp.switch_mlp."))
                #expect(!n.hasPrefix("mtp."), "MTP experts must not land in the target")
                byLayer[layer, default: []].insert(role)
            }
        }
        #expect(byLayer.count == 48)
        for (layer, roles) in byLayer {
            #expect(roles == ["gate", "up", "down"], "layer \(layer) roles \(roles)")
        }
    }

    @Test("The MTP draft is held out for its own sidecar install")
    func mtpIsExcluded() throws {
        let names = try Self.realNames()
        let mtp = names.filter { $0.hasPrefix("mtp.") }
        #expect(mtp.count == 81)
        for n in mtp {
            #expect(
                RepackPlanner.classify(
                    n, numLayers: Self.layers,
                    family: .qwen38flash)
                    == .excludedSidecar)
        }
    }

    @Test("lm_head sits at the top level in this family and stays resident")
    func topLevelLMHead() throws {
        for suffix in ["weight", "scales", "biases"] {
            #expect(
                RepackPlanner.classify(
                    "lm_head.\(suffix)",
                    numLayers: Self.layers,
                    family: .qwen38flash) == .lmResident)
        }
    }

    @Test("The new subsystems land in the resident bucket, not unknown")
    func newSubsystemsAreResident() {
        let probes = [
            "model.language_model.layers.3.attn_hyper_connection.hc_norm",
            "model.language_model.layers.3.mlp_hyper_connection.block_inject_weight.weight",
            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
            "model.language_model.layers.3.self_attn.indexer.index_q_proj.weight",
            "model.language_model.layers.1.ple.conv1d",
            "model.language_model.embed_tokens.weight",
        ]
        for p in probes {
            #expect(
                RepackPlanner.classify(
                    p, numLayers: Self.layers,
                    family: .qwen38flash) == .lmResident,
                "\(p)")
        }
    }

    @Test("qwen36 classification is unchanged by the new family")
    func qwen36Unaffected() {
        #expect(
            RepackPlanner.classify(
                "language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight",
                numLayers: 40, family: .qwen36) == .routedExpert(role: "gate", layer: 3))
        #expect(
            RepackPlanner.classify(
                "language_model.model.embed_tokens.weight",
                numLayers: 40, family: .qwen36) == .lmResident)
        // A qwen4_exp-style name must NOT be accepted by the qwen36 family.
        #expect(
            RepackPlanner.classify(
                "model.language_model.embed_tokens.weight",
                numLayers: 40, family: .qwen36) == .unknown)
    }
}
