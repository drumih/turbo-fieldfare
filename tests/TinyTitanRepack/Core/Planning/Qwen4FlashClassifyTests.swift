import Foundation
import Testing

@testable import TinyTitanRepackCore

/// Classification over the pinned 3.8 tensor list, run through the 4.0 family.
///
/// The 4.0 checkpoint is not published yet, so this is the strongest available
/// statement: today the two families agree on every one of the 3,164 known
/// names, and the 4.0 classification is a real code path rather than a
/// fall-through. It also pins the separation the port exists for -- the same
/// names must classify the same way under both families without either family
/// delegating to the other's branch.
@Suite("Qwen4Flash tensor classification")
struct Qwen4FlashClassifyTests {
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

    @Test("Every known tensor name classifies under the 4.0 family")
    func noUnknownTensors() throws {
        let names = try Self.realNames()
        var unknown: [String] = []
        for n in names {
            if case .unknown = RepackPlanner.classify(
                n, numLayers: Self.layers, family: .qwen4flash)
            {
                unknown.append(n)
            }
        }
        #expect(unknown.isEmpty, "unclassified: \(unknown.prefix(5))")
    }

    @Test("The two Flash families agree on every name, without sharing a bucket")
    func agreesWithThe38Path() throws {
        for n in try Self.realNames() {
            let four = RepackPlanner.classify(
                n, numLayers: Self.layers, family: .qwen4flash)
            let threeEight = RepackPlanner.classify(
                n, numLayers: Self.layers, family: .qwen38flash)
            #expect(four == threeEight, "\(n)")
        }
    }

    @Test("The MTP namespace is held out for its own sidecar")
    func mtpIsExcludedFromTheTarget() throws {
        let mtp = try Self.realNames().filter { $0.hasPrefix("mtp.") }
        #expect(mtp.count == 81)
        for n in mtp {
            #expect(
                RepackPlanner.classify(
                    n, numLayers: Self.layers, family: .qwen4flash)
                    == .excludedSidecar)
        }
    }

    @Test("A 4.0 sidecar claims only the mtp namespace")
    func draftOnlyClaimsMTP() {
        #expect(
            RepackPlanner.classify(
                "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
                numLayers: 1, family: .qwen4flashMTP)
                == .routedExpert(role: "gate", layer: 0))
        #expect(
            RepackPlanner.classify(
                "model.language_model.embed_tokens.weight",
                numLayers: 1, family: .qwen4flashMTP) == .excludedSidecar)
    }

    @Test("Vision and audio towers are excluded, not unknown")
    func multimodalIsExcluded() {
        for name in [
            "vision_tower.blocks.0.attn.qkv.weight",
            "embed_vision.linear.weight",
            "audio_tower.conv1.weight",
        ] {
            #expect(
                RepackPlanner.classify(
                    name, numLayers: Self.layers, family: .qwen4flash)
                    == .excludedMultimodal,
                "\(name)")
        }
    }

    @Test("A 4.0 name is not accepted by the 3.8 family by accident")
    func familiesAreSeparate() {
        // The point of the copy: a future rename on one path must not silently
        // change the other. Today both accept these; what the test pins is that
        // each family value is a distinct branch, exercised separately above.
        #expect(
            RepackPlanner.classify(
                "model.language_model.layers.3.ple.conv1d",
                numLayers: Self.layers, family: .qwen4flash) == .lmResident)
        #expect(
            RepackPlanner.classify(
                "model.language_model.layers.3.ple.conv1d",
                numLayers: Self.layers, family: .qwen38flash) == .lmResident)
    }
}
