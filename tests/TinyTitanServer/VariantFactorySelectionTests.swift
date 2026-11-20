import Foundation
import Testing
import TinyTitanCore
import TinyTitanVariants

@testable import TinyTitan
@testable import TinyTitanLegacyEngine
@testable import TinyTitanServerCore

/// TT-097 Phase 3 Task 10: the server's factory selection. Model-free by
/// construction — the selection is decided from the manifest identity a
/// catalog entry carries, and the factories are compared as selection
/// outcomes, never by running a model. The actual routing flips are Task 11+,
/// each gated on a real install.
@Suite struct VariantFactorySelectionTests {

    @Test("the shipped ornith 4-bit variant selects its dedicated factory")
    func ornith4BitSelectsTheVariant() {
        #expect(
            EngineFactories.gpuSelection(
                family: .qwen36,
                manifestModelID: "ornith-1.5-35b-a3b",
                routedExpertWeightBits: 4) == .ornith35B4Bit)
    }

    @Test("every other GPU identity stays on the 6.x engine")
    func everythingElseStaysLegacy() {
        // The 8-bit ornith variant is not shipped.
        #expect(
            EngineFactories.gpuSelection(
                family: .qwen36, manifestModelID: "ornith-1.5-35b-a3b",
                routedExpertWeightBits: 8) == .legacy)
        // A different qwen36 install is not the ornith variant.
        #expect(
            EngineFactories.gpuSelection(
                family: .qwen36, manifestModelID: "some-private-checkpoint",
                routedExpertWeightBits: 4) == .legacy)
        // Other families have no shipped GPU variant.
        #expect(
            EngineFactories.gpuSelection(
                family: .qwen38flash, manifestModelID: "qwen3.8-flash-next",
                routedExpertWeightBits: 4) == .legacy)
        #expect(
            EngineFactories.gpuSelection(
                family: .laguna, manifestModelID: "laguna-s-2-1",
                routedExpertWeightBits: 4) == .legacy)
    }

    @Test("the ornith variant factory composes the production planner stages")
    func variantFactoryIsComplete() throws {
        let variant = EngineFactories.variantGPUOrnith()
        // The planner and architecture stages are the family-generic
        // production ones; the engine stages are the variant's own. A default
        // EngineFactory refuses with a sentinel ("no load planner" / "no
        // architecture resolver"), so probing a bogus directory distinguishes
        // composed stages from defaults without running a model: a composed
        // stage gets far enough to fail on the read itself.
        let sentinels = ["no load planner", "no architecture resolver"]
        func isSentinel(_ error: Error) -> Bool {
            sentinels.contains { "\(error).".contains($0) }
        }
        do {
            _ = try variant.resolveArchitecture(URL(fileURLWithPath: "/nonexistent"))
            Issue.record("a bogus directory must fail, not resolve")
        } catch {
            #expect(!isSentinel(error), "the variant factory carries the default resolver")
        }
        do {
            _ = try variant.prepareLoad(
                SessionBuildOptions(
                    modelDirectory: URL(fileURLWithPath: "/nonexistent"),
                    context: nil, maxContext: 64, slots: 1,
                    promptCacheMaximumEntries: 0, promptCacheMemoryLimitBytes: 0,
                    promptCacheDiskDirectory: nil, promptCacheDiskLimitBytes: 0,
                    prefillChunkTokens: nil, kvCachePrecision: .int8,
                    hostKVWindowTokens: nil, ropeScalingMode: .none,
                    thinkingMode: .off, reasoningEffort: nil,
                    expertCacheSlots: nil, expertCacheBudgetBytes: nil,
                    mtpModelDirectory: nil, mtpMemoryMiB: 0,
                    integrityPolicy: nil, rdAdvicePolicy: nil, forceLogitsHead: true))
            Issue.record("a bogus directory must fail, not plan")
        } catch {
            #expect(!isSentinel(error), "the variant factory carries the default planner")
        }
    }

    @Test("the router dispatch hands ornith its variant factory, others 6.x")
    func routerDispatchSelectsPerIdentity() {
        // With no Metal context the engine stage is where the two factories
        // differ, and the error names the engine: a model-free probe of
        // which factory the dispatch returned.
        func engineStageError(_ factory: EngineFactory) -> String {
            do {
                _ = try factory.buildModel(
                    URL(fileURLWithPath: "/nonexistent"),
                    ArchConfig.qwen36_35B_A3B,
                    nil)
                return "no error"
            } catch {
                return "\(error)"
            }
        }

        let ornith = EngineFactories.gpuFactory(
            family: .qwen36, manifestModelID: "ornith-1.5-35b-a3b",
            routedExpertWeightBits: 4)
        #expect(
            engineStageError(ornith).contains("ornith variant"),
            "the shipped ornith identity must dispatch to the variant factory")

        let legacy = EngineFactories.gpuFactory(
            family: .qwen36, manifestModelID: "some-private-checkpoint",
            routedExpertWeightBits: 4)
        #expect(
            engineStageError(legacy).contains("6.x"),
            "every other identity stays on the 6.x engine")

        // The dense family routes to the shared dense-GPU engine, whose
        // engine stage names itself (Task 13's flip).
        let dense = EngineFactories.gpuFactory(
            family: .qwen35Dense, manifestModelID: "qwen3.5-4b",
            routedExpertWeightBits: 4)
        #expect(
            engineStageError(dense).contains("dense-GPU"),
            "the dense identity must dispatch to the dense-GPU factory")
    }

    @Test("the selection table pins every dense identity to the dense-GPU engine")
    func denseIdentitiesSelectTheDenseEngine() {
        // Bare and width-suffixed dense installs, both quants — the three
        // widths the registry ships rows for.
        for (modelID, bits) in [
            ("qwen3.5-2b", 4), ("qwen3.5-2b", 8),
            ("qwen3.5-4b", 4), ("qwen3.5-4b", 8),
            ("qwen3.5-9b", 4), ("qwen3.5-9b", 8),
        ] {
            #expect(
                EngineFactories.gpuSelection(
                    family: .qwen35Dense, manifestModelID: modelID,
                    routedExpertWeightBits: bits) == .qwen35DenseGPU,
                "\(modelID) at \(bits)-bit must select the dense-GPU engine")
        }
        // A dense install is dense whatever width the manifest declares —
        // the selection is by family, not by width.
        #expect(
            EngineFactories.gpuSelection(
                family: .qwen35Dense, manifestModelID: "qwen3.5-9b",
                routedExpertWeightBits: 8) == .qwen35DenseGPU)
    }

    @Test("the single-model directory dispatch reads the install identity")
    func directoryDispatchSelectsPerIdentity() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("variant-selection-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let arch: [String: Any] = [
            "hiddenSize": 64, "ffnIntermediate": 128, "moeIntermediateSize": 128,
            "numHeads": 4, "numKVHeads": 2, "numFullKVHeads": 2, "headDim": 32,
            "fullHeadDim": 32, "vocabSize": 1024, "slidingWindow": 1024,
            "finalLogitSoftcap": 0.0, "ropeTheta": 10_000_000.0,
            "fullRopeTheta": 10_000_000.0, "partialRotaryFactor": 0.25,
            "numLayers": 4, "numExperts": 8, "topKExperts": 8,
            "tieWordEmbeddings": false, "attentionKEqV": false,
            "hiddenActivation": "silu", "fullAttentionLayerMask": [2, 1, 2, 1],
            "family": "qwen36",
        ]
        let slot: [String: Any] = [
            "weightBits": 4, "scheme": "affine", "scaleType": "bf16",
            "biasType": "bf16", "groupSize": 64,
        ]
        try JSONSerialization.data(withJSONObject: [
            "magic": "SSDAI", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true], "modelID": "ornith-1.5-35b-a3b",
            "arch": arch, "files": [:], "expertsPerLayer": 8, "numLayers": 4,
            "expertStride": 16384,
            "quant": [
                "embedding": slot, "attention": slot, "router": slot,
                "sharedExpert": slot, "routedExpert": slot,
            ],
        ]).write(to: directory.appendingPathComponent("manifest.json"))
        let factory = try EngineFactories.gpuFactory(directoryURL: directory)
        do {
            _ = try factory.buildModel(directory, ArchConfig.qwen36_35B_A3B, nil)
            Issue.record("the variant requires a Metal context")
        } catch {
            #expect("\(error)".contains("ornith variant"))
        }
        // The CLI and an embedder call `routingFactory`, which must take the
        // same table: a directory naming the Ornith install gets the variant,
        // not the 6.x fallback.
        let routed = EngineFactories.routingFactory(directoryURL: directory)
        do {
            _ = try routed.buildModel(directory, ArchConfig.qwen36_35B_A3B, nil)
            Issue.record("the routed factory must be the variant too")
        } catch {
            #expect("\(error)".contains("ornith variant"))
        }
    }

    @Test("the dense CPU variant is selected per install size and width")
    func denseCPUFactorySelection() throws {
        // Hidden size names the model (2B 2048 / 4B 2560 / 9B 4096); the
        // width its quant twin. Each combination is its own dedicated target.
        #expect(
            try EngineFactories.variantCPUEngineSelection(hiddenSize: 2_048, weightBits: 4)
                == .qwen35Dense2B4Bit)
        #expect(
            try EngineFactories.variantCPUEngineSelection(hiddenSize: 2_048, weightBits: 8)
                == .qwen35Dense2B8Bit)
        #expect(
            try EngineFactories.variantCPUEngineSelection(hiddenSize: 2_560, weightBits: 4)
                == .qwen35Dense4B4Bit)
        #expect(
            try EngineFactories.variantCPUEngineSelection(hiddenSize: 2_560, weightBits: 8)
                == .qwen35Dense4B8Bit)
        #expect(
            try EngineFactories.variantCPUEngineSelection(hiddenSize: 4_096, weightBits: 4)
                == .qwen35Dense9B4Bit)
        #expect(
            try EngineFactories.variantCPUEngineSelection(hiddenSize: 4_096, weightBits: 8)
                == .qwen35Dense9B8Bit)
        // An unknown combination is refused by name, not served by a sibling.
        #expect(throws: (any Error).self) {
            try EngineFactories.variantCPUEngineSelection(hiddenSize: 2_560, weightBits: 6)
        }
        #expect(throws: (any Error).self) {
            try EngineFactories.variantCPUEngineSelection(hiddenSize: 3_072, weightBits: 4)
        }
    }
}
