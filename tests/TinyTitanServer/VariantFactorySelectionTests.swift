import Foundation
import Testing
import TinyTitanCore

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

    @Test("the dense CPU variant factory is built per install width")
    func denseCPUFactorySelection() throws {
        #expect(try EngineFactories.variantCPUEngineSelection(weightBits: 4) == .qwen35Dense4Bit)
        #expect(try EngineFactories.variantCPUEngineSelection(weightBits: 8) == .qwen35Dense8Bit)
        #expect(throws: (any Error).self) {
            try EngineFactories.variantCPUEngineSelection(weightBits: 6)
        }
    }
}
