import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanLib

/// The startup expert warm-up on the toy MoE install (4 layers, 8 experts,
/// top-8) with a synthetic `expert-profile.bin`: the hottest experts land in
/// the slots the decode planner gives them, the LFU seed and the hit/miss
/// statistics are untouched, N is capped by the slot count, 0 disables it,
/// and a generation after (or during) the warm-up is byte-identical to a cold
/// start.
@Suite(.serialized) struct ExpertWarmupTests {
    /// Per-layer counts, rotated by the layer index so every layer ranks a
    /// different set. Two experts per layer were never routed.
    static let skewed: [UInt32] = [5, 0, 40, 10, 30, 0, 20, 1]
    /// Every expert routed at least once, so a cap is the only limit.
    static let dense: [UInt32] = [1, 2, 3, 4, 5, 6, 7, 8]
    static let slotCount = 8

    // MARK: - Policy

    @Test func theDefaultIsTheSlotCountMinusASixteenth() {
        #expect(ExpertWarmup.perLayer(requested: nil, slotCount: 48, expertsPerLayer: 512) == 45)
        #expect(ExpertWarmup.perLayer(requested: nil, slotCount: 16, expertsPerLayer: 512) == 15)
        #expect(ExpertWarmup.perLayer(requested: nil, slotCount: 8, expertsPerLayer: 8) == 7)
        #expect(ExpertWarmup.perLayer(requested: 0, slotCount: 48, expertsPerLayer: 512) == 0)
        #expect(ExpertWarmup.perLayer(requested: -3, slotCount: 48, expertsPerLayer: 512) == 0)
        #expect(ExpertWarmup.perLayer(requested: 100, slotCount: 48, expertsPerLayer: 512) == 48)
        #expect(ExpertWarmup.perLayer(requested: 100, slotCount: 48, expertsPerLayer: 8) == 8)
        #expect(ExpertWarmup.perLayer(requested: 16, slotCount: 48, expertsPerLayer: 512) == 16)
    }

    @Test func theEnvironmentValueIsParsed() {
        let key = ExpertWarmup.environmentKey
        #expect(key == "TINYTITAN_WARM_EXPERTS")
        #expect(ExpertWarmup.requestedPerLayer(environment: [:]) == nil)
        #expect(ExpertWarmup.requestedPerLayer(environment: [key: "0"]) == 0)
        #expect(ExpertWarmup.requestedPerLayer(environment: [key: " 12 "]) == 12)
        #expect(ExpertWarmup.requestedPerLayer(environment: [key: "lots"]) == nil)
    }

    @Test func theLogLineNamesCountBytesAndTime() {
        let report = ExpertWarmupReport(
            perLayer: 45, layersWarmed: 48, layersRanked: 48, expertsRead: 2_160,
            bytesRead: 5_980_815_360, seconds: 3.214, stoppedEarly: false)
        #expect(report.logLine == "TinyTitan warm experts: 45/layer, 5.98 GB read in 3.21 s")
        let stopped = ExpertWarmupReport(
            perLayer: 45, layersWarmed: 7, layersRanked: 48, expertsRead: 315,
            bytesRead: 872_202_240, seconds: 0.5, stoppedEarly: true)
        #expect(stopped.logLine.hasSuffix("(stopped by a request after 7/48 layers)"))
    }

    // MARK: - Residency

    @Test func theHottestExpertsAreResidentInPlannerOrder() async throws {
        let fixture = try Fixture(profile: Self.skewed)
        defer { fixture.remove() }
        let model = try fixture.loadModel()
        let report = try #require(await model.warmRoutedExpertCache(perLayer: 3))
        #expect(report.perLayer == 3)
        #expect(report.layersRanked == fixture.layers)
        #expect(report.layersWarmed == fixture.layers)
        #expect(report.expertsRead == 3 * fixture.layers)
        #expect(!report.stoppedEarly)
        for layer in 0..<fixture.layers {
            let streamer = try model.openStreamer(for: layer)
            let expected = Self.ranked(Self.skewed, layer: layer).prefix(3)
            #expect(streamer.residentExperts() == expected.sorted())
            // An empty cache gives the first miss the first slot: the hottest
            // expert is in slot 0, the next in slot 1, and so on.
            for (rank, expert) in expected.enumerated() {
                let entry = streamer.residencyEntry(expert: expert)
                #expect(entry.state == ExpertResidencyEntry.resident)
                #expect(entry.slot == UInt32(rank))
            }
            // Residency only: the LFU score is still the profile's seed and
            // no plan, hit or miss was charged to a request.
            #expect(
                streamer.expertUseCountsSnapshot()
                    == Self.rotated(Self.skewed, layer: layer).map { Int($0 / 2) })
            let statistics = streamer.statistics()
            #expect(statistics.plans == 0)
            #expect(statistics.hits == 0)
            #expect(statistics.misses == 0)
        }
    }

    @Test func nIsCappedByTheSlotCount() async throws {
        let fixture = try Fixture(profile: Self.dense)
        defer { fixture.remove() }
        let (session, model6) = try await fixture.session(slots: 6)
        await session.startExpertWarmup(requestedPerLayer: 100)
        let report = try #require(await session.awaitExpertWarmup())
        #expect(report.perLayer == 6)
        for layer in 0..<fixture.layers {
            let streamer = try model6.openStreamer(for: layer)
            #expect(
                streamer.residentExperts()
                    == Self.ranked(Self.dense, layer: layer).prefix(6).sorted())
        }
        // Experts the profile never saw are never warmed, cap or not.
        let sparse = try Fixture(profile: Self.skewed)
        defer { sparse.remove() }
        let model = try sparse.loadModel()
        let full = try #require(await model.warmRoutedExpertCache(perLayer: 100))
        #expect(full.perLayer == Self.slotCount)
        #expect(try model.openStreamer(for: 0).residentExperts().count == 6)
    }

    @Test func zeroDisablesTheWarmUp() async throws {
        let fixture = try Fixture(profile: Self.skewed)
        defer { fixture.remove() }
        let (session, model) = try await fixture.session()
        await session.startExpertWarmup(requestedPerLayer: 0)
        #expect(await session.expertWarmupTask == nil)
        #expect(await session.awaitExpertWarmup() == nil)
        #expect(try model.openStreamer(for: 0).residentExperts().isEmpty)
        #expect(try await fixture.loadModel().warmRoutedExpertCache(perLayer: 0) == nil)
    }

    @Test func noProfileMeansNothingToWarm() async throws {
        let fixture = try Fixture(profile: nil)
        defer { fixture.remove() }
        #expect(try await fixture.loadModel().warmRoutedExpertCache(perLayer: 4) == nil)
    }

    // MARK: - Output

    @Test func aGenerationAfterWarmUpMatchesAColdStart() async throws {
        let fixture = try Fixture(profile: Self.skewed)
        defer { fixture.remove() }
        // Both built before either generates: a request rewrites the profile.
        let (cold, coldModel) = try await fixture.session()
        let (warm, warmModel) = try await fixture.session()
        let coldCompletion = try await cold.generate(fixture.request()) { _ in }
        let coldStatistics = coldModel.routedExpertStatistics()

        await warm.startExpertWarmup(requestedPerLayer: 3)
        let report = try #require(await warm.awaitExpertWarmup())
        #expect(report.layersWarmed == fixture.layers)
        let warmCompletion = try await warm.generate(fixture.request()) { _ in }
        let warmStatistics = warmModel.routedExpertStatistics()

        #expect(warmCompletion.content == coldCompletion.content)
        #expect(warmCompletion.reasoning == coldCompletion.reasoning)
        #expect(warmCompletion.finishReason == coldCompletion.finishReason)
        #expect(warmCompletion.usage.completionTokens == coldCompletion.usage.completionTokens)
        #expect(warmCompletion.usage.completionTokens > 0)
        // Same routing, so the same number of experts requested; the warmed
        // ones were hits the cold start had to read.
        #expect(warmStatistics.requestedExperts == coldStatistics.requestedExperts)
        #expect(warmStatistics.hits > coldStatistics.hits)
    }

    @Test func aRequestDuringWarmUpStopsItAndMatchesAColdStart() async throws {
        let fixture = try Fixture(profile: Self.dense)
        defer { fixture.remove() }
        let (cold, _) = try await fixture.session()
        let (racing, racingModel) = try await fixture.session()
        let coldCompletion = try await cold.generate(fixture.request()) { _ in }

        await racing.startExpertWarmup(requestedPerLayer: Self.slotCount)
        // No await on the warm-up: the request has to stop it itself.
        let completion = try await racing.generate(fixture.request()) { _ in }
        #expect(await racing.expertWarmupTask == nil)
        #expect(completion.content == coldCompletion.content)
        #expect(completion.usage.completionTokens == coldCompletion.usage.completionTokens)
        // Every slot the cache holds is a published, single copy.
        for layer in 0..<fixture.layers {
            let streamer = try racingModel.openStreamer(for: layer)
            let resident = streamer.residentExperts()
            #expect(Set(resident).count == resident.count)
            #expect(streamer.statistics().loadingSlots == 0)
        }
    }

    // MARK: - Helpers

    static func rotated(_ base: [UInt32], layer: Int) -> [UInt32] {
        (0..<base.count).map { base[($0 + layer) % base.count] }
    }

    /// The profile's order for one layer, hottest first, never-routed omitted.
    static func ranked(_ base: [UInt32], layer: Int) -> [Int] {
        let counts = rotated(base, layer: layer)
        return counts.indices
            .filter { counts[$0] > 0 }
            .sorted { counts[$0] != counts[$1] ? counts[$0] > counts[$1] : $0 < $1 }
    }
}

/// The toy install with an optional profile beside it, and models and
/// sessions built over it the way the production loader builds them.
private struct Fixture {
    let directory: URL
    let tokenizerFolder: URL
    let context: MetalContext
    let layers: Int

    init(profile base: [UInt32]?) throws {
        directory = try ToyInstall.make()
        tokenizerFolder = try ChatMLTemplateTests.fixtureFolder()
        context = try MetalContext()
        let arch = ArchConfig.qwenToy()
        layers = arch.numLayers
        guard let base else { return }
        var profile = ExpertUsageProfile(
            layers: arch.numLayers, expertsPerLayer: arch.numExperts)
        for layer in 0..<arch.numLayers {
            for (expert, count) in ExpertWarmupTests.rotated(base, layer: layer).enumerated() {
                profile.setCount(layer: layer, expert: expert, to: count)
            }
        }
        try profile.write(to: directory.appendingPathComponent("expert-profile.bin"))
    }

    func remove() {
        try? FileManager.default.removeItem(at: directory)
    }

    func loadModel(slots: Int = ExpertWarmupTests.slotCount) throws -> Model {
        try Model.load(
            directoryURL: directory, device: context.device,
            expecting: .qwenToy(), streamingMode: .pread(slotCount: slots))
    }

    /// The session and the model it runs, read back through the session's
    /// nonisolated runner rather than handed across the actor boundary.
    func session(
        slots: Int = ExpertWarmupTests.slotCount
    ) async throws -> (ServerModelSession, Model) {
        let model = try loadModel(slots: slots)
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 256, slots: 1,
            runtimeConfiguration: try RuntimeConfiguration(
                forceLogitsHead: true, kvCachePrecision: .int8))
        let tokenizer = try await GFTokenizer.load(from: tokenizerFolder)
        let scratch = try RawCompletionScratch(
            context: context, vocab: model.config.vocabSize,
            logitSoftcap: Float(model.config.finalLogitSoftcap))
        let session = ServerModelSession(
            context: context, model: model,
            tokenizer: tokenizer,
            tokenizerFolder: tokenizerFolder,
            loadedReasoning: RequestReasoning(thinkingMode: .off, effort: nil),
            runner: runner, mtpDecoder: nil, scratches: [scratch],
            prefillConfig: .production(chunkTokens: 64),
            expertCacheSlots: slots, slots: 1, maxContext: 256,
            promptCacheMode: .off,
            promptCacheDomain: ServerPromptCacheDomain(
                modelID: "qwen-toy", sourceSnapshotHash: "snapshot",
                runtimeProfileHash: "profile", maximumContext: 256,
                kvStorage: "int8", fp16RingEnabled: false, templateSHA256: "template"),
            promptCache: ServerPromptCache(maximumEntries: 1),
            promptStateStore: nil, promptCheckpointStore: nil, concisePrompt: nil)
        return (session, session.benchmarkRunner.model)
    }

    func request(maxNewTokens: Int = 8) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [
                GFTokenizer.Message(role: .system, content: "You are terse."),
                GFTokenizer.Message(role: .user, content: "Warm or cold?"),
            ],
            tools: [], stream: false, includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: maxNewTokens, temperature: 0),
            maximumCompletionTokens: maxNewTokens)
    }
}
