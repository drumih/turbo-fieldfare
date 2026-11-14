import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore
import TinyTitanCore

@Suite("Model router")
struct ModelRouterTests {
    private typealias Fixture = RoutingFixture

    @Test func switchesToTheNamedModelAndLoadsEachOnce() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(log: log)
        try await router.preload()
        #expect(log.loads == ["load alpha_4-Bit"])

        let first = try await router.generate(Fixture.request(Fixture.alpha.id)) { _ in }
        #expect(first.content == Fixture.alpha.id)
        let second = try await router.generate(Fixture.request(Fixture.small.id)) { _ in }
        #expect(second.content == Fixture.small.id)
        _ = try await router.generate(Fixture.request(Fixture.small.id)) { _ in }

        #expect(log.loads == ["load alpha_4-Bit", "load small-2b"])
        #expect(await router.residentModelID == Fixture.small.id)
        #expect(await router.inFlightCount == 0)
    }

    /// The property the design rests on: a switch never pulls the weights out
    /// from under a generation that is still running on them.
    @Test func aSwitchWaitsForTheInFlightGenerationToDrain() async throws {
        let log = RoutingEventLog()
        let gate = RoutingGate()
        let router = try Fixture.router(log: log, gates: [Fixture.alpha.id: gate])
        try await router.preload()

        let running = Task { try await router.generate(Fixture.request(Fixture.alpha.id)) { _ in } }
        await Fixture.eventually("alpha to start generating") { await gate.isWaiting }
        let switching = Task {
            try await router.generate(Fixture.request(Fixture.small.id)) { _ in }
        }
        await Fixture.eventually("the switch to queue") { await router.waiterCount == 1 }

        #expect(await router.residentModelID == Fixture.alpha.id)
        #expect(!log.loads.contains("load small-2b"))

        await gate.open()
        #expect(try await running.value.content == Fixture.alpha.id)
        #expect(try await switching.value.content == Fixture.small.id)
        let drained = try #require(log.index(of: "end alpha_4-Bit"))
        let loaded = try #require(log.index(of: "load small-2b"))
        #expect(drained < loaded)
        #expect(await router.residentModelID == Fixture.small.id)
    }

    /// Work for the resident model that arrives behind a pending switch waits
    /// its turn, rather than keeping the model busy so the switch never runs.
    @Test func workForTheResidentModelQueuesBehindAPendingSwitch() async throws {
        let log = RoutingEventLog()
        let gate = RoutingGate()
        let router = try Fixture.router(log: log, gates: [Fixture.alpha.id: gate])
        try await router.preload()

        let first = Task { try await router.generate(Fixture.request(Fixture.alpha.id)) { _ in } }
        await Fixture.eventually("alpha to start") { await gate.isWaiting }
        let switching = Task {
            try await router.generate(Fixture.request(Fixture.small.id)) { _ in }
        }
        await Fixture.eventually("the switch to queue") { await router.waiterCount == 1 }
        let later = Task { try await router.generate(Fixture.request(Fixture.alpha.id)) { _ in } }
        await Fixture.eventually("the later request to queue") { await router.waiterCount == 2 }

        await gate.open()
        _ = try await (first.value, switching.value, later.value)
        #expect(log.loads == ["load alpha_4-Bit", "load small-2b", "load alpha_4-Bit"])
        #expect(log.maxConcurrentGenerations == 1)
    }

    @Test func anUnknownModelIsRefusedWithoutLoadingAnything() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(log: log)
        await #expect(throws: ServerRequestError.unknownModel) {
            _ = try await router.generate(Fixture.request("no-such-model")) { _ in }
        }
        #expect(log.loads.isEmpty)
        #expect(throws: ModelRouterError.self) {
            _ = try ModelRouter(
                catalog: Fixture.catalog, initialModelID: "no-such-model",
                reasoning: .off, maximumContext: 4_096
            ) { _, _ in
                RoutedStubModel(id: "x", log: log, gate: nil, delay: nil)
            }
        }
    }

    /// The engine's own requests carry no model and must not force a load.
    @Test func aRequestWithoutAModelRunsOnWhateverIsResident() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(log: log)
        _ = try await router.generate(Fixture.request(Fixture.small.id)) { _ in }
        let engineOwn = try await router.generate(Fixture.request(nil)) { _ in }
        #expect(engineOwn.content == Fixture.small.id)
        #expect(log.loads == ["load small-2b"])
    }

    @Test func aFailedLoadLeavesNothingResidentAndTheNextRequestRetries() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(log: log, failing: [Fixture.flash.id])
        try await router.preload()
        await #expect(throws: RoutingStubFailure.self) {
            _ = try await router.generate(Fixture.request(Fixture.flash.id)) { _ in }
        }
        // The old model was released before the new one was tried: holding
        // both, even briefly, is what the router exists to avoid.
        #expect(await router.residentModelID == nil)
        _ = try await router.generate(Fixture.request(Fixture.alpha.id)) { _ in }
        #expect(log.loads == ["load alpha_4-Bit", "load flash_8-Bit", "load alpha_4-Bit"])
        #expect(await router.inFlightCount == 0)
    }

    @Test func unloadWaitsForInFlightWorkThenReleases() async throws {
        let log = RoutingEventLog()
        let gate = RoutingGate()
        let router = try Fixture.router(log: log, gates: [Fixture.alpha.id: gate])
        try await router.preload()
        let running = Task { try await router.generate(Fixture.request(Fixture.alpha.id)) { _ in } }
        await Fixture.eventually("alpha to start") { await gate.isWaiting }
        let unloading = Task { await router.unload() }
        await Fixture.eventually("the unload to queue") { await router.waiterCount == 1 }
        #expect(await router.residentModelID == Fixture.alpha.id)

        await gate.open()
        _ = try await running.value
        #expect(await unloading.value)
        #expect(await router.residentModelID == nil)
        #expect(await router.unload() == false)
    }

    /// A count names a model but must not switch to it. The resident model
    /// answers for itself; any other is counted from its tokenizer and the
    /// resident model stays loaded. Counting used to route like generation,
    /// so sizing a prompt cost the next request a full reload.
    @Test func tokenCountingNeverSwitchesModels() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(log: log)
        try await router.preload()
        #expect(try await router.countPromptTokens(Fixture.request(Fixture.small.id)) == 5)
        #expect(try await router.countPromptTokens(Fixture.request(Fixture.alpha.id)) == 7)
        #expect(log.events == ["load alpha_4-Bit", "tokenize small-2b", "count alpha_4-Bit"])
        #expect(await router.residentModelID == Fixture.alpha.id)
        #expect(await router.inFlightCount == 0)
    }

    /// With nothing resident, not even the initial model is loaded to count.
    @Test func countingWithNothingResidentLoadsNothing() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(log: log)
        #expect(try await router.countPromptTokens(Fixture.request(nil)) == 5)
        #expect(log.events == ["tokenize alpha_4-Bit"])
        #expect(await router.residentModelID == nil)
    }

    @Test func eachModelAdvertisesItsOwnDefaultsAndContext() throws {
        let router = try Fixture.router(log: RoutingEventLog())
        let alpha = try #require(router.servedModel(named: Fixture.alpha.id))
        let flash = try #require(router.servedModel(named: Fixture.flash.id))
        let small = try #require(router.servedModel(named: Fixture.small.id))
        #expect(alpha.maximumContext == Fixture.configuredContext)
        #expect(flash.sampling.temperature == 1.0)
        // The CPU engine's own ceiling, not the checkpoint's 262k claim.
        #expect(small.maximumContext == CPUModelBackend.contextCeiling)
        #expect(small.displayName == "Small 2B")
        #expect(router.servedModel(named: "small-2b-fast")?.id == Fixture.small.id)
        #expect(router.servedModel(named: "nothing-fast") == nil)
        #expect(
            router.servedModels.map(\.id) == [
                Fixture.alpha.id, Fixture.flash.id,
                Fixture.small.id, Fixture.dense.id,
                "\(Fixture.dense.id)@cpu",
            ])
    }

    @Test func theServerLevelIsFittedToEachModelAndLoadedThatWay() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(reasoning: .on, log: log)
        let flash = try #require(router.servedModel(named: Fixture.flash.id)).reasoningProfile
        #expect(flash.thinkingMode == .on)
        #expect(flash.effectiveEffort == .xhigh)
        let alpha = try #require(router.servedModel(named: Fixture.alpha.id)).reasoningProfile
        #expect(alpha.thinkingMode == .on)
        #expect(alpha.effectiveEffort == nil)

        _ = try await router.generate(Fixture.request(Fixture.flash.id)) { _ in }
        let choice = try #require(log.choices[Fixture.flash.id])
        #expect(choice.requested == .on)
        #expect(choice.effective == .xhigh)
        #expect(choice.effort == .xhigh)
    }

    /// A request names the engine for an install both engines serve.
    ///
    /// The dense Qwen 3.5 installs are the one shape with two engines, so the
    /// router registers `@gpu` (the default, beside the bare id) and `@cpu`
    /// beside each of them; naming one switches engines the same way naming a
    /// different model does.
    @Test func aRequestCanNameTheEngineOnADenseInstall() async throws {
        let log = RoutingEventLog()
        let router = try Fixture.router(log: log)
        try await router.preload()
        #expect(await router.residentModelID == Fixture.alpha.id)

        let cpu = try await router.generate(Fixture.request("\(Fixture.dense.id)@cpu")) { _ in }
        #expect(cpu.content == "\(Fixture.dense.id)@cpu")
        #expect(await router.residentModelID == "\(Fixture.dense.id)@cpu")
        #expect(log.loads.last == "load \(Fixture.dense.id)@cpu")

        // The explicit GPU spelling resolves to the same install, not a second
        // copy of it.
        let gpu = try await router.generate(Fixture.request("\(Fixture.dense.id)@gpu")) { _ in }
        #expect(gpu.content == "\(Fixture.dense.id)@gpu")
        #expect(await router.residentModelID == "\(Fixture.dense.id)@gpu")

        // An install with one engine has no alias, and asking for the other
        // engine is an unknown model rather than a silent wrong-engine load.
        await #expect(throws: ServerRequestError.self) {
            _ = try await router.generate(Fixture.request("\(Fixture.alpha.id)@cpu")) { _ in }
        }
    }
    /// Every queued caller has to be released by the wake that took the gate from
    /// it, and every abandoned one by its cancellation. This is the AUD-143
    /// regression: the waiter array was appended to from the continuation closure,
    /// off the actor's executor, so a wake could land before the append and the
    /// caller slept on a waiter nobody held.
    ///
    /// Progress is counted, not awaited. A job that lost its wake never returns,
    /// so awaiting the jobs would hang the test instead of failing it, and a task
    /// group does not time out either — it joins every child, including the one
    /// blocked on a parked job. The deadline is therefore a polling loop over the
    /// departures the jobs each write on their way out.
    @Test func concurrentSwitchesWakeEveryQueuedCaller() async throws {
        let log = RoutingEventLog()
        let alphaGate = RoutingGate()
        let smallGate = RoutingGate()
        let router = try Fixture.router(
            log: log,
            gates: [Fixture.alpha.id: alphaGate, Fixture.small.id: smallGate])
        try await router.preload()

        var jobs: [Task<Void, Never>] = []
        for index in 0..<12 {
            let id = index.isMultiple(of: 2) ? Fixture.alpha.id : Fixture.small.id
            jobs.append(
                Task {
                    // try?, not try: this job's outcome is the count, and a caller
                    // that left by cancellation has still left.
                    _ = try? await router.generate(Fixture.request(id)) { _ in }
                    log.append("departed \(id)")
                })
        }
        await Fixture.eventually("callers to queue") { await router.waiterCount >= 1 }
        let total = jobs.count
        for job in jobs.suffix(3) { job.cancel() }
        await alphaGate.open()
        await smallGate.open()

        await Fixture.eventually("every caller to be released", timeout: .seconds(20)) {
            let released = log.departures
            let active = await router.inFlightCount
            return released == total && active == 0
        }
        let departed = log.departures
        let waiters = await router.waiterCount
        let inFlight = await router.inFlightCount
        let loads = log.loads.count
        #expect(departed == jobs.count, "\(departed)/\(jobs.count) callers returned")
        #expect(waiters == 0, "\(waiters) queued callers were never dropped")
        #expect(inFlight == 0, "\(inFlight) departed callers kept the count")
        // The stress has to actually switch, or it tests nothing.
        #expect(loads > 1, "\(loads) loads: the callers never forced a switch")
        // Not asserted: one generation at a time. The router allows several callers
        // on the resident model (`inFlight` counts them, and a switch waits for all
        // of them); it is the HTTP coordinator that serialises requests. Measured on
        // the release build, one run in seven overlapped two generations — which is
        // also what made the gate's single stored continuation a hang rather than a
        // curiosity.
    }
}

@Suite("Reasoning fallback")
struct ReasoningFallbackTests {
    private let binary: [ReasoningLevel] = [.off, .on]
    private let efforts: [ReasoningLevel] = [.off, .low, .medium, .xhigh]

    @Test func anEffortOnAnOnOffModelBecomesOn() {
        for level: ReasoningLevel in [.minimal, .low, .medium, .high, .xhigh, .max] {
            #expect(ReasoningFallback.effectiveLevel(level, supported: binary) == .on)
        }
    }

    /// `on` must load what `--thinking on` loads on a single-model server:
    /// the template's default effort, which for Qwen3.8 is extra high. It
    /// used to be the middle effort, so the same flag thought less on a
    /// routed server than on a single-model one.
    @Test func onForAnEffortModelIsItsTemplateDefault() throws {
        let qwen38 = ModelCatalog.Kind.gpu(.qwen38flash)
        #expect(qwen38.levelWhenOn == .xhigh)
        let choice = try ReasoningFallback.choice(for: qwen38, requested: .on)
        #expect(choice.effective == .xhigh)
        #expect(choice.thinking == .on)
        #expect(choice.effort == .xhigh)
        #expect(ModelCatalog.Kind.gpu(.qwen36).levelWhenOn == .on)
        #expect(ModelCatalog.Kind.cpu(.qwen35Dense).levelWhenOn == .on)
    }

    @Test func onWithoutATemplateDefaultFallsToTheMiddleEffort() {
        #expect(ReasoningFallback.effectiveLevel(.on, supported: efforts) == .medium)
    }

    @Test func offIsAlwaysOff() {
        #expect(ReasoningFallback.effectiveLevel(.off, supported: binary) == .off)
        #expect(ReasoningFallback.effectiveLevel(.off, supported: efforts) == .off)
    }

    @Test func aSupportedLevelIsKept() {
        #expect(ReasoningFallback.effectiveLevel(.on, supported: binary) == .on)
        #expect(ReasoningFallback.effectiveLevel(.xhigh, supported: efforts) == .xhigh)
        #expect(ReasoningFallback.effectiveLevel(.low, supported: efforts) == .low)
    }

    @Test func aMissingEffortTakesTheNearestTiesToTheCheaper() {
        #expect(ReasoningFallback.effectiveLevel(.high, supported: efforts) == .medium)
        #expect(ReasoningFallback.effectiveLevel(.max, supported: efforts) == .xhigh)
        #expect(ReasoningFallback.effectiveLevel(.minimal, supported: efforts) == .low)
    }

    @Test func theChoiceCarriesTheRuntimeSettings() throws {
        let flash = try ReasoningFallback.choice(for: .gpu(.qwen38flash), requested: .on)
        #expect(flash.thinking == .on)
        #expect(flash.effort == .xhigh)
        let cpu = try ReasoningFallback.choice(for: .cpu(.qwen35Dense), requested: .high)
        #expect(cpu.effective == .on)
        #expect(cpu.thinking == .on)
        #expect(cpu.effort == nil)
        let off = try ReasoningFallback.choice(for: .gpu(.qwen36), requested: .off)
        #expect(off.thinking == .off)
    }

    @Test func singleModelModeWithoutReasoningKeepsTheOldFlagsVerbatim() throws {
        let arguments = try ServerArguments.parse(
            ["--model", "/nonexistent", "--thinking", "on", "--reasoning-effort", "low"],
            environment: [:])
        // Nothing is read from disk: the path does not exist and this passes.
        let settings = try arguments.singleModelReasoning(
            directory: URL(fileURLWithPath: "/nonexistent"))
        #expect(settings.thinking == .on)
        #expect(settings.effort == .low)
        #expect(arguments.requestedReasoningLevel == .low)
    }

}
