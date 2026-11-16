import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

/// Records how often it was built and lets a load be held open, so residency
/// policy can be tested without a model on disk.
private final class LoadRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _loads = 0
    private var _contexts: [MetalContext] = []
    private var _gate: CheckedContinuation<Void, Never>?
    private var _shouldFail = false

    var loads: Int { lock.withLock { _loads } }
    var contexts: [MetalContext] { lock.withLock { _contexts } }

    func failNextLoad() { lock.withLock { _shouldFail = true } }

    func record(_ context: MetalContext?) throws {
        try lock.withLock {
            _loads += 1
            if let context { _contexts.append(context) }
            if _shouldFail {
                _shouldFail = false
                throw StubError.loadFailed
            }
        }
    }

    enum StubError: Error { case loadFailed }
}

private struct StubBackend: ServerInferenceBackend {
    let maximumContext = 4_096

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("ok"))
        return ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(
                promptTokens: 1,
                completionTokens: 1,
                totalTokens: 2,
                cachedTokens: 0))
    }
}

/// A backend whose generation can be held open, so a manual unload can be
/// observed waiting for an in-flight request to drain.
private actor GatedBackend: ServerInferenceBackend {
    private var continuation: CheckedContinuation<Void, Never>?

    var isWaiting: Bool { continuation != nil }

    func release() {
        continuation?.resume()
        continuation = nil
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        await withCheckedContinuation { continuation = $0 }
        onEvent(.content("ok"))
        return ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(
                promptTokens: 1,
                completionTokens: 1,
                totalTokens: 2,
                cachedTokens: 0))
    }
}

/// Holds a model load open so a manual unload can be issued while it is still
/// in flight.
private actor LoadGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var arrived = false

    var hasArrived: Bool { arrived }

    func wait() async {
        arrived = true
        await withCheckedContinuation { continuation = $0 }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

@Suite("Managed model backend")
struct ManagedModelBackendTests {
    private func plan() -> ModelSessionPlan {
        ModelSessionPlan(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model"),
            maxContext: 4_096,
            promptCacheMode: .multiPrefix,
            promptCacheMaximumEntries: 1,
            promptCacheMemoryLimitBytes: 1_048_576,
            promptCacheDiskDirectory: nil,
            promptCacheDiskLimitBytes: 1_048_576,
            prefillChunkTokens: nil,
            expertCacheSlots: nil,
            mtpModelDirectory: nil,
            mtpMemoryMiB: 0)
    }

    private func facts() -> ModelSessionFacts {
        ModelSessionFacts(
            modelID: "stub",
            prefillChunkTokens: 4_096,
            promptCacheMode: .multiPrefix)
    }

    private func request() -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: "hi")],
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 4, temperature: 0),
            maximumCompletionTokens: 4)
    }

    /// Polls until the reaper has released the session, so the test asserts on
    /// the observable transition rather than sleeping for a fixed interval.
    private func waitForUnload(
        _ managed: ManagedModelBackend,
        timeout: Duration
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if await managed.isLoaded == false { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("model was still loaded after \(timeout)")
    }

    private func backend(
        idleTimeout: Duration? = nil,
        recorder: LoadRecorder
    ) -> ManagedModelBackend {
        ManagedModelBackend(
            plan: plan(),
            facts: facts(),
            idleTimeout: idleTimeout,
            loader: { _, context in
                try recorder.record(context)
                return StubBackend()
            })
    }

    /// The plan-free initializer is what K2's residency path uses: no
    /// `ModelSessionPlan` exists for the family, so the wrapper takes the
    /// facts, the context limit, the sampling row and a loader directly.
    private func k2Backend(
        idleTimeout: Duration? = nil,
        recorder: LoadRecorder
    ) -> ManagedModelBackend {
        ManagedModelBackend(
            facts: ModelSessionFacts(
                modelID: "k2",
                prefillChunkTokens: 0,
                promptCacheMode: .off,
                expertCacheSlots: 16),
            maximumContext: 8_192,
            samplingDefaults: GenerationDefaults.forFamily(.k2Horizon),
            idleTimeout: idleTimeout,
            loader: { context in
                try recorder.record(context)
                return StubBackend()
            })
    }

    @Test func planFreeInitializerDefersAndReportsItsOwnFacts() async throws {
        let recorder = LoadRecorder()
        let managed = k2Backend(recorder: recorder)

        // Before any load: the values a request is validated against.
        #expect(managed.maximumContext == 8_192)
        #expect(managed.samplingDefaults == GenerationDefaults.forFamily(.k2Horizon))
        #expect(managed.samplingDefaults.temperature == 1.0)
        #expect(recorder.loads == 0)

        _ = try await managed.generate(request()) { _ in }
        #expect(recorder.loads == 1)
        #expect(recorder.contexts.count == 1)
        // A second request reuses the resident backend rather than rebuilding.
        _ = try await managed.generate(request()) { _ in }
        #expect(recorder.loads == 1)
    }

    @Test func planFreeInitializerUnloadsAfterIdleAndReloads() async throws {
        let recorder = LoadRecorder()
        let managed = k2Backend(idleTimeout: .milliseconds(30), recorder: recorder)

        _ = try await managed.generate(request()) { _ in }
        #expect(recorder.loads == 1)
        try await waitForUnload(managed, timeout: .seconds(2))

        _ = try await managed.generate(request()) { _ in }
        #expect(recorder.loads == 2)
        // The reload reuses the process's Metal context: one shader compile,
        // one queue, for the life of the process.
        #expect(recorder.contexts.count == 2)
        #expect(recorder.contexts[0] === recorder.contexts[1])
    }

    // MARK: - Deferral

    @Test func doesNotLoadUntilTheFirstRequest() async throws {
        let recorder = LoadRecorder()
        let managed = backend(recorder: recorder)

        // Construction and the banner facts must not touch the model.
        #expect(recorder.loads == 0)
        #expect(await managed.isLoaded == false)
        #expect(managed.maximumContext == 4_096)
        #expect(managed.facts.modelID == "stub")

        _ = try await managed.generate(request()) { _ in }
        #expect(recorder.loads == 1)
        #expect(await managed.isLoaded)
    }

    @Test func concurrentFirstRequestsLoadExactlyOnce() async throws {
        let recorder = LoadRecorder()
        let managed = backend(recorder: recorder)

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask { _ = try? await managed.generate(self.request()) { _ in } }
            }
        }
        #expect(recorder.loads == 1)
    }

    @Test func aFailedLoadIsRetriedRatherThanCached() async throws {
        let recorder = LoadRecorder()
        let managed = backend(recorder: recorder)
        recorder.failNextLoad()

        await #expect(throws: (any Error).self) {
            _ = try await managed.generate(request()) { _ in }
        }
        #expect(await managed.isLoaded == false)
        // The in-flight count must not leak when the load throws, or the
        // reaper would never unload again.
        #expect(await managed.inFlightCount == 0)

        _ = try await managed.generate(request()) { _ in }
        #expect(recorder.loads == 2)
        #expect(await managed.isLoaded)
    }

    // MARK: - MetalContext reuse

    @Test func reloadReusesTheSameMetalContext() async throws {
        let recorder = LoadRecorder()
        let managed = backend(idleTimeout: .milliseconds(1), recorder: recorder)

        _ = try await managed.generate(request()) { _ in }
        await managed.unloadNow()
        _ = try await managed.generate(request()) { _ in }

        #expect(recorder.loads == 2)
        #expect(recorder.contexts.count == 2)
        // Identity, not equality: one MTLCommandQueue and one compiled shader
        // library must survive the unload. Building a fresh context per reload
        // would tear down a queue that MetalContext.deinit documents as having
        // no deinit-safe cleanup.
        #expect(recorder.contexts[0] === recorder.contexts[1])
    }

    // MARK: - Idle policy

    @Test func reaperWaitsWhileTheModelIsBusy() async throws {
        let recorder = LoadRecorder()
        let managed = backend(idleTimeout: .seconds(30), recorder: recorder)
        _ = try await managed.generate(request()) { _ in }

        // Backdate *inside* the in-flight window: if the clock is moved while
        // the reaper is free to run, it can wake between these two awaits,
        // see an idle model, unload it, and make reaperStep() return .stop.
        await managed.withInFlight {
            await managed.backdateActivity(by: .seconds(120))
            // Idle long enough by the clock, but a request is in flight.
            #expect(await managed.reaperStep() == .sleep(.seconds(30)))
        }
        // Once it drains, the same clock reading means unload.
        #expect(await managed.reaperStep() == .stop)
    }

    @Test func reaperSleepsOnlyTheRemainingIdleTime() async throws {
        let recorder = LoadRecorder()
        let managed = backend(idleTimeout: .seconds(60), recorder: recorder)
        _ = try await managed.generate(request()) { _ in }

        await managed.backdateActivity(by: .seconds(45))
        guard case .sleep(let remaining) = await managed.reaperStep() else {
            Issue.record("expected a sleep while still inside the idle window")
            return
        }
        // ~15s left, not a fixed poll interval and not the full timeout.
        #expect(remaining < .seconds(16))
        #expect(remaining > .seconds(14))
    }

    @Test func reaperStopsWhenNothingIsLoaded() async throws {
        let recorder = LoadRecorder()
        let managed = backend(idleTimeout: .seconds(1), recorder: recorder)
        #expect(await managed.reaperStep() == .stop)
    }

    @Test func noIdleTimeoutMeansNoReaper() async throws {
        let recorder = LoadRecorder()
        let managed = backend(recorder: recorder)
        _ = try await managed.generate(request()) { _ in }

        #expect(await managed.hasReaper == false)
        await managed.backdateActivity(by: .seconds(86_400))
        #expect(await managed.reaperStep() == .stop)
        // .stop with no timeout means "nothing to do", not "unload".
        #expect(await managed.isLoaded)
    }

    @Test func unloadsAfterIdleThenReloadsOnDemand() async throws {
        let recorder = LoadRecorder()
        let managed = backend(idleTimeout: .milliseconds(50), recorder: recorder)
        _ = try await managed.generate(request()) { _ in }
        #expect(await managed.isLoaded)

        try await waitForUnload(managed, timeout: .seconds(5))
        #expect(recorder.loads == 1)

        _ = try await managed.generate(request()) { _ in }
        #expect(recorder.loads == 2)
        #expect(await managed.isLoaded)
    }

    // MARK: - Manual unload

    @Test func unloadReleasesTheSessionWhenIdle() async throws {
        let recorder = LoadRecorder()
        let managed = backend(recorder: recorder)
        _ = try await managed.generate(request()) { _ in }
        #expect(await managed.isLoaded)

        let released = await managed.unload()
        #expect(released)
        #expect(await managed.isLoaded == false)

        // A later request reloads on demand.
        _ = try await managed.generate(request()) { _ in }
        #expect(recorder.loads == 2)
        #expect(await managed.isLoaded)
    }

    /// An unload that arrives while the model is still loading must release it.
    ///
    /// `unload()` looped on `while session != nil`, and `session` is nil until a
    /// load finishes — so it returned false immediately while the load went on
    /// to complete and stay resident. The caller was told nothing had been
    /// released and the model it wanted freed stayed mapped.
    @Test func unloadWaitsForAnInFlightLoadAndReleasesIt() async throws {
        let recorder = LoadRecorder()
        let gate = LoadGate()
        let managed = ManagedModelBackend(
            plan: plan(),
            facts: facts(),
            idleTimeout: nil,
            loader: { _, context in
                try recorder.record(context)
                await gate.wait()
                return StubBackend()
            })

        let request = Task { try await managed.generate(self.request()) { _ in } }
        while await gate.hasArrived == false {
            try await Task.sleep(for: .milliseconds(5))
        }

        // The load is parked inside the loader; unload now.
        let unloading = Task { await managed.unload() }
        try await Task.sleep(for: .milliseconds(50))
        await gate.open()

        #expect(
            await unloading.value,
            "unload reported nothing released while a load was in flight")
        #expect(await managed.isLoaded == false)
        _ = try await request.value
    }

    @Test func unloadIsIdempotentWhenNothingIsLoaded() async throws {
        let recorder = LoadRecorder()
        let managed = backend(recorder: recorder)
        #expect(await managed.isLoaded == false)

        let released = await managed.unload()
        #expect(released == false)
        #expect(await managed.isLoaded == false)
        #expect(recorder.loads == 0)
    }

    @Test func unloadCancelsTheReaper() async throws {
        let recorder = LoadRecorder()
        let managed = backend(idleTimeout: .seconds(600), recorder: recorder)
        _ = try await managed.generate(request()) { _ in }
        #expect(await managed.hasReaper)

        let released = await managed.unload()
        #expect(released)
        #expect(await managed.hasReaper == false)
        #expect(await managed.isLoaded == false)

        // The reaper must not respawn itself: with the session gone it would
        // immediately find nothing to reap and chain new reaper tasks forever.
        try await Task.sleep(for: .milliseconds(100))
        #expect(await managed.hasReaper == false)
        #expect(await managed.isLoaded == false)
    }

    @Test func aCancelledUnloadHandsBackTheIdleTimer() async throws {
        let recorder = LoadRecorder()
        let managed = backend(idleTimeout: .seconds(600), recorder: recorder)
        _ = try await managed.generate(request()) { _ in }
        // Hold a request open so the unload has to wait rather than complete.
        await managed.bumpInFlightForTesting(1)

        let attempt = Task { await managed.unload() }
        try await Task.sleep(for: .milliseconds(50))
        attempt.cancel()
        let released = await attempt.value

        #expect(released == false)
        #expect(await managed.isLoaded)
        // The model is still resident, so the idle timer must still be armed.
        // Otherwise a client disconnecting mid-unload silently disables
        // --idle-unload-seconds for the life of the session.
        #expect(await managed.hasReaper)

        await managed.bumpInFlightForTesting(-1)
    }

    @Test func unloadWaitsForInFlightRequestsToDrain() async throws {
        let recorder = LoadRecorder()
        let gated = GatedBackend()
        let managed = ManagedModelBackend(
            plan: plan(),
            facts: facts(),
            idleTimeout: nil,
            loader: { _, context in
                try recorder.record(context)
                return gated
            })

        let generation = Task { try? await managed.generate(request()) { _ in } }
        while await gated.isWaiting == false {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(await managed.inFlightCount == 1)

        let unload = Task { await managed.unload() }
        // The unload must not complete while the generation is in flight.
        try await Task.sleep(for: .milliseconds(100))
        #expect(await managed.isLoaded)

        await gated.release()
        _ = await generation.value
        let released = await unload.value
        #expect(released)
        #expect(await managed.isLoaded == false)
    }

    // MARK: - Lifecycle

    @Test func shutdownCancelsTheReaperAndReleasesTheSession() async throws {
        let recorder = LoadRecorder()
        let managed = backend(idleTimeout: .seconds(600), recorder: recorder)
        _ = try await managed.generate(request()) { _ in }
        #expect(await managed.hasReaper)

        await managed.shutdown()
        #expect(await managed.hasReaper == false)
        #expect(await managed.isLoaded == false)
    }
}

extension ManagedModelBackend {
    /// Runs `body` with the in-flight count raised, mirroring a request that is
    /// mid-generation while the reaper wakes.
    fileprivate func withInFlight(_ body: () async -> Void) async {
        bumpInFlightForTesting(1)
        await body()
        bumpInFlightForTesting(-1)
    }
}
