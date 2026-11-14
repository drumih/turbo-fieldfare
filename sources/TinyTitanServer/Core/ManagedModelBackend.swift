import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

/// Manages when the model is resident: loads it on the first inference request
/// rather than at startup, and optionally releases it again after an idle
/// period. Reloads are transparent to the caller.
///
/// The server otherwise maps ~5 GB of weights at startup and holds them for the
/// process lifetime, which on a 24 GB machine leaves no room for a second
/// workload. `ServerInferenceBackend` has two members, so residency is managed
/// by wrapping it — `TinyTitanHTTPServer` needs no knowledge of any of this.
///
/// Memory is genuinely returned because every large allocation is owned by a
/// refcounted class reachable only from the session: the resident weights are
/// an `mmap` behind an `MTLBuffer` whose deallocator calls `munmap`
/// (`ResidentBuffer`), and the routed-expert slots are `posix_memalign` blocks
/// with a `free` deallocator (`PreadExpertStreamer`). Releasing the session
/// unwinds all of it.
///
/// `MetalContext` is the deliberate exception: it is built once and reused for
/// the process lifetime. `MetalContext.deinit` documents that `MTLCommandQueue`
/// has no deinit-safe cleanup, so tearing one down per unload would be unsafe;
/// it also compiles the whole shader library, which would make every reload pay
/// for a rebuild. It is small next to the weights.
public actor ManagedModelBackend: ServerInferenceBackend, ResidencyManaging, PromptTokenCounting {
    /// Builds a session. Injectable so the residency logic can be tested
    /// against a stub without a model on disk.
    package typealias Loader =
        @Sendable (ModelSessionPlan, MetalContext?) async throws -> any ServerInferenceBackend

    /// What the reaper should do next. Returning a duration rather than
    /// polling on a fixed tick means one wake-up per idle period instead of
    /// one every few seconds, and the unload lands on the deadline instead of
    /// up to a tick late.
    enum ReaperStep: Equatable {
        case sleep(Duration)
        case stop
    }

    /// Builds the object that serves. The Metal context is shared across
    /// reloads; the closure captures whatever else the backend needs (a plan
    /// for the runner graph, an install path for the K2 probe).
    private let loader: @Sendable (MetalContext) async throws -> any ServerInferenceBackend
    private let idleTimeout: Duration?

    /// Answered from configuration and the manifest, so they are valid before
    /// the first load and identical to what the eager path would report.
    public nonisolated let maximumContext: Int
    /// The row a request that omits sampling gets. The runner plans fill it
    /// from their own catalog entry elsewhere; the wrapper answers the house
    /// row unless the caller names one (the K2 probe passes its card row).
    public nonisolated let samplingDefaults: GenerationDefaults.Sampling
    package nonisolated let facts: ModelSessionFacts

    private var session: (any ServerInferenceBackend)?
    /// In-flight load, so concurrent first requests coalesce into one load.
    private var loadTask: Task<any ServerInferenceBackend, any Error>?
    /// Requests currently inside `generate`. Non-zero blocks unloading.
    private var inFlight = 0
    private var lastActivity = ContinuousClock.now
    private var reaper: Task<Void, Never>?
    /// Manual unloads waiting for in-flight requests to drain.
    private var unloadWaiters: [SuspensionSlot] = []
    /// Built on the first load, then reused for the process lifetime.
    private var metalContext: MetalContext?

    package init(
        plan: ModelSessionPlan,
        facts: ModelSessionFacts,
        idleTimeout: Duration?,
        loader: @escaping Loader = { plan, context in
            try await plan.makeSession(reusingContext: context)
        }
    ) {
        self.facts = facts
        self.idleTimeout = idleTimeout
        self.maximumContext = plan.maxContext
        self.samplingDefaults = GenerationDefaults.house
        self.loader = { context in try await loader(plan, context) }
    }

    /// The same residency logic for a backend that is not built from a plan.
    package init(
        facts: ModelSessionFacts,
        maximumContext: Int,
        samplingDefaults: GenerationDefaults.Sampling,
        idleTimeout: Duration?,
        loader:
            @escaping @Sendable (MetalContext) async throws
            -> any ServerInferenceBackend
    ) {
        self.facts = facts
        self.idleTimeout = idleTimeout
        self.maximumContext = maximumContext
        self.samplingDefaults = samplingDefaults
        self.loader = loader
    }

    // MARK: - ServerInferenceBackend

    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let active = try await acquire()
        defer { release() }
        return try await active.generate(request, onEvent: onEvent)
    }

    package func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        let active = try await acquire()
        defer { release() }
        guard let counting = active as? any PromptTokenCounting else {
            throw ServerRequestError.unsupportedOperation("count_tokens")
        }
        return try await counting.countPromptTokens(request)
    }

    /// Releases the model on demand. Waits for in-flight requests to drain
    /// first, so a manual unload never fails a generation; returns true when a
    /// resident session was released. Idempotent: with nothing loaded it
    /// returns false immediately.
    public func unload() async -> Bool {
        // Stop the reaper so it cannot race the manual unload; a later load
        // restarts it.
        reaper?.cancel()
        reaper = nil
        // A load in flight is not "nothing loaded": it is about to become
        // resident. Returning false here left the model mapped while telling the
        // caller nothing had been released, so the request that motivated the
        // endpoint did not free the memory it exists to free.
        //
        // The loop re-checks rather than trusting a single await: a concurrent
        // awaiter of the same task can resume before `residentSession` has
        // assigned `session`, and that function clears `loadTask` in a `defer`
        // after the assignment — so waiting for the field to clear is waiting for
        // the assignment to have happened.
        while let task = loadTask, !Task.isCancelled {
            _ = try? await task.value
            await Task.yield()
        }
        while session != nil {
            if Task.isCancelled {
                // Giving up without unloading: the model is still resident, so
                // hand the idle timer back. Without this, a client that
                // disconnects mid-unload silently disables --idle-unload-seconds
                // for the life of the session.
                startReaper()
                return false
            }
            if inFlight == 0 {
                session = nil
                ServerLog.residency("unloaded")
                return true
            }
            let slot = SuspensionSlot()
            unloadWaiters.append(slot)
            await withTaskCancellationHandler {
                await withCheckedContinuation { slot.handOver($0) }
            } onCancel: {
                Task { await self.cancelUnload(slot) }
            }
        }
        return false
    }

    private func cancelUnload(_ slot: SuspensionSlot) {
        if let index = unloadWaiters.firstIndex(where: { $0 === slot }) {
            unloadWaiters.remove(at: index)
        }
        slot.signal(.cancel)
    }

    // MARK: - Residency

    /// Marks the request in-flight *before* the first suspension point, so the
    /// reaper can never see a zero count while a request is starting up.
    private func acquire() async throws -> any ServerInferenceBackend {
        inFlight += 1
        lastActivity = .now
        do {
            return try await residentSession()
        } catch {
            inFlight -= 1
            noteIdle()
            throw error
        }
    }

    private func release() {
        inFlight -= 1
        // Measured from when a request finished, so a long generation does not
        // count against the idle window.
        lastActivity = .now
        noteIdle()
    }

    /// Wakes manual unloads once the last in-flight request has drained.
    private func noteIdle() {
        guard inFlight == 0, !unloadWaiters.isEmpty else { return }
        let waiters = unloadWaiters
        unloadWaiters.removeAll()
        for waiter in waiters {
            waiter.signal(.wake)
        }
    }

    private func residentSession() async throws -> any ServerInferenceBackend {
        if let session { return session }
        if let loadTask { return try await loadTask.value }

        // Build the Metal context once; every later reload reuses it.
        let context: MetalContext
        if let metalContext {
            context = metalContext
        } else {
            context = try MetalContext()
            metalContext = context
        }

        let loader = self.loader
        let task = Task { try await loader(context) }
        loadTask = task
        defer { loadTask = nil }

        // On failure `session` stays nil, so the next request retries cleanly
        // rather than inheriting a half-built session.
        let loaded = try await task.value
        session = loaded
        lastActivity = .now
        startReaper()
        ServerLog.residency("loaded" + ServerLog.promptCacheField(for: loaded))
        return loaded
    }

    /// Decides the reaper's next action. Split out so the policy is testable
    /// without waiting on real time.
    func reaperStep(now: ContinuousClock.Instant = .now) -> ReaperStep {
        guard let idleTimeout, session != nil else { return .stop }
        // A request in flight or a load in progress: the idle clock is not
        // running, so check again no sooner than a full timeout from now.
        guard inFlight == 0, loadTask == nil else { return .sleep(idleTimeout) }
        let idleFor = now - lastActivity
        guard idleFor >= idleTimeout else { return .sleep(idleTimeout - idleFor) }
        return .stop
    }

    private func startReaper() {
        guard idleTimeout != nil, reaper == nil else { return }
        reaper = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                switch await self.reaperStep() {
                case .sleep(let duration):
                    // A cancelled sleep exits the loop on the next check.
                    try? await Task.sleep(for: duration)
                case .stop:
                    await self.unloadIfIdle()
                    return
                }
            }
        }
    }

    /// Releases the session if it is still idle. Re-checks under actor
    /// isolation because the reaper's decision was made before its sleep.
    private func unloadIfIdle() {
        reaper = nil
        // Nothing to reap: a manual unload or shutdown already released the
        // session while the reaper was waking. Restarting the countdown here
        // would spawn a reaper that immediately finds nothing and respawns
        // itself forever.
        guard session != nil else { return }
        guard case .stop = reaperStep() else {
            // Activity arrived while the reaper was waking; keep the model and
            // restart the countdown.
            startReaper()
            return
        }
        session = nil  // sole strong reference — the weights go here
        ServerLog.residency("unloaded")
    }

    // MARK: - Lifecycle

    /// Stops the reaper and drops the session. Call after the server has shut
    /// down, or the reaper outlives the process's useful life.
    public func shutdown() {
        reaper?.cancel()
        reaper = nil
        session = nil
    }

    // MARK: - Test hooks

    var isLoaded: Bool { session != nil }
    var inFlightCount: Int { inFlight }
    var hasReaper: Bool { reaper != nil }

    /// Pretend the last activity happened `duration` ago, so idle policy can be
    /// tested without sleeping.
    func backdateActivity(by duration: Duration) {
        lastActivity = .now - duration
    }

    func unloadNow() {
        session = nil
    }

    /// Simulates a request sitting inside `generate` while the reaper wakes.
    func bumpInFlightForTesting(_ delta: Int) {
        inFlight += delta
    }
}
