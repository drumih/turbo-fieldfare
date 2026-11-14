import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

// The router's failure modes and their wire descriptions.
//
// Split out of `ModelRouter.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion.
public enum ModelRouterError: Error, CustomStringConvertible {
    case notInCatalog(String)

    public var description: String {
        switch self {
        case .notInCatalog(let model):
            "\(model) is not a model the catalog can serve; run with --catalog to list them"
        }
    }
}

/// Serves every catalog model through one server, keeping at most one resident.
///
/// A request names its model; if that is not the resident one, the router
/// waits for in-flight work on the resident model to drain, releases it, and
/// loads the requested one. It never holds two: on a 26 GB machine two 35B
/// models would swap, so the old weights are dropped before the new ones are
/// mapped, and a switch costs a full load.
///
/// The HTTP coordinator already runs one generation at a time, so a switch
/// normally happens inside that serialised section with nothing else running.
/// The router does not rely on it: token counting and memory consolidation
/// reach the backend outside the coordinator, so residency is guarded here by
/// an in-flight count, exactly as `ManagedModelBackend` guards its unloads.
public actor ModelRouter: ServerInferenceBackend, ResidencyManaging, PromptTokenCounting,
    ModelRouting
{
    /// Builds a backend for one catalog entry. Injectable so switching can be
    /// tested against stubs without a model on disk.
    package typealias Loader =
        @Sendable (ModelCatalog.Entry, ReasoningChoice) async throws -> any ServerInferenceBackend
    /// Counts a request's prompt tokens for a model that is not resident,
    /// from its tokenizer alone. Injectable for the same reason.
    package typealias Counter =
        @Sendable (ModelCatalog.Entry, ReasoningChoice, ValidatedChatRequest) async throws -> Int

    public nonisolated let servedModels: [ServedModel]
    public nonisolated let initialModelID: String
    private nonisolated let initial: ServedModel
    private nonisolated let entries: [String: ModelCatalog.Entry]
    private nonisolated let choices: [String: ReasoningChoice]
    private let loader: Loader
    private let counter: Counter

    /// The protocol's single-model view, answered for the model loaded first.
    /// The HTTP layer asks `servedModel(named:)` instead once it routes.
    public nonisolated var maximumContext: Int { initial.maximumContext }
    public nonisolated var samplingDefaults: GenerationDefaults.Sampling { initial.sampling }

    private struct Resident {
        let id: String
        let backend: any ServerInferenceBackend
    }

    /// One queued caller. The slot is what it suspends on; the kind says whether
    /// the caller is counted in `pendingSwitches`, which is the barrier that keeps
    /// new work for the resident model behind a pending switch.
    private struct QueuedTurn {
        enum Kind {
            case residentWork
            case switcher
        }

        let slot: SuspensionSlot
        let kind: Kind
    }

    private var resident: Resident?
    /// Callers currently using the resident backend. Non-zero blocks a switch.
    private var inFlight = 0
    /// A load is running; nothing is resident until it finishes.
    private var switching = false
    /// Callers waiting to switch away from the resident model. While any wait,
    /// new work for the resident model queues behind them rather than keeping
    /// the in-flight count above zero forever.
    private var pendingSwitches = 0
    private var waiters: [QueuedTurn] = []

    package init(
        catalog: ModelCatalog,
        initialModelID: String,
        reasoning: ReasoningLevel,
        maximumContext: Int,
        loader: @escaping Loader,
        counter: @escaping Counter = ModelRouter.standardCounter
    ) throws {
        var served: [ServedModel] = []
        var choices: [String: ReasoningChoice] = [:]
        var entries: [String: ModelCatalog.Entry] = [:]
        /// Registers an entry under its own id, and remembers it for the
        /// `/v1/models` listing only once per install.
        func register(_ entry: ModelCatalog.Entry, into servedList: inout [ServedModel]) throws {
            let choice = try ReasoningFallback.choice(for: entry.kind, requested: reasoning)
            choices[entry.id] = choice
            entries[entry.id] = entry
            servedList.append(
                ServedModel(
                    id: entry.id,
                    displayName: entry.name,
                    maximumContext: Self.context(for: entry, configured: maximumContext),
                    sampling: entry.sampling,
                    reasoningProfile: Self.profile(for: entry, choice: choice)))
        }
        for entry in catalog.entries {
            try register(entry, into: &served)
            // Every engine the install can be served by, named by a suffix, so
            // one request can ask for the CPU copy of a dense model and the
            // next for the GPU one: `qwen3.5-2b_4-Bit@cpu`. The aliases share
            // the install and are listed beside it.
            for engine in entry.engines {
                let aliasID = "\(entry.id)@\(engine.rawValue)"
                guard aliasID != entry.id,
                    let alias = entry.served(by: engine, id: aliasID)
                else { continue }
                // A single-engine install gets the alias so the explicit
                // spelling resolves, but it is not *listed*: it names the same
                // engine as the bare id, and two entries for one model in
                // `/v1/models` is noise. A second engine is a real choice and
                // is listed.
                if entry.engines.count > 1, engine != entry.backend {
                    try register(alias, into: &served)
                } else {
                    let choice = try ReasoningFallback.choice(
                        for: alias.kind,
                        requested: reasoning)
                    choices[alias.id] = choice
                    entries[alias.id] = alias
                }
            }
        }
        guard let initial = served.first(where: { $0.id == initialModelID }) else {
            throw ModelRouterError.notInCatalog(initialModelID)
        }
        self.servedModels = served
        self.initialModelID = initialModelID
        self.initial = initial
        self.entries = entries
        self.choices = choices
        self.loader = loader
        self.counter = counter
    }

    /// Mirrors `CPUModelBackend`'s own clamp, so validation bounds max_tokens
    /// by the context the CPU engine will actually give the request.
    static func context(for entry: ModelCatalog.Entry, configured: Int) -> Int {
        switch entry.backend {
        case .gpu: configured
        case .cpu:
            min(
                configured, entry.contextLimit ?? CPUModelBackend.contextCeiling,
                CPUModelBackend.contextCeiling)
        }
    }

    /// A CPU family's template has the binary switch Qwen 3.6 has, so it is
    /// validated as that family, as the single-model CPU path already does.
    static func profile(
        for entry: ModelCatalog.Entry,
        choice: ReasoningChoice
    ) -> ServerReasoningProfile {
        let family: ModelFamily
        switch entry.kind {
        case .gpu(let gpuFamily): family = gpuFamily
        case .cpu: family = .qwen36
        }
        return ServerReasoningProfile(
            family: family, thinkingMode: choice.thinking,
            reasoningEffort: choice.effort)
    }

    @available(
        *, deprecated,
        message: "No caller (TT-051): callers index the router's choices map directly."
    )
    public func reasoningChoice(for id: String) -> ReasoningChoice? { choices[id] }

    // MARK: - ServerInferenceBackend

    /// A request with no model -- the engine's own, such as memory
    /// consolidation -- runs on whatever is resident rather than forcing a load.
    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let active = try await acquire(request.model)
        defer { release() }
        return try await active.generate(request, onEvent: onEvent)
    }

    /// A count is a question about text, not a reason to switch. One naming
    /// a model that is not resident is answered from that model's tokenizer
    /// and the resident model stays loaded; routing it like a generation
    /// made the next generation pay a full reload for a number.
    package func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        let target = request.model ?? resident?.id ?? initialModelID
        if target != resident?.id, let entry = entries[target], let choice = choices[target] {
            return try await counter(entry, choice, request)
        }
        let active = try await acquire(request.model)
        defer { release() }
        guard let counting = active as? any PromptTokenCounting else {
            throw ServerRequestError.unsupportedOperation("count_tokens")
        }
        return try await counting.countPromptTokens(request)
    }

    /// Loads the initial model, so a server that was not asked to defer pays
    /// the load at startup rather than on the first request.
    public func preload() async throws {
        _ = try await acquire(initialModelID)
        release()
    }

    // MARK: - ResidencyManaging

    /// Releases the resident model once in-flight work drains. The next
    /// request loads whichever model it names.
    public func unload() async -> Bool {
        while resident != nil || switching {
            if !switching, inFlight == 0, let released = resident {
                resident = nil
                ServerLog.residency("unloaded \(released.id)")
                return true
            }
            if Task.isCancelled { return false }
            let slot = SuspensionSlot()
            waiters.append(QueuedTurn(slot: slot, kind: .residentWork))
            await wait(on: slot)
        }
        return false
    }

    public func shutdown() {
        resident = nil
    }

    public var residentModelID: String? { resident?.id }

    // MARK: - Residency

    /// Marks the caller in flight on the backend for `requested`, switching to
    /// it first when it is not resident. Every exit that does not return a
    /// backend leaves the in-flight count untouched.
    private func acquire(_ requested: String?) async throws -> any ServerInferenceBackend {
        let target = requested ?? resident?.id ?? initialModelID
        guard let entry = entries[target], let choice = choices[target] else {
            throw ServerRequestError.unknownModel
        }
        // Set when this caller is counted in `pendingSwitches` and not yet holding
        // the switch; whoever still holds the count on the way out gives it back.
        var countedSwitcher = false
        while true {
            if Task.isCancelled {
                if countedSwitcher { releaseSwitchClaim() }
                throw CancellationError()
            }
            let step = turn(
                for: target, entry: entry, choice: choice, countedSwitcher: countedSwitcher)
            switch step {
            case .proceed(let backend):
                // Served by the resident model: whatever switch claim this caller
                // held is spent, and leaving it up would block every later caller
                // behind a switch that is no longer coming.
                if countedSwitcher { releaseSwitchClaim() }
                countedSwitcher = false
                return backend
            case .switchNow:
                countedSwitcher = false
                return try await switchTo(entry, choice: choice)
            case .wait(let turn):
                if countedSwitcher, turn.kind != .switcher {
                    releaseSwitchClaim()
                    countedSwitcher = false
                }
                if turn.kind == .switcher { countedSwitcher = true }
                await wait(on: turn.slot)
                if turn.slot.received == .cancel || Task.isCancelled {
                    if countedSwitcher { releaseSwitchClaim() }
                    throw CancellationError()
                }
            }
        }
    }

    /// Decide what this caller may do and, if it must wait, register it — in one
    /// step containing no suspension. That atomicity is the guarantee: a wake
    /// cannot land between reading `switching`/`pendingSwitches` and joining the
    /// queue, which is how a switcher used to be overtaken by fresh work for the
    /// resident model. The claim for a switch is taken here too, because returning
    /// from this method and reaching `switchTo` is itself a suspension point.
    private func turn(
        for target: String,
        entry: ModelCatalog.Entry,
        choice: ReasoningChoice,
        countedSwitcher: Bool
    ) -> Turn {
        if let resident, resident.id == target, !switching, pendingSwitches == 0 {
            inFlight += 1
            return .proceed(resident.backend)
        }
        if resident?.id != target, !switching, inFlight == 0 {
            beginSwitch(counted: countedSwitcher)
            return .switchNow
        }
        let slot = SuspensionSlot()
        let kind: QueuedTurn.Kind = resident?.id != target ? .switcher : .residentWork
        if kind == .switcher && !countedSwitcher { pendingSwitches += 1 }
        let turn = QueuedTurn(slot: slot, kind: kind)
        waiters.append(turn)
        return .wait(turn)
    }

    private enum Turn {
        case proceed(any ServerInferenceBackend)
        case switchNow
        case wait(QueuedTurn)
    }

    /// Raise the barrier for the duration of the switch. The count this caller held
    /// is spent here, so from this step until `switchTo`'s exit nothing can let
    /// resident work past a queued switcher.
    private func beginSwitch(counted: Bool) {
        if counted, pendingSwitches > 0 { pendingSwitches -= 1 }
        switching = true
    }

    /// A queued switcher that is giving up: without this the barrier stays raised
    /// and the resident model starves new work forever.
    private func releaseSwitchClaim() {
        if pendingSwitches > 0 { pendingSwitches -= 1 }
        wakeWaiters()
    }

    private func switchTo(
        _ entry: ModelCatalog.Entry,
        choice: ReasoningChoice
    ) async throws -> any ServerInferenceBackend {
        defer {
            switching = false
            wakeWaiters()
        }
        if let previous = resident {
            // Dropped before the next model is mapped: this reference is the
            // last one, so the old weights are gone before the new ones load.
            resident = nil
            ServerLog.residency("unloaded \(previous.id) to load \(entry.id)")
        }
        // Unstructured, so a client that disconnects mid-load does not abort a
        // load that the requests queued behind it are waiting for.
        let loader = self.loader
        let loaded = try await Task { try await loader(entry, choice) }.value
        resident = Resident(id: entry.id, backend: loaded)
        inFlight += 1
        let fitted =
            choice.effective == choice.requested
            ? "" : " (server level \(choice.requested.rawValue))"
        ServerLog.residency(
            "loaded \(entry.id) on the \(entry.backend.rawValue)"
                + ServerLog.promptCacheField(for: loaded)
                + " reasoning=\(choice.effective.rawValue)\(fitted)")
        return loaded
    }

    /// Marks the caller's wait over and the in-flight count released.
    private func release() {
        inFlight -= 1
        if inFlight == 0 { wakeWaiters() }
    }

    /// Suspend on a slot the actor already holds. Only the slot is reachable from
    /// the continuation closure, and `handOver` is built for a foreign executor:
    /// appending the continuation to `waiters` from here mutated actor state off
    /// the actor, which is ledger AUD-143.
    private func wait(on slot: SuspensionSlot) async {
        await withTaskCancellationHandler {
            await withCheckedContinuation { slot.handOver($0) }
        } onCancel: {
            Task { await self.drop(slot) }
        }
    }

    /// Remove a cancelled caller and wake it. The switch count it may have held is
    /// given back by the caller that owns it, in `acquire`, so this does not touch
    /// `pendingSwitches` and cannot double-count the same departure.
    private func drop(_ slot: SuspensionSlot) {
        if let index = waiters.firstIndex(where: { $0.slot === slot }) {
            waiters.remove(at: index)
        }
        slot.signal(.cancel)
    }

    /// Every waiter re-checks its own condition, so waking all of them is
    /// always safe; the ones that still cannot proceed queue again.
    private func wakeWaiters() {
        let woken = waiters
        waiters.removeAll()
        for waiter in woken {
            waiter.slot.signal(.wake)
        }
    }

    // MARK: - Test hooks

    var inFlightCount: Int { inFlight }
    var waiterCount: Int { waiters.count }
}
