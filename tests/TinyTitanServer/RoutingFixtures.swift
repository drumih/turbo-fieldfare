import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

// Shared by the router and dynamic-serving tests: a three-model catalog that
// exists only in memory, and stub models that record what they were asked.

/// unchecked-invariant: every field is read and written under `lock`.
final class RoutingEventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _events: [String] = []
    private var _requests: [ValidatedChatRequest] = []
    private var _choices: [String: ReasoningChoice] = [:]
    private var active = 0
    private var _maxActive = 0

    func append(_ event: String) { lock.withLock { _events.append(event) } }
    var events: [String] { lock.withLock { _events } }
    var loads: [String] { events.filter { $0.hasPrefix("load ") } }
    var departures: Int { events.filter { $0.hasPrefix("departed ") }.count }
    var releases: Int { events.filter { $0.hasPrefix("released") }.count }
    func index(of event: String) -> Int? { events.firstIndex(of: event) }

    func record(_ request: ValidatedChatRequest) { lock.withLock { _requests.append(request) } }
    var requests: [ValidatedChatRequest] { lock.withLock { _requests } }

    func choose(_ id: String, _ choice: ReasoningChoice) { lock.withLock { _choices[id] = choice } }
    var choices: [String: ReasoningChoice] { lock.withLock { _choices } }

    func enterGeneration() {
        lock.withLock {
            active += 1
            _maxActive = max(_maxActive, active)
        }
    }
    func leaveGeneration() { lock.withLock { active -= 1 } }
    var maxConcurrentGenerations: Int { lock.withLock { _maxActive } }
}

/// Holds a generation open until the test opens it.
///
/// One slot per waiter, for the same reason the engine keeps one per queued
/// caller (ledger AUD-143): the continuation is handed to the slot, never stored
/// in the actor's own memory from inside the suspension closure. A single stored
/// continuation also lost every waiter but the last — two generations on the same
/// model overlapped, and only one of them was ever released.
actor RoutingGate {
    private var slots: [SuspensionSlot] = []
    private var opened = false

    var isWaiting: Bool { !slots.isEmpty }

    func wait() async {
        guard !opened else { return }
        let slot = SuspensionSlot()
        slots.append(slot)
        await withTaskCancellationHandler {
            await withCheckedContinuation { slot.handOver($0) }
        } onCancel: {
            Task { await self.drop(slot) }
        }
    }

    private func drop(_ slot: SuspensionSlot) {
        slots.removeAll { $0 === slot }
        slot.signal(.cancel)
    }

    func open() {
        opened = true
        let woken = slots
        slots.removeAll()
        for slot in woken { slot.signal(.wake) }
    }
}

actor RoutedStubModel: ServerInferenceBackend, PromptTokenCounting {
    let id: String
    let log: RoutingEventLog
    let gate: RoutingGate?
    let delay: Duration?

    init(id: String, log: RoutingEventLog, gate: RoutingGate?, delay: Duration?) {
        self.id = id
        self.log = log
        self.gate = gate
        self.delay = delay
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        log.record(request)
        log.enterGeneration()
        defer { log.leaveGeneration() }
        log.append("start \(id)")
        if let gate { await gate.wait() }
        if let delay { try await Task.sleep(for: delay) }
        log.append("end \(id)")
        onEvent(.content(id))
        return ServerCompletion(
            content: id, toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        log.append("count \(id)")
        return 7
    }
}

struct RoutingStubFailure: Error {}

enum RoutingFixture {
    /// A binary-thinking GPU install at the house settings.
    static let alpha = ModelCatalog.Entry(
        id: "alpha_4-Bit", name: "Alpha 35B", kind: .gpu(.qwen36), quant: 4,
        path: URL(fileURLWithPath: "/models/alpha_4Bit"),
        sampling: GenerationDefaults.house)
    /// An effort-level GPU install whose card asks for temperature 1.0.
    static let flash = ModelCatalog.Entry(
        id: "flash_8-Bit", name: "Flash 125B", kind: .gpu(.qwen38flash), quant: 8,
        path: URL(fileURLWithPath: "/models/flash_8Bit"),
        sampling: GenerationDefaults.Sampling(temperature: 1.0, topK: 20, topP: 0.95))
    /// A dense Qwen 3.5 install: the one shape both engines implement, so it
    /// is listed once and reachable as `@gpu` (its default) and `@cpu`.
    static let dense = ModelCatalog.Entry(
        id: "dense-2b_4-Bit", name: "Dense 2B", kind: .gpu(.qwen35Dense), quant: 4,
        path: URL(fileURLWithPath: "/models/dense_2B_4Bit"),
        sampling: GenerationDefaults.house, engines: [.gpu, .cpu])

    /// A CPU snapshot, whose context the CPU engine caps.
    static let small = ModelCatalog.Entry(
        id: "small-2b", name: "Small 2B", kind: .cpu(.qwen35Dense), quant: 8,
        path: URL(fileURLWithPath: "/models/small_2B_8Bit"),
        sampling: CPUModelFamily.qwen35Dense.samplingDefaults,
        contextLimit: 262_144)

    static var catalog: ModelCatalog {
        ModelCatalog(
            directory: URL(fileURLWithPath: "/models"),
            entries: [alpha, flash, small, dense])
    }

    static let configuredContext = 65_536

    static func router(
        initial: String = alpha.id,
        reasoning: ReasoningLevel = .off,
        log: RoutingEventLog,
        gates: [String: RoutingGate] = [:],
        failing: Set<String> = [],
        delay: Duration? = nil
    ) throws -> ModelRouter {
        try ModelRouter(
            catalog: catalog, initialModelID: initial, reasoning: reasoning,
            maximumContext: configuredContext,
            loader: { entry, choice in
                log.append("load \(entry.id)")
                log.choose(entry.id, choice)
                if failing.contains(entry.id) { throw RoutingStubFailure() }
                return RoutedStubModel(id: entry.id, log: log, gate: gates[entry.id], delay: delay)
            },
            // A different number from the loaded stub's 7, so a test can
            // tell which of the two answered.
            counter: { entry, _, _ in
                log.append("tokenize \(entry.id)")
                return 5
            })
    }

    static func request(_ model: String?) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: "hi")],
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 4, temperature: 0),
            maximumCompletionTokens: 4,
            model: model)
    }

    /// Polls a condition instead of sleeping a fixed time, so a slow machine
    /// makes the test slower rather than wrong.
    static func eventually(
        _ what: String,
        timeout: Duration = .seconds(10),
        _ condition: @Sendable () async -> Bool
    ) async {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if await condition() { return }
            try? await Task.sleep(for: .milliseconds(5))
        }
        Issue.record("timed out waiting for \(what)")
    }
}
