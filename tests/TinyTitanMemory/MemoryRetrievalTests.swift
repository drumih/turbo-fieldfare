import Foundation
import Testing

@testable import TinyTitanMemory

/// The background T7 caller: a search never waits on a judgement, and the
/// hint a later sweep leaves can promote what the token ranking missed.
///
/// The engine is a stub, so the cost that makes this a background caller — a
/// 15.2 s CPU generation per fact — is out of the picture and the wiring is
/// what is under test. `benchmark/side_engine_recall.py` is where the real
/// model's ranking is measured.
@Suite struct MemoryRetrievalTests {
    private func configuration(workspace: String = "repo-r") -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = workspace
        configuration.user = "local"
        return configuration
    }

    private func fact(_ key: String, _ value: String) throws -> MemoryRecord {
        MemoryRecord(key: try MemoryKey(validating: key), value: value)
    }

    /// Runs one `memory_search` and returns the records the model would see.
    private func search(
        _ service: MemoryService,
        _ context: MemorySessionContext,
        _ text: String,
        prefix: String? = nil
    ) async -> [MemoryRecord] {
        var arguments: [String: MemoryToolValue] = ["query": .string(text)]
        if let prefix { arguments["prefix"] = .string(prefix) }
        let result = await service.execute(
            name: "memory_search",
            arguments: arguments,
            in: context)
        guard case .ok(let fields) = result,
            case .records(let records)? = fields["results"]
        else { return [] }
        return records
    }

    @Test func aSearchNeverWaitsForTheEngineAndKeepsTheTokenRanking() async throws {
        let store = InMemoryStore()
        let engine = StubRetrievalEngine { _, _ in true }
        let idle = IdleBox(false)
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            isIdle: { idle.isIdle })
        let context = try #require(await service.beginSession(id: "s-bg"))
        try await store.set(try fact("rules/ferry", "runs only on Sundays"), in: context.scope)
        try await store.set(try fact("setting/town", "Ashgrove"), in: context.scope)

        // No term in the question appears in either fact: the token ranking
        // returns nothing, which is the miss the hint exists to fix.
        let before = await search(service, context, "How often does the boat cross the water?")
        #expect(before.isEmpty)
        // The idle gate is closed, so the background task cannot have asked:
        // the search returned without consulting the engine.
        #expect(engine.retrievalQuestions == 0)

        // The idle window opens. The sweep runs and leaves a hint.
        idle.set(true)
        await service.waitForRetrievalHints()
        #expect(engine.retrievalQuestions > 0)

        let after = await search(service, context, "How often does the boat cross the water?")
        #expect(after.first?.key.rawValue == "rules/ferry")
    }

    @Test func aHintReordersTheFactsTheTokenRankingAlreadyReturned() async throws {
        let store = InMemoryStore()
        // Both facts carry "town", so both match and the key match wins the
        // token ranking. Only the role fact answers the question.
        let engine = StubRetrievalEngine { _, fact in
            fact.key == "characters/ines/role"
        }
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            isIdle: { true })
        let context = try #require(await service.beginSession(id: "s-order"))
        try await store.set(try fact("setting/town", "Ashgrove"), in: context.scope)
        try await store.set(
            try fact("characters/ines/role", "town archivist"),
            in: context.scope)

        let question = "Who keeps town records?"
        let before = await search(service, context, question)
        #expect(before.first?.key.rawValue == "setting/town")

        await service.waitForRetrievalHints()
        let after = await search(service, context, question)
        #expect(after.first?.key.rawValue == "characters/ines/role")
    }

    @Test func aValueThatChangedIsNotPromotedOnAnAnswerAboutTheOldOne() async throws {
        let store = InMemoryStore()
        let engine = StubRetrievalEngine { _, _ in true }
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            isIdle: { true })
        let context = try #require(await service.beginSession(id: "s-stale"))
        try await store.set(try fact("rules/ferry", "runs only on Sundays"), in: context.scope)

        let question = "How often does the boat cross the water?"
        _ = await search(service, context, question)
        await service.waitForRetrievalHints()
        // The hint now answers about the old value.
        let judged = engine.retrievalQuestions

        // "sails twice a day" shares no term with the question, so only the
        // hint could surface it — and the hint is about a value that is gone.
        try await store.set(try fact("rules/ferry", "sails twice a day"), in: context.scope)
        let after = await search(service, context, question)
        await service.waitForRetrievalHints()

        #expect(after.isEmpty)
        // Already judged, so the stale fact is not asked about again.
        #expect(engine.retrievalQuestions == judged)
    }

    @Test func aHintCannotReturnAFactTheQueryExcluded() async throws {
        let store = InMemoryStore()
        // The engine says yes to a fact outside the requested prefix.
        let engine = StubRetrievalEngine { _, fact in fact.key == "gotchas/ferry" }
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            isIdle: { true })
        let context = try #require(await service.beginSession(id: "s-prefix"))
        try await store.set(try fact("gotchas/ferry", "runs only on Sundays"), in: context.scope)

        let question = "How often does the boat cross the water?"
        _ = await search(service, context, question)
        await service.waitForRetrievalHints()

        let filtered = await search(service, context, question, prefix: "rules/")
        #expect(filtered.isEmpty)
    }

    @Test func oneQuestionCoversAtMostTheCoverageLimit() async throws {
        let store = InMemoryStore()
        let engine = StubRetrievalEngine { _, _ in false }
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            isIdle: { true })
        let context = try #require(await service.beginSession(id: "s-cover"))
        for index in 0..<(MemoryRetrievalHinter.coverageLimit + 16) {
            try await store.set(try fact("bulk/fact\(index)", "value \(index)"), in: context.scope)
        }

        _ = await search(service, context, "what is the value?")
        await service.waitForRetrievalHints()

        // The sweep covers what a search returns — the store's own result cap
        // is 50 — and never more than the hinter's own ceiling.
        #expect(engine.retrievalQuestions == MemoryLimits().maximumSearchResults)
        #expect(engine.retrievalQuestions <= MemoryRetrievalHinter.coverageLimit)
    }

    @Test func withoutASideEngineASearchIsTheTokenRankingAlone() async throws {
        let store = InMemoryStore()
        let service = MemoryService(configuration: configuration(), durableStore: store)
        let context = try #require(await service.beginSession(id: "s-none"))
        try await store.set(try fact("setting/town", "Ashgrove"), in: context.scope)

        let records = await search(service, context, "Which town is this?")
        #expect(records.map(\.key.rawValue) == ["setting/town"])
        await service.waitForRetrievalHints()
    }

    @Test func theSweepLeavesALogLineWithoutTheQuestionOrAValue() async throws {
        let store = InMemoryStore()
        let log = RetrievalLogCollector()
        let engine = StubRetrievalEngine { _, _ in true }
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            isIdle: { true },
            log: { log.append($0) })
        let context = try #require(await service.beginSession(id: "s-log"))
        try await store.set(try fact("rules/ferry", "runs only on Sundays"), in: context.scope)

        _ = await search(service, context, "How often does the boat cross the water?")
        await service.waitForRetrievalHints()

        let lines = log.messages()
        #expect(lines.contains { $0.contains("retrieval hints") })
        for line in lines {
            #expect(!line.contains("boat"))
            #expect(!line.contains("Sundays"))
        }
    }
}

/// Answers retrieval questions from a closure and counts what it was asked.
///
/// unchecked-invariant: `asked` is only ever touched under `lock`.
private final class StubRetrievalEngine: MemorySideEngine, @unchecked Sendable {
    private let lock = NSLock()
    private let answer: @Sendable (String, MemoryFact) -> Bool?
    private var asked: [(String, MemoryFact)] = []

    init(answer: @escaping @Sendable (String, MemoryFact) -> Bool?) {
        self.answer = answer
    }

    var retrievalQuestions: Int { lock.withLock { asked.count } }

    func isDurable(_ fact: MemoryFact) async -> Bool? { nil }
    func duplicates(_ stored: MemoryFact, _ new: MemoryFact) async -> Bool? { nil }
    func contradicts(_ stored: MemoryFact, _ new: MemoryFact) async -> Bool? { nil }
    func supersedes(
        _ stored: MemoryFact, _ new: MemoryFact,
        rule: String?
    ) async -> MemorySupersession? { nil }

    func couldAnswer(_ question: String, _ fact: MemoryFact) async -> Bool? {
        lock.withLock { asked.append((question, fact)) }
        return answer(question, fact)
    }
}

/// A test-owned idle signal.
///
/// unchecked-invariant: `value` is only ever touched under `lock`.
private final class IdleBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Bool

    init(_ value: Bool) { self.value = value }

    var isIdle: Bool { lock.withLock { value } }
    func set(_ newValue: Bool) { lock.withLock { value = newValue } }
}

/// Collects the service's log events.
///
/// unchecked-invariant: `events` is only ever touched under `lock`.
private final class RetrievalLogCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [MemoryLogEvent] = []

    func append(_ event: MemoryLogEvent) { lock.withLock { events.append(event) } }
    func messages() -> [String] { lock.withLock { events.map(\.message) } }
}
