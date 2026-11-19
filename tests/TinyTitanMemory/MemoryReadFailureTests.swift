import Foundation
import Testing

@testable import TinyTitanMemory

/// A memory *read* that failed must not be reported as a memory that is empty.
///
/// AUD-134 is the family of `try?`-and-`[]` sites in the service: a store that
/// threw was answered with an empty list, so the log said "this workspace has
/// no facts on file" and the caller went on deduping against a pool that was
/// never read. The fix splits in two — the reads a caller can act on now throw
/// (`recordedFacts`, the journal's three reads), and the reads that still have
/// to produce a value say so once per operation and workspace. These tests are
/// the second half, plus one line each for the first.
@Suite struct MemoryReadFailureTests {

    private func configuration() -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        return configuration
    }

    private func scope(_ workspace: String = "repo-a") throws -> MemoryScope {
        try MemoryScope(namespace: "tinytitan", user: "local", workspace: workspace)
    }

    private func fact(_ key: String, _ value: String) throws -> MemoryRecord {
        MemoryRecord(key: try MemoryKey(validating: key), value: value)
    }

    /// An in-process store that answers as if it were unreachable.
    ///
    /// Only the reads are wired to fail: a failing *write* would set
    /// `isDegraded`, and `activeStore(for:)` then swaps in the local store, so
    /// the test would be measuring the fallback rather than the read.
    private struct FlakyStore: MemoryStore {
        let inner: InMemoryStore
        /// Operations that throw in every workspace.
        let failing: Set<String>
        /// Workspaces whose `search` throws, for a failure that is local to one
        /// scope — which is what makes the shared pool and the project pool
        /// separately observable.
        let searchFailsIn: Set<String>

        init(
            _ inner: InMemoryStore = InMemoryStore(),
            failing: Set<String> = [],
            searchFailsIn: Set<String> = []
        ) {
            self.inner = inner
            self.failing = failing
            self.searchFailsIn = searchFailsIn
        }

        private func check(_ operation: String, _ scope: MemoryScope) throws {
            if failing.contains(operation)
                || (operation == "search" && searchFailsIn.contains(scope.workspace))
            {
                throw MemoryError.backendUnavailable("connection refused")
            }
        }

        func get(_ key: MemoryKey, in scope: MemoryScope) async throws -> MemoryRecord? {
            try check("get", scope)
            return try await inner.get(key, in: scope)
        }

        func set(_ record: MemoryRecord, in scope: MemoryScope) async throws {
            try await inner.set(record, in: scope)
        }

        func delete(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
            try check("delete", scope)
            return try await inner.delete(key, in: scope)
        }

        func exists(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
            try check("exists", scope)
            return try await inner.exists(key, in: scope)
        }

        func list(
            prefix: String, limit: Int, in scope: MemoryScope
        ) async throws -> [MemoryKey] {
            try check("list", scope)
            return try await inner.list(prefix: prefix, limit: limit, in: scope)
        }

        func search(_ query: MemoryQuery, in scope: MemoryScope) async throws -> [MemoryRecord] {
            try check("search", scope)
            return try await inner.search(query, in: scope)
        }

        func append(
            _ text: String, to key: MemoryKey, in scope: MemoryScope
        ) async throws -> MemoryRecord {
            try check("append", scope)
            return try await inner.append(text, to: key, in: scope)
        }

        func sessionInit(
            _ session: MemorySession, in scope: MemoryScope
        ) async throws -> MemoryBootstrap {
            try await inner.sessionInit(session, in: scope)
        }
    }

    /// Never answers, so every rule-based decision falls through to the write.
    ///
    /// unchecked-invariant: `pairs` is only ever touched under `lock`.
    private final class SilentSideEngine: MemorySideEngine, @unchecked Sendable {
        private let lock = NSLock()
        private var storedPairs: [(MemoryFact, MemoryFact)] = []
        var pairs: [(MemoryFact, MemoryFact)] { lock.withLock { storedPairs } }

        func isDurable(_ fact: MemoryFact) async -> Bool? { nil }
        func supersedes(
            _ stored: MemoryFact, _ new: MemoryFact, rule: String?
        ) async -> MemorySupersession? { nil }

        func duplicates(_ stored: MemoryFact, _ new: MemoryFact) async -> Bool? {
            lock.withLock { storedPairs.append((stored, new)) }
            return nil
        }

        func contradicts(_ stored: MemoryFact, _ new: MemoryFact) async -> Bool? { nil }
        func couldAnswer(_ question: String, _ fact: MemoryFact) async -> Bool? { nil }
    }

    /// AUD-138, the same defect in the background T7 sweep: the coverage read
    /// used to be `try?` into an empty candidate list, so an unreadable store
    /// and a workspace with nothing worth judging produced the same zero-hint
    /// run. The queue is the only evidence, and one of those two is a bug.
    @Test func theHinterReportsAnUnreadableCoveragePool() async throws {
        let events = EventBox()
        let service = MemoryService(
            configuration: configuration(),
            durableStore: FlakyStore(failing: ["search"]),
            sideEngine: SilentSideEngine(),
            isIdle: { true },
            log: { events.append($0) })
        let context = try #require(await service.beginSession(id: "s-hint"))
        _ = await service.execute(
            name: "memory_search",
            arguments: ["query": .string("how often does the boat cross")],
            in: context)
        await service.waitForRetrievalHints()

        let reports = events.containing("degraded during hint-coverage")
        #expect(reports.count == 1, "one report per scope, got \(reports)")
        #expect(
            reports.first?.contains("no facts were judged") == true,
            "the line must say what the failure cost: \(reports)")
        // Once per workspace: a second question to the same broken store adds
        // no line.
        _ = await service.execute(
            name: "memory_search",
            arguments: ["query": .string("what colour are her eyes")],
            in: context)
        await service.waitForRetrievalHints()
        #expect(events.containing("degraded during hint-coverage").count == 1)
    }

    /// unchecked-invariant: `events` is only ever touched under `lock`.
    private final class EventBox: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [MemoryLogEvent] = []
        func append(_ event: MemoryLogEvent) { lock.withLock { events.append(event) } }
        func messages() -> [String] { lock.withLock { events.map(\.message) } }
        func containing(_ fragment: String) -> [String] {
            messages().filter { $0.contains(fragment) }
        }
    }

    // MARK: - The reads that propagate

    /// The old shape was `try?` and `[]`, which the server rendered as "this
    /// workspace has no facts on file" — the sentence that sends someone to
    /// re-explain their project to a model that still remembers it.
    @Test func theFactListThrowsRatherThanAnsweringEmpty() async throws {
        let service = MemoryService(
            configuration: configuration(),
            durableStore: FlakyStore(failing: ["search"]))
        await #expect(throws: MemoryError.self) {
            try await service.recordedFacts(in: try scope())
        }
    }

    // MARK: - The reads that still have to answer

    /// An unreadable dedup pool is reported and the fact is still written: the
    /// caller has already spent a session on the extraction, so the loss is the
    /// dedup, not the fact.
    @Test func anUnreadableDedupPoolIsReportedAndTheFactStillLands() async throws {
        let store = FlakyStore(failing: ["search"])
        let events = EventBox()
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: SilentSideEngine(),
            log: { events.append($0) })
        let context = try #require(await service.beginSession(id: "s-pool"))

        let written = await service.storeConsolidation(
            [try fact("decisions/sync", "FooManager stays.")], in: context)

        #expect(written == 1, "a failed read must not cost the fact")
        #expect(
            try await store.inner.get(
                MemoryKey(validating: "decisions/sync"), in: context.scope) != nil)
        let reports = events.containing("degraded during dedup-pool")
        #expect(reports.count == 1, "exactly one report, got \(reports)")
        #expect(
            reports.first?.contains("answering as unknown") == true,
            "the log must not read like an empty workspace: \(reports)")
        #expect(reports.first?.contains("repo-a") == true)
    }

    /// Two facts, one unreadable `get`: one line, not two. The counter is the
    /// same once-per-operation-and-workspace rule the journal failure uses, so
    /// a store that stays broken produces one line rather than one per fact.
    @Test func anUnreadableCurrentValueIsReportedOnceForTheWorkspace() async throws {
        let store = FlakyStore(failing: ["get"])
        let events = EventBox()
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            log: { events.append($0) })
        let context = try #require(await service.beginSession(id: "s-current"))

        let written = await service.storeConsolidation(
            [
                try fact("decisions/sync", "FooManager stays."),
                try fact("decisions/auth", "Tokens rotate weekly."),
            ],
            in: context)

        #expect(written == 2)
        #expect(events.containing("degraded during current-value").count == 1)
    }

    /// The shared workspace's pool, read for a global fact, is a different
    /// store from the project's. When it cannot be read the pool is empty and
    /// *not* the project's records: deduping a person's preference against
    /// another workspace's facts judges it against the wrong store, and here it
    /// would silently ask the engine about `rules/ferry` — the project's.
    @Test func anUnreadableSharedPoolIsNotReplacedByTheProjectPool() async throws {
        let inner = InMemoryStore()
        let project = try scope()
        try await inner.set(
            try fact("preferences/tone", "Dry, no exclamation marks."), in: project)
        let store = FlakyStore(inner, searchFailsIn: [MemoryConfiguration.sharedWorkspace])
        let events = EventBox()
        let engine = SilentSideEngine()
        let service = MemoryService(
            configuration: configuration(),
            durableStore: store,
            sideEngine: engine,
            log: { events.append($0) })
        let context = try #require(await service.beginSession(id: "s-shared"))

        var global = try fact("preferences/language", "Answers in German.")
        global.isGlobal = true
        let written = await service.storeConsolidation([global], in: context)

        #expect(written == 1)
        #expect(engine.pairs.isEmpty, "the project's pool was used for a shared fact")
        let reports = events.containing("degraded during dedup-pool")
        #expect(reports.count == 1, "the shared pool failure was not reported: \(reports)")
        #expect(reports.first?.contains(MemoryConfiguration.sharedWorkspace) == true)
        let shared = try #require(
            await store.inner.get(
                MemoryKey(validating: "preferences/language"),
                in: try scope(MemoryConfiguration.sharedWorkspace)))
        #expect(shared.value == "Answers in German.")
    }
}
