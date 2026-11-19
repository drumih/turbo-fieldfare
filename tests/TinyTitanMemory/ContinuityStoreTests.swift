import ContinuityCore
import Foundation
import Testing

@testable import TinyTitanMemory

/// The durable backend, which used to be Valkey and is now the in-process
/// continuity engine.
///
/// It has to satisfy the same contract `InMemoryStore` does, and two things
/// the reference store never had to: an address mapping that round-trips, and
/// state that survives a restart.
@Suite struct ContinuityStoreTests {
    private func scope(
        _ workspace: String = "repo-a",
        user: String = "local",
        namespace: String = "tinytitan"
    ) throws -> MemoryScope {
        try MemoryScope(namespace: namespace, user: user, workspace: workspace)
    }

    private func key(_ raw: String) throws -> MemoryKey { try MemoryKey(validating: raw) }

    private func makeStore() -> ContinuityStore {
        ContinuityStore(engine: ContinuityEngine())
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("continuity-store-\(UUID().uuidString)")
    }

    // MARK: - The contract

    @Test func setThenGetPreservesEveryField() async throws {
        let store = makeStore()
        let scope = try scope()
        // A session first: the writing session is carried as provenance now,
        // not copied into the value, so it can only be reported for a session
        // the store has actually seen.
        _ = try await store.sessionInit(MemorySession(id: "session-1"), in: scope)
        let record = MemoryRecord(
            key: try key("decisions/sync"),
            value: "FooManager stays; it prevents a sync race.",
            importance: 0.9,
            confidence: 0.6,
            tags: ["sync", "concurrency"],
            sourceSession: "session-1")
        try await store.set(record, in: scope)

        let loaded = try #require(try await store.get(try key("decisions/sync"), in: scope))
        #expect(loaded.value == record.value)
        #expect(loaded.importance == 0.9)
        #expect(loaded.confidence == 0.6)
        #expect(loaded.sourceSession == "session-1")
        #expect(loaded.tags == ["sync", "concurrency"])
    }

    /// A workspace's *second* session is the one that loses its name.
    ///
    /// `loadLabels` fills the id→name map once per task, by walking whichever
    /// sessions exist at that moment, and `sessionInit` opens a session without
    /// labelling it — so any session opened after the workspace's first read
    /// writes facts that read back saying nobody wrote them, which is what
    /// `memory_get` and `memory_list` then show the model as `source_session`.
    @Test func aSessionOpenedAfterTheFirstReadStillSignsItsFacts() async throws {
        let store = makeStore()
        let scope = try scope()
        _ = try await store.sessionInit(MemorySession(id: "session-1"), in: scope)
        _ = try await store.sessionInit(MemorySession(id: "session-2"), in: scope)
        try await store.set(
            MemoryRecord(
                key: try key("decisions/sync"),
                value: "FooManager stays; it prevents a sync race.",
                importance: 0.9,
                sourceSession: "session-2"),
            in: scope)

        let loaded = try #require(try await store.get(try key("decisions/sync"), in: scope))
        #expect(loaded.sourceSession == "session-2")
    }

    @Test func scopesDoNotSeeEachOther() async throws {
        let store = makeStore()
        let first = try scope("repo-a")
        let second = try scope("repo-b")
        try await store.set(MemoryRecord(key: try key("k"), value: "a"), in: first)
        try await store.set(MemoryRecord(key: try key("k"), value: "b"), in: second)
        #expect(try await store.get(try key("k"), in: first)?.value == "a")
        #expect(try await store.get(try key("k"), in: second)?.value == "b")

        let byUser = try scope("repo-a", user: "other")
        #expect(try await store.get(try key("k"), in: byUser) == nil)
        let byNamespace = try scope("repo-a", namespace: "other")
        #expect(try await store.get(try key("k"), in: byNamespace) == nil)
    }

    @Test func missingKeysReadAsNil() async throws {
        let store = makeStore()
        #expect(try await store.get(try key("nothing/here"), in: try scope()) == nil)
        #expect(try await store.exists(try key("nothing/here"), in: try scope()) == false)
    }

    @Test func deleteReportsWhetherSomethingWasThere() async throws {
        let store = makeStore()
        let scope = try scope()
        try await store.set(MemoryRecord(key: try key("gotchas/build"), value: "x"), in: scope)
        #expect(try await store.delete(try key("gotchas/build"), in: scope))
        #expect(try await store.get(try key("gotchas/build"), in: scope) == nil)
        // Deleting twice is not an error, and the second call says nothing
        // was removed.
        #expect(try await store.delete(try key("gotchas/build"), in: scope) == false)
    }

    @Test func listFiltersByPrefixNewestFirst() async throws {
        let store = makeStore()
        let scope = try scope()
        for name in ["decisions/a", "decisions/b", "gotchas/c"] {
            try await store.set(MemoryRecord(key: try key(name), value: name), in: scope)
        }
        let decisions = try await store.list(prefix: "decisions/", limit: 10, in: scope)
        #expect(Set(decisions.map(\.rawValue)) == ["decisions/a", "decisions/b"])
        let all = try await store.list(prefix: "", limit: 10, in: scope)
        #expect(all.count == 3)
        let limited = try await store.list(prefix: "", limit: 1, in: scope)
        #expect(limited.count == 1)
    }

    /// `limit` bounds the result set on the durable backend too.
    ///
    /// It used to be computed and then dropped: `MemoryRanking.rank` never read
    /// it, so a query asking for two got every match up to the 2000-candidate
    /// scan, while the in-memory backend honoured it. One tool result could push
    /// thousands of records into the model's context.
    @Test func searchHonoursTheLimit() async throws {
        let store = makeStore()
        let scope = try scope()
        for index in 0..<6 {
            try await store.set(
                MemoryRecord(
                    key: try key("notes/sync\(index)"),
                    value: "the sync race number \(index)",
                    importance: 0.5), in: scope)
        }

        let all = try await store.search(MemoryQuery(text: "sync", limit: 10), in: scope)
        #expect(all.count == 6)
        let two = try await store.search(MemoryQuery(text: "sync", limit: 2), in: scope)
        #expect(two.count == 2)
    }

    @Test func searchRanksByRelevance() async throws {
        let store = makeStore()
        let scope = try scope()
        try await store.set(
            MemoryRecord(
                key: try key("decisions/sync"),
                value: "the sync race is prevented by FooManager",
                importance: 0.9, tags: ["sync"]), in: scope)
        try await store.set(
            MemoryRecord(
                key: try key("notes/colour"),
                value: "the palette is warm"), in: scope)

        let hits = try await store.search(MemoryQuery(text: "sync", limit: 5), in: scope)
        #expect(hits.first?.key.rawValue == "decisions/sync")
        // A query the store cannot match returns nothing rather than
        // everything, which is the failure that quietly fills a context window.
        let empty = try await store.search(MemoryQuery(text: "kubernetes", limit: 5), in: scope)
        #expect(empty.isEmpty)
    }

    /// Seed the same fact in both backends and ask both the same question.
    ///
    /// The parity is the point. `MemoryStore.search` promises "records matching
    /// a query" and the query type's own comment says text matching is
    /// "substring-and-token based", with `MemoryRanking` scoring each term
    /// separately — but the durable backend handed the whole `text` to the
    /// engine first, and the engine's predicate is one case-insensitive
    /// substring of address and value. So the backend the server runs answered a
    /// different question from the backend every memory test was written
    /// against. Returns the keys each one matched, keyed by backend name.
    private func matchedByBothBackends(
        value: String, tags: [String] = [], asking text: String
    ) async throws -> [String: [String]] {
        let backends: [(String, any MemoryStore)] = [
            ("durable", ContinuityStore(engine: ContinuityEngine())),
            ("reference", InMemoryStore()),
        ]
        var answers: [String: [String]] = [:]
        for (name, store) in backends {
            let storeScope = try scope("search-\(name)")
            try await store.set(
                MemoryRecord(
                    key: try key("decisions/sync"), value: value, importance: 0.9, tags: tags),
                in: storeScope)
            answers[name] =
                try await store.search(MemoryQuery(text: text, limit: 5), in: storeScope)
                .map(\.key.rawValue)
        }
        return answers
    }

    /// Two terms in a query are two terms, in any order, not a phrase.
    ///
    /// Reversed order is the shape that cannot be satisfied by any substring of
    /// a sentence that does not already read "race sync".
    @Test func aQueryIsASetOfTermsNotAPhrase() async throws {
        let answers = try await matchedByBothBackends(
            value: "the sync race is prevented by FooManager holding the lock",
            asking: "race sync")
        #expect(
            answers["reference"] == ["decisions/sync"],
            "the reference store lost the token matching it already had: \(answers)")
        #expect(
            answers["durable"] == ["decisions/sync"],
            "the durable store read the query as a phrase and answered nothing: \(answers)")
    }

    /// A tag is part of what a record says about itself, and the ranker reads it.
    ///
    /// `MemoryRanking.haystack` is key + value + tags; the engine's predicate is
    /// address + value only, so a fact filed under `concurrency` whose sentence
    /// never uses the word was findable on one backend and not the other.
    @Test func aTermThatAppearsOnlyInATagStillMatches() async throws {
        let answers = try await matchedByBothBackends(
            value: "FooManager prevents the race",
            tags: ["concurrency"],
            asking: "concurrency")
        #expect(answers["reference"] == ["decisions/sync"], "\(answers)")
        #expect(
            answers["durable"] == ["decisions/sync"],
            "the durable backend cannot see its own tags from a text query: \(answers)")
    }

    @Test func appendExtendsAnExistingRecordAndCreatesAMissingOne() async throws {
        let store = makeStore()
        let scope = try scope()
        let created = try await store.append("first", to: try key("log/notes"), in: scope)
        #expect(created.value == "first")
        let extended = try await store.append("second", to: try key("log/notes"), in: scope)
        #expect(extended.value == "first\nsecond")
        #expect(try await store.get(try key("log/notes"), in: scope)?.value == "first\nsecond")
    }

    @Test func oversizedValuesAreRefusedWithTheCallersLimit() async throws {
        let store = ContinuityStore(
            engine: ContinuityEngine(),
            limits: TinyTitanMemory.MemoryLimits(maximumValueBytes: 64))
        let scope = try scope()
        await #expect(throws: MemoryError.self) {
            try await store.set(
                MemoryRecord(
                    key: try self.key("big"),
                    value: String(repeating: "x", count: 65)),
                in: scope)
        }
    }

    @Test func bootstrapIsBounded() async throws {
        let limits = TinyTitanMemory.MemoryLimits(bootstrapRecords: 2, bootstrapBytes: 4096)
        let store = ContinuityStore(engine: ContinuityEngine(), limits: limits)
        let scope = try scope()
        for index in 0..<6 {
            try await store.set(
                MemoryRecord(
                    key: try key("k\(index)"),
                    value: "value \(index)",
                    importance: Double(index) / 10),
                in: scope)
        }
        let bootstrap = try await store.sessionInit(MemorySession(id: "s1"), in: scope)
        #expect(bootstrap.records.count == 2)
        #expect(bootstrap.omittedCount == 4)
        // The most important records are the ones that survive the bound.
        #expect(bootstrap.records.first?.key.rawValue == "k5")
    }

    // MARK: - Address mapping

    @Test func keysMapToAddressesAndBack() throws {
        let mapped = ContinuityStore.address(for: try key("decisions/sync"))
        #expect(mapped == ContinuityStore.Address(namespace: "k.decisions", key: "sync"))

        let deep = ContinuityStore.address(for: try key("a/b/c/d"))
        #expect(deep == ContinuityStore.Address(namespace: "k.a.b.c", key: "d"))

        // A one-segment key does not collide with a two-segment one, which is
        // what the leading marker segment is for.
        let flat = ContinuityStore.address(for: try key("sync"))
        #expect(flat == ContinuityStore.Address(namespace: "k", key: "sync"))
        #expect(flat != mapped)
    }

    @Test func foldingIsIdempotentSoAReturnedKeyStillResolves() async throws {
        let store = makeStore()
        let scope = try scope()
        try await store.set(
            MemoryRecord(key: try key("Decisions/Sync.v2"), value: "held"),
            in: scope)

        // The stored key is the folded one, and using it verbatim works.
        let listed = try await store.list(prefix: "", limit: 5, in: scope)
        #expect(listed.map(\.rawValue) == ["decisions/sync-v2"])
        let round = try await store.get(listed[0], in: scope)
        #expect(round?.value == "held")
        // And the original spelling still finds it.
        #expect(try await store.get(try key("Decisions/Sync.v2"), in: scope)?.value == "held")
    }

    @Test func aScopeAlwaysResolvesToTheSameTask() throws {
        let first = ContinuityStore.taskIdentifier(for: try scope("repo-a"))
        let again = ContinuityStore.taskIdentifier(for: try scope("repo-a"))
        let other = ContinuityStore.taskIdentifier(for: try scope("repo-b"))
        #expect(first == again)
        #expect(first != other)
    }

    // MARK: - Durability

    @Test func factsSurviveARestart() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.ndjson")
        let scope = try scope()

        do {
            let engine = ContinuityEngine(journal: try FileJournal(url: url))
            try await engine.start()
            let store = ContinuityStore(engine: engine)
            try await store.set(
                MemoryRecord(
                    key: try key("decisions/storage"),
                    value: "native swift, same process",
                    importance: 0.95,
                    tags: ["architecture"]),
                in: scope)
            await engine.shutDown()
        }

        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        let loaded = try #require(try await store.get(try key("decisions/storage"), in: scope))
        #expect(loaded.value == "native swift, same process")
        #expect(loaded.importance == 0.95)
        #expect(loaded.tags == ["architecture"])
    }

    /// The other half of the label map: after a restart it is rebuilt by
    /// walking the sessions the journal replayed.
    ///
    /// A regression pin rather than a failing test — it asserts behavior AUD-207
    /// did not change, and the point of asserting it is that the map has two
    /// writers and only the open-time one had a test.
    @Test func aRestartedWorkspaceStillKnowsWhichSessionWroteWhat() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.ndjson")
        let scope = try scope()

        do {
            let engine = ContinuityEngine(journal: try FileJournal(url: url))
            try await engine.start()
            let store = ContinuityStore(engine: engine)
            _ = try await store.sessionInit(MemorySession(id: "session-1"), in: scope)
            _ = try await store.sessionInit(MemorySession(id: "session-2"), in: scope)
            try await store.set(
                MemoryRecord(
                    key: try key("decisions/storage"),
                    value: "native swift, same process",
                    importance: 0.95,
                    sourceSession: "session-2"),
                in: scope)
            await engine.shutDown()
        }

        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        let loaded = try #require(try await store.get(try key("decisions/storage"), in: scope))
        #expect(loaded.sourceSession == "session-2")
    }

    /// A delete has to be a delete from the model's point of view even though
    /// the engine keeps the chain, or a "forget that" leaves the fact in every
    /// later prompt.
    @Test func aDeletedKeyStaysGoneAcrossARestart() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.ndjson")
        let scope = try scope()

        do {
            let engine = ContinuityEngine(journal: try FileJournal(url: url))
            try await engine.start()
            let store = ContinuityStore(engine: engine)
            try await store.set(MemoryRecord(key: try key("temp/thing"), value: "x"), in: scope)
            #expect(try await store.delete(try key("temp/thing"), in: scope))
            await engine.shutDown()
        }

        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        #expect(try await store.get(try key("temp/thing"), in: scope) == nil)
        #expect(try await store.list(prefix: "", limit: 10, in: scope).isEmpty)
    }
}

@Suite struct ContinuityJournalStoreTests {
    private func scope() throws -> MemoryScope {
        try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
    }

    private func pair() -> (ContinuityStore, ContinuityJournalStore) {
        let engine = ContinuityEngine()
        let store = ContinuityStore(engine: engine)
        return (store, ContinuityJournalStore(engine: engine, store: store))
    }

    private func turn(
        _ index: Int,
        session: String = "s1",
        prompt: String,
        reply: String
    ) -> JournalTurn {
        JournalTurn(
            session: session, workspace: "repo-a", index: index,
            prompt: prompt, reply: reply, model: "qwen35b",
            promptTokens: 10, completionTokens: 20,
            latencyMilliseconds: 30, stopReason: "stop")
    }

    @Test func turnsComeBackNewestFirstWithTheirMeasurements() async throws {
        let (_, journal) = pair()
        let scope = try scope()
        await journal.record(turn(0, prompt: "first", reply: "one"), in: scope)
        await journal.record(turn(1, prompt: "second", reply: "two"), in: scope)

        let stored = try await journal.turns(session: "s1", limit: 10, in: scope)
        #expect(stored.map(\.prompt) == ["second", "first"])
        #expect(stored.first?.reply == "two")
        #expect(stored.first?.model == "qwen35b")
        #expect(stored.first?.completionTokens == 20)
        #expect(stored.first?.stopReason == "stop")
        #expect(stored.first?.workspace == "repo-a")
    }

    @Test func sessionsAreSummarisedNewestFirst() async throws {
        let (_, journal) = pair()
        let scope = try scope()
        await journal.record(turn(0, session: "s1", prompt: "a", reply: "1"), in: scope)
        await journal.record(turn(0, session: "s2", prompt: "b", reply: "2"), in: scope)
        await journal.record(turn(1, session: "s2", prompt: "c", reply: "3"), in: scope)

        let summaries = try await journal.sessions(limit: 10, in: scope)
        #expect(Set(summaries.map(\.session)) == ["s1", "s2"])
        let second = try #require(summaries.first { $0.session == "s2" })
        #expect(second.turnCount == 2)
        #expect(second.model == "qwen35b")
    }

    @Test func searchFindsTurnsAcrossSessions() async throws {
        let (_, journal) = pair()
        let scope = try scope()
        await journal.record(
            turn(
                0, session: "s1", prompt: "about sync races",
                reply: "kept FooManager"), in: scope)
        await journal.record(
            turn(
                0, session: "s2", prompt: "about colour",
                reply: "warm palette"), in: scope)

        let hits = try await journal.search("SYNC", limit: 10, in: scope)
        #expect(hits.count == 1)
        #expect(hits.first?.session == "s1")
        #expect(try await journal.search("kubernetes", limit: 10, in: scope).isEmpty)
    }

    /// The two stores describe one session, not two that share a name. A
    /// fact written during a conversation and the turns of that conversation
    /// have to be attributable to each other.
    @Test func theJournalReusesTheSessionTheStoreOpened() async throws {
        let (store, journal) = pair()
        let scope = try scope()
        _ = try await store.sessionInit(
            MemorySession(id: "shared", modelID: "qwen35b"),
            in: scope)
        await journal.record(
            turn(0, session: "shared", prompt: "hello", reply: "hi"),
            in: scope)

        let summaries = try await journal.sessions(limit: 10, in: scope)
        #expect(summaries.count == 1)
        #expect(summaries.first?.session == "shared")
        #expect(summaries.first?.turnCount == 1)
    }

    @Test func retentionBoundsTurnsPerSession() async throws {
        let engine = ContinuityEngine()
        let store = ContinuityStore(engine: engine)
        let journal = ContinuityJournalStore(
            engine: engine, store: store,
            limits: JournalLimits(turnsPerSession: 3))
        let scope = try scope()
        for index in 0..<10 {
            await journal.record(
                turn(index, prompt: "ask \(index)", reply: "reply \(index)"),
                in: scope)
        }
        let stored = try await journal.turns(session: "s1", limit: 50, in: scope)
        #expect(stored.count == 3)
        // The newest survive, which is the only useful direction to trim.
        #expect(stored.first?.prompt == "ask 9")
        #expect(stored.last?.prompt == "ask 7")
    }

    /// A session the log pruned away must not stay cached under its name.
    ///
    /// `record` prunes a workspace's oldest sessions, but the store keeps its
    /// own external-id-to-session cache and never invalidates it, so a pruned
    /// name kept resolving to a session the log no longer knows. Every later
    /// turn of that conversation threw `unknownSession` into the `try?` in
    /// `record` — lost, with nothing on the engine's failure channel to say so,
    /// for the rest of the process's life.
    @Test func aPrunedSessionStillAcceptsItsTurns() async throws {
        let engine = ContinuityEngine()
        let store = ContinuityStore(engine: engine)
        let journal = ContinuityJournalStore(
            engine: engine, store: store,
            limits: JournalLimits(turnsPerSession: 10, sessionsPerWorkspace: 1))
        let scope = try scope()
        await journal.record(
            turn(0, session: "s-1", prompt: "one", reply: "a"),
            in: scope)
        await journal.record(
            turn(0, session: "s-2", prompt: "two", reply: "b"),
            in: scope)
        await journal.record(
            turn(1, session: "s-1", prompt: "three", reply: "c"),
            in: scope)

        let stored = try await journal.turns(session: "s-1", limit: 10, in: scope)
        #expect(stored.map(\.prompt) == ["three"])
        // The loss has to be closed by the turn landing, not by a failure
        // being reported instead: the append channel must stay clean.
        #expect(await engine.journalFailure == nil)
    }

    /// The same dangling id can arrive through the *store's* cache instead of
    /// the journal's, and the journal has no way to invalidate what it does not
    /// own — so it has to refuse to write through it.
    ///
    /// `sessionInit` opens the session and remembers its id; a later turn under
    /// another name prunes the oldest session, which is that one; the journal
    /// then asks the store for the session shared with it and is handed a
    /// session the log dropped.
    @Test func aPrunedStoreSessionDoesNotStrandTheJournal() async throws {
        let engine = ContinuityEngine()
        let store = ContinuityStore(engine: engine)
        let journal = ContinuityJournalStore(
            engine: engine, store: store,
            limits: JournalLimits(turnsPerSession: 10, sessionsPerWorkspace: 1))
        let scope = try scope()
        _ = try await store.sessionInit(
            MemorySession(id: "shared", modelID: "qwen35b"), in: scope)
        let opened = try #require(await store.continuitySession(for: "shared"))
        let taskID = try await store.taskID(for: scope)
        await journal.record(
            turn(0, session: "other", prompt: "one", reply: "a"),
            in: scope)
        // The prune that turn ends with dropped "shared", and the store still
        // names it.
        #expect(await engine.sessions(taskID: taskID).count == 1)

        await journal.record(
            turn(1, session: "shared", prompt: "two", reply: "b"),
            in: scope)

        let stored = try await journal.turns(session: "shared", limit: 10, in: scope)
        #expect(stored.map(\.prompt) == ["two"])
        #expect(await engine.session(externalID: "shared")?.id != opened)
        #expect(await engine.journalFailure == nil)
    }

    /// The same stale id sits in front of the *store's* own write path, and
    /// there the throw is not swallowed — it reaches the model.
    ///
    /// `set` takes the writing session's id out of `sessionIDs` to carry as
    /// provenance, `engine.remember(sessionID:)` rejects a session the log has
    /// dropped, and the error the caller gets back names the backend rather
    /// than the stale entry. The fact the model asked to keep was not written.
    @Test func aFactWrittenAfterItsSessionWasPrunedStillLands() async throws {
        let engine = ContinuityEngine()
        let store = ContinuityStore(engine: engine)
        let journal = ContinuityJournalStore(
            engine: engine, store: store,
            limits: JournalLimits(turnsPerSession: 10, sessionsPerWorkspace: 1))
        let scope = try scope()
        _ = try await store.sessionInit(MemorySession(id: "shared"), in: scope)
        await journal.record(
            turn(0, session: "other", prompt: "one", reply: "a"),
            in: scope)

        try await store.set(
            MemoryRecord(
                key: try MemoryKey(validating: "decisions/sync"),
                value: "FooManager stays; it prevents a sync race.",
                importance: 0.9,
                sourceSession: "shared"),
            in: scope)

        let loaded = try #require(
            await store.get(try MemoryKey(validating: "decisions/sync"), in: scope))
        #expect(loaded.value == "FooManager stays; it prevents a sync race.")
        // The conversation kept its name, so the fact stays attributable to it:
        // provenance is carried on the session reopened under that name rather
        // than dropped with the one that was pruned.
        #expect(loaded.sourceSession == "shared")
        #expect(await engine.journalFailure == nil)
    }
}

/// The path the server actually takes: a `MemoryService` built from a
/// configuration, with no store injected.
///
/// Every other service test injects a double, so this is the only thing that
/// proves the default wiring reaches disk at all.
@Suite struct MemoryServiceDefaultWiringTests {
    private func configuration(directory: URL) -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "wiring-test"
        configuration.user = "local"
        configuration.namespace = "tinytitan"
        configuration.storage.directory = directory
        return configuration
    }

    @Test func factsWrittenThroughTheServiceOutliveIt() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-wiring-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration = configuration(directory: directory)

        do {
            let service = MemoryService(configuration: configuration)
            let context = try #require(
                await service.beginSession(
                    id: "s1",
                    modelID: "qwen35b"))
            #expect(await service.isDurable)
            let result = await service.execute(
                name: "memory_set",
                arguments: [
                    "key": .string("decisions/storage"),
                    "value": .string("native swift, same process"),
                    "importance": .number(0.9),
                ],
                in: context)
            guard case .ok = result else {
                Issue.record("the write failed: \(result)")
                return
            }
            // A restart is a new process, and the kernel drops the journal's
            // exclusive lock when the old one exits. In one process the
            // service has to be told, or its background work can keep the
            // lock held a moment longer -- which, under full-suite load, made
            // the second service fall back to empty local storage and read as
            // a lost fact rather than a held lock.
            await service.shutDown()
        }

        // A second service over the same directory is a restart.
        let service = MemoryService(configuration: configuration)
        let context = try #require(await service.beginSession(id: "s2"))
        #expect(
            await service.isDurable,
            "the restarted service must own the journal, or nothing below means anything")
        #expect(context.bootstrap.records.count == 1)
        #expect(context.bootstrap.records.first?.key.rawValue == "decisions/storage")
        #expect(context.bootstrap.records.first?.value == "native swift, same process")
        // And the fragment tells the model what it already knows.
        #expect(await service.instructions(for: context).contains("decisions/storage"))
    }

    @Test func theJournalCapturesTurnsAndSurvivesARestart() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-wiring-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration = configuration(directory: directory)

        do {
            let service = MemoryService(configuration: configuration)
            let context = try #require(
                await service.beginSession(
                    id: "s1",
                    modelID: "qwen35b"))
            await service.recordTurn(
                session: context, index: 0,
                prompt: "write pong in swift",
                reply: "done, 800 by 600",
                model: "qwen35b", promptTokens: 12,
                completionTokens: 40, latencyMilliseconds: 900,
                stopReason: "stop")
            // As above: a restart releases the lock; one process has to ask.
            await service.shutDown()
        }

        let service = MemoryService(configuration: configuration)
        _ = await service.beginSession(id: "s2")
        #expect(await service.isDurable)
        let journal = try #require(await service.journalStore())
        let scope = try #require(configuration.scope())
        let sessions = try await journal.sessions(limit: 10, in: scope)
        #expect(sessions.contains { $0.session == "s1" })
        let turns = try await journal.turns(session: "s1", limit: 10, in: scope)
        #expect(turns.first?.prompt == "write pong in swift")
        #expect(turns.first?.completionTokens == 40)
    }

    @Test func aDirectoryThatCannotBeWrittenDegradesInsteadOfFailing() async throws {
        var configuration = configuration(directory: URL(fileURLWithPath: "/dev/null/nope"))
        configuration.degradesToLocalStore = true
        let service = MemoryService(configuration: configuration)
        // The session still starts; the model is told its writes will not last.
        let context = try #require(await service.beginSession(id: "s1"))
        #expect(context.isDurable == false)
        #expect(await service.instructions(for: context).contains("lasts only for this session"))
    }
}

/// Reads have to stay bounded and prefixes have to mean what they say.
@Suite struct ContinuityStoreReadTests {
    private func scope(_ workspace: String = "repo-a") throws -> MemoryScope {
        try MemoryScope(namespace: "tinytitan", user: "local", workspace: workspace)
    }

    private func key(_ raw: String) throws -> MemoryKey { try MemoryKey(validating: raw) }

    @Test func prefixesArePushedDownOnlyWhenThatIsSafe() {
        // Whole segments: the engine can answer it exactly.
        let whole = ContinuityStore.pushDown(prefix: "decisions/")
        #expect(whole.namespace == "k.decisions")
        #expect(whole.isExact)

        // A partial last segment: the namespace narrows the scan, the residual
        // decides the answer.
        let partial = ContinuityStore.pushDown(prefix: "decisions/sy")
        #expect(partial.namespace == "k.decisions")
        #expect(partial.isExact == false)
        #expect(partial.matches("decisions/sync"))
        #expect(partial.matches("decisions/other") == false)

        // A partial first segment cannot be pushed down at all: "dec" could
        // be the head of a namespace or of a bare key, and guessing wrong
        // would silently return nothing.
        let head = ContinuityStore.pushDown(prefix: "dec")
        #expect(head.namespace == nil)
        #expect(head.matches("decisions/sync"))

        let empty = ContinuityStore.pushDown(prefix: "")
        #expect(empty.namespace == nil)
        #expect(empty.matches("anything"))
    }

    @Test func listMatchesPartialSegmentsCorrectly() async throws {
        let store = ContinuityStore(engine: ContinuityEngine())
        let scope = try scope()
        for name in ["decisions/sync", "decisions/storage", "deploy/steps", "gotchas/build"] {
            try await store.set(MemoryRecord(key: try key(name), value: name), in: scope)
        }

        let byCategory = try await store.list(prefix: "decisions/", limit: 10, in: scope)
        #expect(Set(byCategory.map(\.rawValue)) == ["decisions/sync", "decisions/storage"])

        // The case the naive push-down got wrong: a prefix that stops in the
        // middle of the first segment.
        let byHead = try await store.list(prefix: "de", limit: 10, in: scope)
        #expect(
            Set(byHead.map(\.rawValue))
                == ["decisions/sync", "decisions/storage", "deploy/steps"])

        let byPartialKey = try await store.list(prefix: "decisions/st", limit: 10, in: scope)
        #expect(byPartialKey.map(\.rawValue) == ["decisions/storage"])

        #expect(try await store.list(prefix: "nothing", limit: 10, in: scope).isEmpty)
    }

    /// The record used to be stored as a JSON envelope, so a search for a
    /// word that appears in the metadata would match every record in the
    /// workspace.
    @Test func searchDoesNotMatchTheStorageFormat() async throws {
        let store = ContinuityStore(engine: ContinuityEngine())
        let scope = try scope()
        try await store.set(
            MemoryRecord(
                key: try key("notes/a"), value: "the palette is warm",
                importance: 0.5, tags: ["colour"]),
            in: scope)
        try await store.set(
            MemoryRecord(
                key: try key("notes/b"), value: "the sync race",
                importance: 0.5),
            in: scope)

        for term in ["importance", "createdAt", "updatedAt", "tags", "value"] {
            let hits = try await store.search(MemoryQuery(text: term, limit: 10), in: scope)
            #expect(hits.isEmpty, "'\(term)' is a field name, not content")
        }
        #expect(
            try await store.search(MemoryQuery(text: "palette", limit: 10), in: scope)
                .count == 1)
    }

    @Test func theBootstrapReadsABoundedSlice() async throws {
        let limits = TinyTitanMemory.MemoryLimits(bootstrapRecords: 5, bootstrapBytes: 1 << 20)
        let store = ContinuityStore(engine: ContinuityEngine(), limits: limits)
        let scope = try scope()
        for index in 0..<200 {
            try await store.set(
                MemoryRecord(
                    key: try key("k\(index)"),
                    value: "value \(index)",
                    importance: Double(index) / 200),
                in: scope)
        }
        let bootstrap = try await store.sessionInit(MemorySession(id: "s1"), in: scope)
        #expect(bootstrap.records.count == 5)
        // Ranked by importance, so the slice that is read is the one worth
        // reading rather than whatever sorted first.
        #expect(bootstrap.records.first?.key.rawValue == "k199")
    }

    @Test func theValueIsStoredAsWrittenNotWrapped() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("plain-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        try await store.set(
            MemoryRecord(
                key: try key("decisions/sync"),
                value: "FooManager prevents a race"),
            in: try scope())
        await engine.shutDown()

        // Readable in the file, which is what makes a journal inspectable
        // with ordinary tools.
        let contents = String(data: try Data(contentsOf: url), encoding: .utf8) ?? ""
        #expect(contents.contains("FooManager prevents a race"))
        #expect(contents.contains("\\\"value\\\"") == false)
    }
}

/// One workspace, one file, one writer.
@Suite struct MemoryWorkspaceIsolationTests {
    private func configuration(
        directory: URL,
        workspace: String = "repo-a"
    ) -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = workspace
        configuration.user = "local"
        configuration.namespace = "tinytitan"
        configuration.storage.directory = directory
        return configuration
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-ws-\(UUID().uuidString)")
    }

    /// A request that names another workspace used to have its facts written
    /// into the default workspace's file, so deleting one project's memory
    /// would have deleted another's.
    @Test func perRequestWorkspacesGetTheirOwnFile() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var configuration = configuration(directory: directory)
        configuration.allowsPerRequestWorkspace = true

        let service = MemoryService(configuration: configuration)
        let home = try #require(await service.beginSession(id: "s1"))
        let other = try #require(
            await service.beginSession(
                id: "s2",
                workspaceOverride: "repo-b"))
        #expect(home.scope.workspace == "repo-a")
        #expect(other.scope.workspace == "repo-b")

        _ = await service.execute(
            name: "memory_set",
            arguments: [
                "key": .string("here"),
                "value": .string("belongs to repo-a"),
            ],
            in: home)
        _ = await service.execute(
            name: "memory_set",
            arguments: [
                "key": .string("here"),
                "value": .string("belongs to repo-b"),
            ],
            in: other)
        await service.shutDown()

        let first = directory.appendingPathComponent("tinytitan/local/repo-a.ndjson")
        let second = directory.appendingPathComponent("tinytitan/local/repo-b.ndjson")
        #expect(FileManager.default.fileExists(atPath: first.path))
        #expect(FileManager.default.fileExists(atPath: second.path))

        let a = String(data: try Data(contentsOf: first), encoding: .utf8) ?? ""
        let b = String(data: try Data(contentsOf: second), encoding: .utf8) ?? ""
        #expect(a.contains("belongs to repo-a"))
        #expect(a.contains("belongs to repo-b") == false)
        #expect(b.contains("belongs to repo-b"))
        #expect(b.contains("belongs to repo-a") == false)
    }

    /// Two servers launched from one directory. The second must not write
    /// into the first's file, and must not claim its writes will last.
    @Test func aSecondServerOnOneWorkspaceRunsWithoutPersistence() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration = configuration(directory: directory)

        let first = MemoryService(configuration: configuration)
        let firstContext = try #require(await first.beginSession(id: "s1"))
        #expect(firstContext.isDurable)
        _ = await first.execute(
            name: "memory_set",
            arguments: ["key": .string("k"), "value": .string("from first")],
            in: firstContext)

        let second = MemoryService(configuration: configuration)
        let secondContext = try #require(await second.beginSession(id: "s2"))
        #expect(secondContext.isDurable == false)
        // And it says so where the model will read it.
        let instructions = await second.instructions(for: secondContext)
        #expect(instructions.contains("lasts only for this session"))

        // Its writes work for the session and do not reach the other's file.
        _ = await second.execute(
            name: "memory_set",
            arguments: [
                "key": .string("k2"),
                "value": .string("from second"),
            ],
            in: secondContext)
        await first.shutDown()
        await second.shutDown()

        let file = directory.appendingPathComponent("tinytitan/local/repo-a.ndjson")
        let contents = String(data: try Data(contentsOf: file), encoding: .utf8) ?? ""
        #expect(contents.contains("from first"))
        #expect(contents.contains("from second") == false)
    }

    /// The ceiling is what memory adds to the process, not what each
    /// workspace may take. Per-workspace limits alone would multiply it by
    /// the number of repositories a session touched.
    @Test func theCeilingCoversEveryWorkspaceTogether() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var configuration = configuration(directory: directory)
        configuration.allowsPerRequestWorkspace = true
        configuration.storage.maximumMemoryBytes = 128 << 10
        configuration.limits.maximumValueBytes = 8 << 10

        let service = MemoryService(configuration: configuration)
        let filler = String(repeating: "x", count: 4 << 10)
        for index in 0..<12 {
            let context = try #require(
                await service.beginSession(
                    id: "s\(index)",
                    workspaceOverride: "repo-\(index)"))
            for entry in 0..<8 {
                _ = await service.execute(
                    name: "memory_set",
                    arguments: [
                        "key": .string("k\(entry)"),
                        "value": .string(filler),
                    ],
                    in: context)
            }
            // Checked after every workspace, not just at the end: a ceiling
            // that only holds once the work has stopped is not a ceiling.
            let resident = await service.residentBytes()
            #expect(
                resident <= 128 << 10,
                "resident \(resident) after workspace \(index)")
        }
        await service.shutDown()

        // Everything written is still on disk; only residency was bounded.
        let first = directory.appendingPathComponent("tinytitan/local/repo-0.ndjson")
        #expect(FileManager.default.fileExists(atPath: first.path))
        let contents = try Data(contentsOf: first)
        #expect(contents.count > 0)
    }

    /// A workspace closed to stay inside the ceiling has to come back
    /// complete, or the bound is a data-loss bug wearing a budget's clothes.
    @Test func aClosedWorkspaceReopensWithItsFacts() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var configuration = configuration(directory: directory)
        configuration.allowsPerRequestWorkspace = true
        configuration.storage.maximumMemoryBytes = 128 << 10
        configuration.limits.maximumValueBytes = 8 << 10

        let service = MemoryService(configuration: configuration)
        let early = try #require(
            await service.beginSession(
                id: "s0",
                workspaceOverride: "repo-early"))
        _ = await service.execute(
            name: "memory_set",
            arguments: [
                "key": .string("decisions/storage"),
                "value": .string("native swift"),
            ],
            in: early)

        // Enough other workspaces to push the first one out of residency.
        let filler = String(repeating: "x", count: 4 << 10)
        for index in 0..<12 {
            let context = try #require(
                await service.beginSession(
                    id: "f\(index)",
                    workspaceOverride: "repo-\(index)"))
            for entry in 0..<8 {
                _ = await service.execute(
                    name: "memory_set",
                    arguments: [
                        "key": .string("k\(entry)"),
                        "value": .string(filler),
                    ],
                    in: context)
            }
        }

        let returning = try #require(
            await service.beginSession(
                id: "s1",
                workspaceOverride: "repo-early"))
        #expect(returning.bootstrap.records.contains { $0.key.rawValue == "decisions/storage" })
        let result = await service.execute(
            name: "memory_get",
            arguments: ["key": .string("decisions/storage")],
            in: returning)
        #expect(result.jsonString().contains("native swift"))
        await service.shutDown()
    }

    @Test func shuttingDownHandsTheWorkspaceOver() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration = configuration(directory: directory)

        let first = MemoryService(configuration: configuration)
        let context = try #require(await first.beginSession(id: "s1"))
        #expect(context.isDurable)
        await first.shutDown()

        let second = MemoryService(configuration: configuration)
        let handedOver = try #require(await second.beginSession(id: "s2"))
        #expect(handedOver.isDurable)
        await second.shutDown()
    }
}

/// The bootstrap's order under pressure.
@Suite struct MemoryBootstrapOrderTests {
    @Test func amongEqualImportanceTheOlderFactWins() throws {
        let early = Date(timeIntervalSince1970: 1_000)
        let late = Date(timeIntervalSince1970: 9_000)
        let bible = MemoryRecord(
            key: try MemoryKey(validating: "characters/rosa/eyes"),
            value: "hazel", importance: 0.9,
            createdAt: early, updatedAt: early)
        let state = MemoryRecord(
            key: try MemoryKey(validating: "continuity/last_scene"),
            value: "Rosa on the shore", importance: 0.9,
            createdAt: late, updatedAt: late)
        let limits = TinyTitanMemory.MemoryLimits(bootstrapRecords: 1, bootstrapBytes: 1 << 16)
        let bootstrap = MemoryBootstrap.build(from: [state, bible], limits: limits)
        #expect(bootstrap.records.map(\.key.rawValue) == ["characters/rosa/eyes"])
        #expect(bootstrap.omittedCount == 1)
    }

    @Test func importanceStillComesFirst() throws {
        let early = Date(timeIntervalSince1970: 1_000)
        let late = Date(timeIntervalSince1970: 9_000)
        let minor = MemoryRecord(
            key: try MemoryKey(validating: "notes/aside"), value: "x",
            importance: 0.2, createdAt: early, updatedAt: early)
        let major = MemoryRecord(
            key: try MemoryKey(validating: "rules/weather"),
            value: "never rains", importance: 0.95,
            createdAt: late, updatedAt: late)
        let limits = TinyTitanMemory.MemoryLimits(bootstrapRecords: 1, bootstrapBytes: 1 << 16)
        #expect(
            MemoryBootstrap.build(from: [minor, major], limits: limits)
                .records.map(\.key.rawValue) == ["rules/weather"])
    }
}

/// Project files must not pile up.
@Suite struct MemoryRetentionTests {
    private func configuration(directory: URL, days: Int = 30, cap: Int = 100)
        -> MemoryConfiguration
    {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "live"
        configuration.user = "local"
        configuration.namespace = "tinytitan"
        configuration.storage.directory = directory
        configuration.storage.retentionDays = days
        configuration.storage.maximumWorkspaces = cap
        return configuration
    }

    /// Writes a project file with a chosen last-write time.
    private func plant(_ name: String, in directory: URL, daysOld: Int) throws -> URL {
        let folder = directory.appendingPathComponent("tinytitan/local")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("\(name).ndjson")
        try Data("{}\n".utf8).write(to: url)
        try Data().write(to: url.appendingPathExtension("lock"))
        let when = Date().addingTimeInterval(-Double(daysOld) * 86_400)
        try FileManager.default.setAttributes([.modificationDate: when], ofItemAtPath: url.path)
        return url
    }

    private func names(in directory: URL) -> Set<String> {
        let folder = directory.appendingPathComponent("tinytitan/local")
        let files = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        return Set(files.filter { $0.hasSuffix(".ndjson") })
    }

    /// Retention never deletes: an old project keeps its file and its facts.
    /// Only the cap removes a project. The transcript expiry itself is
    /// covered in MemoryRetentionKeepsFactsTests.
    @Test func filesUntouchedForThirtyDaysAreKeptNotDeleted() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("retention-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        _ = try plant("old-project", in: directory, daysOld: 45)
        _ = try plant("recent-project", in: directory, daysOld: 3)
        let service = MemoryService(configuration: configuration(directory: directory))
        await service.sweepStaleWorkspaces()
        #expect(names(in: directory) == ["old-project.ndjson", "recent-project.ndjson"])
    }

    @Test func beyondTheCapTheOldestGoFirst() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("retention-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        for index in 0..<6 {
            _ = try plant("p\(index)", in: directory, daysOld: index)  // p0 newest
        }
        let service = MemoryService(configuration: configuration(directory: directory, cap: 4))
        await service.sweepStaleWorkspaces()
        #expect(names(in: directory) == ["p0.ndjson", "p1.ndjson", "p2.ndjson", "p3.ndjson"])
    }

    @Test func anOpenWorkspaceIsNeverSweptAndCountsAgainstTheCap() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("retention-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        for index in 0..<3 { _ = try plant("p\(index)", in: directory, daysOld: 60) }
        let service = MemoryService(
            configuration: configuration(
                directory: directory,
                days: 0, cap: 2))
        // Opening the live workspace creates its file and runs the sweep:
        // the live one plus one planted file fit the cap of two.
        let context = try #require(await service.beginSession(id: "s1"))
        #expect(context.isDurable)
        let kept = names(in: directory)
        #expect(kept.contains("live.ndjson"))
        #expect(kept.count == 2)
        // And the live file is older-proof: backdate it and sweep again.
        let live = directory.appendingPathComponent("tinytitan/local/live.ndjson")
        try FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(-90 * 86_400)], ofItemAtPath: live.path)
        await service.sweepStaleWorkspaces()
        #expect(names(in: directory).contains("live.ndjson"))
        await service.shutDown()
    }

    @Test func zeroKeepsEverything() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("retention-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        for index in 0..<5 { _ = try plant("p\(index)", in: directory, daysOld: 400) }
        let service = MemoryService(
            configuration: configuration(
                directory: directory,
                days: 0, cap: 0))
        await service.sweepStaleWorkspaces()
        #expect(names(in: directory).count == 5)
    }

    @Test func retentionIsReadFromTheEnvironment() {
        let configuration = MemoryConfiguration.fromEnvironment([
            "TINYTITAN_MEMORY": "1", "TINYTITAN_MEMORY_RETENTION_DAYS": "7",
            "TINYTITAN_MEMORY_MAX_WORKSPACES": "12",
        ])
        #expect(configuration.storage.retentionDays == 7)
        #expect(configuration.storage.maximumWorkspaces == 12)
        let defaults = MemoryConfiguration.fromEnvironment(["TINYTITAN_MEMORY": "1"])
        #expect(defaults.storage.retentionDays == 30)
        #expect(defaults.storage.maximumWorkspaces == 100)
    }
}

/// v3: retention keeps the facts, and the project file can be read back
/// without a lock.
@Suite struct MemoryRetentionKeepsFactsTests {
    @Test func anOldProjectLosesItsSessionsAndKeepsItsFacts() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("retention-facts-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let folder = directory.appendingPathComponent("tinytitan/local")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("old-novel.ndjson")

        // A project with a bible and two sessions of transcript.
        do {
            let engine = ContinuityEngine(journal: try FileJournal(url: url))
            try await engine.start()
            let task = try await engine.createTask(title: "The Photograph")
            for index in 0..<2 {
                let session = try await engine.beginSession(taskID: task.id)
                try await engine.recordUserPrompt(sessionID: session.id, text: "chapter \(index)")
                try await engine.recordAssistantResponse(
                    sessionID: session.id,
                    text: String(repeating: "prose ", count: 200))
                try await engine.remember(
                    sessionID: session.id, namespace: "k.characters.rosa",
                    key: "eyes", value: "hazel")
                _ = try await engine.endSession(session.id)
            }
            await engine.shutDown()
        }
        let before = try MemoryProjectFile.load(url)
        #expect(before.sessionCount == 2)
        #expect(before.facts.count == 1)
        try FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(-45 * 86_400)], ofItemAtPath: url.path)

        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "live"
        configuration.user = "local"
        configuration.namespace = "tinytitan"
        configuration.storage.directory = directory
        configuration.storage.retentionDays = 30
        let service = MemoryService(configuration: configuration)
        await service.sweepStaleWorkspaces()

        // Still there, smaller, facts intact, transcript gone.
        #expect(FileManager.default.fileExists(atPath: url.path))
        let after = try MemoryProjectFile.load(url)
        #expect(after.facts.map(\.key) == ["characters/rosa/eyes"])
        #expect(after.facts.first?.value == "hazel")
        #expect(after.sessionCount == 0)
        #expect(after.bytesOnDisk < before.bytesOnDisk)
    }

    @Test func theProjectFileReaderFoldsHistoryAndResolvesPrefixes() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("reader-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let folder = directory.appendingPathComponent("tinytitan/local")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("photograph-851a1c1a.ndjson")
        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let task = try await engine.createTask(title: "Novel")
        try await engine.remember(
            taskID: task.id, namespace: "k.state", key: "inn", value: "standing")
        try await engine.remember(
            taskID: task.id, namespace: "k.state", key: "inn", value: "burned")
        // Read while the writer still holds the lock: no lock is taken.
        let file = try MemoryProjectFile.load(url)
        await engine.shutDown()
        let inn = try #require(file.facts.first { $0.key == "state/inn" })
        #expect(inn.value == "burned")
        #expect(inn.version == 2)
        #expect(inn.history.map(\.value) == ["standing"])
        #expect(file.title == "Novel")

        let files = MemoryProjectFile.discover(in: directory)
        #expect(files.count == 1)
        if case .success(let found) = MemoryProjectFile.resolve("photo", among: files) {
            #expect(found.workspace == "photograph-851a1c1a")
        } else {
            Issue.record("prefix did not resolve")
        }
        if case .failure = MemoryProjectFile.resolve("nothing", among: files) {
        } else {
            Issue.record("an unknown name resolved")
        }
    }
}
