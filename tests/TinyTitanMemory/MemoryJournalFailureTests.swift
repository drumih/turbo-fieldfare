import ContinuityCore
import Foundation
import Testing

@testable import TinyTitanMemory

/// A journal that refuses writes on demand, standing in for a full disk, an
/// I/O error or a descriptor closed underneath it.
///
/// `kinds` narrows the refusal to one record kind, which is what separates the
/// three ways a turn can be lost: the task of the workspace, the session it
/// belongs to, and the turn's own content. Refusing everything cannot tell
/// them apart, and they are not the same path.
private actor RefusingJournal: ContinuityJournal {
    private(set) var records: [JournalRecord] = []
    private var refusing = false
    private var kinds: Set<String>?

    func refuse(_ value: Bool, kinds: Set<String>? = nil) {
        refusing = value
        self.kinds = kinds
    }

    func append(_ record: JournalRecord) async throws {
        try check(record)
        records.append(record)
    }

    func replay() async throws -> [JournalRecord] { records }

    func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {
        try check(.checkpoint(sessionLog, memory))
        records = [.checkpoint(sessionLog, memory)]
    }

    func truncate() async throws { records = [] }

    private func check(_ record: JournalRecord) throws {
        guard refusing, kinds.map({ $0.contains(Self.kind(of: record)) }) ?? true else { return }
        throw JournalError.writeFailed(URL(fileURLWithPath: "/journal.ndjson"), errno: ENOSPC)
    }

    private static func kind(of record: JournalRecord) -> String {
        switch record {
        case .task: "task"
        case .session: "session"
        case .event: "event"
        case .memory: "memory"
        case .memoryVersion: "memoryVersion"
        case .checkpoint: "checkpoint"
        }
    }
}

/// `memory_set` answering "stored" is a promise that the fact survives a
/// restart. These drive the real engine and store over a journal that
/// refuses on demand, which is the only way to see what the model is told
/// when the disk fills in the middle of a session.
@Suite struct MemoryJournalFailureTests {
    private struct Harness {
        let journal: RefusingJournal
        let service: MemoryService
        let events: LogCollector
        let context: MemorySessionContext
    }

    private func harness(
        journalingTurns: Bool = false,
        compactionThreshold: Int? = nil
    ) async throws -> Harness {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .full
        let journal = RefusingJournal()
        let engine = ContinuityEngine(
            configuration: compactionThreshold.map {
                ContinuityConfiguration(compactionThreshold: $0)
            } ?? ContinuityConfiguration(),
            journal: journal)
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        let events = LogCollector()
        let service = MemoryService(
            configuration: configuration,
            durableStore: store,
            journal: journalingTurns ? ContinuityJournalStore(engine: engine, store: store) : nil,
            log: { events.append($0) })
        let context = try #require(await service.beginSession(id: "s1"))
        return Harness(journal: journal, service: service, events: events, context: context)
    }

    private func set(_ key: String, _ value: String, in harness: Harness) async
        -> MemoryToolResult
    {
        await harness.service.execute(
            name: "memory_set",
            arguments: ["key": .string(key), "value": .string(value)],
            in: harness.context)
    }

    private func journalFailureLines(_ harness: Harness) -> Int {
        harness.events.messages().filter { $0.contains("degraded during journal") }.count
    }

    private func compactionFailureLines(_ harness: Harness) -> Int {
        harness.events.messages().filter { $0.contains("degraded during compaction") }.count
    }

    private struct JournalCase {
        let journal: RefusingJournal
        let store: ContinuityStore
        let journalStore: ContinuityJournalStore
        let scope: MemoryScope
    }

    /// The journal and the two stores over it, with one record kind refused
    /// before anything is written.
    private func journalCase(refusing kind: String) async throws -> JournalCase {
        let journal = RefusingJournal()
        await journal.refuse(true, kinds: [kind])
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        return JournalCase(
            journal: journal, store: store,
            journalStore: ContinuityJournalStore(engine: engine, store: store),
            scope: try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a"))
    }

    private func turn(session: String = "s1", index: Int) -> JournalTurn {
        JournalTurn(
            session: session, workspace: "repo-a", index: index,
            prompt: "why is FooManager here?",
            reply: "It prevents a race in background sync.")
    }

    /// The records that are a turn's own content, as opposed to the structure
    /// around it. One refused record kind leaves the rest flowing, so "this
    /// turn was lost" has to mean its prompt and reply rather than any event.
    private func contentEvents(_ records: [JournalRecord]) -> [SessionEvent] {
        records.compactMap { record in
            guard case .event(let event) = record, Self.carriesTurn(event.kind) else { return nil }
            return event
        }
    }

    private static func carriesTurn(_ kind: SessionEventKind) -> Bool {
        switch kind {
        case .userPrompt, .assistantResponse, .assistantResponseChunk,
            .assistantResponseCompleted:
            return true
        case .sessionStarted, .sessionEnded, .assistantResponseStarted, .memoryWritten,
            .contextAssembled:
            return false
        }
    }

    @Test func aWriteThatReachesTheJournalIsStillReportedAsStored() async throws {
        let harness = try await harness()
        let result = await set("decisions/db", "postgres", in: harness)

        #expect(result.jsonString().contains("\"stored\":true"))
        #expect(harness.context.isDurable)
        #expect(await harness.service.isDurable)
        #expect(!harness.events.messages().contains { $0.contains("degraded") })
        let records = await harness.journal.records
        #expect(
            records.contains { record in
                if case .memory(let item) = record { return item.value == "postgres" }
                return false
            })
    }

    @Test func aSetTheJournalRefusesIsAFailureAndNotDurable() async throws {
        let harness = try await harness()
        _ = await set("decisions/db", "postgres", in: harness)
        await harness.journal.refuse(true)

        let refused = await set("decisions/cache", "redis", in: harness)
        #expect(refused.isFailure)
        #expect(!refused.jsonString().contains("\"stored\":true"))
        #expect(await harness.service.isDurable == false)

        // Not sent to the empty local store: the engine still holds every
        // fact, and a local retry would have answered "stored".
        let again = await set("decisions/queue", "sqs", in: harness)
        #expect(again.isFailure)
        let read = await harness.service.execute(
            name: "memory_get",
            arguments: ["key": .string("decisions/db")],
            in: harness.context)
        #expect(read.jsonString().contains("postgres"))
        #expect(journalFailureLines(harness) == 1)

        // The disk recovering does not make the lost write durable, so the
        // next session is not told its memory persists.
        await harness.journal.refuse(false)
        let next = try #require(await harness.service.beginSession(id: "s2"))
        #expect(!next.isDurable)
        #expect(journalFailureLines(harness) == 1)
    }

    @Test func aDeleteTheJournalRefusesIsAFailure() async throws {
        let harness = try await harness()
        _ = await set("decisions/db", "postgres", in: harness)
        await harness.journal.refuse(true)

        let result = await harness.service.execute(
            name: "memory_delete",
            arguments: ["key": .string("decisions/db")],
            in: harness.context)
        #expect(result.isFailure)
        #expect(!result.jsonString().contains("\"deleted\":true"))
        #expect(await harness.service.isDurable == false)
    }

    @Test func aTurnTheJournalRefusesFlipsDurabilityWithoutFailingIt() async throws {
        let harness = try await harness(journalingTurns: true)
        await harness.journal.refuse(true)

        for index in 0..<2 {
            await harness.service.recordTurn(
                session: harness.context, index: index,
                prompt: "prompt \(index)", reply: "reply \(index)",
                model: nil, promptTokens: 1, completionTokens: 1,
                latencyMilliseconds: 1, stopReason: "stop")
        }

        #expect(await harness.service.isDurable == false)
        #expect(journalFailureLines(harness) == 1)
    }

    /// `record` can lose a turn *before* its content is written: it needs the
    /// workspace's task, and when that write fails it swallows the throw and
    /// returns. Nothing at the caller sees the loss, so the only trace is the
    /// engine's own failure — which is what the service reads back after every
    /// turn, in `journalFailed(in:)`. Measured on the journal rather than
    /// through `recordTurn`, because a session resolves its task during
    /// `sessionInit`, so that guard is unreachable from the service.
    @Test func aTurnLostToAFailedTaskRecordIsStillReported() async throws {
        let testCase = try await journalCase(refusing: "task")
        let first = turn(index: 0)

        await testCase.journalStore.record(first, in: testCase.scope)

        #expect(contentEvents(await testCase.journal.records).isEmpty, "the turn was written")
        let failure = await testCase.store.journalFailure
        #expect(failure != nil, "a swallowed task write left no trace to report")

        // The task is in RAM even though its record never landed, so the loss
        // is the one turn that met the failure rather than the workspace: the
        // next turn of the same session journals normally.
        let second = turn(index: 1)
        await testCase.journalStore.record(second, in: testCase.scope)
        #expect(contentEvents(await testCase.journal.records).count == 2)
    }

    /// The second pre-write failure: a turn whose session id the memory store
    /// never opened has to be begun by the journal, and that begin is itself a
    /// write. The task write succeeded here, so the refusal can only be felt
    /// at the session — which is what makes this a distinct path from the
    /// task's, and it fails for a reason a caller cannot see either.
    @Test func aTurnLostToAFailedSessionRecordIsStillReported() async throws {
        let testCase = try await journalCase(refusing: "session")
        let lost = turn(session: "s-never-opened", index: 0)

        await testCase.journalStore.record(lost, in: testCase.scope)

        let records = await testCase.journal.records
        #expect(records.contains { if case .task = $0 { true } else { false } })
        #expect(contentEvents(records).isEmpty, "the turn was written")
        let failure = await testCase.store.journalFailure
        #expect(failure != nil, "a swallowed session begin left no trace to report")
        // Reading it back gives the honest answer rather than a swallowed one:
        // this session genuinely has no turns, and `turns` throws only when it
        // could not ask.
        #expect(
            try await testCase.journalStore.turns(
                session: "s-never-opened", limit: 10, in: testCase.scope
            )
            .isEmpty)
    }

    /// The one write nobody asked for. Compaction fires from the engine's own
    /// record counter, so there is no caller to fail and, before this, no
    /// trace to read: a journal that could append but no longer collapse kept
    /// every prompt and reply on the disk while the workspace went on
    /// reporting itself healthy. Refusing only the checkpoint kind is what
    /// separates that from a full disk — here every write still lands, so the
    /// answer is a log line, not a durability flip.
    @Test func aStalledCompactionIsSaidOnceAndDoesNotUndoDurability() async throws {
        let harness = try await harness(journalingTurns: true, compactionThreshold: 2)
        await harness.journal.refuse(true, kinds: ["checkpoint"])

        for index in 0..<5 {
            let result = await set("decisions/k\(index)", "v\(index)", in: harness)
            #expect(result.jsonString().contains("\"stored\":true"))
        }
        #expect(
            compactionFailureLines(harness) == 1,
            "a stalled journal went unreported at a memory tool call")
        await harness.service.recordTurn(
            session: harness.context, index: 0,
            prompt: "prompt", reply: "reply",
            model: nil, promptTokens: 1, completionTokens: 1,
            latencyMilliseconds: 1, stopReason: "stop")

        // Once per workspace, on the precedent of the journal failure line.
        #expect(compactionFailureLines(harness) == 1, "the turn boundary repeated the line")
        // Distinct from a refused write: nothing was lost, so the workspace
        // still persists, and the log line is the whole of the report.
        #expect(journalFailureLines(harness) == 0)
        #expect(await harness.service.isDurable)
    }

    /// The path a server actually takes: a model that never calls a memory
    /// tool still grows the journal with every turn's prompt and reply, so the
    /// turn boundary has to be a reporting point on its own. Without it the
    /// warning would depend on someone writing a fact.
    @Test func aStalledCompactionIsSaidOnATurnThatUsesNoMemoryTool() async throws {
        let harness = try await harness(journalingTurns: true, compactionThreshold: 2)
        await harness.journal.refuse(true, kinds: ["checkpoint"])

        for index in 0..<3 {
            await harness.service.recordTurn(
                session: harness.context, index: index,
                prompt: "prompt \(index)", reply: "reply \(index)",
                model: nil, promptTokens: 1, completionTokens: 1,
                latencyMilliseconds: 1, stopReason: "stop")
        }

        #expect(compactionFailureLines(harness) == 1, "a turn-only stall went unreported")
        #expect(journalFailureLines(harness) == 0)
        #expect(await harness.service.isDurable)
    }
}

/// Collects log events from the service's `@Sendable` callback.
///
/// unchecked-invariant: every access is under `lock`.
private final class LogCollector: @unchecked Sendable {
    private var events: [MemoryLogEvent] = []
    private let lock = NSLock()

    func append(_ event: MemoryLogEvent) {
        lock.withLock { events.append(event) }
    }

    func messages() -> [String] {
        lock.withLock { events.map(\.message) }
    }
}
