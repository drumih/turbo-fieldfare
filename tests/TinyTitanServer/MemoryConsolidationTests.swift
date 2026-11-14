import ContinuityCore
import Foundation
import Testing
import TinyTitan
import TinyTitanMemory
import TinyTitanCore

@testable import TinyTitanLib
@testable import TinyTitanServerCore

/// The engine writing memory on its own, and the loop answering when its
/// rounds are gone. Both exist because a real model was measured not doing
/// the thing the design assumed it would.
@Suite struct MemoryConsolidationTests {
    /// unchecked-invariant: every access to `script` and `seen` is under `lock`.
    private final class ScriptedBackend: ServerInferenceBackend, @unchecked Sendable {
        private var script: [ServerCompletion]
        private var seen: [ValidatedChatRequest] = []
        private let lock = NSLock()
        init(_ script: [ServerCompletion]) { self.script = script }
        func generate(
            _ request: ValidatedChatRequest,
            onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
        ) async throws
            -> ServerCompletion
        {
            lock.withLock { seen.append(request) }
            let completion = lock.withLock { script.isEmpty ? nil : script.removeFirst() }
            guard let completion else {
                return ServerCompletion(
                    content: "", toolCalls: [], finishReason: "stop",
                    usage: OpenAIUsage(
                        promptTokens: 0, completionTokens: 0,
                        totalTokens: 0))
            }
            return completion
        }
        var requests: [ValidatedChatRequest] { lock.withLock { seen } }
    }

    private func completion(
        _ content: String, calls: [ParsedToolCall] = [],
        finish: String = "stop"
    ) -> ServerCompletion {
        ServerCompletion(
            content: content, toolCalls: calls, finishReason: finish,
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    private func call(_ name: String, _ arguments: [String: String]) -> ParsedToolCall {
        ParsedToolCall(
            id: "call-\(name)-\(UUID().uuidString.prefix(4))", name: name,
            arguments: .object(arguments.mapValues { .string($0) }),
            argumentsJSON: "{}")
    }

    private func request(_ text: String) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: text)],
            tools: [], stream: false, includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 32),
            maximumCompletionTokens: 32)
    }

    private func configuration(
        rounds: Int = 2,
        tools: MemoryToolSurface = .full,
        consolidation: Bool,
        idleSeconds: Double = 0.05
    ) -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.maximumToolRounds = rounds
        configuration.toolSurface = tools
        configuration.sessionConsolidation = consolidation
        configuration.consolidationIdleSeconds = idleSeconds
        // These sessions are a line each; the trivial-session guard has its
        // own test and would otherwise skip every one of them.
        configuration.consolidationMinimumCharacters = 0
        return configuration
    }

    // MARK: - Round exhaustion

    /// Measured: a model that wanted memory on every round got its preamble
    /// returned as the answer -- 31 tokens where ten chapters should have
    /// been. Now the last calls are answered and it gets one more turn.
    @Test func exhaustedRoundsEndInAnAnswerNotAPreamble() async throws {
        let inner = ScriptedBackend([
            completion("I need to check memory first.", calls: [call("memory_get", ["key": "a"])]),
            completion("Still checking.", calls: [call("memory_get", ["key": "b"])]),
            // Rounds are gone and the model still asks. This is where the
            // old loop returned the preamble above as the answer.
            completion("One more look.", calls: [call("memory_get", ["key": "c"])]),
            // The final, tool-free generation.
            completion("Chapter 11\nThe tide came in."),
        ])
        let configuration = configuration(rounds: 2, consolidation: false)
        let service = MemoryService(
            configuration: configuration,
            durableStore: InMemoryStore())
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        let result = try await backend.generate(request("write chapter 11"), onEvent: { _ in })
        #expect(result.content.contains("The tide came in."))
        #expect(result.finishReason == "stop")
        #expect(result.toolCalls.isEmpty)

        // Four generations: two rounds, the one that overran, and the answer.
        // The last request carries the exhaustion notice and still carries
        // the tools, so the prompt prefix did not move.
        let requests = inner.requests
        #expect(requests.count == 4)
        let last = try #require(requests.last)
        #expect(last.messages.last?.role == .user)
        #expect(last.messages.last?.content?.contains("used up") == true)
        #expect(last.tools.contains { $0.name == "memory_get" })
    }

    @Test func aToolCallOnTheFinalTurnIsDroppedRatherThanLooped() async throws {
        let inner = ScriptedBackend([
            completion("", calls: [call("memory_get", ["key": "a"])]),
            completion("", calls: [call("memory_get", ["key": "b"])]),
            completion("", calls: [call("memory_get", ["key": "c"])]),
            // The final turn answers and, wrongly, calls a tool as well.
            completion("Here is the answer anyway.", calls: [call("memory_get", ["key": "d"])]),
        ])
        let configuration = configuration(rounds: 2, consolidation: false)
        let service = MemoryService(
            configuration: configuration,
            durableStore: InMemoryStore())
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)
        let result = try await backend.generate(request("go"), onEvent: { _ in })
        #expect(result.content == "Here is the answer anyway.")
        #expect(result.toolCalls.isEmpty)
        #expect(inner.requests.count == 4)
    }

    // MARK: - Consolidation

    private static let extraction = """
        ```json
        [
          {"key": "characters/marcus", "value": "Marcus has grey eyes", "importance": 0.9},
          {"key": "state/inn", "value": "The inn burned down in chapter 34", "importance": 0.8},
          {"key": "Bad Key!", "value": "skipped", "importance": 0.1}
        ]
        ```
        """

    /// After a turn, the idle timer fires and the engine writes facts the
    /// model never chose to write.
    @Test func theIdleTimerDistilsTheSessionIntoMemory() async throws {
        let inner = ScriptedBackend([
            completion("Chapter 34: the inn burned."),  // the turn
            completion(Self.extraction),  // the consolidation
        ])
        let configuration = configuration(tools: .off, consolidation: true)
        let store = InMemoryStore()
        let service = MemoryService(
            configuration: configuration, durableStore: store,
            journal: InMemoryJournal())
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)
        _ = try await backend.generate(request("write chapter 34"), onEvent: { _ in })

        let scope = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        var written: [MemoryRecord] = []
        for _ in 0..<100 {
            try await Task.sleep(for: .milliseconds(25))
            written = await store.allRecords(in: scope)
            if !written.isEmpty { break }
        }
        #expect(written.count == 2, "two valid facts, the malformed key skipped")
        #expect(written.contains { $0.key.rawValue == "state/inn" && $0.value.contains("burned") })

        // The extraction request is its own conversation: no tools, a system
        // message that names the job, and the session's transcript.
        let extraction = try #require(inner.requests.last)
        #expect(extraction.tools.isEmpty)
        #expect(extraction.messages.first?.role == .system)
        #expect(extraction.messages.last?.content?.contains("the inn burned") == true)
        await backend.shutDown()
    }

    /// A new session before the timer fires is a rollover. The previous
    /// session is consolidated after the new session's request returns --
    /// never before it, because a person is waiting on that request.
    @Test func aRolloverConsolidatesThePreviousSessionAfterTheTurn() async throws {
        let inner = ScriptedBackend([
            completion("Chapter 1."),  // session A, turn
            completion("Chapter 11."),  // session B, turn (rollover)
            completion(Self.extraction),  // consolidation of A
        ])
        // A long idle so only the rollover can trigger it.
        let configuration = configuration(tools: .off, consolidation: true, idleSeconds: 60)
        let store = InMemoryStore()
        let service = MemoryService(
            configuration: configuration, durableStore: store,
            journal: InMemoryJournal())
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        _ = try await backend.generate(request("write chapter 1"), onEvent: { _ in })
        #expect(inner.requests.count == 1, "nothing consolidates while the session is live")
        let second = try await backend.generate(request("write chapter 11"), onEvent: { _ in })
        #expect(second.content == "Chapter 11.", "the rollover request got its own answer")

        let scope = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        var written: [MemoryRecord] = []
        for _ in 0..<100 {
            try await Task.sleep(for: .milliseconds(25))
            written = await store.allRecords(in: scope)
            if !written.isEmpty { break }
        }
        #expect(written.count == 2)
        // The extraction came after session B's answer, and it was A's
        // transcript that was distilled.
        #expect(inner.requests.count == 3)
        #expect(inner.requests.last?.messages.last?.content?.contains("chapter 1") == true)
        await backend.shutDown()
    }

    @Test func consolidationOffMeansNoExtraGeneration() async throws {
        let inner = ScriptedBackend([completion("done")])
        let configuration = configuration(tools: .off, consolidation: false)
        let service = MemoryService(
            configuration: configuration, durableStore: InMemoryStore(),
            journal: InMemoryJournal())
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)
        _ = try await backend.generate(request("hi"), onEvent: { _ in })
        try await Task.sleep(for: .milliseconds(200))
        #expect(inner.requests.count == 1)
    }

    /// AUD-134, server side. The journal read used to answer `[]` when the
    /// file would not open, and the head of a session whose transcript was
    /// never fetched moved anyway: the turns were consolidated into nothing,
    /// once, forever. A failed read now stops the run before the generation
    /// and says which of the two it was.
    @Test func anUnreadableJournalStopsTheRunAndSaysSo() async throws {
        let inner = ScriptedBackend([completion("Chapter 34: the inn burned.")])
        let configuration = configuration(tools: .off, consolidation: true)
        let lines = LogBox()
        ServerLog.useSink { lines.append($0) }
        defer { ServerLog.useSink(nil) }
        let service = MemoryService(
            configuration: configuration, durableStore: InMemoryStore(),
            journal: UnreadableJournal())
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)
        _ = try await backend.generate(request("write chapter 34"), onEvent: { _ in })
        try await Task.sleep(for: .milliseconds(250))

        let messages = lines.messages()
        #expect(
            messages.contains { $0.contains("the journal could not be read") },
            "the log should name the failed read: \(messages)")
        #expect(
            !messages.contains { $0.contains("no new turns") },
            "an unreadable journal is not an empty session")
        #expect(
            inner.requests.count == 1,
            "a transcript that was never read got distilled anyway")
        await backend.shutDown()
    }

    /// unchecked-invariant: `lines` is only ever touched under `lock`.
    private final class LogBox: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []
        func append(_ line: String) { lock.withLock { lines.append(line) } }
        func messages() -> [String] { lock.withLock { lines } }
    }

    /// A journal whose transcript read fails, standing in for a file the
    /// engine could not replay.
    private struct UnreadableJournal: SessionJournal {
        func record(_: JournalTurn, in _: MemoryScope) async {}
        func turns(session _: String, limit _: Int, in _: MemoryScope) async throws
            -> [JournalTurn]
        {
            throw MemoryError.backendUnavailable("journal file unreadable")
        }

        func sessions(limit _: Int, in _: MemoryScope) async throws
            -> [JournalSessionSummary]
        {
            []
        }

        func search(_: String, limit _: Int, in _: MemoryScope) async throws -> [JournalTurn] {
            []
        }
    }

    // MARK: - The parser

    @Test func extractionOutputIsParsedLeniently() {
        let fenced = ServerMemory.consolidationRecords(from: Self.extraction)
        #expect(fenced.map(\.key.rawValue) == ["characters/marcus", "state/inn"])
        #expect(fenced.first?.importance == 0.9)

        let bare = ServerMemory.consolidationRecords(
            from: "Here you go: [{\"key\": \"decisions/x\", \"value\": 42}] hope that helps")
        #expect(bare.first?.key.rawValue == "decisions/x")
        #expect(bare.first?.value == "42")

        #expect(ServerMemory.consolidationRecords(from: "[]").isEmpty)
        #expect(ServerMemory.consolidationRecords(from: "no json here").isEmpty)
        #expect(ServerMemory.consolidationRecords(from: "[{\"value\": \"no key\"}]").isEmpty)
    }

    @Test func minimalSurfaceIncludesList() {
        #expect(MemoryToolSurface.minimal.toolNames == ["memory_set", "memory_get", "memory_list"])
    }

    @Test func consolidationIsOnByDefaultWithMemory() {
        let on = MemoryConfiguration.fromEnvironment(["TINYTITAN_MEMORY": "1"])
        #expect(on.sessionConsolidation)
        #expect(on.consolidationIdleSeconds == 30)
        let off = MemoryConfiguration.fromEnvironment([
            "TINYTITAN_MEMORY": "1",
            "TINYTITAN_MEMORY_CONSOLIDATION": "0",
        ])
        #expect(!off.sessionConsolidation)
        let quick = MemoryConfiguration.fromEnvironment(
            ["TINYTITAN_MEMORY": "1", "TINYTITAN_MEMORY_CONSOLIDATION_IDLE_SECONDS": "5"])
        #expect(quick.consolidationIdleSeconds == 5)
    }
}

/// The generation gate: a consolidation may never overlap a person's turn.
@Suite struct MemoryGenerationGateTests {
    /// A backend that fails the test if it is ever entered twice at once.
    /// unchecked-invariant: `active`, `peak` and `calls` are only touched
    /// under `lock`.
    private final class OverlapDetector: ServerInferenceBackend, @unchecked Sendable {
        private let lock = NSLock()
        private var active = 0
        private(set) var peak = 0
        private(set) var calls = 0
        private let reply: @Sendable (Int) -> String
        init(reply: @escaping @Sendable (Int) -> String) { self.reply = reply }
        func generate(
            _ request: ValidatedChatRequest,
            onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
        ) async throws
            -> ServerCompletion
        {
            let index: Int = lock.withLock {
                active += 1
                peak = max(peak, active)
                calls += 1
                return calls
            }
            // Leave on every path, including a cancelled sleep: a detector
            // that forgets to leave reports an overlap that never happened.
            defer { lock.withLock { active -= 1 } }
            try await Task.sleep(for: .milliseconds(60))
            return ServerCompletion(
                content: reply(index), toolCalls: [], finishReason: "stop",
                usage: OpenAIUsage(
                    promptTokens: 1, completionTokens: 1,
                    totalTokens: 2))
        }
        var maximumConcurrency: Int { lock.withLock { peak } }
        var callCount: Int { lock.withLock { calls } }
    }

    private func request(_ text: String) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: text)],
            tools: [], stream: false, includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 32),
            maximumCompletionTokens: 32)
    }

    @Test func consolidationNeverOverlapsATurn() async throws {
        let extraction = "```json\n[{\"key\": \"a/b\", \"value\": \"c\", \"importance\": 0.5}]\n```"
        let long = String(repeating: "a substantial reply. ", count: 40)
        // Turns get long replies (so the session is worth distilling);
        // consolidations get the extraction.
        let inner = OverlapDetector { index in index % 2 == 1 ? long : extraction }
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .off
        configuration.sessionConsolidation = true
        configuration.consolidationIdleSeconds = 0.01
        let service = MemoryService(
            configuration: configuration, durableStore: InMemoryStore(),
            journal: InMemoryJournal())
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        // Fire turns fast enough that each idle timer lands while the next
        // turn is running.
        for index in 0..<6 {
            _ = try await backend.generate(request("turn \(index) " + long), onEvent: { _ in })
            try await Task.sleep(for: .milliseconds(15))
        }
        try await Task.sleep(for: .milliseconds(400))
        #expect(inner.maximumConcurrency == 1, "a consolidation overlapped a turn")
        #expect(inner.callCount > 6, "no consolidation ever ran")
        await backend.shutDown()
    }

    @Test func trivialSessionsAreNotDistilled() async throws {
        let inner = OverlapDetector { _ in "OK" }
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .off
        configuration.sessionConsolidation = true
        configuration.consolidationIdleSeconds = 0.01
        let service = MemoryService(
            configuration: configuration, durableStore: InMemoryStore(),
            journal: InMemoryJournal())
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)
        _ = try await backend.generate(request("Say OK."), onEvent: { _ in })
        try await Task.sleep(for: .milliseconds(300))
        // One generation: the turn. A "say OK" session buys no second one.
        #expect(inner.callCount == 1)
        await backend.shutDown()
    }
}

/// What a hundred-chapter run taught the extraction.
@Suite struct MemoryConsolidationExtractionTests {
    @Test func aTruncatedArrayStillYieldsTheCompleteObjects() {
        // The output cap landed inside the seventeenth object.
        let cut = """
            ```json
            [
              {"key": "characters/marcus/eyes", "value": "grey", "importance": 0.9},
              {"key": "state/inn", "value": "burned down in chapter 34", "importance": 0.8},
              {"key": "state/tomas", "value": "found alive in the ligh
            """
        let records = ServerMemory.consolidationRecords(from: cut)
        #expect(records.map(\.key.rawValue) == ["characters/marcus/eyes", "state/inn"])
    }

    @Test func placeholdersAreNeverWrittenOverAFact() {
        let output = """
            [
              {"key": "characters/halvorsen/eyes", "value": "not specified", "importance": 0.5},
              {"key": "state/ferry_day", "value": "N/A", "importance": 0.5},
              {"key": "state/ferry_running", "value": false, "importance": 0.7},
              {"key": "state/anyone_left", "value": null, "importance": 0.7},
              {"key": "rules/weather", "value": "it never rains", "importance": 0.9}
            ]
            """
        let records = ServerMemory.consolidationRecords(from: output)
        #expect(records.map(\.key.rawValue) == ["state/ferry_running", "rules/weather"])
        #expect(records.first?.value == "false")
    }

    @Test func thePromptShowsValuesAndAsksOnlyForChanges() throws {
        let existing = [
            MemoryRecord(key: try MemoryKey(validating: "characters/marcus/eyes"), value: "grey"),
            MemoryRecord(key: try MemoryKey(validating: "state/inn"), value: "standing"),
        ]
        let turn = JournalTurn(
            session: "s", workspace: "w", index: 0,
            prompt: "write chapter 34", reply: "The inn burned.")
        let request = ServerMemory.consolidationRequest(
            turns: [turn], existing: existing,
            workspace: "w")
        let user = request.messages.last?.content ?? ""
        let system = request.messages.first?.content ?? ""
        // The touched namespace by name with its value; the untouched one
        // summarised. The every-key list was v2's whole extra cost.
        #expect(user.contains("- state/inn"))
        #expect(user.contains("state/inn = standing"))
        #expect(user.contains("- characters/ (1 key, not touched by this session)"))
        #expect(user.contains("- characters/marcus/eyes") == false)
        #expect(system.contains("ONLY facts this session added or changed"))
        #expect(system.contains("omit the key instead"))
        #expect(request.maximumCompletionTokens >= 2000)
        #expect(request.tools.isEmpty)
    }
}

/// Two more things the hundred chapters taught.
@Suite struct MemoryConsolidationReconcileTests {
    private func key(_ raw: String) throws -> MemoryKey { try MemoryKey(validating: raw) }

    @Test func aBareObjectIsOneFact() {
        let one = ServerMemory.consolidationRecords(
            from:
                "{ \"key\": \"continuity/halvorsen_confessed\", \"value\": true, \"importance\": 1 }"
        )
        #expect(one.map(\.key.rawValue) == ["continuity/halvorsen_confessed"])
        #expect(one.first?.value == "true")

        let several = ServerMemory.consolidationRecords(
            from: "{\"key\": \"a/b\", \"value\": 1}\n{\"key\": \"a/c\", \"value\": 2}")
        #expect(several.map(\.key.rawValue) == ["a/b", "a/c"])
    }

    @Test func aParallelNameForAKnownFactIsRoutedToIt() throws {
        let existing = [
            MemoryRecord(key: try key("state/inn_status"), value: "burned"),
            MemoryRecord(key: try key("characters/marcus/eyes"), value: "grey"),
            MemoryRecord(key: try key("characters/ines/eyes"), value: "green"),
        ]
        let incoming = [
            MemoryRecord(key: try key("continuity/inn_status"), value: "standing"),
            MemoryRecord(key: try key("characters/rosa/eyes"), value: "hazel"),
            MemoryRecord(key: try key("state/inn_status"), value: "rebuilt"),
            MemoryRecord(key: try key("continuity/ferry_running"), value: "false"),
        ]
        let (records, merged) = ServerMemory.reconcile(incoming, existing: existing)
        #expect(
            records.map(\.key.rawValue)
                == [
                    "state/inn_status", "characters/rosa/eyes", "state/inn_status",
                    "continuity/ferry_running",
                ])
        // `inn_status` names one thing and is merged; `eyes` names every
        // character and is never merged; an exact match passes through; a
        // genuinely new fact is left alone.
        #expect(merged.count == 1)
        #expect(merged.first?.from == "continuity/inn_status")
        #expect(merged.first?.to == "state/inn_status")
    }

    /// The false positive that gave Marcus a fact before chapter 60: a
    /// shared last word is not a shared fact when the entity differs.
    @Test func aPerEntityAttributeIsNeverRoutedToAnotherEntity() throws {
        let existing = [
            MemoryRecord(key: try key("characters/marcus/knows_photo_content"), value: "false"),
            MemoryRecord(key: try key("setting/location"), value: "Ashgrove"),
        ]
        let incoming = [
            MemoryRecord(key: try key("characters/ines/knows_photo_content"), value: "true"),
            MemoryRecord(key: try key("characters/tomas/location"), value: "lighthouse"),
            // A renamed namespace for the same path is still routed.
            MemoryRecord(key: try key("world/location"), value: "Ashgrove, coastal"),
        ]
        let (records, merged) = ServerMemory.reconcile(incoming, existing: existing)
        #expect(
            records.map(\.key.rawValue)
                == [
                    "characters/ines/knows_photo_content", "characters/tomas/location",
                    "setting/location",
                ])
        #expect(merged.count == 1)
        #expect(merged.first?.from == "world/location")
    }

    @Test func aBasenameSharedByTwoKeysIsNotMerged() throws {
        let existing = [
            MemoryRecord(key: try key("state/ferry_running"), value: "true"),
            MemoryRecord(key: try key("continuity/ferry_running"), value: "false"),
        ]
        let incoming = [MemoryRecord(key: try key("plot/ferry_running"), value: "false")]
        let (records, merged) = ServerMemory.reconcile(incoming, existing: existing)
        #expect(records.first?.key.rawValue == "plot/ferry_running")
        #expect(merged.isEmpty)
    }
}

/// The bootstrap is ranked by the request, the last session's changes come
/// first, and a reversion is a dispute rather than a silent overwrite.
@Suite struct MemoryBootstrapQualityTests {
    private func scope() throws -> MemoryScope {
        try MemoryScope(namespace: "tinytitan", user: "local", workspace: "novel")
    }
    private func key(_ raw: String) throws -> MemoryKey { try MemoryKey(validating: raw) }

    @Test func theRequestDecidesWhatIsInTheWindow() async throws {
        let limits = TinyTitanMemory.MemoryLimits(bootstrapRecords: 2, bootstrapBytes: 1 << 16)
        let store = ContinuityStore(engine: ContinuityEngine(), limits: limits)
        let scope = try scope()
        // Running state rated highest, the way an extraction rates events.
        try await store.set(
            MemoryRecord(
                key: try key("state/ferry"), value: "stopped running",
                importance: 1.0), in: scope)
        try await store.set(
            MemoryRecord(
                key: try key("state/lighthouse"), value: "dark",
                importance: 1.0), in: scope)
        try await store.set(
            MemoryRecord(
                key: try key("characters/rosa/eyes"), value: "hazel",
                importance: 0.8), in: scope)
        try await store.set(
            MemoryRecord(
                key: try key("characters/marcus/eyes"), value: "grey",
                importance: 0.8), in: scope)

        // A static ranking would show the two state facts. The request is
        // about Rosa, and characters outrank state in any case.
        let bootstrap = try await store.sessionInit(
            MemorySession(id: "s1", focus: "Write the chapter where Rosa closes the inn."),
            in: scope)
        let shown = bootstrap.records.map(\.key.rawValue)
        #expect(shown.contains("characters/rosa/eyes"))
        #expect(shown.contains("state/ferry") == false)
    }

    @Test func theLastSessionsChangesAreListedFirst() async throws {
        let store = ContinuityStore(engine: ContinuityEngine())
        let scope = try scope()
        _ = try await store.sessionInit(MemorySession(id: "s1"), in: scope)
        try await store.set(
            MemoryRecord(
                key: try key("rules/weather"), value: "never rains",
                sourceSession: "s1"), in: scope)
        try await store.set(
            MemoryRecord(
                key: try key("state/inn"), value: "burned",
                sourceSession: "s1"), in: scope)

        let next = try await store.sessionInit(
            MemorySession(id: "s2", focus: "continue"),
            in: scope)
        #expect(Set(next.recent.map(\.key.rawValue)) == ["rules/weather", "state/inn"])
        let text = MemoryPrompt.instructions(
            scope: scope, session: MemorySession(id: "s2"),
            bootstrap: next)
        #expect(text.contains("Changed in the most recent session:"))
        // Listed once, under "changed", not again under "already known".
        #expect(text.components(separatedBy: "`state/inn`").count == 2)
    }

    @Test func aReversionIsWrittenButDisputed() async throws {
        let store = ContinuityStore(engine: ContinuityEngine())
        let scope = try scope()
        try await store.set(MemoryRecord(key: try key("state/inn"), value: "standing"), in: scope)
        try await store.set(MemoryRecord(key: try key("state/inn"), value: "burned"), in: scope)
        // A later consolidation, written against a stale bootstrap, says
        // "standing" again.
        let flagged = try await store.set(
            MemoryRecord(key: try key("state/inn"), value: "Standing."),
            in: scope, flaggingReversions: true)
        #expect(flagged)
        let record = try #require(try await store.get(try key("state/inn"), in: scope))
        #expect(record.isDisputed)
        let text = MemoryPrompt.instructions(
            scope: scope, session: MemorySession(id: "s3"),
            bootstrap: try await store.sessionInit(MemorySession(id: "s3"), in: scope))
        #expect(text.contains("[disputed"))

        // A genuinely new value is not a reversion, and settles the dispute.
        let again = try await store.set(
            MemoryRecord(key: try key("state/inn"), value: "rebuilt"),
            in: scope, flaggingReversions: true)
        #expect(again == false)
        #expect(try await store.get(try key("state/inn"), in: scope)?.isDisputed == false)
    }

    @Test func keysAreMentionedBySubject() {
        #expect(ServerMemory.isMentioned("state/inn_status", in: "rosa lit the inn's lamps"))
        #expect(ServerMemory.isMentioned("characters/rosa/eyes", in: "rosa stood at the door"))
        #expect(ServerMemory.isMentioned("state/ferry_running", in: "the tide came in") == false)
        // Whole words: "inn" is a subject, "beginning" is not the inn.
        #expect(ServerMemory.isMentioned("state/inn", in: "at the beginning") == false)
        #expect(ServerMemory.isMentioned("x/eyes", in: "her eyes") == true)
        #expect(ServerMemory.isMentioned("a/b", in: "a b c") == false)
    }
}

/// v3: the extraction sees a bounded key list, unchanged facts are not
/// rewritten, a long session is distilled incrementally, and a fact about
/// the person lands in the shared workspace.
@Suite struct MemoryV3Tests {
    /// unchecked-invariant: every access to `script` and `seen` is under `lock`.
    private final class ScriptedBackend: ServerInferenceBackend, @unchecked Sendable {
        private var script: [ServerCompletion]
        private var seen: [ValidatedChatRequest] = []
        private let lock = NSLock()
        init(_ script: [ServerCompletion]) { self.script = script }
        func generate(
            _ request: ValidatedChatRequest,
            onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
        ) async throws
            -> ServerCompletion
        {
            lock.withLock { seen.append(request) }
            let completion = lock.withLock { script.isEmpty ? nil : script.removeFirst() }
            return completion
                ?? ServerCompletion(
                    content: "[]", toolCalls: [], finishReason: "stop",
                    usage: OpenAIUsage(promptTokens: 0, completionTokens: 0, totalTokens: 0))
        }
        var requests: [ValidatedChatRequest] { lock.withLock { seen } }
    }

    /// Answers each generation from the request itself, and blocks the first
    /// consolidation generation until the test releases it. Holding one
    /// generation under the backend's gate is what makes "a second session
    /// distils while the first is still writing" deterministic instead of a
    /// race. Content is derived from the request rather than taken from a
    /// queue, because which generation arrives first is the thing under test.
    private final class HeldBackend: ServerInferenceBackend, @unchecked Sendable {
        private let responder: @Sendable (ValidatedChatRequest) -> String
        private var seen: [ValidatedChatRequest] = []
        private var heldOne = false
        private var released = false
        private let lock = NSLock()
        init(responder: @escaping @Sendable (ValidatedChatRequest) -> String) {
            self.responder = responder
        }

        private static func extraction(_ request: ValidatedChatRequest) -> Bool {
            request.messages.count == 2
                && request.messages.first?.content?.hasPrefix("You distil") == true
        }

        func generate(
            _ request: ValidatedChatRequest,
            onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
        ) async throws
            -> ServerCompletion
        {
            let (holdThisOne, content) = lock.withLock { () -> (Bool, String) in
                seen.append(request)
                let text = responder(request)
                guard Self.extraction(request), !heldOne else { return (false, text) }
                heldOne = true
                return (true, text)
            }
            while holdThisOne, !lock.withLock({ released }) {
                try await Task.sleep(for: .milliseconds(5))
            }
            return ServerCompletion(
                content: content, toolCalls: [], finishReason: "stop",
                usage: OpenAIUsage(
                    promptTokens: 1, completionTokens: 1,
                    totalTokens: 2))
        }

        func release() { lock.withLock { released = true } }
        var requests: [ValidatedChatRequest] { lock.withLock { seen } }
        var extractions: [ValidatedChatRequest] { lock.withLock { seen.filter(Self.extraction) } }
    }

    private func completion(_ content: String) -> ServerCompletion {
        ServerCompletion(
            content: content, toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
    private func request(_ text: String) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: text)],
            tools: [], stream: false, includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 32),
            maximumCompletionTokens: 32)
    }
    private func key(_ raw: String) throws -> MemoryKey { try MemoryKey(validating: raw) }
    private func configuration() -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .off
        configuration.sessionConsolidation = true
        configuration.consolidationIdleSeconds = 0.05
        configuration.consolidationMinimumCharacters = 0
        return configuration
    }
    /// An extraction is its own two-message conversation whose system prompt
    /// names the job. A turn also opens with a system message -- the memory
    /// fragment -- so the role alone does not tell them apart.
    private func isExtraction(_ request: ValidatedChatRequest) -> Bool {
        request.messages.count == 2
            && request.messages.first?.content?.hasPrefix("You distil") == true
    }
    private func waitForConsolidations(_ inner: ScriptedBackend, atLeast count: Int) async throws {
        for _ in 0..<200 {
            if inner.requests.filter(isExtraction).count >= count { return }
            try await Task.sleep(for: .milliseconds(20))
        }
    }

    // MARK: 1. the key list

    @Test func onlyTouchedNamespacesAreListedInFull() throws {
        let existing =
            try (0..<30).map {
                MemoryRecord(key: try key("decisions/d\($0)"), value: "v")
            } + [
                MemoryRecord(key: try key("state/inn_status"), value: "burned"),
                MemoryRecord(key: try key("characters/rosa/eyes"), value: "hazel"),
            ]
        let turn = JournalTurn(
            session: "s", workspace: "w", index: 0,
            prompt: "write the scene at the inn", reply: "Rosa lit the lamps.")
        let request = ServerMemory.consolidationRequest(
            turns: [turn], existing: existing,
            workspace: "w")
        let user = request.messages.last?.content ?? ""
        // Touched namespaces by name, untouched ones as one line with a count.
        #expect(user.contains("- state/inn_status"))
        #expect(user.contains("- characters/rosa/eyes"))
        #expect(user.contains("- decisions/ (30 keys, not touched by this session)"))
        #expect(user.contains("- decisions/d7") == false)
    }

    @Test func aTouchedNamespaceIsCappedNotUnbounded() throws {
        let existing = try (0..<80).map {
            MemoryRecord(key: try key("state/thing\($0)"), value: "v")
        }
        let listing = ServerMemory.keyListing(existing, mentioned: [existing[0]], maximumListed: 10)
        #expect(listing.components(separatedBy: "\n").count == 11)
        #expect(listing.contains("and 70 more keys"))
    }

    // MARK: 2. unchanged facts are not rewritten

    @Test func aFactWithTheSameValueIsNotWrittenAgain() async throws {
        let configuration = configuration()
        let store = ContinuityStore(engine: ContinuityEngine())
        let service = MemoryService(configuration: configuration, durableStore: store)
        let scope = try #require(configuration.scope())
        let context = try #require(await service.beginSession(id: "s1"))
        _ = await service.storeConsolidation(
            [MemoryRecord(key: try key("characters/rosa/eyes"), value: "hazel")], in: context)
        let written = await service.storeConsolidation(
            [
                MemoryRecord(key: try key("characters/rosa/eyes"), value: "Hazel."),
                MemoryRecord(key: try key("state/inn"), value: "burned"),
            ], in: context)
        #expect(written == 1)
        let rosa = try #require(try await store.get(try key("characters/rosa/eyes"), in: scope))
        #expect(rosa.value == "hazel")
    }

    // MARK: 3. incremental consolidation

    @Test func aSecondConsolidationReadsOnlyTheNewTurns() async throws {
        let inner = ScriptedBackend([
            completion("Chapter one."),  // turn 1
            completion("[]"),  // consolidation of turn 1
            completion("Chapter two."),  // turn 2
            completion("[]"),  // consolidation of turn 2
            completion("Chapter three."),  // turn 3
            completion("[]"),  // consolidation of turn 3
        ])
        let service = MemoryService(
            configuration: configuration(), durableStore: InMemoryStore(),
            journal: InMemoryJournal())
        let backend = MemoryBackend(
            wrapping: inner, service: service, configuration: configuration())
        // One conversation growing turn by turn: the first user message stays
        // the same, so it is one session with three turns, not three sessions.
        func conversation(_ prompts: [String], _ replies: [String]) -> ValidatedChatRequest {
            var messages: [GFTokenizer.Message] = []
            for (index, prompt) in prompts.enumerated() {
                messages.append(GFTokenizer.Message(role: .user, content: prompt))
                if index < replies.count {
                    messages.append(GFTokenizer.Message(role: .assistant, content: replies[index]))
                }
            }
            return ValidatedChatRequest(
                messages: messages, tools: [], stream: false,
                includeUsage: false,
                generationConfig: GenerationConfig(maxNewTokens: 32),
                maximumCompletionTokens: 32)
        }
        _ = try await backend.generate(conversation(["write chapter one"], []), onEvent: { _ in })
        try await waitForConsolidations(inner, atLeast: 1)
        _ = try await backend.generate(
            conversation(
                ["write chapter one", "write chapter two"],
                ["Chapter one."]), onEvent: { _ in })
        try await waitForConsolidations(inner, atLeast: 2)
        _ = try await backend.generate(
            conversation(
                [
                    "write chapter one", "write chapter two",
                    "write chapter three",
                ],
                ["Chapter one.", "Chapter two."]),
            onEvent: { _ in })
        try await waitForConsolidations(inner, atLeast: 3)

        let extractions = inner.requests.filter(isExtraction)
        #expect(extractions.count == 3)
        let third = extractions[2].messages.last?.content ?? ""
        // Turn three plus one turn of context, and not turn one.
        #expect(third.contains("write chapter three"))
        #expect(third.contains("write chapter two"))
        #expect(third.contains("write chapter one") == false)
        await backend.shutDown()
    }

    // MARK: TT-035

    /// A later session that names an amended fact under a new prefix must land
    /// on the address already holding it, not beside it. This is the shape the
    /// `contract` world produced: session 1 distilled `msa/governing_law`,
    /// session 2 distilled `agreement/governing_law`, and both stayed live, so
    /// no supersession could fire.
    @Test func anAmendedFactUnderANewPrefixLandsOnTheExistingKey() async throws {
        let inner = ScriptedBackend([
            completion("Drafted the MSA."),
            completion(
                "[{\"key\": \"msa/governing_law\", \"value\": \"singapore\", "
                    + "\"importance\": 0.9, \"source\": \"user\"}]"),
            completion("Amended the governing law."),
            completion(
                "[{\"key\": \"agreement/governing_law\", \"value\": \"england\", "
                    + "\"importance\": 0.9, \"source\": \"user\"}]"),
        ])
        var configuration = configuration()
        // The durable path, not the injected in-memory store: the two differ in
        // how a consolidation reads back what it already holds, and that read
        // is what this test is about.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tt035-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        configuration.storage.directory = directory
        let service = MemoryService(configuration: configuration)
        let backend = MemoryBackend(wrapping: inner, service: service, configuration: configuration)
        func turn(_ text: String) -> ValidatedChatRequest {
            ValidatedChatRequest(
                messages: [GFTokenizer.Message(role: .user, content: text)],
                tools: [], stream: false, includeUsage: false,
                generationConfig: GenerationConfig(maxNewTokens: 32),
                maximumCompletionTokens: 32)
        }
        _ = try await backend.generate(
            turn("draft the master services agreement"),
            onEvent: { _ in })
        try await waitForConsolidations(inner, atLeast: 1)
        _ = try await backend.generate(
            turn("the client proposed an amendment to the agreement"),
            onEvent: { _ in })
        try await waitForConsolidations(inner, atLeast: 2)

        let scope = try #require(configuration.scope())
        let facts = try await service.recordedFacts(in: scope, limit: 50)
        let keys = facts.map(\.key.rawValue)
        #expect(keys.contains("msa/governing_law"))
        #expect(keys.contains("agreement/governing_law") == false)
        #expect(facts.first { $0.key.rawValue == "msa/governing_law" }?.value == "england")
        await backend.shutDown()
    }

    /// The other half of TT-035: two sessions in one scope may be in flight
    /// together, and the later extraction must not read memory before the
    /// earlier one has written. Here session 1's distillation is held open
    /// while session 2's is requested, which is the state the `contract`
    /// benchmark reached for real (its two prompts were built four seconds
    /// apart and both saw an empty store). The later prompt must list
    /// `msa/governing_law`, and the amendment must land on it.
    @Test func aLaterConsolidationReadsMemoryOnlyAfterTheEarlierOneWrote() async throws {
        let held = HeldBackend { request in
            let user = request.messages.last?.content ?? ""
            let isExtraction = request.messages.first?.content?.hasPrefix("You distil") == true
            guard isExtraction else { return "Noted." }
            return user.contains("master services agreement")
                ? "[{\"key\": \"msa/governing_law\", \"value\": \"singapore\", "
                    + "\"importance\": 0.9, \"source\": \"user\"}]"
                : "[{\"key\": \"agreement/governing_law\", \"value\": \"england\", "
                    + "\"importance\": 0.9, \"source\": \"user\"}]"
        }
        var configuration = configuration()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tt035-race-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        configuration.storage.directory = directory
        let service = MemoryService(configuration: configuration)
        let backend = MemoryBackend(wrapping: held, service: service, configuration: configuration)

        // Both turns land before the first idle timer fires, so both
        // distillations are requested while neither has written: session
        // one's on the rollover, session two's on its own idle timer. Session
        // two names the subject of the fact it amends, so the address the
        // earlier session wrote is one the extraction is shown.
        _ = try await backend.generate(
            request("draft the master services agreement"),
            onEvent: { _ in })
        _ = try await backend.generate(
            request("the client proposed an amendment to the governing law of the agreement"),
            onEvent: { _ in })
        try await Task.sleep(for: .milliseconds(400))
        held.release()
        for _ in 0..<400 where held.extractions.count < 2 {
            try await Task.sleep(for: .milliseconds(20))
        }

        let extractions = held.extractions
        #expect(extractions.count == 2)
        #expect(extractions[1].messages.last?.content?.contains("msa/governing_law") == true)

        let scope = try #require(configuration.scope())
        let facts = try await service.recordedFacts(in: scope, limit: 50)
        let keys = facts.map(\.key.rawValue)
        #expect(keys.contains("msa/governing_law"))
        #expect(keys.contains("agreement/governing_law") == false)
        #expect(facts.first { $0.key.rawValue == "msa/governing_law" }?.value == "england")
        await backend.shutDown()
    }

    // MARK: 8. a fact about the person

    @Test func aGlobalFactLandsInTheSharedWorkspaceAndEveryBootstrap() async throws {
        let configuration = configuration()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("shared-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        var stored = configuration
        stored.storage.directory = directory
        stored.allowsPerRequestWorkspace = true
        let service = MemoryService(configuration: stored)
        let context = try #require(await service.beginSession(id: "s1"))
        var preference = MemoryRecord(
            key: try key("preferences/language"),
            value: "answer in British English")
        preference.isGlobal = true
        let written = await service.storeConsolidation(
            [preference, MemoryRecord(key: try key("state/inn"), value: "burned")], in: context)
        #expect(written == 2)

        // Another project sees the preference and not the inn.
        let other = try #require(await service.beginSession(id: "s2", workspaceOverride: "repo-b"))
        #expect(other.bootstrap.shared.map(\.key.rawValue) == ["preferences/language"])
        #expect(other.bootstrap.records.contains { $0.key.rawValue == "state/inn" } == false)
        let text = await service.instructions(for: other)
        #expect(text.contains("About this person, in every project:"))
        #expect(text.contains("British English"))

        // The shared workspace cannot be named by a request.
        #expect(await service.beginSession(id: "s3", workspaceOverride: "_global") == nil)
        await service.shutDown()
        let files = (try? FileManager.default.subpathsOfDirectory(atPath: directory.path)) ?? []
        #expect(files.contains { $0.hasSuffix("_global.ndjson") })
    }

    @Test func theExtractionCanMarkAFactGlobal() {
        let records = ServerMemory.consolidationRecords(
            from: "[{\"key\": \"preferences/tabs\", \"value\": \"uses tabs\", \"global\": true}, "
                + "{\"key\": \"state/x\", \"value\": \"y\"}]")
        #expect(records.map(\.isGlobal) == [true, false])
    }
}
