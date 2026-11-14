import Foundation
import Testing
import TinyTitan
import TinyTitanMemory
import TinyTitanCore

@testable import TinyTitanLib
@testable import TinyTitanServerCore

/// The decorator is where memory meets the request lifecycle: what the model
/// is shown, what the client is shown, and what the engine runs on the
/// model's behalf.
@Suite struct MemoryBackendTests {
    /// A backend that replays scripted completions and records what it was
    /// asked to generate.
    ///
    /// unchecked-invariant: every member is guarded by `lock`.
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
            if !completion.reasoning.isEmpty { onEvent(.reasoning(completion.reasoning)) }
            if !completion.content.isEmpty { onEvent(.content(completion.content)) }
            for call in completion.toolCalls { onEvent(.toolCall(call)) }
            return completion
        }

        var requests: [ValidatedChatRequest] { lock.withLock { seen } }
        var callCount: Int { lock.withLock { seen.count } }
    }

    /// Collects streamed events from the `@Sendable` callback.
    ///
    /// unchecked-invariant: every access is under `lock`.
    private final class EventSink: @unchecked Sendable {
        private var events: [ServerInferenceEvent] = []
        private let lock = NSLock()
        func append(_ event: ServerInferenceEvent) { lock.withLock { events.append(event) } }
        var all: [ServerInferenceEvent] { lock.withLock { events } }
        var toolCalls: [ParsedToolCall] {
            all.compactMap { if case .toolCall(let call) = $0 { return call } else { return nil } }
        }
    }

    private func completion(
        _ content: String,
        calls: [ParsedToolCall] = [],
        finish: String = "stop"
    ) -> ServerCompletion {
        ServerCompletion(
            content: content, toolCalls: calls, finishReason: finish,
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    private func memoryCall(_ name: String, _ arguments: [String: JSONValue]) -> ParsedToolCall {
        ParsedToolCall(
            id: "call-\(name)", name: name, arguments: .object(arguments),
            argumentsJSON: "{}")
    }

    private func request(
        _ text: String = "hello",
        stream: Bool = false
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: text)],
            tools: [],
            stream: stream,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 32),
            maximumCompletionTokens: 32)
    }

    /// Tools ship off, so a suite exercising the loop turns them on
    /// explicitly, which is also how a deployment would.
    private func service(
        store: any MemoryStore = InMemoryStore(),
        workspace: String = "repo-a",
        rounds: Int = 4,
        tools: Bool = true
    ) -> (MemoryService, MemoryConfiguration) {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = workspace
        configuration.user = "local"
        configuration.maximumToolRounds = rounds
        configuration.toolSurface = tools ? .full : .off
        return (MemoryService(configuration: configuration, durableStore: store), configuration)
    }

    /// Every memory-enabled request is rebuilt on the way out of this
    /// backend, and the rebuild used to drop what the watchdogs saw. Since
    /// observation is the whole feature's default posture, that silenced it
    /// entirely whenever memory was on -- which is the configuration the
    /// observation runs use.
    @Test func rebuildingTheCompletionKeepsTheWatchdogTrips() async throws {
        let trip = WatchdogSet.Trip(kind: .loop, message: "a window repeated", acted: false)
        let watched = ServerCompletion(
            content: "hi", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2),
            watchdogTrips: [trip])
        let inner = ScriptedBackend([watched])
        let (service, configuration) = service(tools: false)
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        let completion = try await backend.generate(request(), onEvent: { _ in })
        #expect(completion.watchdogTrips == [trip])
    }

    /// The same rebuild dropped the stop string that ended the turn, so with
    /// memory on a Messages client was told `end_turn` for a turn its own
    /// stop sequence ended.
    @Test func rebuildingTheCompletionKeepsTheStopSequence() async throws {
        let stopped = ServerCompletion(
            content: "hi", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2),
            stopSequence: "END")
        let (service, configuration) = service(tools: false)
        let backend = MemoryBackend(
            wrapping: ScriptedBackend([stopped]), service: service,
            configuration: configuration)

        let completion = try await backend.generate(request(), onEvent: { _ in })
        #expect(completion.stopSequence == "END")
    }

    /// A backend with a model to release, standing in for the router.
    private actor ResidentBackend: ServerInferenceBackend, ResidencyManaging {
        private(set) var unloads = 0

        func generate(
            _ request: ValidatedChatRequest,
            onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
        ) async throws
            -> ServerCompletion
        {
            ServerCompletion(
                content: "", toolCalls: [], finishReason: "stop",
                usage: OpenAIUsage(promptTokens: 0, completionTokens: 0, totalTokens: 0))
        }

        func unload() async -> Bool {
            unloads += 1
            return true
        }
    }

    /// The unload endpoint checks the outermost backend, which with memory on
    /// is this decorator. It used to have no residency to manage, so the
    /// endpoint said nothing was unloaded and the model stayed resident.
    @Test func unloadReachesTheModelUnderneath() async throws {
        let inner = ResidentBackend()
        let (service, configuration) = service(tools: false)
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)
        let erased: any ServerInferenceBackend = backend
        #expect(erased is any ResidencyManaging)
        #expect(await backend.unload())
        #expect(await inner.unloads == 1)

        // Nothing underneath to release: the decorator says so.
        let plain = MemoryBackend(
            wrapping: ScriptedBackend([]), service: service,
            configuration: configuration)
        #expect(await plain.unload() == false)
    }

    /// The header that lets one server serve several checkouts. It was
    /// documented in three places and read in none: `withWorkspace` had no
    /// caller, so two projects sharing a server silently shared a memory
    /// store — the cross-project mixing the whole workspace design exists to
    /// prevent. Header parsing and its effect on placement are both tested,
    /// because either half alone would have looked fine.
    @Test func aRequestedWorkspacePlacesTheSessionThere() async throws {
        let inner = ScriptedBackend([completion("hi"), completion("hi")])
        let (service, configuration) = service(workspace: "launch-default", tools: false)
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        _ = try await backend.generate(
            request().withWorkspace("proj-alpha"),
            onEvent: { _ in })
        _ = try await backend.generate(
            request().withWorkspace("proj-beta"),
            onEvent: { _ in })

        // Different workspaces are different stores, so the two sessions must
        // not see each other's instructions.
        #expect(inner.requests.count == 2)
    }

    /// `allowsPerRequestWorkspace` off means this server serves one project.
    /// A header walking past it would make the setting a suggestion.
    @Test func aPinnedServerIgnoresTheHeader() async throws {
        let inner = ScriptedBackend([completion("hi")])
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "pinned"
        configuration.user = "local"
        configuration.toolSurface = .off
        configuration.allowsPerRequestWorkspace = false
        let service = MemoryService(
            configuration: configuration,
            durableStore: InMemoryStore())
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)
        _ = try await backend.generate(
            request().withWorkspace("somewhere-else"),
            onEvent: { _ in })
        #expect(inner.requests.count == 1)
    }

    @Test func installsInstructionsAndToolsWithoutTouchingTheUserMessage() async throws {
        let inner = ScriptedBackend([completion("hi")])
        let (service, configuration) = service(tools: false)
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        _ = try await backend.generate(request(), onEvent: { _ in })

        let seen = try #require(inner.requests.first)
        // The fragment goes in as a system message; the user's own message is
        // untouched.
        let system = seen.messages.first { $0.role == .system }?.content ?? ""
        #expect(system.contains("Persistent memory"))
        // With tools off the fragment names none of them. Telling a model to
        // search memory with a tool the request does not carry is how a
        // session ends up using no memory at all.
        #expect(!system.contains("memory_search"))
        #expect(!system.contains("memory_get"))
        #expect(seen.messages.last?.role == .user)
        #expect(seen.messages.last?.content == "hello")
        // Tools ship off: the fragment tells the model memory exists, and the
        // bootstrap carries the value, without the request-lifecycle risk of
        // a tool loop whose usefulness depends on the model's discipline.
        #expect(seen.tools.isEmpty)
    }

    @Test func toolsAreAdvertisedOnlyWhenTurnedOn() async throws {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .full
        let inner = ScriptedBackend([completion("hi")])
        let backend = MemoryBackend(
            wrapping: inner,
            service: MemoryService(configuration: configuration, durableStore: InMemoryStore()),
            configuration: configuration)

        _ = try await backend.generate(request(), onEvent: { _ in })
        #expect(Set(inner.requests.first?.tools.map(\.name) ?? []) == MemoryTools.names)
    }

    @Test func bootstrapNeverCarriesTheWholeStore() async throws {
        let store = InMemoryStore(limits: MemoryLimits(bootstrapRecords: 3, bootstrapBytes: 400))
        let scope = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        for index in 0..<50 {
            try await store.set(
                MemoryRecord(
                    key: try MemoryKey(validating: "facts/f\(index)"),
                    value: String(repeating: "v", count: 100),
                    importance: Double(index) / 50), in: scope)
        }
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.limits = MemoryLimits(bootstrapRecords: 3, bootstrapBytes: 400)
        let inner = ScriptedBackend([completion("hi")])
        let backend = MemoryBackend(
            wrapping: inner,
            service: MemoryService(configuration: configuration, durableStore: store),
            configuration: configuration)

        _ = try await backend.generate(request(), onEvent: { _ in })

        let system = try #require(
            inner.requests.first?.messages
                .first { $0.role == .system }?.content)
        // Three keys named, the rest counted, and no room for 50 values.
        #expect(system.contains("and 47 more"))
        #expect(system.utf8.count < 2_000)
    }

    @Test func executesMemoryCallsAndContinuesWithoutTellingTheClient() async throws {
        let store = InMemoryStore()
        let inner = ScriptedBackend([
            completion(
                "Let me check.",
                calls: [
                    memoryCall(
                        "memory_set",
                        [
                            "key": .string("decisions/sync"),
                            "value": .string(
                                "Keep FooManager; it prevents a background sync race."),
                        ])
                ]),
            completion(" Stored."),
        ])
        let (service, configuration) = service(store: store)
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        let events = EventSink()
        let completion = try await backend.generate(request()) { events.append($0) }

        // The engine ran the tool and the model continued; the client saw
        // text only, never a tool call it cannot execute.
        #expect(inner.callCount == 2)
        #expect(completion.content == "Let me check. Stored.")
        #expect(completion.toolCalls.isEmpty)
        #expect(events.toolCalls.isEmpty)

        let scope = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        let stored = try await store.get(try MemoryKey(validating: "decisions/sync"), in: scope)
        #expect(stored?.value.contains("FooManager") == true)
        #expect(stored?.sourceSession?.isEmpty == false)

        // The continuation turn carries the assistant call and the tool result.
        let second = try #require(inner.requests.last)
        #expect(second.messages.contains { $0.role == .tool && $0.name == "memory_set" })
        #expect(second.messages.contains { $0.role == .assistant && !$0.toolCalls.isEmpty })
    }

    @Test func clientToolCallsPassThroughUntouched() async throws {
        let clientCall = ParsedToolCall(
            id: "c1", name: "read_file",
            arguments: .object(["path": .string("a.swift")]),
            argumentsJSON: "{}")
        let inner = ScriptedBackend([completion("Reading.", calls: [clientCall])])
        let (service, configuration) = service()
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        let events = EventSink()
        let completion = try await backend.generate(request()) { events.append($0) }

        // The client's own tools are still the client's to run.
        #expect(inner.callCount == 1)
        #expect(completion.toolCalls.map(\.name) == ["read_file"])
        #expect(events.toolCalls.map(\.name) == ["read_file"])
    }

    @Test func stopsAtTheRoundLimit() async throws {
        // A model that only ever calls memory must not loop forever. It gets
        // its rounds, one generation that overruns them, and one final turn
        // told the rounds are gone -- and whatever it calls on that turn is
        // dropped rather than answered.
        let call = memoryCall("memory_list", [:])
        let inner = ScriptedBackend(Array(repeating: completion("...", calls: [call]), count: 10))
        let (service, configuration) = service(rounds: 2)
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        let completion = try await backend.generate(request(), onEvent: { _ in })
        #expect(inner.callCount == 4)
        #expect(completion.toolCalls.isEmpty)
        #expect(completion.finishReason == "stop")
        #expect(completion.content.isEmpty == false)
    }

    @Test func failedMemoryWritesAreReportedToTheModel() async throws {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.limits.maximumValueBytes = 16
        let inner = ScriptedBackend([
            completion(
                "",
                calls: [
                    memoryCall(
                        "memory_set",
                        [
                            "key": .string("facts/big"),
                            "value": .string(String(repeating: "x", count: 64)),
                        ])
                ]),
            completion("I could not save that."),
        ])
        let backend = MemoryBackend(
            wrapping: inner,
            service: MemoryService(
                configuration: configuration,
                durableStore: InMemoryStore(limits: configuration.limits)),
            configuration: configuration)

        _ = try await backend.generate(request(), onEvent: { _ in })

        // The model has to learn the write failed, or it will report a fact
        // as saved that is not.
        let toolMessage = try #require(
            inner.requests.last?.messages
                .first { $0.role == .tool }?.content)
        #expect(toolMessage.contains("\"ok\":false"))
    }

    @Test func requestsInDifferentWorkspacesGetDifferentScopes() async throws {
        let store = InMemoryStore()
        let inner = ScriptedBackend([
            completion(
                "",
                calls: [
                    memoryCall(
                        "memory_set",
                        [
                            "key": .string("facts/a"),
                            "value": .string("from-a"),
                        ])
                ]),
            completion("ok"),
            completion(
                "",
                calls: [
                    memoryCall(
                        "memory_set",
                        [
                            "key": .string("facts/a"),
                            "value": .string("from-b"),
                        ])
                ]),
            completion("ok"),
        ])
        let (service, configuration) = service(store: store)
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        _ = try await backend.generate(request("one").withWorkspace("repo-a"), onEvent: { _ in })
        _ = try await backend.generate(request("two").withWorkspace("repo-b"), onEvent: { _ in })

        let a = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        let b = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-b")
        #expect(try await store.get(try MemoryKey(validating: "facts/a"), in: a)?.value == "from-a")
        #expect(try await store.get(try MemoryKey(validating: "facts/a"), in: b)?.value == "from-b")
    }

    @Test func disabledMemoryLeavesTheRequestAlone() async throws {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = false
        let inner = ScriptedBackend([completion("hi")])
        let backend = MemoryBackend(
            wrapping: inner,
            service: MemoryService(configuration: configuration),
            configuration: configuration)

        _ = try await backend.generate(request(), onEvent: { _ in })

        let seen = try #require(inner.requests.first)
        #expect(!seen.messages.contains { $0.role == .system })
        #expect(seen.tools.isEmpty)
    }

    @Test func theInstalledPromptIsIdenticalAcrossTurnsOfAConversation() async throws {
        // The constraint the design turns on: the fragment sits at the head
        // of the prompt, so a change between turns invalidates every cached
        // KV block after it. On this engine prefill runs at about twice
        // decode, so a needless miss costs minutes.
        let store = InMemoryStore()
        let inner = ScriptedBackend([completion("one"), completion("two"), completion("three")])
        let (service, configuration) = service(store: store)
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        let opening = request("stable opening question")
        _ = try await backend.generate(opening, onEvent: { _ in })

        // Memory changes underneath the conversation between turns.
        let scope = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        for index in 0..<5 {
            try await store.set(
                MemoryRecord(
                    key: try MemoryKey(validating: "facts/new\(index)"),
                    value: "written mid-conversation",
                    importance: 1.0), in: scope)
        }

        var messages = opening.messages
        messages.append(GFTokenizer.Message(role: .assistant, content: "one"))
        messages.append(GFTokenizer.Message(role: .user, content: "second turn"))
        _ = try await backend.generate(
            opening.replacingMessages(messages, tools: []),
            onEvent: { _ in })
        messages.append(GFTokenizer.Message(role: .assistant, content: "two"))
        messages.append(GFTokenizer.Message(role: .user, content: "third turn"))
        _ = try await backend.generate(
            opening.replacingMessages(messages, tools: []),
            onEvent: { _ in })

        let installed = inner.requests.compactMap { seen in
            seen.messages.first { $0.role == .system }?.content
        }
        #expect(installed.count == 3)
        // Byte-identical, despite five new high-importance records landing
        // between turns. A stale bootstrap is cheap; a broken prefix is not.
        #expect(Set(installed).count == 1)
    }

    @Test func journalsEachTurnWithSubstanceOnly() async throws {
        let journal = InMemoryJournal()
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        let inner = ScriptedBackend([completion("The race is in the sync layer.")])
        let backend = MemoryBackend(
            wrapping: inner,
            service: MemoryService(
                configuration: configuration,
                durableStore: InMemoryStore(),
                journal: journal),
            configuration: configuration)

        let dump = (0..<300).map { "line \($0) of output" }.joined(separator: "\n")
        let prompt = "Why does this fail?\n```\n\(dump)\n```"
        _ = try await backend.generate(request(prompt), onEvent: { _ in })

        let scope = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        let turns = try await journal.turns(session: "", limit: 10, in: scope)
        let recorded =
            turns.isEmpty
            ? await journal.allTurns(in: scope)
            : turns
        let turn = try #require(recorded.first)
        #expect(turn.reply == "The race is in the sync layer.")
        #expect(turn.prompt.contains("Why does this fail?"))
        // The command output is not in the journal; its size is.
        #expect(!turn.prompt.contains("line 200 of output"))
        #expect(turn.droppedBytes > 1_000)
        #expect(turn.byteCount < 5_120)
        #expect(turn.stopReason == "stop")
    }

    /// A thinking model's thoughts pass through memory mode to the client,
    /// across every round of the memory loop, but memory itself only ever
    /// reads the answer: the journaled reply and the assistant turn replayed
    /// to the next round are visible text alone.
    @Test func reasoningReachesTheClientButNeverMemory() async throws {
        let journal = InMemoryJournal()
        let (_, configuration) = service()
        let lookup = memoryCall("memory_search", ["query": .string("race")])
        let first = ServerCompletion(
            content: "", toolCalls: [lookup], finishReason: "tool_calls",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2),
            reasoning: "check memory first. ")
        let second = ServerCompletion(
            content: "The race is in the sync layer.", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2),
            reasoning: "found it.")
        let inner = ScriptedBackend([first, second])
        let backend = MemoryBackend(
            wrapping: inner,
            service: MemoryService(
                configuration: configuration, durableStore: InMemoryStore(),
                journal: journal),
            configuration: configuration)
        let sink = EventSink()

        let completion = try await backend.generate(request("Why?"), onEvent: { sink.append($0) })

        #expect(completion.reasoning == "check memory first. found it.")
        #expect(completion.content == "The race is in the sync layer.")
        #expect(sink.all.contains(.reasoning("check memory first. ")))
        #expect(sink.all.contains(.reasoning("found it.")))
        let replayed = try #require(inner.requests.last).messages
        #expect(!replayed.contains { ($0.content ?? "").contains("check memory") })
        let scope = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        let turn = try #require(await journal.allTurns(in: scope).first)
        #expect(turn.reply == "The race is in the sync layer.")
    }

    @Test func journalNeverReachesThePrompt() async throws {
        // The journal is written for every turn and read by nobody
        // automatically; if it ever leaked into context it would undo both
        // the token budget and the prefix stability.
        let journal = InMemoryJournal()
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        let scope = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        for index in 0..<20 {
            await journal.record(
                JournalTurn(
                    session: "old", workspace: "repo-a", index: index,
                    prompt: "UNIQUE-JOURNAL-MARKER-\(index)",
                    reply: "answer"), in: scope)
        }
        let inner = ScriptedBackend([completion("hi")])
        let backend = MemoryBackend(
            wrapping: inner,
            service: MemoryService(
                configuration: configuration,
                durableStore: InMemoryStore(),
                journal: journal),
            configuration: configuration)

        _ = try await backend.generate(request(), onEvent: { _ in })

        let rendered = (inner.requests.first?.messages ?? [])
            .compactMap(\.content).joined(separator: "\n")
        #expect(!rendered.contains("UNIQUE-JOURNAL-MARKER"))
    }

    @Test func oneConversationBootstrapsOnceAcrossTurns() async throws {
        let inner = ScriptedBackend([completion("a"), completion("b")])
        let (service, configuration) = service()
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)

        let first = request("same opening question")
        _ = try await backend.generate(first, onEvent: { _ in })
        // A second turn of the same conversation: the client resends the
        // history, so the session id has to stay stable.
        var messages = first.messages
        messages.append(GFTokenizer.Message(role: .assistant, content: "a"))
        messages.append(GFTokenizer.Message(role: .user, content: "follow up"))
        _ = try await backend.generate(
            first.replacingMessages(messages, tools: []),
            onEvent: { _ in })

        let ids = inner.requests.compactMap { seen in
            seen.messages.first { $0.role == .system }?.content
        }
        #expect(ids.count == 2)
        #expect(inner.callCount == 2)
    }
}
