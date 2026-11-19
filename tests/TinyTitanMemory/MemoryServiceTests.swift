import Foundation
import Testing

@testable import TinyTitanMemory

/// The service is where "memory is optional" is actually decided: what
/// happens when durable storage is gone, what the model is told, and what a tool call
/// is allowed to reach.
@Suite struct MemoryServiceTests {
    private func configuration(
        enabled: Bool = true,
        workspace: String = "repo-a",
        degrades: Bool = true,
        tools: Bool = true
    ) -> MemoryConfiguration {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = enabled
        configuration.workspace = workspace
        configuration.user = "local"
        configuration.degradesToLocalStore = degrades
        configuration.toolSurface = tools ? .full : .off
        return configuration
    }

    /// A store whose operations fail, standing in for storage that cannot be written.
    private struct UnavailableStore: MemoryStore {
        func get(_: MemoryKey, in _: MemoryScope) async throws -> MemoryRecord? {
            throw MemoryError.backendUnavailable("connection refused")
        }
        func set(_: MemoryRecord, in _: MemoryScope) async throws {
            throw MemoryError.backendUnavailable("connection refused")
        }
        func delete(_: MemoryKey, in _: MemoryScope) async throws -> Bool {
            throw MemoryError.backendUnavailable("connection refused")
        }
        func exists(_: MemoryKey, in _: MemoryScope) async throws -> Bool {
            throw MemoryError.backendUnavailable("connection refused")
        }
        func list(prefix _: String, limit _: Int, in _: MemoryScope) async throws -> [MemoryKey] {
            throw MemoryError.backendUnavailable("connection refused")
        }
        func search(_: MemoryQuery, in _: MemoryScope) async throws -> [MemoryRecord] {
            throw MemoryError.backendUnavailable("connection refused")
        }
        func append(_: String, to _: MemoryKey, in _: MemoryScope) async throws -> MemoryRecord {
            throw MemoryError.backendUnavailable("connection refused")
        }
        func sessionInit(_: MemorySession, in _: MemoryScope) async throws -> MemoryBootstrap {
            throw MemoryError.backendUnavailable("connection refused")
        }
    }

    @Test func disabledServiceStartsNoSession() async {
        let service = MemoryService(configuration: configuration(enabled: false))
        #expect(await service.beginSession(id: "s1") == nil)
        #expect(await service.toolDefinitions().isEmpty)
    }

    @Test func sessionCarriesScopeAndBoundedBootstrap() async throws {
        let store = InMemoryStore()
        let scope = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        try await store.set(
            MemoryRecord(
                key: try MemoryKey(validating: "decisions/sync"),
                value: "Keep FooManager.", importance: 0.9), in: scope)
        let service = MemoryService(configuration: configuration(), durableStore: store)

        let context = try #require(await service.beginSession(id: "s1", modelID: "qwen3.6"))
        #expect(context.scope.workspace == "repo-a")
        #expect(context.session.modelID == "qwen3.6")
        #expect(context.bootstrap.records.count == 1)
        #expect(await service.isDurable)
    }

    @Test func toolCallsUseTheSessionScopeNotTheArguments() async throws {
        let store = InMemoryStore()
        let service = MemoryService(
            configuration: configuration(workspace: "repo-a"),
            durableStore: store)
        let context = try #require(await service.beginSession(id: "s1"))

        // A workspace named in the arguments is simply not a parameter the
        // tools accept; the write has to land in the session's own scope.
        _ = await service.execute(
            name: "memory_set",
            arguments: [
                "key": .string("decisions/db"),
                "value": .string("postgres"),
                "workspace": .string("repo-b"),
            ],
            in: context)

        let repoA = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-a")
        let repoB = try MemoryScope(namespace: "tinytitan", user: "local", workspace: "repo-b")
        #expect(try await store.get(try MemoryKey(validating: "decisions/db"), in: repoA) != nil)
        #expect(try await store.get(try MemoryKey(validating: "decisions/db"), in: repoB) == nil)
    }

    @Test func malformedKeysComeBackAsToolErrors() async throws {
        let service = MemoryService(configuration: configuration(), durableStore: InMemoryStore())
        let context = try #require(await service.beginSession(id: "s1"))

        for key in ["../escape", "/absolute", "a//b", "with space"] {
            let result = await service.execute(
                name: "memory_set",
                arguments: [
                    "key": .string(key),
                    "value": .string("x"),
                ],
                in: context)
            #expect(result.isFailure, "\(key) should be rejected")
        }
    }

    @Test func unavailableBackendDegradesAndSaysSo() async throws {
        let service = MemoryService(
            configuration: configuration(), durableStore: UnavailableStore())
        let context = try #require(await service.beginSession(id: "s1"))

        // Serving continues, but the session knows it is not persisting and
        // the prompt says so, because a model told its write succeeded when
        // it did not is worse than one with no memory at all.
        #expect(!context.isDurable)
        let prompt = await service.instructions(for: context)
        #expect(prompt.contains("unreachable"))

        let write = await service.execute(
            name: "memory_set",
            arguments: [
                "key": .string("decisions/x"),
                "value": .string("v"),
            ],
            in: context)
        #expect(!write.isFailure)
        let read = await service.execute(
            name: "memory_get",
            arguments: ["key": .string("decisions/x")],
            in: context)
        #expect(!read.isFailure)
    }

    @Test func withoutLocalFallbackAnUnavailableBackendDisablesTheSession() async {
        let service = MemoryService(
            configuration: configuration(degrades: false),
            durableStore: UnavailableStore())
        #expect(await service.beginSession(id: "s1") == nil)
    }

    @Test func failedWritesAreReportedNotSwallowed() async throws {
        // The store rejects the value; the model must see a failure rather
        // than a confirmation.
        var config = configuration()
        config.limits.maximumValueBytes = 16
        let service = MemoryService(
            configuration: config,
            durableStore: InMemoryStore(limits: config.limits))
        let context = try #require(await service.beginSession(id: "s1"))

        let result = await service.execute(
            name: "memory_set",
            arguments: [
                "key": .string("big"), "value": .string(String(repeating: "x", count: 64)),
            ],
            in: context)
        #expect(result.isFailure)
        #expect(result.jsonString().contains("\"ok\":false"))
    }

    @Test func logsCarryNoMemoryContents() async throws {
        // Memory can hold anything the model wrote; the operational log must
        // not become a copy of it.
        let events = EventCollector()
        let service = MemoryService(
            configuration: configuration(),
            durableStore: InMemoryStore(),
            log: { event in events.append(event) })
        let context = try #require(await service.beginSession(id: "s1"))
        _ = await service.execute(
            name: "memory_set",
            arguments: [
                "key": .string("secrets/note"),
                "value": .string("SUPER-SECRET-VALUE"),
            ],
            in: context)
        await service.endSession(context)

        let messages = events.messages()
        #expect(!messages.isEmpty)
        #expect(!messages.contains { $0.contains("SUPER-SECRET-VALUE") })
    }

    @Test func toolsAreOffUnlessTurnedOn() async {
        // The loop ships off: it is where the request-lifecycle risk sits,
        // and whether a 3B-active model uses six tools well is a measurement
        // rather than a claim. The prompt and the bootstrap work without it.
        let quiet = MemoryService(
            configuration: configuration(tools: false),
            durableStore: InMemoryStore())
        #expect(await quiet.toolDefinitions().isEmpty)

        var withTools = configuration()
        withTools.toolSurface = .full
        let service = MemoryService(configuration: withTools, durableStore: InMemoryStore())
        #expect(Set(await service.toolDefinitions().map(\.name)) == MemoryTools.names)
    }

    @Test func searchAndListRoundTripThroughTheService() async throws {
        let service = MemoryService(configuration: configuration(), durableStore: InMemoryStore())
        let context = try #require(await service.beginSession(id: "s1"))
        _ = await service.execute(
            name: "memory_set",
            arguments: [
                "key": .string("decisions/sync"),
                "value": .string("Keep FooManager for the race."),
                "tags": .stringArray(["sync"]),
            ],
            in: context)

        let search = await service.execute(
            name: "memory_search",
            arguments: ["query": .string("FooManager")],
            in: context)
        #expect(!search.isFailure)
        #expect(search.jsonString().contains("decisions/sync"))

        let list = await service.execute(
            name: "memory_list",
            arguments: ["prefix": .string("decisions/")],
            in: context)
        #expect(list.jsonString().contains("decisions/sync"))
    }

    @Test func consolidationStampsTheSessionAndCounts() async throws {
        let store = InMemoryStore()
        let service = MemoryService(configuration: configuration(), durableStore: store)
        let context = try #require(await service.beginSession(id: "s-consolidate"))

        let written = await service.storeConsolidation(
            [
                MemoryRecord(key: try MemoryKey(validating: "decisions/a"), value: "one"),
                MemoryRecord(key: try MemoryKey(validating: "gotchas/b"), value: "two"),
            ], in: context)

        #expect(written == 2)
        let stored = try await store.get(
            try MemoryKey(validating: "decisions/a"), in: context.scope)
        #expect(stored?.sourceSession == "s-consolidate")
    }
}

/// Collects log events from the service's `@Sendable` callback.
///
/// unchecked-invariant: every access is under `lock`.
private final class EventCollector: @unchecked Sendable {
    private var events: [MemoryLogEvent] = []
    private let lock = NSLock()

    func append(_ event: MemoryLogEvent) {
        lock.withLock { events.append(event) }
    }

    func messages() -> [String] {
        lock.withLock { events.map(\.message) }
    }
}
