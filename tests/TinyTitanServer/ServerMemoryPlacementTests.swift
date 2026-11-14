import Foundation
import Testing
import TinyTitan
import TinyTitanMemory
import TinyTitanCore

@testable import TinyTitanLib
@testable import TinyTitanServerCore

/// Which project a conversation's memory lands in, and how the server tells.
@Suite struct ServerMemoryPlacementTests {
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
        _ content: String,
        calls: [ParsedToolCall] = []
    ) -> ServerCompletion {
        ServerCompletion(
            content: content, toolCalls: calls, finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    private func toolCall(_ name: String, _ arguments: [String: String]) -> ServerCompletion {
        let call = ParsedToolCall(
            id: "call-\(name)", name: name,
            arguments: .object(arguments.mapValues { .string($0) }),
            argumentsJSON: "{}")
        return completion("", calls: [call])
    }

    private func request(_ messages: [GFTokenizer.Message]) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages, tools: [], stream: false, includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 32),
            maximumCompletionTokens: 32)
    }

    private func system(_ text: String) -> GFTokenizer.Message {
        GFTokenizer.Message(role: .system, content: text)
    }
    private func user(_ text: String) -> GFTokenizer.Message {
        GFTokenizer.Message(role: .user, content: text)
    }

    @Test func claudeCodeStyleWorkingDirectoryIsRecognised() {
        let messages = [
            system(
                """
                You are a coding assistant.

                # Environment
                 - Primary working directory: /Users/ada/novels/photograph
                 - Is a git repository: true
                """),
            user("write chapter 41"),
        ]
        #expect(
            ServerMemory.declaredWorkingDirectory(in: messages)
                == "/Users/ada/novels/photograph")

        let plain = [
            system("Working directory: /Users/ada/src/tinytitan\nPlatform: darwin"),
            user("hi"),
        ]
        #expect(ServerMemory.declaredWorkingDirectory(in: plain) == "/Users/ada/src/tinytitan")
    }

    @Test func codexStyleEnvironmentContextIsRecognised() {
        let messages = [
            system(
                "<environment_context>\n  <cwd>/Users/ada/src/tinytitan</cwd>\n  <shell>zsh</shell>\n</environment_context>"
            ),
            user("fix the build"),
        ]
        #expect(ServerMemory.declaredWorkingDirectory(in: messages) == "/Users/ada/src/tinytitan")
    }

    @Test func onlySystemMessagesAndAbsolutePathsCount() {
        // A user pasting a transcript must not be able to move their memory.
        let pasted = [
            system("You are helpful."),
            user("Working directory: /Users/eve/secret"),
        ]
        #expect(ServerMemory.declaredWorkingDirectory(in: pasted) == nil)

        let relative = [system("Working directory: src/tinytitan"), user("hi")]
        #expect(ServerMemory.declaredWorkingDirectory(in: relative) == nil)

        let none = [system("You are helpful."), user("hi")]
        #expect(ServerMemory.declaredWorkingDirectory(in: none) == nil)
    }

    /// The whole point: one server, two clients in two directories, two
    /// fact stores, and neither bootstrap ever shows the other's facts.
    @Test func declaredDirectoriesKeepProjectsApart() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("placement-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }

        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.toolSurface = .full
        configuration.workspace = "launch-dir"
        configuration.user = "local"
        configuration.namespace = "tinytitan"
        configuration.storage.directory = directory
        let service = MemoryService(configuration: configuration)

        let novelPrompt = system("Working directory: /Users/ada/novels/photograph")
        let codePrompt = system("<cwd>/Users/ada/src/tinytitan</cwd>")

        // Session one: the novel writes a fact.
        let inner = ScriptedBackend([
            toolCall("memory_set", ["key": "plot/brother", "value": "missing until act three"]),
            completion("noted"),
        ])
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)
        _ = try await backend.generate(
            request([novelPrompt, user("remember the brother")]),
            onEvent: { _ in })

        // Session two, same server: the codebase must not see it.
        let inner2 = ScriptedBackend([completion("ok")])
        let backend2 = MemoryBackend(
            wrapping: inner2, service: service,
            configuration: configuration)
        _ = try await backend2.generate(
            request([codePrompt, user("fix the build")]),
            onEvent: { _ in })
        let codeSystem = inner2.requests.first?.messages.first { $0.role == .system }?.content ?? ""
        #expect(!codeSystem.contains("plot/brother"))

        // And a fresh novel session does.
        let inner3 = ScriptedBackend([completion("ok")])
        let backend3 = MemoryBackend(
            wrapping: inner3, service: service,
            configuration: configuration)
        _ = try await backend3.generate(
            request([novelPrompt, user("chapter 42")]),
            onEvent: { _ in })
        let novelSystem =
            inner3.requests.first?.messages.first { $0.role == .system }?.content ?? ""
        #expect(novelSystem.contains("plot/brother"))

        await service.shutDown()
        let files = (try? FileManager.default.subpathsOfDirectory(atPath: directory.path)) ?? []
        // One file per declared project, named for it, and nothing landed in
        // the launch workspace.
        #expect(files.contains { $0.contains("/photograph-") && $0.hasSuffix(".ndjson") })
        #expect(files.contains { $0.contains("/tinytitan-") && $0.hasSuffix(".ndjson") })
        #expect(!files.contains { $0.contains("launch-dir") })
    }

    @Test func aDeclaredHomeDirectoryFallsBackToTheLaunchWorkspace() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("placement-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "launch-dir"
        configuration.user = "local"
        configuration.namespace = "tinytitan"
        configuration.storage.directory = directory
        let service = MemoryService(configuration: configuration)
        let home = FileManager.default.homeDirectoryForCurrentUser.path

        let inner = ScriptedBackend([completion("ok")])
        let backend = MemoryBackend(
            wrapping: inner, service: service,
            configuration: configuration)
        _ = try await backend.generate(
            request([
                system("Working directory: \(home)"),
                user("hi"),
            ]),
            onEvent: { _ in })
        await service.shutDown()
        let files = (try? FileManager.default.subpathsOfDirectory(atPath: directory.path)) ?? []
        #expect(files.contains { $0.contains("launch-dir.ndjson") })
    }
}
