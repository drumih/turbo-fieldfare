import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// K3 serving: request validation/mapping into the renderer-facing payload,
/// response surfacing (reasoning_content, tool_calls), SSE reasoning deltas,
/// and exact-prefix cache matching. Uses a scripted backend — no real model.
@Suite struct K3ServerTests {

    private static let modelID = "kimi-k3"

    private actor K3CapturingBackend: ServerInferenceBackend {
        private(set) var received: ValidatedChatRequest?
        let events: [ServerInferenceEvent]
        let completion: ServerCompletion

        init(events: [ServerInferenceEvent] = [], completion: ServerCompletion) {
            self.events = events
            self.completion = completion
        }

        func generate(
            _ request: ValidatedChatRequest,
            onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
        ) async throws -> ServerCompletion {
            received = request
            for event in events {
                onEvent(event)
            }
            return completion
        }
    }

    private static func plainCompletion(
        _ content: String = "hello",
        reasoning: String? = nil,
        toolCalls: [ParsedToolCall] = [],
        cachedTokens: Int = 0
    ) -> ServerCompletion {
        ServerCompletion(
            content: content,
            toolCalls: toolCalls,
            finishReason: toolCalls.isEmpty ? "stop" : "tool_calls",
            usage: OpenAIUsage(promptTokens: 12, completionTokens: 3, totalTokens: 15,
                               cachedTokens: cachedTokens),
            reasoningContent: reasoning)
    }

    private func startServer(
        backend: K3CapturingBackend
    ) async throws -> (TurboFieldfareHTTPServer, Int) {
        let server = TurboFieldfareHTTPServer(
            modelID: Self.modelID,
            queueLimit: 1,
            backend: backend,
            requestValidation: .k3)
        let channel = try await server.start(port: 0)
        return try (server, #require(channel.localAddress?.port))
    }

    private func post(_ port: Int, _ body: String) async throws -> (Data, Int) {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(body.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        return (data, (response as? HTTPURLResponse)?.statusCode ?? -1)
    }

    @Test func k3PromptCacheRequiresExactTokenPrefix() {
        #expect(K3ServerPromptCache.isExactPrefix([1, 2, 3], of: [1, 2, 3, 4]))
        #expect(K3ServerPromptCache.isExactPrefix([1, 2, 3], of: [1, 2, 3]))
        #expect(!K3ServerPromptCache.isExactPrefix([], of: [1]))
        #expect(!K3ServerPromptCache.isExactPrefix([1, 2, 4], of: [1, 2, 3, 4]))
        #expect(!K3ServerPromptCache.isExactPrefix([1, 2, 3], of: [1, 2]))
    }

    @Test func modelsEndpointListsK3ModelID() async throws {
        let (server, port) = try await startServer(
            backend: K3CapturingBackend(completion: Self.plainCompletion()))
        let data = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0
        #expect(String(decoding: data, as: UTF8.self).contains(Self.modelID))
        try await server.shutdown()
    }

    @Test func chatCompletionMapsOntoRendererInput() async throws {
        let backend = K3CapturingBackend(completion: Self.plainCompletion())
        let (server, port) = try await startServer(backend: backend)
        let (_, status) = try await post(port, #"""
        {
          "model": "kimi-k3",
          "reasoning_effort": "high",
          "tool_choice": "required",
          "messages": [
            {"role": "system", "content": "You can call tools."},
            {"role": "user", "content": "Weather in Seoul?"},
            {"role": "assistant", "reasoning_content": "Need the weather call.",
             "content": "Checking.",
             "tool_calls": [{"id": "call_0", "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": "{\"city\": \"Seoul\"}"}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\": 25}"},
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": [{"type": "text", "text": "And "},
                                          {"type": "text", "text": "Busan?"}]}
          ],
          "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}]
        }
        """#)
        #expect(status == 200)

        let k3 = try #require(await backend.received?.k3)
        // Options: reasoning_effort → thinkingEffort; tool_choice passthrough.
        #expect(k3.options.thinking)
        #expect(k3.options.thinkingEffort == .high)
        #expect(k3.options.toolChoice == .required)
        #expect(k3.options.tools == [
            K3ToolDeclaration(
                name: "get_weather",
                description: "Get the current weather for a city.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object(["city": .object(["type": .string("string")])]),
                    "required": .array([.string("city")]),
                ])),
        ])

        // Message mapping, in order.
        #expect(k3.messages.count == 6)
        #expect(k3.messages[0] == K3ChatMessage(
            role: .system, content: .text("You can call tools.")))
        #expect(k3.messages[1] == K3ChatMessage(
            role: .user, content: .text("Weather in Seoul?")))
        #expect(k3.messages[2] == K3ChatMessage(
            role: .assistant,
            content: .text("Checking."),
            reasoningContent: "Need the weather call.",
            toolCalls: [K3ToolCall(id: "call_0", name: "get_weather",
                                   arguments: .jsonString("{\"city\": \"Seoul\"}"))]))
        #expect(k3.messages[3] == K3ChatMessage(
            role: .tool, content: .text("{\"temp\": 25}"), toolCallID: "call_0"))
        // developer folds into system.
        #expect(k3.messages[4] == K3ChatMessage(
            role: .system, content: .text("Be terse.")))
        // Text parts stay parts (they encode as separate segments).
        #expect(k3.messages[5] == K3ChatMessage(
            role: .user, content: .parts(["And ", "Busan?"])))

        // The Gemma-shaped fields stay empty on the K3 path.
        let received = try #require(await backend.received)
        #expect(received.messages.isEmpty)
        #expect(received.tools.isEmpty)
        #expect(received.generationConfig.temperature == 0)
        try await server.shutdown()
    }

    @Test func reasoningAndToolCallsSurfaceInResponse() async throws {
        let call = ParsedToolCall(
            id: "call_abc",
            name: "get_weather",
            arguments: .object(["city": .string("Seoul")]),
            argumentsJSON: #"{"city":"Seoul"}"#)
        let backend = K3CapturingBackend(completion: Self.plainCompletion(
            "", reasoning: "I should check the weather.", toolCalls: [call]))
        let (server, port) = try await startServer(backend: backend)
        let (data, status) = try await post(port, #"""
        {"model": "kimi-k3",
         "messages": [{"role": "user", "content": "Weather?"}]}
        """#)
        #expect(status == 200)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["reasoning_content"] as? String == "I should check the weather.")
        // Content is null when the turn is pure tool calls (house shape).
        #expect(message["content"] is NSNull)
        let toolCalls = try #require(message["tool_calls"] as? [[String: Any]])
        let function = try #require(toolCalls[0]["function"] as? [String: Any])
        #expect(function["name"] as? String == "get_weather")
        #expect(function["arguments"] as? String == #"{"city":"Seoul"}"#)
        #expect(choices[0]["finish_reason"] as? String == "tool_calls")
        try await server.shutdown()
    }

    @Test func completionReportsBackendCachedTokenCount() async throws {
        let backend = K3CapturingBackend(
            completion: Self.plainCompletion(cachedTokens: 7))
        let (server, port) = try await startServer(backend: backend)
        let (data, status) = try await post(port, #"""
        {"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]}
        """#)
        #expect(status == 200)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try #require(object["usage"] as? [String: Any])
        #expect(usage["prompt_tokens"] as? Int == 12)
        let details = try #require(usage["prompt_tokens_details"] as? [String: Any])
        #expect(details["cached_tokens"] as? Int == 7)
        try await server.shutdown()
    }

    @Test func streamingEmitsReasoningAndContentDeltas() async throws {
        let backend = K3CapturingBackend(
            events: [.reasoning("thinking "), .reasoning("hard"), .content("the answer")],
            completion: Self.plainCompletion("the answer", reasoning: "thinking hard"))
        let (server, port) = try await startServer(backend: backend)
        let (data, status) = try await post(port, #"""
        {"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}],
         "stream": true, "stream_options": {"include_usage": true}}
        """#)
        #expect(status == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""reasoning_content":"thinking ""#))
        #expect(text.contains(#""reasoning_content":"hard""#))
        #expect(text.contains(#""content":"the answer""#))
        #expect(text.contains(#""cached_tokens":0"#))
        #expect(text.hasSuffix("data: [DONE]\n\n"))
        try await server.shutdown()
    }

    @Test func stopSubstringsRejected() async throws {
        let backend = K3CapturingBackend(completion: Self.plainCompletion())
        let (server, port) = try await startServer(backend: backend)
        let (data, status) = try await post(port, #"""
        {"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}],
         "stop": ["\n"]}
        """#)
        #expect(status == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("unsupported_value"))
        try await server.shutdown()
    }

    @Test func invalidReasoningEffortRejected() async throws {
        let backend = K3CapturingBackend(completion: Self.plainCompletion())
        let (server, port) = try await startServer(backend: backend)
        let (_, status) = try await post(port, #"""
        {"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}],
         "reasoning_effort": "medium"}
        """#)
        #expect(status == 400)
        try await server.shutdown()
    }

    @Test func toolChoiceNoneDropsToolsAndSetsRendererChoice() async throws {
        let backend = K3CapturingBackend(completion: Self.plainCompletion())
        let (server, port) = try await startServer(backend: backend)
        let (_, status) = try await post(port, #"""
        {"model": "kimi-k3",
         "messages": [{"role": "user", "content": "hi"}],
         "tool_choice": "none",
         "tools": [{"type": "function", "function": {
            "name": "get_weather", "parameters": {"type": "object"}}}]}
        """#)
        #expect(status == 200)
        let k3 = try #require(await backend.received?.k3)
        #expect(k3.options.tools.isEmpty)
        #expect(k3.options.toolChoice == K3ToolChoice.none)
        try await server.shutdown()
    }

    @Test func namedToolChoiceRejected() async throws {
        let backend = K3CapturingBackend(completion: Self.plainCompletion())
        let (server, port) = try await startServer(backend: backend)
        let (_, status) = try await post(port, #"""
        {"model": "kimi-k3",
         "messages": [{"role": "user", "content": "hi"}],
         "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}
        """#)
        #expect(status == 400)
        try await server.shutdown()
    }

    @Test func emptyToolNameRejected() async throws {
        let backend = K3CapturingBackend(completion: Self.plainCompletion())
        let (server, port) = try await startServer(backend: backend)
        let (_, status) = try await post(port, #"""
        {"model": "kimi-k3",
         "messages": [{"role": "user", "content": "hi"}],
         "tools": [{"type": "function", "function": {
            "name": "", "parameters": {"type": "object"}}}]}
        """#)
        #expect(status == 400)
        try await server.shutdown()
    }
}
