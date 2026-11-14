import Foundation
import NIOCore
import Testing
import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

// Backends scripted for the two API surfaces. Each records the validated
// request it was handed so a test can assert what the model would have seen.

private final class RequestLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _requests: [ValidatedChatRequest] = []
    var requests: [ValidatedChatRequest] { lock.withLock { _requests } }
    func record(_ request: ValidatedChatRequest) { lock.withLock { _requests.append(request) } }
}

private actor TextBackend: ServerInferenceBackend, PromptTokenCounting {
    let log = RequestLog()
    let finishReason: String
    let stopSequence: String?

    init(finishReason: String = "stop", stopSequence: String? = nil) {
        self.finishReason = finishReason
        self.stopSequence = stopSequence
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        log.record(request)
        onEvent(.content("hel"))
        onEvent(.content("lo"))
        return ServerCompletion(
            content: "hello", toolCalls: [], finishReason: finishReason,
            usage: OpenAIUsage(
                promptTokens: 12, completionTokens: 2, totalTokens: 14, cachedTokens: 4),
            stopSequence: stopSequence)
    }

    func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        log.record(request)
        return 42
    }
}

private actor ToolBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let call = ParsedToolCall(
            id: "call_000000000000000000000009",
            name: "read",
            arguments: .object(["path": .string("/tmp/a")]),
            argumentsJSON: #"{"path":"/tmp/a"}"#)
        onEvent(.content("Reading."))
        onEvent(.toolCall(call))
        return ServerCompletion(
            content: "Reading.", toolCalls: [call], finishReason: "tool_calls",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 8, totalTokens: 11))
    }
}

private actor FailingBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("partial"))
        throw ServerRequestError.invalid(
            message: "synthetic failure", param: "messages", code: "synthetic")
    }
}

private struct SSEEvent {
    let name: String?
    let object: [String: Any]
}

private func sseEvents(_ text: String) throws -> [SSEEvent] {
    try text.components(separatedBy: "\n\n").compactMap { block in
        var name: String?
        var data: String?
        for line in block.split(separator: "\n") {
            if line.hasPrefix("event: ") { name = String(line.dropFirst(7)) }
            if line.hasPrefix("data: ") { data = String(line.dropFirst(6)) }
        }
        guard let data, data != "[DONE]" else { return nil }
        let object = try #require(
            JSONSerialization.jsonObject(with: Data(data.utf8)) as? [String: Any])
        return SSEEvent(name: name, object: object)
    }
}

private func post(
    _ port: Int, _ path: String, _ json: String,
    headers: [String: String] = [:]
) async throws -> (Data, HTTPURLResponse) {
    var request = URLRequest(url: try localURL(port: port, path))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "content-type")
    for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
    request.httpBody = Data(json.utf8)
    let (data, response) = try await URLSession.shared.data(for: request)
    return (data, try #require(response as? HTTPURLResponse))
}

private func call(
    _ port: Int, _ method: String, _ path: String,
    headers: [String: String] = [:]
) async throws -> (Data, HTTPURLResponse) {
    var request = URLRequest(url: try localURL(port: port, path))
    request.httpMethod = method
    for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
    let (data, response) = try await URLSession.shared.data(for: request)
    return (data, try #require(response as? HTTPURLResponse))
}

private func json(_ data: Data) throws -> [String: Any] {
    try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

private func withServer<T>(
    _ backend: any ServerInferenceBackend,
    _ body: (Int) async throws -> T
) async throws -> T {
    let server = TinyTitanHTTPServer(modelID: "test-model", queueLimit: 2, backend: backend)
    let channel = try await server.start(port: 0)
    let port = try #require(channel.localAddress?.port)
    do {
        let result = try await body(port)
        try await server.shutdown()
        return result
    } catch {
        try await server.shutdown()
        throw error
    }
}

@Suite("Responses API over HTTP", .serialized)
struct ResponsesAPIHTTPTests {
    @Test func stringInputNonStreamingResponseObject() async throws {
        let backend = TextBackend()
        try await withServer(backend) { port in
            let (data, response) = try await post(
                port, "/v1/responses",
                """
                {"model":"test-model","input":"hi","instructions":"Be brief.","metadata":{"k":"v"},
                 "user":"u1","max_output_tokens":50,"truncation":"auto"}
                """)
            #expect(response.statusCode == 200)
            let object = try json(data)
            #expect(object["object"] as? String == "response")
            #expect(object["status"] as? String == "completed")
            #expect(object["instructions"] as? String == "Be brief.")
            #expect(object["max_output_tokens"] as? Int == 50)
            #expect(object["truncation"] as? String == "auto")
            #expect(object["user"] as? String == "u1")
            #expect((object["metadata"] as? [String: Any])?["k"] as? String == "v")
            #expect(object["completed_at"] is Int)
            #expect(object["incomplete_details"] is NSNull)
            let output = try #require(object["output"] as? [[String: Any]])
            #expect(output.count == 1)
            #expect(output[0]["type"] as? String == "message")
            #expect((output[0]["id"] as? String)?.hasPrefix("msg_") == true)
            let content = try #require(output[0]["content"] as? [[String: Any]])
            #expect(content[0]["type"] as? String == "output_text")
            #expect(content[0]["text"] as? String == "hello")
            #expect(content[0]["logprobs"] is [Any])
            let usage = try #require(object["usage"] as? [String: Any])
            #expect(usage["input_tokens"] as? Int == 12)
            #expect(
                (usage["input_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 4)
            #expect(
                (usage["output_tokens_details"] as? [String: Any])?["reasoning_tokens"] as? Int == 0
            )
            #expect(usage["total_tokens"] as? Int == 14)
            // The system message carries the instructions; the user turn the string input.
            let seen = backend.log.requests[0].messages
            #expect(seen.map(\.role) == [.system, .user])
            #expect(seen[1].content == "hi")
        }
    }

    @Test func streamingEventsAreNumberedAndEndWithoutDone() async throws {
        try await withServer(TextBackend()) { port in
            let (data, response) = try await post(
                port, "/v1/responses",
                """
                {"model":"test-model","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"stream":true}
                """)
            #expect(response.statusCode == 200)
            let text = data.lossyUTF8String
            #expect(!text.contains("[DONE]"))
            #expect(!text.contains("response.content_part.delta"))
            let events = try sseEvents(text)
            let types = events.compactMap { $0.object["type"] as? String }
            #expect(
                types == [
                    "response.created", "response.in_progress",
                    "response.output_item.added", "response.content_part.added",
                    "response.output_text.delta", "response.output_text.delta",
                    "response.output_text.done", "response.content_part.done",
                    "response.output_item.done", "response.completed",
                ])
            #expect(events.map { $0.name } == types)
            let sequence = events.compactMap { $0.object["sequence_number"] as? Int }
            #expect(sequence == Array(0..<events.count))
            let delta = events[4].object
            #expect(delta["delta"] as? String == "hel")
            #expect(
                delta["item_id"] as? String == (events[2].object["item"] as? [String: Any])?["id"]
                    as? String)
            #expect(delta["output_index"] as? Int == 0)
            #expect(delta["content_index"] as? Int == 0)
            #expect(delta["logprobs"] is [Any])
            let done = events[6].object
            #expect(done["text"] as? String == "hello")
            let completed = try #require(events.last?.object["response"] as? [String: Any])
            #expect(completed["status"] as? String == "completed")
            #expect((completed["usage"] as? [String: Any])?["output_tokens"] as? Int == 2)
        }
    }

    @Test func toolCallsStreamTheirWholeLifecycle() async throws {
        try await withServer(ToolBackend()) { port in
            let (data, _) = try await post(
                port, "/v1/responses",
                """
                {"model":"test-model","input":"read a","stream":true,
                 "tools":[{"type":"function","name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}},"strict":true}]}
                """)
            let events = try sseEvents(data.lossyUTF8String)
            let types = events.compactMap { $0.object["type"] as? String }
            #expect(types.contains("response.function_call_arguments.delta"))
            let doneIndex = try #require(
                types.firstIndex(of: "response.function_call_arguments.done"))
            #expect(events[doneIndex].object["arguments"] as? String == #"{"path":"/tmp/a"}"#)
            #expect(events[doneIndex].object["name"] as? String == "read")
            #expect(types[doneIndex + 1] == "response.output_item.done")
            let item = try #require(events[doneIndex + 1].object["item"] as? [String: Any])
            #expect(item["type"] as? String == "function_call")
            #expect(item["call_id"] as? String == "call_000000000000000000000009")
            #expect((item["id"] as? String)?.hasPrefix("fc_") == true)
            #expect(item["output_index"] == nil)
            // The message came first, so the call is output index 1.
            #expect(events[doneIndex].object["output_index"] as? Int == 1)
            let final = try #require(events.last?.object["response"] as? [String: Any])
            let output = try #require(final["output"] as? [[String: Any]])
            #expect(output.map { $0["type"] as? String } == ["message", "function_call"])
            let tools = try #require(final["tools"] as? [[String: Any]])
            #expect(tools[0]["name"] as? String == "read")
            #expect(tools[0]["strict"] as? Bool == true)
        }
    }

    @Test func incompleteWhenTheOutputCapEndsGeneration() async throws {
        try await withServer(TextBackend(finishReason: "length")) { port in
            let (data, _) = try await post(
                port, "/v1/responses",
                #"{"model":"test-model","input":"hi"}"#)
            let object = try json(data)
            #expect(object["status"] as? String == "incomplete")
            #expect(
                (object["incomplete_details"] as? [String: Any])?["reason"] as? String
                    == "max_output_tokens")
            let (stream, _) = try await post(
                port, "/v1/responses",
                #"{"model":"test-model","input":"hi","stream":true}"#)
            let events = try sseEvents(stream.lossyUTF8String)
            #expect(events.last?.object["type"] as? String == "response.incomplete")
        }
    }

    @Test func storedResponsesChainAndCanBeRetrieved() async throws {
        let backend = TextBackend()
        try await withServer(backend) { port in
            let (first, _) = try await post(
                port, "/v1/responses",
                #"{"model":"test-model","input":"first"}"#)
            let firstID = try #require(try json(first)["id"] as? String)

            let (second, _) = try await post(
                port, "/v1/responses",
                """
                {"model":"test-model","input":"second","previous_response_id":"\(firstID)"}
                """)
            let secondObject = try json(second)
            #expect(secondObject["previous_response_id"] as? String == firstID)
            let seen = backend.log.requests[1].messages
            #expect(seen.map(\.role) == [.user, .assistant, .user])
            #expect(seen[1].content == "hello")
            #expect(seen[2].content == "second")

            let (fetched, status) = try await call(port, "GET", "/v1/responses/\(firstID)")
            #expect(status.statusCode == 200)
            #expect(try json(fetched)["id"] as? String == firstID)

            let (items, _) = try await call(port, "GET", "/v1/responses/\(firstID)/input_items")
            let list = try json(items)
            #expect(list["object"] as? String == "list")
            #expect((list["data"] as? [[String: Any]])?.count == 1)
            #expect(list["has_more"] as? Bool == false)

            let (cancel, cancelStatus) = try await call(
                port, "POST", "/v1/responses/\(firstID)/cancel")
            #expect(cancelStatus.statusCode == 400)
            #expect(cancel.lossyUTF8String.contains("background"))

            let (deleted, _) = try await call(port, "DELETE", "/v1/responses/\(firstID)")
            #expect(try json(deleted)["deleted"] as? Bool == true)
            let (_, gone) = try await call(port, "GET", "/v1/responses/\(firstID)")
            #expect(gone.statusCode == 404)

            let (_, missing) = try await post(
                port, "/v1/responses",
                """
                {"model":"test-model","input":"x","previous_response_id":"resp_nope"}
                """)
            #expect(missing.statusCode == 404)

            let (unstored, _) = try await post(
                port, "/v1/responses",
                #"{"model":"test-model","input":"x","store":false}"#)
            let unstoredID = try #require(try json(unstored)["id"] as? String)
            let (_, unstoredStatus) = try await call(port, "GET", "/v1/responses/\(unstoredID)")
            #expect(unstoredStatus.statusCode == 404)
        }
    }

    @Test func replayedReasoningAndOutputTextItemsAreAccepted() async throws {
        let backend = TextBackend()
        try await withServer(backend) { port in
            let (_, response) = try await post(
                port, "/v1/responses",
                """
                {"model":"test-model","input":[
                   {"type":"message","role":"user","content":"a"},
                   {"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque"},
                   {"type":"message","role":"assistant","id":"msg_1","status":"completed",
                    "content":[{"type":"output_text","text":"b","annotations":[]}]},
                   {"type":"message","role":"user","content":"c"}
                 ]}
                """)
            #expect(response.statusCode == 200)
            let seen = backend.log.requests[0].messages
            #expect(seen.map(\.role) == [.user, .assistant, .user])
            #expect(seen[1].content == "b")
        }
    }

    /// `text.format` reaches the validator: it used to be refused as an
    /// unsupported feature, and now it is a schema the sampler is masked with.
    @Test func aStructuredOutputFormatReachesTheValidator() async throws {
        let backend = TextBackend()
        try await withServer(backend) { port in
            let (_, response) = try await post(
                port, "/v1/responses",
                """
                {"model":"test-model","input":"x","text":{"format":{"type":"json_object"}}}
                """)
            #expect(response.statusCode == 200)
            #expect(
                backend.log.requests[0].jsonSchema
                    == .object(properties: [:], required: [], additional: true))
        }
    }

    @Test func unsupportedFeaturesAreRefusedByName() async throws {
        try await withServer(TextBackend()) { port in
            for body in [
                #"{"model":"test-model","input":"x","background":true}"#,
                #"{"model":"test-model","input":"x","prompt":{"id":"p"}}"#,
                #"{"model":"test-model","input":"x","text":{"format":{"type":"xml"}}}"#,
                #"{"model":"test-model","input":[{"role":"user","content":[{"type":"input_image","image_url":"http://x"}]}]}"#,
            ] {
                let (data, response) = try await post(port, "/v1/responses", body)
                #expect(response.statusCode == 400, Comment(rawValue: body))
                let error = try #require(try json(data)["error"] as? [String: Any])
                #expect(
                    error["type"] as? String == "invalid_request_error", Comment(rawValue: body))
            }
        }
    }

    @Test func failuresMidStreamEndWithResponseFailed() async throws {
        try await withServer(FailingBackend()) { port in
            let (data, _) = try await post(
                port, "/v1/responses",
                #"{"model":"test-model","input":"x","stream":true}"#)
            let text = data.lossyUTF8String
            #expect(!text.contains("[DONE]"))
            let events = try sseEvents(text)
            let last = try #require(events.last?.object)
            #expect(last["type"] as? String == "response.failed")
            let failed = try #require(last["response"] as? [String: Any])
            #expect(failed["status"] as? String == "failed")
            #expect((failed["error"] as? [String: Any])?["code"] as? String == "synthetic")
        }
    }
}

@Suite("Anthropic Messages API over HTTP", .serialized)
struct AnthropicMessagesHTTPTests {
    private let version = ["anthropic-version": "2023-06-01", "x-api-key": "unused"]

    @Test func nonStreamingMessageObject() async throws {
        let backend = TextBackend(stopSequence: "END")
        try await withServer(backend) { port in
            let (data, response) = try await post(
                port, "/v1/messages",
                """
                {"model":"test-model","max_tokens":32,"system":"Be terse.","stop_sequences":["END"],
                 "messages":[{"role":"user","content":"hi"}]}
                """, headers: version)
            #expect(response.statusCode == 200)
            #expect(response.value(forHTTPHeaderField: "request-id")?.hasPrefix("req_") == true)
            let object = try json(data)
            #expect(object["type"] as? String == "message")
            #expect(object["role"] as? String == "assistant")
            #expect((object["id"] as? String)?.hasPrefix("msg_") == true)
            #expect(object["model"] as? String == "test-model")
            #expect(object["stop_reason"] as? String == "stop_sequence")
            #expect(object["stop_sequence"] as? String == "END")
            let content = try #require(object["content"] as? [[String: Any]])
            #expect(content.count == 1)
            #expect(content[0]["type"] as? String == "text")
            #expect(content[0]["text"] as? String == "hello")
            let usage = try #require(object["usage"] as? [String: Any])
            #expect(usage["input_tokens"] as? Int == 8)
            #expect(usage["cache_read_input_tokens"] as? Int == 4)
            #expect(usage["output_tokens"] as? Int == 2)
            let seen = backend.log.requests[0]
            #expect(seen.messages.map(\.role) == [.system, .user])
            #expect(seen.generationConfig.stopStrings == ["END"])
            #expect(seen.maximumCompletionTokens == 32)
        }
    }

    @Test func streamingEventOrder() async throws {
        try await withServer(TextBackend()) { port in
            let (data, response) = try await post(
                port, "/v1/messages",
                """
                {"model":"test-model","max_tokens":32,"stream":true,
                 "messages":[{"role":"user","content":"hi"}]}
                """, headers: version)
            #expect(response.statusCode == 200)
            #expect(
                response.value(forHTTPHeaderField: "content-type")?.hasPrefix("text/event-stream")
                    == true)
            let text = data.lossyUTF8String
            #expect(!text.contains("[DONE]"))
            let events = try sseEvents(text)
            let types = events.compactMap { $0.object["type"] as? String }
            #expect(
                types == [
                    "message_start", "content_block_start", "content_block_delta",
                    "content_block_delta", "content_block_stop", "message_delta", "message_stop",
                ])
            #expect(events.map { $0.name } == types)
            let start = try #require(events[0].object["message"] as? [String: Any])
            #expect(start["type"] as? String == "message")
            #expect((start["content"] as? [Any])?.isEmpty == true)
            #expect(start["stop_reason"] is NSNull)
            let block = try #require(events[1].object["content_block"] as? [String: Any])
            #expect(block["type"] as? String == "text")
            #expect(events[1].object["index"] as? Int == 0)
            let delta = try #require(events[2].object["delta"] as? [String: Any])
            #expect(delta["type"] as? String == "text_delta")
            #expect(delta["text"] as? String == "hel")
            let messageDelta = events[5].object
            #expect(
                (messageDelta["delta"] as? [String: Any])?["stop_reason"] as? String == "end_turn")
            #expect((messageDelta["usage"] as? [String: Any])?["output_tokens"] as? Int == 2)
        }
    }

    @Test func toolUseStreamsAsItsOwnBlock() async throws {
        try await withServer(ToolBackend()) { port in
            let (data, _) = try await post(
                port, "/v1/messages",
                """
                {"model":"test-model","max_tokens":32,"stream":true,
                 "tools":[{"name":"read","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],
                 "messages":[{"role":"user","content":"read a"}]}
                """, headers: version)
            let events = try sseEvents(data.lossyUTF8String)
            let types = events.compactMap { $0.object["type"] as? String }
            #expect(
                types == [
                    "message_start", "content_block_start", "content_block_delta",
                    "content_block_stop", "content_block_start", "content_block_delta",
                    "content_block_stop", "message_delta", "message_stop",
                ])
            let toolStart = try #require(events[4].object["content_block"] as? [String: Any])
            #expect(toolStart["type"] as? String == "tool_use")
            #expect(toolStart["name"] as? String == "read")
            #expect(toolStart["id"] as? String == "call_000000000000000000000009")
            #expect(events[4].object["index"] as? Int == 1)
            let jsonDelta = try #require(events[5].object["delta"] as? [String: Any])
            #expect(jsonDelta["type"] as? String == "input_json_delta")
            #expect(jsonDelta["partial_json"] as? String == #"{"path":"/tmp/a"}"#)
            #expect(
                (events[7].object["delta"] as? [String: Any])?["stop_reason"] as? String
                    == "tool_use")

            let (plain, _) = try await post(
                port, "/v1/messages",
                """
                {"model":"test-model","max_tokens":32,
                 "tools":[{"name":"read","input_schema":{"type":"object"}}],
                 "messages":[{"role":"user","content":"read a"}]}
                """, headers: version)
            let object = try json(plain)
            let content = try #require(object["content"] as? [[String: Any]])
            #expect(content.map { $0["type"] as? String } == ["text", "tool_use"])
            #expect((content[1]["input"] as? [String: Any])?["path"] as? String == "/tmp/a")
            #expect(object["stop_reason"] as? String == "tool_use")
        }
    }

    @Test func errorsUseTheAnthropicEnvelope() async throws {
        try await withServer(TextBackend()) { port in
            let (missing, missingStatus) = try await post(
                port, "/v1/messages",
                """
                {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
                """, headers: version)
            #expect(missingStatus.statusCode == 400)
            let envelope = try json(missing)
            #expect(envelope["type"] as? String == "error")
            let error = try #require(envelope["error"] as? [String: Any])
            #expect(error["type"] as? String == "invalid_request_error")
            #expect((error["message"] as? String)?.hasPrefix("max_tokens:") == true)
            #expect((envelope["request_id"] as? String)?.hasPrefix("req_") == true)

            let (wrong, wrongStatus) = try await post(
                port, "/v1/messages",
                """
                {"model":"claude-opus-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}
                """, headers: version)
            #expect(wrongStatus.statusCode == 404)
            #expect(
                (try json(wrong)["error"] as? [String: Any])?["type"] as? String
                    == "not_found_error")

            let (malformed, malformedStatus) = try await post(
                port, "/v1/messages", "{", headers: version)
            #expect(malformedStatus.statusCode == 400)
            #expect(try json(malformed)["type"] as? String == "error")

            let (route, routeStatus) = try await call(port, "GET", "/v1/nothing", headers: version)
            #expect(routeStatus.statusCode == 404)
            #expect(
                (try json(route)["error"] as? [String: Any])?["type"] as? String
                    == "not_found_error")
        }
    }

    @Test func failuresMidStreamSendAnErrorEvent() async throws {
        try await withServer(FailingBackend()) { port in
            let (data, _) = try await post(
                port, "/v1/messages",
                """
                {"model":"test-model","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]}
                """, headers: version)
            let events = try sseEvents(data.lossyUTF8String)
            let last = try #require(events.last)
            #expect(last.name == "error")
            #expect(last.object["type"] as? String == "error")
            #expect(
                (last.object["error"] as? [String: Any])?["type"] as? String
                    == "invalid_request_error")
        }
    }

    @Test func countTokensAndModels() async throws {
        let backend = TextBackend()
        try await withServer(backend) { port in
            let (count, status) = try await post(
                port, "/v1/messages/count_tokens",
                """
                {"model":"test-model","system":"s","messages":[{"role":"user","content":"hi"}]}
                """, headers: version)
            #expect(status.statusCode == 200)
            #expect(try json(count)["input_tokens"] as? Int == 42)
            #expect(backend.log.requests[0].messages.map(\.role) == [.system, .user])

            let (models, _) = try await call(port, "GET", "/v1/models", headers: version)
            let list = try json(models)
            #expect(list["has_more"] as? Bool == false)
            let data = try #require(list["data"] as? [[String: Any]])
            #expect(data[0]["type"] as? String == "model")
            #expect(data[0]["id"] as? String == "test-model")
            #expect(list["first_id"] as? String == "test-model")

            let (one, oneStatus) = try await call(
                port, "GET", "/v1/models/test-model", headers: version)
            #expect(oneStatus.statusCode == 200)
            #expect(try json(one)["display_name"] as? String == "test-model")

            // Without the header the same path keeps its OpenAI shape.
            let (openAI, _) = try await call(port, "GET", "/v1/models")
            #expect(try json(openAI)["object"] as? String == "list")
        }
    }

    @Test func countTokensWithoutACountingBackendIs501() async throws {
        try await withServer(ToolBackend()) { port in
            let (data, status) = try await post(
                port, "/v1/messages/count_tokens",
                """
                {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
                """, headers: version)
            #expect(status.statusCode == 501)
            #expect((try json(data)["error"] as? [String: Any])?["type"] as? String == "api_error")
        }
    }
}
