import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import Testing
@testable import TurboFieldfareServerCore

/// End-to-end tests for the Anthropic Messages endpoint, backed by the fake
/// generator. No Metal, no model files.
@Suite("Anthropic messages route")
struct AnthropicRoutesTests {
    private func makeApp(generator: FakeTokenGenerator = FakeTokenGenerator(),
                         maxContext: Int = 64) -> some ApplicationProtocol {
        let session = GenerationSession(modelID: "test-model",
                                        maxContext: maxContext,
                                        created: 1_700_000_000,
                                        generator: generator)
        return Application(router: buildServerRouter(session: session))
    }

    private func jsonBody(_ json: String) -> ByteBuffer {
        ByteBuffer(string: json)
    }

    private func jsonObject(_ buffer: ByteBuffer) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: buffer) as? [String: Any])
    }

    @Test func nonStreamingHappyPath() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"""
                {"model": "claude-x", "max_tokens": 10,
                 "messages": [{"role": "user", "content": "hi"}]}
                """#)
            ) { response in
                #expect(response.status == .ok)
                #expect(response.head.headerFields[.contentType] == "application/json")
                let object = try jsonObject(response.body)
                #expect(object["type"] as? String == "message")
                #expect(object["role"] as? String == "assistant")
                #expect(object["model"] as? String == "test-model")
                #expect((object["id"] as? String)?.hasPrefix("msg_") == true)
                let content = try #require(object["content"] as? [[String: Any]])
                #expect(content.count == 1)
                #expect(content[0]["type"] as? String == "text")
                #expect(content[0]["text"] as? String == "Hello, world")
                #expect(object["stop_reason"] as? String == "end_turn")
                #expect(object["stop_sequence"] is NSNull)
                let usage = try #require(object["usage"] as? [String: Any])
                #expect(usage["input_tokens"] as? Int == 8)
                #expect(usage["output_tokens"] as? Int == 2)
            }
        }
    }

    @Test func finishReasonLengthMapsToMaxTokens() async throws {
        let generator = FakeTokenGenerator()
        generator.finishReason = .length
        try await makeApp(generator: generator).test(.router) { client in
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"""
                {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
                """#)
            ) { response in
                #expect(response.status == .ok)
                let object = try jsonObject(response.body)
                #expect(object["stop_reason"] as? String == "max_tokens")
            }
        }
    }

    @Test func systemStringAndBlocksBecomeLeadingSystemMessage() async throws {
        let generator = FakeTokenGenerator()
        try await makeApp(generator: generator).test(.router) { client in
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"""
                {"max_tokens": 10, "system": "be terse",
                 "messages": [{"role": "user", "content": "hi"}]}
                """#)
            ) { response in
                #expect(response.status == .ok)
            }
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"""
                {"max_tokens": 10,
                 "system": [{"type": "text", "text": "be "}, {"type": "text", "text": "terse"}],
                 "messages": [{"role": "user", "content": "hi"}]}
                """#)
            ) { response in
                #expect(response.status == .ok)
            }
        }
        let calls = await generator.calls
        #expect(calls.count == 2)
        for call in calls {
            guard case .chat(let messages) = call.input else {
                Issue.record("expected chat input")
                continue
            }
            #expect(messages.first?.role == "system")
            #expect(messages.first?.content == "be terse")
            #expect(messages.count == 2)
            #expect(messages[1].role == "user")
        }
    }

    @Test func contentBlocksConcatenated() async throws {
        let generator = FakeTokenGenerator()
        try await makeApp(generator: generator).test(.router) { client in
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"""
                {"max_tokens": 10,
                 "messages": [{"role": "user",
                               "content": [{"type": "text", "text": "Hello"},
                                           {"type": "text", "text": " world"}]}]}
                """#)
            ) { response in
                #expect(response.status == .ok)
            }
        }
        let calls = await generator.calls
        guard case .chat(let messages) = calls.first?.input else {
            Issue.record("expected chat input")
            return
        }
        #expect(messages.first?.role == "user")
        #expect(messages.first?.content == "Hello world")
    }

    @Test func missingMaxTokensRejected() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"{"messages": [{"role": "user", "content": "hi"}]}"#)
            ) { response in
                #expect(response.status == .badRequest)
                let object = try jsonObject(response.body)
                #expect(object["type"] as? String == "error")
                let error = try #require(object["error"] as? [String: Any])
                #expect(error["type"] as? String == "invalid_request_error")
                #expect((error["message"] as? String)?.contains("max_tokens") == true)
            }
        }
    }

    @Test func zeroMaxTokensRejected() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"""
                {"max_tokens": 0, "messages": [{"role": "user", "content": "hi"}]}
                """#)
            ) { response in
                #expect(response.status == .badRequest)
                let object = try jsonObject(response.body)
                let error = try #require(object["error"] as? [String: Any])
                #expect(error["type"] as? String == "invalid_request_error")
                #expect((error["message"] as? String)?.contains("max_tokens") == true)
            }
        }
    }

    @Test func toolRoleRejected() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"""
                {"max_tokens": 10, "messages": [{"role": "tool", "content": "x"}]}
                """#)
            ) { response in
                #expect(response.status == .badRequest)
                let object = try jsonObject(response.body)
                let error = try #require(object["error"] as? [String: Any])
                #expect(error["type"] as? String == "invalid_request_error")
                #expect((error["message"] as? String)?.contains("tool") == true)
            }
        }
    }

    @Test func unsupportedParametersRejected() async throws {
        try await makeApp().test(.router) { client in
            for field in [#""tools": []"#, #""tool_choice": {"type": "auto"}"#,
                          #""thinking": {"type": "enabled"}"#, #""metadata": {}"#] {
                try await client.execute(
                    uri: "/v1/messages",
                    method: .post,
                    headers: [.contentType: "application/json"],
                    body: jsonBody(#"{"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}], \#(field)}"#)
                ) { response in
                    #expect(response.status == .badRequest)
                    let object = try jsonObject(response.body)
                    let error = try #require(object["error"] as? [String: Any])
                    #expect(error["type"] as? String == "invalid_request_error")
                }
            }
        }
    }

    @Test func streamingEventOrder() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"""
                {"max_tokens": 10, "stream": true,
                 "messages": [{"role": "user", "content": "hi"}]}
                """#)
            ) { response in
                #expect(response.status == .ok)
                #expect(response.head.headerFields[.contentType] == "text/event-stream")
                let body = String(buffer: response.body)
                #expect(!body.contains("[DONE]"))

                let frames = body.components(separatedBy: "\n\n").filter { !$0.isEmpty }
                // message_start, content_block_start, 2 deltas,
                // content_block_stop, message_delta, message_stop
                #expect(frames.count == 7)

                let events = frames.map { frame -> String in
                    #expect(frame.hasPrefix("event: "))
                    return String(frame.dropFirst("event: ".count).prefix { $0 != "\n" })
                }
                #expect(events == ["message_start", "content_block_start",
                                 "content_block_delta", "content_block_delta",
                                 "content_block_stop", "message_delta", "message_stop"])

                func frameData(_ index: Int) throws -> [String: Any] {
                    let frame = frames[index]
                    let dataStart = try #require(frame.range(of: "\ndata: "))
                    return try jsonObject(ByteBuffer(string: String(frame[dataStart.upperBound...])))
                }

                let messageStart = try frameData(0)
                #expect(messageStart["type"] as? String == "message_start")
                let message = try #require(messageStart["message"] as? [String: Any])
                #expect((message["id"] as? String)?.hasPrefix("msg_") == true)
                #expect(message["type"] as? String == "message")
                #expect(message["role"] as? String == "assistant")
                #expect(message["model"] as? String == "test-model")
                #expect((message["content"] as? [[String: Any]])?.isEmpty == true)
                #expect(message["stop_reason"] is NSNull)
                #expect(message["stop_sequence"] is NSNull)
                let startUsage = try #require(message["usage"] as? [String: Any])
                #expect(startUsage["input_tokens"] as? Int == 8)
                #expect(startUsage["output_tokens"] as? Int == 1)

                let blockStart = try frameData(1)
                #expect(blockStart["type"] as? String == "content_block_start")
                #expect(blockStart["index"] as? Int == 0)
                let block = try #require(blockStart["content_block"] as? [String: Any])
                #expect(block["type"] as? String == "text")
                #expect(block["text"] as? String == "")

                let firstDelta = try frameData(2)
                #expect(firstDelta["type"] as? String == "content_block_delta")
                let delta = try #require(firstDelta["delta"] as? [String: Any])
                #expect(delta["type"] as? String == "text_delta")
                #expect(delta["text"] as? String == "Hello")

                let blockStop = try frameData(4)
                #expect(blockStop["type"] as? String == "content_block_stop")
                #expect(blockStop["index"] as? Int == 0)

                let messageDelta = try frameData(5)
                #expect(messageDelta["type"] as? String == "message_delta")
                let stopDelta = try #require(messageDelta["delta"] as? [String: Any])
                #expect(stopDelta["stop_reason"] as? String == "end_turn")
                #expect(stopDelta["stop_sequence"] is NSNull)
                let deltaUsage = try #require(messageDelta["usage"] as? [String: Any])
                #expect(deltaUsage["output_tokens"] as? Int == 2)

                let messageStop = try frameData(6)
                #expect(messageStop["type"] as? String == "message_stop")
            }
        }
    }

    @Test func generationFailureReturnsSanitizedAPIError() async throws {
        struct Boom: Error {}
        let generator = FakeTokenGenerator()
        generator.thrownError = Boom()
        try await makeApp(generator: generator).test(.router) { client in
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"""
                {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
                """#)
            ) { response in
                #expect(response.status == .internalServerError)
                let object = try jsonObject(response.body)
                #expect(object["type"] as? String == "error")
                let error = try #require(object["error"] as? [String: Any])
                #expect(error["type"] as? String == "api_error")
                #expect(error["message"] as? String == "generation failed")
            }
        }
    }

    @Test func malformedJSONReturnsAnthropicError() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/messages",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody("{not json")
            ) { response in
                #expect(response.status == .badRequest)
                let object = try jsonObject(response.body)
                #expect(object["type"] as? String == "error")
                let error = try #require(object["error"] as? [String: Any])
                #expect(error["type"] as? String == "invalid_request_error")
                #expect(error["message"] != nil)
            }
        }
    }
}
