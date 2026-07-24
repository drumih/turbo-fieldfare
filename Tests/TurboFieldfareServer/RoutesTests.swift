import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import Testing
@testable import TurboFieldfareServerCore

/// End-to-end route tests against the router, backed by the fake generator.
/// No Metal, no model files.
@Suite("HTTP routes")
struct RoutesTests {
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

    @Test func health() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(uri: "/health", method: .get) { response in
                #expect(response.status == .ok)
                let object = try jsonObject(response.body)
                #expect(object["status"] as? String == "ok")
            }
        }
    }

    @Test func models() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(uri: "/v1/models", method: .get) { response in
                #expect(response.status == .ok)
                let object = try jsonObject(response.body)
                #expect(object["object"] as? String == "list")
                let data = try #require(object["data"] as? [[String: Any]])
                #expect(data.count == 1)
                #expect(data[0]["id"] as? String == "test-model")
                #expect(data[0]["object"] as? String == "model")
                #expect(data[0]["created"] as? Int == 1_700_000_000)
                #expect(data[0]["owned_by"] as? String == "local")
            }
        }
    }

    @Test func chatCompletionNonStreaming() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"{"messages": [{"role": "user", "content": "hi"}]}"#)
            ) { response in
                #expect(response.status == .ok)
                let object = try jsonObject(response.body)
                #expect(object["object"] as? String == "chat.completion")
                #expect(object["model"] as? String == "test-model")
                #expect((object["id"] as? String)?.hasPrefix("chatcmpl-") == true)
                let choices = try #require(object["choices"] as? [[String: Any]])
                #expect(choices.count == 1)
                let message = try #require(choices[0]["message"] as? [String: Any])
                #expect(message["role"] as? String == "assistant")
                #expect(message["content"] as? String == "Hello, world")
                #expect(choices[0]["finish_reason"] as? String == "stop")
                let usage = try #require(object["usage"] as? [String: Any])
                #expect(usage["prompt_tokens"] as? Int == 8)
                #expect(usage["completion_tokens"] as? Int == 2)
                #expect(usage["total_tokens"] as? Int == 10)
            }
        }
    }

    @Test func chatCompletionRejectsNGreaterThanOne() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"{"messages": [{"role": "user", "content": "hi"}], "n": 2}"#)
            ) { response in
                #expect(response.status == .badRequest)
                let object = try jsonObject(response.body)
                let error = try #require(object["error"] as? [String: Any])
                #expect(error["type"] as? String == "invalid_request_error")
                #expect(error["message"] != nil)
            }
        }
    }

    @Test func chatCompletionRejectsNZero() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"{"messages": [{"role": "user", "content": "hi"}], "n": 0}"#)
            ) { response in
                #expect(response.status == .badRequest)
            }
        }
    }

    @Test func chatCompletionRejectsUnsupportedSemanticFields() async throws {
        let bodies = [
            #"{"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}"#,
            #"{"messages": [{"role": "user", "content": "hi"}], "tool_choice": "auto"}"#,
            #"{"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}}"#,
        ]
        for body in bodies {
            try await makeApp().test(.router) { client in
                try await client.execute(
                    uri: "/v1/chat/completions",
                    method: .post,
                    headers: [.contentType: "application/json"],
                    body: jsonBody(body)
                ) { response in
                    #expect(response.status == .badRequest)
                    let object = try jsonObject(response.body)
                    let error = try #require(object["error"] as? [String: Any])
                    #expect(error["type"] as? String == "invalid_request_error")
                }
            }
        }
    }

    @Test func chatCompletionRejectsHugeMaxTokens() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                // Int.max must yield a 400, not an integer-overflow trap.
                body: jsonBody(#"{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 9223372036854775807}"#)
            ) { response in
                #expect(response.status == .badRequest)
                let object = try jsonObject(response.body)
                let error = try #require(object["error"] as? [String: Any])
                #expect(error["type"] as? String == "invalid_request_error")
            }
        }
    }

    @Test func chatCompletionRejectsToolRole() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"{"messages": [{"role": "tool", "content": "x"}]}"#)
            ) { response in
                #expect(response.status == .badRequest)
                let object = try jsonObject(response.body)
                let error = try #require(object["error"] as? [String: Any])
                #expect(error["type"] as? String == "invalid_request_error")
                #expect((error["message"] as? String)?.contains("tool") == true)
            }
        }
    }

    @Test func chatCompletionRejectsContextOverflow() async throws {
        try await makeApp(maxContext: 10).test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}"#)
            ) { response in
                #expect(response.status == .badRequest)
                let object = try jsonObject(response.body)
                let error = try #require(object["error"] as? [String: Any])
                #expect((error["message"] as? String)?.contains("context overflow") == true)
            }
        }
    }

    @Test func chatCompletionStreaming() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"""
                {
                  "messages": [{"role": "user", "content": "hi"}],
                  "stream": true,
                  "stream_options": {"include_usage": true}
                }
                """#)
            ) { response in
                #expect(response.status == .ok)
                #expect(response.head.headerFields[.contentType] == "text/event-stream")
                let body = String(buffer: response.body)
                #expect(body.hasSuffix("data: [DONE]\n\n"))

                let frames = body.components(separatedBy: "\n\n").filter { !$0.isEmpty }
                // role chunk + 2 content chunks + final chunk + usage chunk + [DONE]
                #expect(frames.count == 6)
                for frame in frames.dropLast() {
                    #expect(frame.hasPrefix("data: "))
                }
                #expect(frames.last == "data: [DONE]")

                let roleChunk = try jsonObject(
                    ByteBuffer(string: String(frames[0].dropFirst("data: ".count))))
                #expect(roleChunk["object"] as? String == "chat.completion.chunk")
                let roleChoices = try #require(roleChunk["choices"] as? [[String: Any]])
                #expect((roleChoices[0]["delta"] as? [String: Any])?["role"] as? String == "assistant")

                let firstDelta = try jsonObject(
                    ByteBuffer(string: String(frames[1].dropFirst("data: ".count))))
                let deltaChoices = try #require(firstDelta["choices"] as? [[String: Any]])
                #expect((deltaChoices[0]["delta"] as? [String: Any])?["content"] as? String == "Hello")

                let finalChunk = try jsonObject(
                    ByteBuffer(string: String(frames[3].dropFirst("data: ".count))))
                let finalChoices = try #require(finalChunk["choices"] as? [[String: Any]])
                #expect(finalChoices[0]["finish_reason"] as? String == "stop")

                let usageChunk = try jsonObject(
                    ByteBuffer(string: String(frames[4].dropFirst("data: ".count))))
                let usage = try #require(usageChunk["usage"] as? [String: Any])
                #expect(usage["total_tokens"] as? Int == 10)
                #expect((usageChunk["choices"] as? [[String: Any]])?.isEmpty == true)
            }
        }
    }

    @Test func completionNonStreaming() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"{"prompt": "once upon a time", "max_tokens": 10}"#)
            ) { response in
                #expect(response.status == .ok)
                let object = try jsonObject(response.body)
                #expect(object["object"] as? String == "text_completion")
                #expect((object["id"] as? String)?.hasPrefix("cmpl-") == true)
                let choices = try #require(object["choices"] as? [[String: Any]])
                #expect(choices[0]["text"] as? String == "Hello, world")
                #expect(choices[0]["finish_reason"] as? String == "stop")
                let usage = try #require(object["usage"] as? [String: Any])
                #expect(usage["total_tokens"] as? Int == 10)
            }
        }
    }

    @Test func completionStreamingFinishReasonLength() async throws {
        let generator = FakeTokenGenerator()
        generator.finishReason = .length
        try await makeApp(generator: generator).test(.router) { client in
            try await client.execute(
                uri: "/v1/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody(#"{"prompt": "p", "stream": true}"#)
            ) { response in
                #expect(response.status == .ok)
                #expect(response.head.headerFields[.contentType] == "text/event-stream")
                let body = String(buffer: response.body)
                #expect(body.hasSuffix("data: [DONE]\n\n"))
                let frames = body.components(separatedBy: "\n\n").filter { !$0.isEmpty }
                // 2 content chunks + final chunk + [DONE]
                #expect(frames.count == 4)
                let finalChunk = try jsonObject(
                    ByteBuffer(string: String(frames[2].dropFirst("data: ".count))))
                #expect(finalChunk["object"] as? String == "text_completion")
                let choices = try #require(finalChunk["choices"] as? [[String: Any]])
                #expect(choices[0]["finish_reason"] as? String == "length")
            }
        }
    }

    @Test func malformedJSONReturnsOpenAIError() async throws {
        try await makeApp().test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions",
                method: .post,
                headers: [.contentType: "application/json"],
                body: jsonBody("{not json")
            ) { response in
                #expect(response.status == .badRequest)
                let object = try jsonObject(response.body)
                let error = try #require(object["error"] as? [String: Any])
                #expect(error["type"] as? String == "invalid_request_error")
            }
        }
    }
}
