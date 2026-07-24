import Foundation
import Testing
@testable import TurboFieldfareServerCore

@Suite("ChatCompletionRequest decoding")
struct ChatCompletionRequestDecodingTests {
    private func decode(_ json: String) throws -> ChatCompletionRequest {
        try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
    }

    @Test func fullBody() throws {
        let request = try decode(#"""
        {
          "model": "gemma",
          "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"}
          ],
          "temperature": 0.7,
          "top_p": 0.9,
          "max_tokens": 128,
          "stop": ["</s>", "END"],
          "seed": 42,
          "stream": true,
          "stream_options": {"include_usage": true},
          "n": 1
        }
        """#)
        #expect(request.model == "gemma")
        #expect(request.messages.count == 2)
        #expect(request.messages[0] == ChatMessage(role: "system", content: "be brief"))
        #expect(request.temperature == 0.7)
        #expect(request.topP == 0.9)
        #expect(request.maxTokens == 128)
        #expect(request.stop == .multiple(["</s>", "END"]))
        #expect(request.seed == 42)
        #expect(request.stream == true)
        #expect(request.streamOptions?.includeUsage == true)
        #expect(request.n == 1)
    }

    @Test func minimalBody() throws {
        let request = try decode(#"{"messages": [{"role": "user", "content": "hi"}]}"#)
        #expect(request.messages.count == 1)
        #expect(request.model == nil)
        #expect(request.temperature == nil)
        #expect(request.maxTokens == nil)
        #expect(request.stop == nil)
        #expect(request.stream == nil)
        #expect(request.n == nil)
    }

    @Test func stopAsString() throws {
        let request = try decode(#"{"messages": [], "stop": "halt"}"#)
        #expect(request.stop == .single("halt"))
        #expect(request.stop?.values == ["halt"])
    }

    @Test func stopAsArray() throws {
        let request = try decode(#"{"messages": [], "stop": ["a", "b"]}"#)
        #expect(request.stop?.values == ["a", "b"])
    }

    @Test func maxCompletionTokensAlias() throws {
        let request = try decode(#"{"messages": [], "max_completion_tokens": 64}"#)
        #expect(request.maxTokens == 64)
    }

    @Test func maxTokensWinsOverAlias() throws {
        let request = try decode(#"{"messages": [], "max_tokens": 32, "max_completion_tokens": 64}"#)
        #expect(request.maxTokens == 32)
    }

    @Test func unknownFieldsIgnored() throws {
        let request = try decode(#"""
        {
          "messages": [{"role": "user", "content": "hi"}],
          "frequency_penalty": 0.5,
          "logit_bias": {"1": 2},
          "tools": [{"type": "function"}],
          "response_format": {"type": "json_object"}
        }
        """#)
        #expect(request.messages.count == 1)
    }

    @Test func nIsDecoded() throws {
        let request = try decode(#"{"messages": [], "n": 2}"#)
        #expect(request.n == 2)
    }
}

@Suite("CompletionRequest decoding")
struct CompletionRequestDecodingTests {
    @Test func promptAndOptions() throws {
        let request = try JSONDecoder().decode(
            CompletionRequest.self,
            from: Data(#"{"prompt": "once upon a time", "max_tokens": 10, "stream": true}"#.utf8))
        #expect(request.prompt == "once upon a time")
        #expect(request.maxTokens == 10)
        #expect(request.stream == true)
    }
}
