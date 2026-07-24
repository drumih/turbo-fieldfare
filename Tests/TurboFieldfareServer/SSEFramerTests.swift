import Foundation
import Testing
@testable import TurboFieldfareServerCore

@Suite("SSE framing")
struct SSEFramerTests {
    @Test func frameWrapsJSONWithDataPrefixAndBlankLine() throws {
        struct Probe: Encodable { var value: Int }
        let frame = try SSEFramer.frame(Probe(value: 7))
        let text = String(decoding: frame, as: UTF8.self)
        #expect(text.hasPrefix("data: "))
        #expect(text.hasSuffix("\n\n"))
        #expect(text == "data: {\"value\":7}\n\n")
    }

    @Test func doneSentinel() {
        #expect(String(decoding: SSEFramer.doneFrame, as: UTF8.self) == "data: [DONE]\n\n")
    }

    @Test func chatChunkSerializesOpenAIShape() throws {
        let chunk = ChatCompletionChunk(
            id: "chatcmpl-x",
            created: 123,
            model: "m",
            choices: [ChatChunkChoice(index: 0,
                                      delta: ChatChunkDelta(content: "hi"),
                                      finishReason: nil)])
        let frame = try SSEFramer.frame(chunk)
        let text = String(decoding: frame, as: UTF8.self)
        #expect(text.hasPrefix("data: "))
        #expect(text.hasSuffix("\n\n"))

        let jsonText = String(text.dropFirst("data: ".count).dropLast(2))
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(jsonText.utf8)) as? [String: Any])
        #expect(object["object"] as? String == "chat.completion.chunk")
        #expect(object["id"] as? String == "chatcmpl-x")
        #expect(object["created"] as? Int == 123)
        #expect(object["model"] as? String == "m")
        let choices = try #require(object["choices"] as? [[String: Any]])
        #expect(choices.count == 1)
        #expect(choices[0]["index"] as? Int == 0)
        // finish_reason must be present as explicit null in in-flight chunks.
        #expect(choices[0].keys.contains("finish_reason"))
        #expect(choices[0]["finish_reason"] is NSNull)
        let delta = try #require(choices[0]["delta"] as? [String: Any])
        #expect(delta["content"] as? String == "hi")
        // usage omitted unless requested
        #expect(object["usage"] == nil)
    }

    @Test func completionChunkUsesTextCompletionObject() throws {
        let chunk = CompletionChunk(
            id: "cmpl-x",
            created: 1,
            model: "m",
            choices: [CompletionChoice(index: 0, text: "abc", finishReason: "stop")])
        let frame = try SSEFramer.frame(chunk)
        let jsonText = String(decoding: frame, as: UTF8.self)
            .dropFirst("data: ".count).dropLast(2)
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(jsonText.utf8)) as? [String: Any])
        #expect(object["object"] as? String == "text_completion")
        let choices = try #require(object["choices"] as? [[String: Any]])
        #expect(choices[0]["text"] as? String == "abc")
        #expect(choices[0]["finish_reason"] as? String == "stop")
    }

    @Test func usageChunkCarriesUsageWithEmptyChoices() throws {
        let chunk = ChatCompletionChunk(
            id: "chatcmpl-x",
            created: 1,
            model: "m",
            choices: [],
            usage: Usage(promptTokens: 3, completionTokens: 4))
        let frame = try SSEFramer.frame(chunk)
        let jsonText = String(decoding: frame, as: UTF8.self)
            .dropFirst("data: ".count).dropLast(2)
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(jsonText.utf8)) as? [String: Any])
        let usage = try #require(object["usage"] as? [String: Any])
        #expect(usage["prompt_tokens"] as? Int == 3)
        #expect(usage["completion_tokens"] as? Int == 4)
        #expect(usage["total_tokens"] as? Int == 7)
    }
}
