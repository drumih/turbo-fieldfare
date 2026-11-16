import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

@Suite struct ResponsesAPIMapperTests {
    private func decode(_ json: String) throws -> ResponsesAPIRequest {
        try JSONDecoder().decode(ResponsesAPIRequest.self, from: Data(json.utf8))
    }

    @Test func omittedSamplingControlsUseProductionDefaults() throws {
        let request = try decode(
            """
            {"model": "m", "input": [{"role": "user", "content": "hi"}]}
            """)

        let chat = try ResponsesAPIMapper.chatRequest(request)
        let validated = try OpenAIRequestValidator.validate(chat, modelID: "m")

        #expect(validated.generationConfig.temperature == 0.6)
        #expect(validated.generationConfig.topK == 20)
        #expect(validated.generationConfig.topP == 0.95)
        #expect(validated.generationConfig.presencePenalty == 0)
    }

    /// An omitted sampling field must stay nil through the mapper, so the
    /// validator can resolve it from the *served model's* profile. Filling it
    /// with a fixed number here made the profile unreachable, and Qwen3.8-
    /// Flash-Next -- whose card says temperature 1.0 -- sampled at 0.6 on this
    /// surface while /v1/chat/completions honoured the profile.
    @Test func omittedSamplingFollowsTheServedModelNotAFixedDefault() throws {
        let request = try decode(
            """
            {"model": "m", "input": [{"role": "user", "content": "hi"}]}
            """)

        let chat = try ResponsesAPIMapper.chatRequest(request)
        // Deliberately not the house default: that is the only way this tells
        // "the mapper left it alone" apart from "the mapper hardcoded 0.6".
        let validated = try OpenAIRequestValidator.validate(
            chat, modelID: "m",
            sampling: GenerationDefaults.Sampling(temperature: 1.0, topK: 20, topP: 0.95))

        #expect(validated.generationConfig.temperature == 1.0)
    }

    @Test func samplingControlsMapExplicitly() throws {
        let request = try decode(
            """
            {"model":"m","input":[{"role":"user","content":"hi"}],
             "temperature":0.7,"top_p":0.9,"top_k":12,"presence_penalty":0.0}
            """)

        let chat = try ResponsesAPIMapper.chatRequest(request)
        let validated = try OpenAIRequestValidator.validate(chat, modelID: "m")

        #expect(validated.generationConfig.temperature == 0.7)
        #expect(validated.generationConfig.topK == 12)
        #expect(validated.generationConfig.topP == 0.9)
        #expect(validated.generationConfig.presencePenalty == 0)
    }

    /// The other half of C11: the *request* stays nil so the profile supplies
    /// the value, but the *Response object* must report the number the server
    /// actually sampled with. The schema marks all four required numbers, so a
    /// null or an absent field fails every client's validation. The profile
    /// here is deliberately not the house default, so the echo has to have
    /// come from the resolved config rather than from the mapper or a constant.
    @Test func responseEchoesTheResolvedSampling() throws {
        let request = try decode(
            """
            {"model": "m", "input": [{"role": "user", "content": "hi"}]}
            """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        let validated = try OpenAIRequestValidator.validate(
            chat, modelID: "m",
            sampling: GenerationDefaults.Sampling(temperature: 1.0, topK: 20, topP: 0.8))

        let echo = ResponsesAPIEcho(
            request: request, effectiveEffort: nil,
            applied: validated.generationConfig)
        let object = ResponsesAPIBuilder.responseObject(
            id: "resp_1", created: 0, model: "m", status: "completed",
            output: [], usage: nil, echo: echo)

        // Numbers, not null and not absent.
        #expect(object["temperature"] as? Float == 1.0)
        #expect(object["top_p"] as? Float == 0.8)
        #expect(object["presence_penalty"] as? Float == 0)
        #expect(object["frequency_penalty"] as? Float == 0)
    }

    @Test func instructionsAndDeveloperMergeIntoOneLeadingSystemMessage() throws {
        let request = try decode(
            """
            {
              "model": "ornith-1.5-35b-a3b",
              "instructions": "You are a coding agent.",
              "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Be precise."}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
              ]
            }
            """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        #expect(chat.messages.count == 2)
        #expect(chat.messages[0].role == "system")
        #expect(
            try chat.messages[0].content?.textValue() == "You are a coding agent.\n\nBe precise.")
        #expect(chat.messages[1].role == "user")
    }

    @Test func functionCallMapsToAssistantToolCalls() throws {
        let request = try decode(
            """
            {
              "model": "m",
              "input": [
                {"type": "function_call", "call_id": "call_1", "name": "exec_command",
                 "arguments": "{\\"command\\": \\"ls\\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "file.txt"}
              ]
            }
            """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        #expect(chat.messages.count == 2)
        #expect(chat.messages[0].role == "assistant")
        #expect(chat.messages[0].toolCalls?.first?.function.name == "exec_command")
        #expect(chat.messages[1].role == "tool")
        #expect(chat.messages[1].toolCallID == "call_1")
    }

    @Test func toolsMapToChatFunctionTools() throws {
        let request = try decode(
            """
            {
              "model": "m",
              "input": [],
              "tools": [
                {"type": "function", "name": "apply_patch", "description": "edit files",
                 "parameters": {"type": "object", "properties": {}}}
              ]
            }
            """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        #expect(chat.tools?.count == 1)
        #expect(chat.tools?.first?.function.name == "apply_patch")
        #expect(chat.tools?.first?.type == "function")
    }

    @Test func outputTokensAndStreamMapThrough() throws {
        let request = try decode(
            """
            {"model": "m", "input": [], "max_output_tokens": 512, "stream": true}
            """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        #expect(chat.maxTokens == 512)
        #expect(chat.stream == true)
        #expect(chat.messages.isEmpty)
    }

    @Test func nonTextPartsAreRejected() throws {
        let request = try decode(
            """
            {
              "model": "m",
              "input": [{"type": "message", "role": "user",
                         "content": [{"type": "input_image", "image_url": "x"}]}]
            }
            """)
        #expect(throws: ServerRequestError.self) {
            _ = try ResponsesAPIMapper.chatRequest(request)
        }
    }

    @Test func inputItemsWithoutTypeAreInferred() throws {
        // OpenCode omits the item "type" field and relies on role+content.
        let request = try decode(
            """
            {
              "model": "m",
              "input": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
                {"call_id": "call_1", "name": "bash", "arguments": "{}"},
                {"call_id": "call_1", "output": "ok"}
              ]
            }
            """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        #expect(chat.messages.count == 4)
        #expect(chat.messages[0].role == "system")
        #expect(chat.messages[1].role == "user")
        #expect(chat.messages[2].role == "assistant")
        #expect(chat.messages[2].toolCalls?.first?.function.name == "bash")
        #expect(chat.messages[3].role == "tool")
        #expect(chat.messages[3].toolCallID == "call_1")
    }

    @Test func unsupportedInputItemIsRejected() throws {
        let request = try decode(
            """
            {"model": "m", "input": [{"type": "computer_call", "action": "click"}]}
            """)
        #expect(throws: ServerRequestError.self) {
            _ = try ResponsesAPIMapper.chatRequest(request)
        }
    }
}
