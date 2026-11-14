import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

/// Structured output across the three surfaces: one parser, three spellings.
///
/// The grammar itself is tested in `JSONGrammarTests` and the mask in
/// `JSONConstraintTests`; what is tested here is that a request for JSON is
/// understood on every surface, that an unsupported schema is refused *before*
/// a model is touched, and that thinking is turned off for it -- a grammar
/// constrains every token, so a thought cannot be written beside the document.
@Suite("Structured output requests")
struct StructuredOutputRequestTests {
    private func chat(_ json: String) throws -> ValidatedChatRequest {
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
        return try OpenAIRequestValidator.validate(request, modelID: "m")
    }

    @Test func plainTextAndAnUnknownShapeStayFreeText() throws {
        #expect(
            try chat(
                #"""
                {"model":"m","messages":[{"role":"user","content":"x"}],"response_format":{"type":"text"}}
                """#
            ).jsonSchema == nil)
        // An unrecognized shape was never a refusal on this surface, and it
        // still is not.
        #expect(
            try chat(
                #"""
                {"model":"m","messages":[{"role":"user","content":"x"}],"response_format":{"format":"json"}}
                """#
            ).jsonSchema == nil)
        #expect(
            try chat(
                #"""
                {"model":"m","messages":[{"role":"user","content":"x"}]}
                """#
            ).jsonSchema == nil)
    }

    @Test func jsonObjectMeansAnObjectAtTheTopLevel() throws {
        let validated = try chat(
            #"""
            {"model":"m","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"}}
            """#)
        #expect(
            validated.jsonSchema
                == .object(properties: [:], required: [], additional: true))
    }

    @Test func aSupportedSchemaIsCompiledDuringValidation() throws {
        let validated = try chat(
            #"""
            {"model":"m","messages":[{"role":"user","content":"x"}],
             "response_format":{"type":"json_schema","json_schema":{"name":"n","strict":true,
               "schema":{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],
                         "additionalProperties":false}}}}
            """#)
        #expect(
            validated.jsonSchema
                == .object(
                    properties: ["a": .scalar([.integer])], required: ["a"], additional: false))
    }

    @Test func anUnsupportedKeywordIsRefusedByName() throws {
        do {
            _ = try chat(
                #"""
                {"model":"m","messages":[{"role":"user","content":"x"}],
                 "response_format":{"type":"json_schema","json_schema":{"schema":{
                   "type":"string","pattern":"^a+$"}}}}
                """#)
            Issue.record("expected a refusal")
        } catch let error as ServerRequestError {
            guard case .invalid(let message, let param, let code) = error else {
                Issue.record("unexpected \(error)")
                return
            }
            #expect(message.contains("pattern"))
            #expect(param == "response_format.json_schema.schema")
            #expect(code == "unsupported_value")
        }
    }

    @Test func aMissingSchemaAndAnUnknownFormatAreRefused() throws {
        #expect(throws: ServerRequestError.self) {
            try chat(
                #"""
                {"model":"m","messages":[{"role":"user","content":"x"}],
                 "response_format":{"type":"json_schema","json_schema":{"name":"n"}}}
                """#)
        }
        #expect(throws: ServerRequestError.self) {
            try chat(
                #"""
                {"model":"m","messages":[{"role":"user","content":"x"}],
                 "response_format":{"type":"xml"}}
                """#)
        }
    }

    /// The grammar constrains every token, so the thought would have to be
    /// written as part of the JSON. Thinking is off, and the note says why.
    @Test func thinkingIsOffForAConstrainedRequest() throws {
        let on = ServerReasoningProfile(
            family: .qwen38flash, thinkingMode: .on,
            reasoningEffort: .xhigh)
        let request = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(
                #"""
                {"model":"m","messages":[{"role":"user","content":"x"}],
                 "reasoning_effort":"xhigh","response_format":{"type":"json_object"}}
                """#.utf8))
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "m",
            reasoningProfile: on)
        #expect(validated.reasoning == RequestReasoning(thinkingMode: .off, effort: nil))
        #expect(validated.reasoningNotes.contains { $0.contains("thinking is off") })
        // Without a format the same request keeps the level it asked for.
        let free = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(
                #"""
                {"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":"xhigh"}
                """#.utf8))
        let freeValidated = try OpenAIRequestValidator.validate(
            free, modelID: "m",
            reasoningProfile: on)
        #expect(freeValidated.reasoning?.thinkingMode == .on)
    }

    // MARK: - The other two spellings

    @Test func theResponsesSurfaceReshapesTextFormatIntoResponseFormat() throws {
        let object = try ResponsesAPIMapper.responseFormat(
            try JSONDecoder().decode(JSONValue.self, from: Data(#"{"type":"json_object"}"#.utf8)))
        #expect(object == .object(["type": .string("json_object")]))
        let schema = try ResponsesAPIMapper.responseFormat(
            try JSONDecoder().decode(
                JSONValue.self,
                from: Data(
                    #"""
                    {"type":"json_schema","name":"n","strict":true,"schema":{"type":"integer"}}
                    """#.utf8)))
        #expect(
            schema
                == .object([
                    "type": .string("json_schema"),
                    "json_schema": .object([
                        "name": .string("n"), "strict": .bool(true),
                        "schema": .object(["type": .string("integer")]),
                    ]),
                ]))
        #expect(
            try ResponsesAPIMapper.responseFormat(
                try JSONDecoder().decode(JSONValue.self, from: Data(#"{"type":"text"}"#.utf8)))
                == nil)
    }

    @Test func theMessagesSurfaceReshapesOutputConfigFormat() throws {
        let schema = try AnthropicMapper.responseFormat(
            try JSONDecoder().decode(
                JSONValue.self,
                from: Data(
                    #"""
                    {"type":"json_schema","schema":{"type":"object","properties":{}}}
                    """#.utf8)))
        #expect(
            schema
                == .object([
                    "type": .string("json_schema"),
                    "json_schema": .object([
                        "schema": .object([
                            "type": .string("object"),
                            "properties": .object([:]),
                        ])
                    ]),
                ]))
        #expect(
            try AnthropicMapper.responseFormat(
                try JSONDecoder().decode(
                    JSONValue.self, from: Data(#"{"type":"json_object"}"#.utf8)))
                == .object(["type": .string("json_object")]))
    }

    /// The same Messages request, all the way through the mapper the handler
    /// calls, so the schema reaches the validator.
    @Test func aMessagesRequestCarriesItsSchemaIntoTheChatRequest() throws {
        let decoded = try JSONDecoder().decode(
            AnthropicMessagesRequest.self,
            from: Data(
                #"""
                {"model":"m","max_tokens":64,
                 "output_config":{"format":{"type":"json_object"}},
                 "messages":[{"role":"user","content":"hi"}]}
                """#.utf8))
        let chat = try AnthropicMapper.chatRequest(decoded)
        let validated = try OpenAIRequestValidator.validate(chat, modelID: "m")
        #expect(
            validated.jsonSchema
                == .object(properties: [:], required: [], additional: true))
    }

    /// A Responses request, through its mapper.
    @Test func aResponsesRequestCarriesItsSchemaIntoTheChatRequest() throws {
        let decoded = try JSONDecoder().decode(
            ResponsesAPIRequest.self,
            from: Data(
                #"""
                {"model":"m","input":"hi","text":{"format":{"type":"json_schema",
                  "name":"n","schema":{"type":"object","properties":{"a":{"type":"string"}}}}}}
                """#.utf8))
        let chat = try ResponsesAPIMapper.chatRequest(decoded)
        let validated = try OpenAIRequestValidator.validate(chat, modelID: "m")
        guard case .object(let properties, _, _)? = validated.jsonSchema else {
            Issue.record(
                "expected an object schema, got \(String(describing: validated.jsonSchema))")
            return
        }
        #expect(properties["a"] == .scalar([.string]))
    }
}
