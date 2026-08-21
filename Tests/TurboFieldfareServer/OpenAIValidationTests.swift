import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("OpenAI request validation")
struct OpenAIValidationTests {
    @Test func capturedOpenCodeInitialRequestValidates() throws {
        let request = try fixture("opencode-1.15.11-initial.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "gemma-4-26b-a4b-it")
        #expect(validated.stream)
        #expect(validated.includeUsage)
        #expect(validated.tools.count == 1)
        #expect(validated.maximumCompletionTokens == 4096)
    }

    @Test func capturedOpenCodeToolResultValidates() throws {
        let request = try fixture("opencode-1.15.11-tool-result.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "gemma-4-26b-a4b-it")
        #expect(validated.messages.count == 4)
        #expect(validated.messages[2].toolCalls.count == 1)
        #expect(validated.messages[3].toolCallID == "call_0123456789abcdef01234567")
    }

    @Test func capturedOpenCodePromptFits16KWith4096Completion() async throws {
        let request = try fixture("opencode-1.15.11-tool-result.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "gemma-4-26b-a4b-it")
        let tokenizer = try await GFTokenizer.load()
        let ids = try tokenizer.encodeToolChat(
            messages: validated.messages, tools: validated.tools)
        #expect(ids.count <= 16_384 - 4_096)
    }

    @Test func capturedDshInitialRequestValidatesAndRenders() async throws {
        // DeepSeek Harness 0.1.1-rc.1's `workflow` tool declares its `args`
        // parameter as a bare object node with `additionalProperties` — the
        // shape that crashed template rendering with "upper filter requires
        // string" (PR 138). Rendering here is the regression: the fixture
        // must survive the full validate + render path.
        let request = try fixture("dsh-0.1.1-rc.1-initial.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "gemma-4-26b-a4b-it")
        #expect(validated.stream)
        #expect(validated.includeUsage)
        #expect(validated.tools.count == 25)
        let workflow = try #require(validated.tools.first { $0.name == "workflow" })
        let args = workflow.parameters.objectValue?["properties"]?
            .objectValue?["args"]?.objectValue
        #expect(args?["properties"] == .object([:]))
        let tokenizer = try await GFTokenizer.load()
        _ = try tokenizer.encodeToolChat(
            messages: validated.messages, tools: validated.tools)
    }

    @Test func capturedDshToolResultValidatesAndRenders() async throws {
        let request = try fixture("dsh-0.1.1-rc.1-tool-result.json")
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "gemma-4-26b-a4b-it")
        #expect(validated.messages.count == 5)
        let call = try #require(
            validated.messages.first { !$0.toolCalls.isEmpty }?.toolCalls.first)
        #expect(call.id == "call_0123456789abcdef01234567")
        let tokenizer = try await GFTokenizer.load()
        _ = try tokenizer.encodeToolChat(
            messages: validated.messages, tools: validated.tools)
    }

    @Test func requiredToolChoiceIsRejected() throws {
        let data = Data(#"""
        {"model":"m","messages":[{"role":"user","content":"x"}],"tool_choice":"required"}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m")
        }
    }

    @Test func hyphenatedToolNamesValidateInDefinitionsAndHistory() throws {
        let data = Data(#"""
        {
          "model":"m",
          "messages":[
            {"role":"user","content":"resolve it"},
            {"role":"assistant","tool_calls":[{
              "id":"call_0123456789abcdef01234567",
              "type":"function",
              "function":{"name":"resolve-library-id","arguments":"{\"name\":\"swift\"}"}
            }]},
            {"role":"tool","tool_call_id":"call_0123456789abcdef01234567","content":"42"}
          ],
          "tools":[{
            "type":"function",
            "function":{
              "name":"resolve-library-id",
              "parameters":{"type":"object","properties":{"name":{"type":"string"}}}
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        #expect(validated.tools.first?.name == "resolve-library-id")
        #expect(validated.messages[1].toolCalls.first?.name == "resolve-library-id")
    }

    @Test func invalidToolNameErrorIdentifiesTheName() throws {
        for invalid in ["bad name", "bad.name", "bad@name"] {
            let data = Data(#"""
            {
              "model":"m",
              "messages":[{"role":"user","content":"x"}],
              "tools":[{
                "type":"function",
                "function":{"name":"\#(invalid)","parameters":{"type":"object"}}
              }]
            }
            """#.utf8)
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
            do {
                _ = try OpenAIRequestValidator.validate(request, modelID: "m")
                Issue.record("invalid tool name was accepted: \(invalid)")
            } catch let error as ServerRequestError {
                #expect(error.envelope.error.code == "invalid_tool_name")
                #expect(error.envelope.error.message.contains(String(reflecting: invalid)))
            }
        }
    }

    @Test func acceptsLeadingSystemAndDeveloperGuidance() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"system","content":"system"},
          {"role":"developer","content":"developer"},
          {"role":"user","content":"hello"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        #expect(validated.messages.map(\.role) == [.system, .developer, .user])
    }

    @Test func rejectsLateDeveloperGuidance() throws {
        let data = Data(#"""
        {"model":"m","messages":[
          {"role":"user","content":"hello"},
          {"role":"developer","content":"late"}
        ]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m")
        }
    }

    @Test func wideIntegerToolArgumentsRoundTripExactly() async throws {
        let expected = "9007199254740993"
        let parsed = try GemmaToolCallParser().parse(
            "call:lookup{id:\(expected)}",
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.argumentsJSON.contains(#""id":\#(expected)"#))
        let signedMinimum = String(Int64.min)
        let signedMaximum = String(Int64.max)
        let unsignedMaximum = String(UInt64.max)
        let edges = try GemmaToolCallParser().parse(
            "call:lookup{minimum:\(signedMinimum),maximum:\(signedMaximum),unsigned:\(unsignedMaximum)}",
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234568")
        #expect(edges.arguments.objectValue?["minimum"] == .integer(.min))
        #expect(edges.arguments.objectValue?["maximum"] == .integer(.max))
        #expect(edges.arguments.objectValue?["unsigned"] == .unsignedInteger(.max))
        let encodedEdges = try edges.arguments.encoded()
        #expect(encodedEdges.contains(signedMinimum))
        #expect(encodedEdges.contains(signedMaximum))
        #expect(encodedEdges.contains(unsignedMaximum))
        #expect(try JSONDecoder().decode(
            JSONValue.self,
            from: Data(encodedEdges.utf8)) == edges.arguments)
        for malformed in ["+1", "01", "1.", ".1", "1e", "--1"] {
            #expect(throws: GemmaToolCallParserError.self) {
                try GemmaToolCallParser().parse(
                    "call:lookup{id:\(malformed)}",
                    allowedTools: ["lookup"],
                    id: "call_0123456789abcdef01234570")
            }
        }

        let data = Data(#"""
        {
          "model":"m",
          "messages":[
            {"role":"user","content":"lookup"},
            {"role":"assistant","tool_calls":[{
              "id":"call_0123456789abcdef01234567",
              "type":"function",
              "function":{"name":"lookup","arguments":"{\"id\":9007199254740993}"}
            }]}
          ],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{"type":"object","properties":{"id":{"type":"integer"}}}
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let call = try #require(validated.messages[1].toolCalls.first)
        #expect(try call.arguments.encoded().contains(#""id":\#(expected)"#))
        let tokenizer = try await GFTokenizer.load()
        let rendered = tokenizer.decode(
            try tokenizer.encodeToolChat(
                messages: validated.messages,
                tools: validated.tools),
            skipSpecialTokens: false)
        #expect(rendered.contains(expected))

        let unrepresentableHistory = Data(#"""
        {
          "model":"m",
          "messages":[
            {"role":"user","content":"lookup"},
            {"role":"assistant","tool_calls":[{
              "id":"call_0123456789abcdef01234569",
              "type":"function",
              "function":{"name":"lookup","arguments":"{\"id\":18446744073709551615}"}
            }]}
          ],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{"type":"object","properties":{"id":{"type":"integer"}}}
            }
          }]
        }
        """#.utf8)
        let rejected = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: unrepresentableHistory)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(rejected, modelID: "m")
        }
    }

    @Test func acceptedNonIdentifierParameterKeysParseAndRender() async throws {
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"lookup"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{
                "type":"object",
                "properties":{
                  "$id":{"type":"string"},
                  "file-path":{"type":"string"},
                  "nested":{"type":"object","properties":{"child-key":{"type":"integer"}}}
                }
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let tokenizer = try await GFTokenizer.load()
        _ = try tokenizer.encodeToolChat(
            messages: validated.messages,
            tools: validated.tools)
        let parsed = try GemmaToolCallParser().parse(
            #"call:lookup{$id:<|"|>item<|"|>,file-path:<|"|>/tmp/x<|"|>,nested:{child-key:7}}"#,
            allowedTools: ["lookup"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.arguments.objectValue?["$id"] == .string("item"))
        #expect(parsed.arguments.objectValue?["file-path"] == .string("/tmp/x"))
    }

    @Test func stringConstantUnionAdaptsToEnumAndRenders() async throws {
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"search"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"vcc_recall",
              "description":"",
              "parameters":{
                "type":"object",
                "properties":{
                  "scope":{
                    "anyOf":[
                      {"type":"string","const":"lineage"},
                      {"type":"string","const":"all"}
                    ],
                    "description":""
                  }
                }
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let tool = try #require(validated.tools.first)
        let properties = try #require(tool.parameters.objectValue?["properties"]?.objectValue)
        let scope = try #require(properties["scope"]?.objectValue)
        #expect(scope["type"] == .string("string"))
        #expect(scope["enum"] == .array([.string("lineage"), .string("all")]))
        #expect(scope["anyOf"] == nil)

        let tokenizer = try await GFTokenizer.load()
        let rendered = tokenizer.decode(
            try tokenizer.encodeToolChat(
                messages: validated.messages,
                tools: validated.tools),
            skipSpecialTokens: false)
        #expect(rendered.contains("lineage"))
        #expect(rendered.contains("all"))
    }

    @Test func nullableToolSchemasAdaptWithoutChangingConstraints() throws {
        let typeArray = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {
          "type":"object",
          "properties":{
            "name":{"type":["null","string"],"minLength":2}
          }
        }
        """#.utf8))
        let adapted = try GemmaToolSchema.adapted(typeArray, toolName: "lookup")
        let name = adapted.objectValue?["properties"]?.objectValue?["name"]?.objectValue
        #expect(name?["type"] == .string("string"))
        #expect(name?["nullable"] == .bool(true))
        #expect(name?["minLength"] == .integer(2))
        #expect(try GemmaToolSchema.adapted(adapted, toolName: "lookup") == adapted)

        let anyOf = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {
          "type":"object",
          "properties":{
            "limit":{"description":"limit","anyOf":[
              {"type":"integer","minimum":1},
              {"type":"null"}
            ]}
          }
        }
        """#.utf8))
        let anyOfAdapted = try GemmaToolSchema.adapted(anyOf, toolName: "lookup")
        let limit = anyOfAdapted.objectValue?["properties"]?.objectValue?["limit"]?.objectValue
        #expect(limit?["type"] == .string("integer"))
        #expect(limit?["nullable"] == .bool(true))
        #expect(limit?["minimum"] == .integer(1))
        #expect(limit?["description"] == .string("limit"))
        #expect(limit?["anyOf"] == nil)

        let nestedOneOf = try JSONDecoder().decode(JSONValue.self, from: Data(#"""
        {
          "type":"object",
          "properties":{
            "names":{"type":"array","items":{"oneOf":[
              {"type":"null"},
              {"type":"string","minLength":1}
            ]}}
          }
        }
        """#.utf8))
        let nestedAdapted = try GemmaToolSchema.adapted(nestedOneOf, toolName: "lookup")
        let item = nestedAdapted.objectValue?["properties"]?.objectValue?["names"]?
            .objectValue?["items"]?.objectValue
        #expect(item?["type"] == .string("string"))
        #expect(item?["nullable"] == .bool(true))
        #expect(item?["minLength"] == .integer(1))
        #expect(item?["oneOf"] == nil)
    }

    @Test func bareObjectNodesWithSiblingKeywordsRender() async throws {
        // Regression: the chat template routes an object node without
        // `properties` through its filter_keys branch, which iterates the
        // node's own keys as property schemas; preserved keywords such as
        // `additionalProperties`, `default`, or `title` then hit
        // `value['type'] | upper` on a non-string and rendering fails with
        // Jinja runtime("upper filter requires string") — the 500 reported by
        // a DeepSeek Harness user in PR 138. Without the injected empty
        // `properties` mapping, encodeToolChat throws here.
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"go"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"probe",
              "parameters":{
                "type":"object",
                "properties":{
                  "closed":{"type":"object","additionalProperties":false},
                  "annotated":{"type":"object","default":{},"title":"Config"},
                  "bare":{"type":"object"}
                }
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let tokenizer = try await GFTokenizer.load()
        let rendered = tokenizer.decode(
            try tokenizer.encodeToolChat(
                messages: validated.messages,
                tools: validated.tools),
            skipSpecialTokens: false)
        // All three nodes render through the same branch a bare object node
        // already used, so the output shape stays `properties:{}` (the
        // detokenized text carries token-boundary spaces).
        #expect(rendered.contains("closed:{ properties:{ },type:"))
        #expect(rendered.contains("annotated:{ properties:{ },type:"))
        #expect(rendered.contains("bare:{ properties:{ },type:"))
    }

    @Test func adaptedObjectNodesAlwaysCarryProperties() throws {
        // Invariant: no adapted object node reaches the template without a
        // `properties` mapping, so the template's key-iterating fallback
        // branch is unreachable for adapter output.
        let schemas = [
            #"{"type":"object","properties":{"v":{"type":"object","additionalProperties":true}}}"#,
            #"{"type":"object","properties":{"v":{"type":"array","items":{"type":"object","default":{"a":1}}}}}"#,
            #"{"type":"object","properties":{"v":{"anyOf":[{"type":"object","examples":{"a":1}},{"type":"null"}]}}}"#,
            #"{"type":"object","properties":{"v":{"type":"object","properties":{"w":{"type":"object"}}}}}"#,
        ]
        for encoded in schemas {
            let schema = try JSONDecoder().decode(JSONValue.self, from: Data(encoded.utf8))
            let adapted = try GemmaToolSchema.adapted(schema, toolName: "probe")
            try assertObjectNodesCarryProperties(adapted, path: "parameters")
            let again = try GemmaToolSchema.adapted(adapted, toolName: "probe")
            #expect(again == adapted)
        }
    }

    private func assertObjectNodesCarryProperties(
        _ schema: JSONValue, path: String
    ) throws {
        guard case .object(let object) = schema else { return }
        if object["type"] == .string("object") {
            let properties = object["properties"]
            #expect(properties?.objectValue != nil,
                    "object node at \(path) lacks a properties mapping")
        }
        if case .object(let definitions)? = object["properties"] {
            for (key, value) in definitions {
                try assertObjectNodesCarryProperties(value, path: "\(path).properties.\(key)")
            }
        }
        if let items = object["items"] {
            try assertObjectNodesCarryProperties(items, path: "\(path).items")
        }
    }

    @Test func unsupportedToolSchemaUnionsFailClosed() throws {
        let schemas = [
            #"{"type":"object","properties":{"v":{"anyOf":[{"type":"string"},{"type":"object"}]}}}"#,
            #"{"type":"object","properties":{"args":{"anyOf":[{"type":"string"},{"type":"object","properties":{},"additionalProperties":true}]}}}"#,
            #"{"type":"object","properties":{"v":{"oneOf":[{"type":"integer"},{"type":"number"}]}}}"#,
            #"{"type":"object","properties":{"v":{"allOf":[{"type":"string"}]}}}"#,
            #"{"type":"object","properties":{"v":{"description":"missing"}}}"#,
            #"{"type":"object","properties":{"v":{"type":["string","number"]}}}"#,
            #"{"type":"object","properties":{"v":{"type":["string","null"],"nullable":false}}}"#,
            #"{"type":"object","properties":{"v":true}}"#,
        ]
        for encoded in schemas {
            let schema = try JSONDecoder().decode(JSONValue.self, from: Data(encoded.utf8))
            do {
                _ = try GemmaToolSchema.adapted(schema, toolName: "unsafe")
                Issue.record("unsupported schema was accepted: \(encoded)")
            } catch let error as ServerRequestError {
                #expect(error.envelope.error.code == "invalid_tool_schema")
                #expect(error.envelope.error.param == "tools")
            }
        }
    }

    @Test func semanticsChangingNullableSchemasFailClosed() throws {
        let schemas = [
            #"{"type":["object","null"],"properties":{}}"#,
            #"{"type":"object","properties":{"v":{"oneOf":[{"type":["string","null"]},{"type":"null"}]}}}"#,
            #"{"type":"object","properties":{"v":{"oneOf":[{"type":"string","const":"same"},{"type":"string","const":"same"}]}}}"#,
        ]
        for encoded in schemas {
            let schema = try JSONDecoder().decode(JSONValue.self, from: Data(encoded.utf8))
            #expect(throws: ServerRequestError.self) {
                try GemmaToolSchema.adapted(schema, toolName: "unsafe")
            }
        }
    }

    @Test func ambiguousParameterKeysFailValidation() throws {
        let data = Data(#"""
        {
          "model":"m",
          "messages":[{"role":"user","content":"lookup"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"lookup",
              "parameters":{
                "type":"object",
                "allOf":[{
                  "type":"object",
                  "properties":{"bad:key":{"type":"string"}}
                }]
              }
            }
          }]
        }
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m")
        }
    }

    private func fixture(_ name: String) throws -> OpenAIChatRequest {
        let url = try #require(Bundle.module.url(
            forResource: name, withExtension: nil, subdirectory: "Fixtures"))
        return try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(contentsOf: url))
    }
}

@Suite("Gemma tool calls")
struct GemmaToolCallTests {
    @Test func parsesNestedArgumentsAndGemmaQuotes() throws {
        let parsed = try GemmaToolCallParser().parse(
            #"call:read{path:<|"|>/tmp/ü"<|"|>,options:{lines:[1,2],exact:true}}"#,
            allowedTools: ["read"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.name == "read")
        #expect(parsed.argumentsJSON.contains(#""path":"/tmp/ü\"""#))
        #expect(parsed.argumentsJSON.contains(#""exact":true"#))
    }

    @Test func unknownToolFailsClosed() {
        #expect(throws: GemmaToolCallParserError.unknownTool("write")) {
            try GemmaToolCallParser().parse(
                "call:write{path:<|\"|>/tmp/x<|\"|>}",
                allowedTools: ["read"],
                id: "call_0123456789abcdef01234567")
        }
    }

    @Test func parsesJSONUnicodeEscapesAndSurrogatePairs() throws {
        let parsed = try GemmaToolCallParser().parse(
            #"call:read{path:"\u00fc-\ud83c\udf33",note:"a\b\f"}"#,
            allowedTools: ["read"],
            id: "call_0123456789abcdef01234567")
        #expect(parsed.argumentsJSON.contains(#""path":"ü-🌳""#))
        #expect(parsed.argumentsJSON.contains(#""note":"a\b\f""#))
    }

    @Test func suppressesThoughtBlockAndExposesTextAfterChannelClose() async throws {
        let tokenizer = try await GFTokenizer.load()
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [])
        #expect(try decoder.consume(tokenID: tokenizer.channelStartID, delta: "").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "thought").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "\n").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "private").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.channelEndID, delta: "").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "visible") == [
            .content("visible"),
        ])
    }

    @Test func routesControlTokenDeltaThroughCurrentChannel() async throws {
        // A non-empty delta on a control token is text the detokenizer held
        // back from before that token; it belongs to the channel in effect
        // now and must not vanish with the control token's early return.
        let tokenizer = try await GFTokenizer.load()
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [])
        #expect(try decoder.consume(tokenID: tokenizer.channelStartID, delta: "leftover") == [
            .content("leftover"),
        ])
        // The channel switch still happened: this resolves the label.
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "thought\n").isEmpty)
        // In the thought channel the routed delta is correctly dropped.
        #expect(try decoder.consume(tokenID: tokenizer.channelEndID, delta: "hidden").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "ok") == [.content("ok")])
    }

    @Test func tailDuringThoughtChannelIsSuppressed() async throws {
        let tokenizer = try await GFTokenizer.load()
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [])
        #expect(try decoder.consume(tokenID: tokenizer.channelStartID, delta: "").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.bosID, delta: "thought\n").isEmpty)
        #expect(try decoder.consumeTail("secret").isEmpty)
        #expect(try decoder.consume(tokenID: tokenizer.channelEndID, delta: "").isEmpty)
        #expect(try decoder.consumeTail("ok") == [.content("ok")])
    }

    @Test func tailDuringUnresolvedLabelEmitsNothing() async throws {
        // Generation ended before the channel label line completed; the text
        // cannot be attributed, so nothing may surface.
        let tokenizer = try await GFTokenizer.load()
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [])
        #expect(try decoder.consume(tokenID: tokenizer.channelStartID, delta: "").isEmpty)
        #expect(try decoder.consumeTail("final-but-no-newline").isEmpty)
    }

    @Test func heldBytesBeforeChannelMarkerStayInTheirChannel() async throws {
        // Thought text ending in a byte-fallback character right before
        // <channel|> must not leak into the visible answer. The barrier
        // detokenizer commits the held character as the marker's delta, and
        // consume routes it under the still-thought channel.
        let tokenizer = try await GFTokenizer.load()
        var detok = GFDetokenizer(tokenizer: tokenizer,
                                  barrierTokenIDs: tokenizer.structuralMarkerIDs)
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [])
        var events: [StructuredAssistantEvent] = []
        func feed(_ id: Int32) throws {
            events += try decoder.consume(tokenID: id, delta: detok.push(id))
        }

        try feed(tokenizer.channelStartID)
        for id in tokenizer.encode("thought\n", addBOS: false) { try feed(id) }
        for token in ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"] {
            try feed(GFTokenizer.requireTokenID(tokenizer.tokenizer, token))
        }
        try feed(tokenizer.channelEndID)
        for id in tokenizer.encode("ok", addBOS: false) { try feed(id) }
        events += try decoder.consumeTail(detok.flush())

        let visible = events.compactMap { event -> String? in
            if case .content(let text) = event { return text }
            return nil
        }.joined()
        #expect(visible == "ok", "thought-channel bytes leaked: '\(visible)'")
    }

    @Test func tailAfterFailureThrows() async throws {
        let tokenizer = try await GFTokenizer.load()
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: [])
        #expect(throws: GemmaToolCallParserError.self) {
            try decoder.consume(tokenID: tokenizer.toolCallEndID, delta: "")
        }
        #expect(throws: GemmaToolCallParserError.self) {
            try decoder.consumeTail("x")
        }
    }
}

@Suite("Streaming stop matcher")
struct StreamingStopMatcherTests {
    @Test func withholdsCrossChunkStop() {
        var matcher = StreamingStopMatcher(stops: ["END"])
        #expect(matcher.push("hello E") == "hello ")
        #expect(matcher.push("N") == "")
        #expect(matcher.push("D ignored") == "")
        #expect(matcher.isStopped)
    }

    @Test func flushesUnicodeTail() {
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("hello 🌳") == "hello ")
        #expect(matcher.finish() == "🌳")
    }
}

@Suite("Server arguments")
struct ServerArgumentTests {
    @Test func defaults() throws {
        let arguments = try ServerArguments.parse(["--model", "model.gturbo"])
        #expect(arguments.port == 8080)
        #expect(arguments.maxContext == 16_384)
        #expect(arguments.queueLimit == 4)
        #expect(arguments.promptCacheMode == .singlePrefix)
        #expect(arguments.expertCacheSlots == 16)
        #expect(arguments.expertCachePolicy == .lfu)
        #expect(arguments.prefillPolicy == .chunked)
        #expect(arguments.prefillChunkTokens == 128)
        #expect(arguments.rdadvisePolicy == .off)
    }

    @Test func parsesSinglePrefixModeAndRejectsUnknownMode() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--prompt-cache-mode", "single-prefix",
        ])
        #expect(arguments.promptCacheMode == .singlePrefix)
        let rollback = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--prompt-cache-mode", "off",
        ])
        #expect(rollback.promptCacheMode == .off)
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse([
                "--model", "model.gturbo",
                "--prompt-cache-mode", "many",
            ])
        }
    }

    @Test func runtimeFlagsReachTheResolvedConfiguration() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--expert-cache-slots", "32",
            "--expert-cache-policy", "lru",
            "--prefill", "on",
            "--prefill-chunk-tokens", "64",
            "--rdadvise", "adaptive",
        ])
        #expect(arguments.expertCacheSlots == 32)
        #expect(arguments.expertCachePolicy == .lru)
        #expect(arguments.prefillPolicy == .chunked)
        #expect(arguments.prefillChunkTokens == 64)
        #expect(arguments.rdadvisePolicy == .adaptive)

        let configuration = try arguments.resolvedRuntimeConfiguration()
        #expect(configuration.expertCacheSlots == 32)
        #expect(configuration.expertCachePolicy == .lru)
        #expect(configuration.prefillPolicy == .chunked)
        #expect(configuration.prefillChunkTokens == 64)
        #expect(configuration.rdadvisePolicy == .adaptive)
    }

    @Test func prefillOffIsResolvableBelowTheChunkedPrefillSlotFloor() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--expert-cache-slots", "8",
            "--prefill", "off",
        ])
        let configuration = try arguments.resolvedRuntimeConfiguration()
        #expect(configuration.expertCacheSlots == 8)
        #expect(configuration.prefillPolicy == .off)
    }

    @Test func chunkedPrefillBelowTheSlotFloorIsRejected() throws {
        let arguments = try ServerArguments.parse([
            "--model", "model.gturbo",
            "--expert-cache-slots", "8",
            "--prefill", "on",
        ])
        #expect(throws: ServerArgumentError.self) {
            try arguments.resolvedRuntimeConfiguration()
        }
    }

    @Test(arguments: [
        ["--expert-cache-slots", "12"],
        ["--expert-cache-policy", "mru"],
        ["--prefill", "maybe"],
        ["--prefill-chunk-tokens", "256"],
        ["--rdadvise", "eager"],
    ])
    func rejectsUnsupportedRuntimeValues(flag: [String]) throws {
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "model.gturbo"] + flag)
        }
    }
}
