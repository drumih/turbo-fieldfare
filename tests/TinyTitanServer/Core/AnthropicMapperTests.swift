import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

@Suite struct AnthropicMapperTests {
    private func decode(_ json: String) throws -> AnthropicMessagesRequest {
        try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(json.utf8))
    }

    private func map(_ json: String) throws -> OpenAIChatRequest {
        try AnthropicMapper.chatRequest(try decode(json))
    }

    @Test func systemAndStringContentBecomeChatMessages() throws {
        let chat = try map(
            """
            {"model":"m","max_tokens":64,"system":"Be terse.",
             "messages":[{"role":"user","content":"hi"}],
             "stop_sequences":["END"],"temperature":0.4,"top_p":0.8,"top_k":9}
            """)
        #expect(chat.messages.map(\.role) == ["system", "user"])
        #expect(chat.messages[0].content == .text("Be terse."))
        #expect(chat.maxTokens == 64)
        #expect(chat.stop == .many(["END"]))
        #expect(chat.temperature == 0.4)
        #expect(chat.topP == 0.8)
        #expect(chat.topK == 9)
        #expect(chat.stream == false)
        let validated = try OpenAIRequestValidator.validate(chat, modelID: "m")
        #expect(validated.generationConfig.stopStrings == ["END"])
    }

    /// See the note in `ResponsesAPIMapperTests`: an omitted field stays nil so
    /// the served model's profile decides, and the Anthropic surface is the
    /// second place that used to hardcode the house defaults instead.
    @Test func omittedSamplingFollowsTheServedModelNotAFixedDefault() throws {
        let chat = try map(
            """
            {"model":"m","max_tokens":64,
             "messages":[{"role":"user","content":"hi"}]}
            """)
        let validated = try OpenAIRequestValidator.validate(
            chat, modelID: "m",
            sampling: GenerationDefaults.Sampling(temperature: 1.0, topK: 20, topP: 0.95))
        #expect(validated.generationConfig.temperature == 1.0)
    }

    @Test func systemTextBlocksJoin() throws {
        let chat = try map(
            """
            {"model":"m","max_tokens":8,
             "system":[{"type":"text","text":"A","cache_control":{"type":"ephemeral"}},{"type":"text","text":"B"}],
             "messages":[{"role":"user","content":"hi"}]}
            """)
        #expect(chat.messages[0].content == .text("A\n\nB"))
    }

    @Test func maxTokensIsRequired() throws {
        #expect(
            throws: ServerRequestError.invalid(
                message: "field required", param: "max_tokens", code: "invalid_value")
        ) {
            try map(#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
        }
    }

    @Test func toolUseAndToolResultRoundTripAsChatToolCalls() throws {
        let chat = try map(
            """
            {"model":"m","max_tokens":8,
             "tools":[{"name":"read","description":"Read a file",
                       "input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
             "messages":[
               {"role":"user","content":"read it"},
               {"role":"assistant","content":[{"type":"text","text":"Sure."},
                                              {"type":"tool_use","id":"toolu_1","name":"read","input":{"path":"/a"}}]},
               {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"contents"},
                                         {"type":"text","text":"now summarise"}]}
             ]}
            """)
        #expect(chat.messages.map(\.role) == ["user", "assistant", "tool", "user"])
        let assistant = chat.messages[1]
        #expect(assistant.content == .text("Sure."))
        #expect(assistant.toolCalls?.count == 1)
        #expect(assistant.toolCalls?[0].id == "toolu_1")
        #expect(assistant.toolCalls?[0].function.name == "read")
        #expect(assistant.toolCalls?[0].function.arguments == #"{"path":"/a"}"#)
        #expect(chat.messages[2].toolCallID == "toolu_1")
        #expect(chat.messages[2].content == .text("contents"))
        #expect(chat.messages[3].content == .text("now summarise"))
        #expect(chat.tools?.count == 1)
        #expect(chat.tools?[0].function.name == "read")
        #expect(chat.tools?[0].function.description == "Read a file")
        // The chat validator accepts the mapped conversation as a whole.
        let validated = try OpenAIRequestValidator.validate(chat, modelID: "m")
        #expect(validated.tools.count == 1)
        #expect(validated.messages.count == 4)
    }

    @Test func erroredToolResultIsMarked() throws {
        let chat = try map(
            """
            {"model":"m","max_tokens":8,
             "messages":[
               {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"read","input":{}}]},
               {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","is_error":true,"content":"no such file"}]}
             ]}
            """)
        #expect(chat.messages[1].content == .text("Error: no such file"))
        #expect(chat.messages[0].content == nil)
    }

    @Test func thinkingBlocksInHistoryAreSkipped() throws {
        let chat = try map(
            """
            {"model":"m","max_tokens":8,
             "messages":[
               {"role":"user","content":"q"},
               {"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"sig"},
                                              {"type":"redacted_thinking","data":"x"},
                                              {"type":"text","text":"answer"}]},
               {"role":"user","content":"more"}
             ]}
            """)
        #expect(chat.messages.map(\.role) == ["user", "assistant", "user"])
        #expect(chat.messages[1].content == .text("answer"))
    }

    @Test func midConversationSystemMessagesJoinTheSystemBlock() throws {
        let chat = try map(
            """
            {"model":"m","max_tokens":8,"system":"lead",
             "messages":[{"role":"user","content":"one"},{"role":"system","content":"later"},
                         {"role":"assistant","content":"two"},{"role":"user","content":"three"}]}
            """)
        #expect(chat.messages.map(\.role) == ["system", "user", "assistant", "user"])
        #expect(chat.messages[0].content == .text("lead\n\nlater"))
    }

    @Test func maxTokensIsClampedToTheContextWindow() throws {
        let request = try decode(
            #"{"model":"m","max_tokens":32000,"messages":[{"role":"user","content":"hi"}]}"#)
        let chat = try AnthropicMapper.chatRequest(request, maxContext: 8192)
        #expect(chat.maxTokens == 8192)
    }

    @Test func consecutiveUserTurnsCombine() throws {
        let chat = try map(
            """
            {"model":"m","max_tokens":8,
             "messages":[{"role":"user","content":"one"},{"role":"user","content":"two"}]}
            """)
        #expect(chat.messages.count == 1)
        #expect(chat.messages[0].content == .text("one\n\ntwo"))
    }

    @Test func imageBlocksAreRefusedByPath() throws {
        do {
            _ = try map(
                """
                {"model":"m","max_tokens":8,
                 "messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"http://x/y.png"}}]}]}
                """)
            Issue.record("expected a refusal")
        } catch let error as ServerRequestError {
            guard case .invalid(let message, let param, let code) = error else {
                Issue.record("unexpected \(error)")
                return
            }
            #expect(param == "messages.0.content.0")
            #expect(code == "unsupported_value")
            #expect(message.contains("text-only"))
            let envelope = AnthropicErrorEnvelope.from(error, requestID: "req_1")
            #expect(envelope.type == "error")
            #expect(envelope.error.type == "invalid_request_error")
            #expect(envelope.error.message.hasPrefix("messages.0.content.0: "))
            #expect(envelope.httpStatus == 400)
        }
    }

    @Test func builtInToolsAndForcedChoicesAreRefused() throws {
        #expect(throws: ServerRequestError.self) {
            try map(
                """
                {"model":"m","max_tokens":8,"tools":[{"type":"bash_20250124","name":"bash"}],
                 "messages":[{"role":"user","content":"hi"}]}
                """)
        }
        #expect(throws: ServerRequestError.self) {
            try map(
                """
                {"model":"m","max_tokens":8,"tool_choice":{"type":"any"},
                 "tools":[{"name":"f","input_schema":{"type":"object"}}],
                 "messages":[{"role":"user","content":"hi"}]}
                """)
        }
        let none = try map(
            """
            {"model":"m","max_tokens":8,"tool_choice":{"type":"none"},
             "tools":[{"name":"f","input_schema":{"type":"object"}}],
             "messages":[{"role":"user","content":"hi"}]}
            """)
        #expect(none.toolChoice == .string("none"))
    }

    /// Thinking is a **per-request** control on this surface, as it is on the
    /// OpenAI ones: the block's type and budget become the level the generation
    /// runs at. The mapper takes no server profile at all now — it used to read
    /// one to refuse `enabled` when the server had been started with thinking
    /// off, which made the same harness behave two ways depending on the API a
    /// client spoke. The HTTP-level test that the request really overrides the
    /// loaded profile lives in `ReasoningSurfaceTests`.
    @Test func thinkingComesFromTheRequestAlone() throws {
        func effort(_ body: String) throws -> String? {
            try map(body).reasoningEffort
        }
        let budgets: [(Int, String)] = [(2048, "low"), (8192, "medium"), (32768, "xhigh")]
        for (budget, expected) in budgets {
            #expect(
                try effort(
                    """
                    {"model":"m","max_tokens":65536,
                     "thinking":{"type":"enabled","budget_tokens":\(budget)},
                     "messages":[{"role":"user","content":"hi"}]}
                    """) == expected)
        }
        // Adaptive leaves the server's own setting alone: Claude Code sends it
        // on every request as "you decide", so it must not force a level.
        #expect(
            try effort(
                """
                {"model":"m","max_tokens":4096,"thinking":{"type":"adaptive","display":"omitted"},
                 "context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},
                 "output_config":{"effort":"high"},
                 "messages":[{"role":"user","content":"hi"}]}
                """) == nil)
        // Disabled is a real request for off — the half that matters for a
        // client that over-thinks a turn.
        #expect(
            try effort(
                """
                {"model":"m","max_tokens":8,"thinking":{"type":"disabled"},
                 "messages":[{"role":"user","content":"hi"}]}
                """) == "off")
        // An omitted block asks for nothing, so the loaded profile decides.
        #expect(
            try effort(
                """
                {"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}
                """) == nil)
    }

    /// The budget rules stay Anthropic's own: required when enabled, at least
    /// 1024, and below max_tokens. A budget outside them is a malformed request,
    /// not a mapping choice — this server renders levels, not token counts.
    @Test func enabledWithoutAUsableBudgetIsRefused() throws {
        #expect(throws: ServerRequestError.self) {
            try map(
                """
                {"model":"m","max_tokens":4096,"thinking":{"type":"enabled"},
                 "messages":[{"role":"user","content":"hi"}]}
                """)
        }
        #expect(throws: ServerRequestError.self) {
            try map(
                """
                {"model":"m","max_tokens":4096,"thinking":{"type":"enabled","budget_tokens":512},
                 "messages":[{"role":"user","content":"hi"}]}
                """)
        }
        #expect(throws: ServerRequestError.self) {
            try map(
                """
                {"model":"m","max_tokens":1000,"thinking":{"type":"enabled","budget_tokens":2048},
                 "messages":[{"role":"user","content":"hi"}]}
                """)
        }
    }

    @Test func prefillIsRefused() throws {
        #expect(throws: ServerRequestError.self) {
            try map(
                """
                {"model":"m","max_tokens":8,
                 "messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"The answer is"}]}
                """)
        }
    }

    /// `output_config.format` is this surface's spelling of structured output.
    /// It used to be refused outright; it is now reshaped into the Chat
    /// Completions spelling, which is the one the validator parses.
    @Test func structuredOutputIsCarriedIntoTheChatRequest() throws {
        let chat = try map(
            """
            {"model":"m","max_tokens":8,"output_config":{"format":{"type":"json_schema","schema":{}}},
             "messages":[{"role":"user","content":"hi"}]}
            """)
        #expect(
            chat.responseFormat
                == .object([
                    "type": .string("json_schema"),
                    "json_schema": .object(["schema": .object([:])]),
                ]))
        let object = try map(
            """
            {"model":"m","max_tokens":8,"output_config":{"format":{"type":"json_object"}},
             "messages":[{"role":"user","content":"hi"}]}
            """)
        #expect(object.responseFormat == .object(["type": .string("json_object")]))
    }

    @Test func stopReasonsFollowTheCompletion() {
        let usage = OpenAIUsage(
            promptTokens: 10, completionTokens: 2, totalTokens: 12, cachedTokens: 4)
        let plain = ServerCompletion(
            content: "x", toolCalls: [], finishReason: "stop", usage: usage)
        #expect(AnthropicBuilder.stopReason(for: plain).reason == "end_turn")
        let capped = ServerCompletion(
            content: "x", toolCalls: [], finishReason: "length", usage: usage)
        #expect(AnthropicBuilder.stopReason(for: capped).reason == "max_tokens")
        let stopped = ServerCompletion(
            content: "x", toolCalls: [], finishReason: "stop",
            usage: usage, stopSequence: "END")
        let reason = AnthropicBuilder.stopReason(for: stopped)
        #expect(reason.reason == "stop_sequence")
        #expect(reason.sequence == "END")
        let call = ParsedToolCall(
            id: "c1", name: "read", arguments: .object([:]), argumentsJSON: "{}")
        let tool = ServerCompletion(
            content: "", toolCalls: [call], finishReason: "tool_calls", usage: usage)
        #expect(AnthropicBuilder.stopReason(for: tool).reason == "tool_use")
        let usageObject = AnthropicBuilder.usageObject(usage)
        #expect(usageObject["input_tokens"] as? Int == 6)
        #expect(usageObject["cache_read_input_tokens"] as? Int == 4)
        #expect(usageObject["output_tokens"] as? Int == 2)
    }

    @Test func countTokensBodyMapsWithoutGenerationFields() throws {
        let request = try JSONDecoder().decode(
            AnthropicCountTokensRequest.self,
            from: Data(
                """
                {"model":"m","system":"s","messages":[{"role":"user","content":"hi"}]}
                """.utf8))
        let chat = try AnthropicMapper.chatRequest(counting: request)
        #expect(chat.messages.map(\.role) == ["system", "user"])
        #expect(chat.maxTokens == 1)
    }
}
