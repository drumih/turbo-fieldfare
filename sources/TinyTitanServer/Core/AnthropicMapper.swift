import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

// The Anthropic Messages request mapping onto the chat request.
//
// Split out of `AnthropicModels.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion; the mapper's two
// `private` error helpers moved with it.
// MARK: - Messages -> chat mapping

public enum AnthropicMapper {
    private static func invalid(_ message: String, _ param: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: "invalid_value")
    }

    private static func unsupported(_ message: String, _ param: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: "unsupported_value")
    }

    /// The text of a `system` prompt: a string or text blocks.
    static func systemText(_ system: JSONValue?) throws -> String? {
        guard let system else { return nil }
        switch system {
        case .string(let text):
            return text
        case .array(let blocks):
            var out: [String] = []
            for (index, block) in blocks.enumerated() {
                guard case .object(let dict) = block,
                    case .string("text")? = dict["type"],
                    case .string(let text)? = dict["text"]
                else {
                    throw invalid("system blocks must be text blocks", "system.\(index)")
                }
                out.append(text)
            }
            return out.joined(separator: "\n\n")
        default:
            throw invalid("system must be a string or an array of text blocks", "system")
        }
    }

    /// A tool_result's content: a string, or text blocks (other block kinds
    /// inside a result are refused, as TinyTitan cannot show the model an image).
    static func toolResultText(_ content: JSONValue?, param: String) throws -> String {
        guard let content else { return "" }
        switch content {
        case .string(let text):
            return text
        case .array(let blocks):
            var out: [String] = []
            for (index, block) in blocks.enumerated() {
                guard case .object(let dict) = block, case .string(let type)? = dict["type"] else {
                    throw invalid("content blocks must have a type", "\(param).content.\(index)")
                }
                guard type == "text", case .string(let text)? = dict["text"] else {
                    throw unsupported(
                        "\(type) blocks in tool results are not supported; this server is text-only",
                        "\(param).content.\(index)")
                }
                out.append(text)
            }
            return out.joined(separator: "\n")
        default:
            throw invalid(
                "tool_result content must be a string or an array of blocks", "\(param).content")
        }
    }

    /// One conversation message, in the order the chat template needs it:
    /// a user message with tool results becomes tool messages (then any text
    /// as a user message); an assistant message becomes one assistant message
    /// carrying its text and tool calls.
    static func chatMessages(
        for message: AnthropicMessagesRequest.Message,
        index: Int
    ) throws -> [OpenAIChatMessage] {
        let param = "messages.\(index)"
        guard ["user", "assistant", "system"].contains(message.role) else {
            throw invalid("role must be user, assistant or system", "\(param).role")
        }
        switch message.content {
        case .string(let text):
            return [
                OpenAIChatMessage(
                    role: message.role, content: .text(text),
                    toolCalls: nil, toolCallID: nil, name: nil)
            ]
        case .array(let blocks):
            var text: [String] = []
            var toolCalls: [OpenAIToolCall] = []
            var toolResults: [OpenAIChatMessage] = []
            for (blockIndex, block) in blocks.enumerated() {
                let blockParam = "\(param).content.\(blockIndex)"
                guard case .object(let dict) = block, case .string(let type)? = dict["type"] else {
                    throw invalid("content blocks must have a type", blockParam)
                }
                switch type {
                case "text":
                    guard case .string(let value)? = dict["text"] else {
                        throw invalid("text block requires text", blockParam)
                    }
                    text.append(value)
                case "tool_use":
                    guard message.role == "assistant" else {
                        throw invalid("tool_use blocks belong to assistant messages", blockParam)
                    }
                    guard case .string(let id)? = dict["id"], case .string(let name)? = dict["name"]
                    else {
                        throw invalid("tool_use requires id and name", blockParam)
                    }
                    let input = dict["input"] ?? .object([:])
                    let arguments = (try? input.encoded(sortedKeys: true)) ?? "{}"
                    toolCalls.append(
                        OpenAIToolCall(
                            id: id, type: "function",
                            function: OpenAIFunctionCall(name: name, arguments: arguments)))
                case "tool_result":
                    guard message.role == "user" else {
                        throw invalid("tool_result blocks belong to user messages", blockParam)
                    }
                    guard case .string(let useID)? = dict["tool_use_id"] else {
                        throw invalid("tool_result requires tool_use_id", blockParam)
                    }
                    var result = try toolResultText(dict["content"], param: blockParam)
                    if case .bool(true)? = dict["is_error"], !result.hasPrefix("Error") {
                        result = "Error: " + result
                    }
                    toolResults.append(
                        OpenAIChatMessage(
                            role: "tool", content: .text(result),
                            toolCalls: nil, toolCallID: useID, name: nil))
                case "thinking", "redacted_thinking":
                    // Replayed thoughts from an earlier turn. TinyTitan never
                    // renders a model's prior thinking into its prompt.
                    continue
                case "image", "document", "search_result", "server_tool_use",
                    "web_search_tool_result", "container_upload":
                    throw unsupported(
                        "\(type) blocks are not supported; this server is text-only", blockParam)
                default:
                    throw unsupported("unsupported content block type \(type)", blockParam)
                }
            }
            var out = toolResults
            if message.role == "assistant" {
                let joined = text.joined(separator: "\n")
                out.append(
                    OpenAIChatMessage(
                        role: "assistant",
                        content: joined.isEmpty && !toolCalls.isEmpty ? nil : .text(joined),
                        toolCalls: toolCalls.isEmpty ? nil : toolCalls,
                        toolCallID: nil, name: nil))
            } else if !text.isEmpty || toolResults.isEmpty {
                out.append(
                    OpenAIChatMessage(
                        role: "user", content: .text(text.joined(separator: "\n")),
                        toolCalls: nil, toolCallID: nil, name: nil))
            }
            return out
        default:
            throw invalid("content must be a string or an array of blocks", "\(param).content")
        }
    }

    /// Function definitions from Anthropic tool objects. Built-in tool types
    /// (bash, text editor, web search, computer use, ...) are refused: nothing
    /// on this server executes them.
    static func tools(_ tools: [JSONValue]?) throws -> [OpenAITool]? {
        guard let tools, !tools.isEmpty else { return nil }
        return try tools.enumerated().map { index, tool in
            let param = "tools.\(index)"
            guard case .object(let dict) = tool else {
                throw invalid("tools must be objects", param)
            }
            if case .string(let type)? = dict["type"], type != "custom" {
                throw unsupported(
                    "tool type \(type) is not supported; only custom (function) tools are available",
                    "\(param).type")
            }
            guard case .string(let name)? = dict["name"] else {
                throw invalid("tool requires a name", "\(param).name")
            }
            let description: String?
            if case .string(let text)? = dict["description"] {
                description = text
            } else {
                description = nil
            }
            let schema =
                dict["input_schema"]
                ?? .object(["type": .string("object"), "properties": .object([:])])
            return OpenAITool(
                type: "function",
                function: OpenAIFunctionDefinition(
                    name: name, description: description,
                    parameters: schema))
        }
    }

    /// The chat-side tool_choice for an Anthropic one. `auto` and `none` are
    /// honoured; `any` and `tool` force a call, which the decoder cannot do.
    static func toolChoice(_ choice: JSONValue?) throws -> JSONValue? {
        guard let choice else { return nil }
        guard case .object(let dict) = choice, case .string(let type)? = dict["type"] else {
            throw invalid("tool_choice must be an object with a type", "tool_choice")
        }
        if case .bool(true)? = dict["disable_parallel_tool_use"] {
            throw unsupported(
                "disable_parallel_tool_use is not supported",
                "tool_choice.disable_parallel_tool_use")
        }
        switch type {
        case "auto": return .string("auto")
        case "none": return .string("none")
        case "any", "tool":
            throw unsupported(
                "tool_choice \(type) is not supported; the model chooses whether to call a tool",
                "tool_choice.type")
        default:
            throw invalid("tool_choice type must be auto, any, tool or none", "tool_choice.type")
        }
    }

    /// The reasoning level a Messages `thinking` block asks for, or nil when the
    /// request names none.
    ///
    /// Thinking is a **per-request** control here, exactly as `reasoning_effort`
    /// is on the OpenAI surfaces: the level travels into generation, which
    /// resolves a tokenizer for it, so a Messages client may turn thinking on,
    /// off, or to another effort between turns of one session. That is also why
    /// `chatRequest` takes no `ServerReasoningProfile`: nothing the server was
    /// loaded with may contribute to this answer. The parameter it used to take
    /// only read the loaded mode, to refuse an `enabled` request when thinking
    /// was off; the validator still maps the resulting level onto what the
    /// served model renders, on the same path a Chat Completions request takes.
    ///
    /// `adaptive` returns nil rather than a level: Claude Code sends it on every
    /// request to mean "you decide", so it must leave the server's own setting
    /// alone instead of forcing anything. `enabled` maps its `budget_tokens`
    /// onto the one effort ladder — a token budget is not enforceable here, the
    /// template renders levels — and the levels a model does not define are
    /// mapped to the nearest by the same rule the OpenAI path uses.
    ///
    /// The *response* carries the thought as a `thinking` block whenever the
    /// model produced one, signed with `AnthropicBuilder.thinkingSignature` —
    /// the empty string, because an Anthropic signature is an attestation this
    /// server cannot produce and a made-up token would only pretend to be
    /// verifiable. Turning the level off is therefore what removes the block.
    static func requestedThinking(_ thinking: JSONValue?, maxTokens: Int) throws -> ReasoningLevel?
    {
        guard let thinking else { return nil }
        guard case .object(let dict) = thinking, case .string(let type)? = dict["type"] else {
            throw invalid("thinking must be an object with a type", "thinking")
        }
        switch type {
        case "disabled":
            return .off
        case "adaptive":
            return nil
        case "enabled":
            guard case .integer(let budget)? = dict["budget_tokens"] else {
                throw invalid(
                    "thinking.budget_tokens is required when thinking is enabled",
                    "thinking.budget_tokens")
            }
            guard budget >= 1024 else {
                throw invalid("budget_tokens must be at least 1024", "thinking.budget_tokens")
            }
            guard budget < maxTokens else {
                throw invalid(
                    "budget_tokens must be less than max_tokens", "thinking.budget_tokens")
            }
            // Three rungs for the Qwen effort template (low|medium|xhigh); K2's
            // set ends at high and the request path maps the nearest level it
            // supports. Binary templates map any effort to "think". The
            // boundaries are the published budgets Claude Code itself sends:
            // 4k is its small setting, 16k its large.
            if budget < 4096 { return .low }
            if budget < 16384 { return .medium }
            return .xhigh
        default:
            throw invalid("thinking type must be enabled, adaptive or disabled", "thinking.type")
        }
    }

    /// This surface's `output_config.format`, reshaped into the Chat
    /// Completions `response_format` spelling the one validator parses.
    ///
    /// Anthropic's object carries the schema directly
    /// (`{"type": "json_schema", "schema": {...}}`); Chat Completions nests it
    /// under `json_schema`. Nil -- plain text -- is also what an unrecognized
    /// shape means, which is how it has always been treated.
    static func responseFormat(_ format: JSONValue?) throws -> JSONValue? {
        guard let format, case .object(let dict) = format,
            case .string(let type)? = dict["type"]
        else {
            return nil
        }
        switch type {
        case "text":
            return nil
        case "json_object":
            return format
        case "json_schema":
            guard let schema = dict["schema"] else {
                throw invalid("json_schema requires a schema", "output_config.format.schema")
            }
            var wrapper: [String: JSONValue] = ["schema": schema]
            if let name = dict["name"] { wrapper["name"] = name }
            return .object([
                "type": .string("json_schema"),
                "json_schema": .object(wrapper),
            ])
        default:
            throw unsupported(
                "output_config.format \(type) is not supported; use json_object or json_schema",
                "output_config.format.type")
        }
    }

    /// Build the chat-completions request for a Messages request, so the one
    /// validator and the one generation path serve both APIs. No server profile
    /// is taken: everything the served model contributes is applied by the
    /// validator, and the reasoning level is the request's own.
    package static func chatRequest(
        _ request: AnthropicMessagesRequest,
        maxContext: Int = Int.max
    ) throws -> OpenAIChatRequest {
        guard let requestedMaxTokens = request.maxTokens else {
            throw invalid("field required", "max_tokens")
        }
        guard requestedMaxTokens > 0 else {
            throw invalid("max_tokens must be greater than 0", "max_tokens")
        }
        // Clients set max_tokens to the model's ceiling (Claude Code sends
        // 32,000); a server with a smaller context window serves what it has
        // rather than refusing every request.
        let maxTokens = min(requestedMaxTokens, maxContext)
        if let temperature = request.temperature, !(0...1).contains(temperature) {
            throw invalid("temperature must be between 0 and 1", "temperature")
        }
        if let topP = request.topP, !(0...1).contains(topP) {
            throw invalid("top_p must be between 0 and 1", "top_p")
        }
        if request.outputFormat != nil {
            // Not the same field as `output_config.format`, which is the
            // spelling this surface's structured output uses. `output_format`
            // is a legacy name from a draft of the API; accepting it would be
            // guessing at a contract nobody publishes.
            throw unsupported(
                "output_format is not supported; structured output goes in output_config.format",
                "output_format")
        }
        // `output_config.format` is this surface's spelling of the same rule
        // Chat Completions puts in `response_format`; it is reshaped into that
        // spelling so the one validator parses it.
        let outputFormat: JSONValue?
        if case .object(let config)? = request.outputConfig {
            outputFormat = config["format"]
        } else {
            outputFormat = nil
        }
        let responseFormat = try AnthropicMapper.responseFormat(outputFormat)
        if request.container != nil {
            throw unsupported("containers are not supported", "container")
        }
        if request.mcpServers != nil {
            throw unsupported("MCP servers are not supported", "mcp_servers")
        }
        // context_management asks the API to clear old thinking or tool
        // results from the context. This server never renders earlier
        // thinking into a prompt and keeps every turn the client sends, so
        // the edits have nothing to do; accepting them keeps Claude Code,
        // which sends them always.
        //
        // The thinking block is the request's own reasoning level, on the same
        // per-request path the OpenAI surfaces use: `disabled` is off,
        // `enabled` maps its budget onto a rung, and `adaptive` (Claude Code's
        // every-turn default) leaves the server's setting alone.
        let thinkingLevel = try requestedThinking(request.thinking, maxTokens: maxTokens)
        guard !request.messages.isEmpty else {
            throw invalid("at least one message is required", "messages")
        }
        if request.messages.last(where: { $0.role != "system" })?.role == "assistant" {
            throw unsupported("a trailing assistant message (prefill) is not supported", "messages")
        }

        var messages: [OpenAIChatMessage] = []
        var systemParts: [String] = []
        if let system = try systemText(request.system), !system.isEmpty {
            systemParts.append(system)
        }
        for (index, message) in request.messages.enumerated() {
            // A system message inside the conversation (Claude Code's
            // mid-conversation guidance) joins the leading system block: the
            // chat template renders exactly one, ahead of the turns.
            if message.role == "system" {
                let text = try systemText(message.content) ?? ""
                if !text.isEmpty { systemParts.append(text) }
                continue
            }
            for chat in try chatMessages(for: message, index: index) {
                // Consecutive user text turns combine into one, as the API
                // itself documents; the template renders one turn per role.
                if chat.role == "user", let previous = messages.last, previous.role == "user",
                    case .text(let earlier)? = previous.content,
                    case .text(let later)? = chat.content
                {
                    messages[messages.count - 1] = OpenAIChatMessage(
                        role: "user", content: .text(earlier + "\n\n" + later),
                        toolCalls: nil, toolCallID: nil, name: nil)
                } else {
                    messages.append(chat)
                }
            }
        }
        if !systemParts.isEmpty {
            messages.insert(
                OpenAIChatMessage(
                    role: "system", content: .text(systemParts.joined(separator: "\n\n")),
                    toolCalls: nil, toolCallID: nil, name: nil), at: 0)
        }
        guard messages.contains(where: { $0.role != "system" }) else {
            throw invalid("at least one user message is required", "messages")
        }
        let stop: OpenAIStop? =
            (request.stopSequences?.isEmpty ?? true) ? nil : .many(request.stopSequences ?? [])
        return OpenAIChatRequest(
            model: request.model,
            messages: messages,
            stream: request.stream ?? false,
            streamOptions: nil,
            // Nil, not `GenerationDefaults`: see the same note in
            // `ResponsesAPIModels`. The validator fills each omitted field from
            // the loaded model's profile, and a hardcoded 0.6 here would override
            // a model whose card says otherwise.
            temperature: request.temperature,
            topP: request.topP,
            maxTokens: maxTokens,
            maxCompletionTokens: nil,
            stop: stop,
            seed: nil,
            tools: try tools(request.tools),
            toolChoice: try toolChoice(request.toolChoice),
            parallelToolCalls: nil,
            topK: request.topK,
            repetitionPenalty: nil,
            n: 1,
            logprobs: nil,
            presencePenalty: nil,
            frequencyPenalty: nil,
            reasoningEffort: thinkingLevel?.rawValue,
            responseFormat: responseFormat)
    }

    /// The count_tokens body, as a Messages request without generation.
    package static func chatRequest(counting request: AnthropicCountTokensRequest) throws
        -> OpenAIChatRequest
    {
        let full = AnthropicMessagesRequest(
            model: request.model, messages: request.messages, maxTokens: 1,
            system: request.system, metadata: nil, stopSequences: nil, stream: false,
            temperature: nil, topK: nil, topP: nil, tools: request.tools,
            toolChoice: request.toolChoice, thinking: nil, serviceTier: nil,
            outputConfig: nil, outputFormat: nil, container: nil, mcpServers: nil,
            contextManagement: nil)
        return try chatRequest(full)
    }
}
