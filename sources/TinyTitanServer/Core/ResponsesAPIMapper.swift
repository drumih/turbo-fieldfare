import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

// The Responses API to chat mapping: request fields onto the chat request,
// tools and sampling that the server can honour.
//
// Split out of `ResponsesAPIModels.swift` (2026-09-28) under the 500-line-per-
// file rule (Task 8 of the cleanup runbook) as pure code motion.
// MARK: - Responses -> chat mapping

public enum ResponsesAPIMapper {
    /// Text of a message's content: a plain string or an array of text parts.
    /// `input_text`, `output_text` and `refusal` parts carry text; images and
    /// files are refused because TinyTitan is text-only.
    public static func messageText(_ content: JSONValue?, param: String = "input") throws -> String
    {
        guard let content else { return "" }
        switch content {
        case .string(let text):
            return text
        case .array(let parts):
            var out = ""
            for part in parts {
                guard case .object(let dict) = part,
                    case .string(let type)? = dict["type"]
                else {
                    throw ServerRequestError.invalid(
                        message: "content parts must be objects with a type",
                        param: param, code: "invalid_value")
                }
                switch type {
                case "input_text", "output_text":
                    if case .string(let text)? = dict["text"] { out += text }
                case "refusal":
                    if case .string(let text)? = dict["refusal"] { out += text }
                case "input_image", "input_file", "input_audio":
                    throw ServerRequestError.invalid(
                        message: "\(type) parts are not supported; this server is text-only",
                        param: param, code: "unsupported_content")
                default:
                    throw ServerRequestError.invalid(
                        message: "unsupported content part type \(type)",
                        param: param, code: "unsupported_content")
                }
            }
            return out
        default:
            throw ServerRequestError.invalid(
                message: "content must be a string or an array of parts",
                param: param, code: "invalid_value")
        }
    }

    /// This surface's `text.format`, reshaped into the Chat Completions
    /// `response_format` spelling the one validator parses.
    ///
    /// OpenAI's two spellings differ only in where the schema lives: the
    /// Responses object puts `name`, `schema` and `strict` at the top level of
    /// the format, Chat Completions nests them under `json_schema`. Nothing
    /// else about the rule differs, so nothing else is duplicated.
    public static func responseFormat(_ format: JSONValue?) throws -> JSONValue? {
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
            var wrapper: [String: JSONValue] = [:]
            for key in ["name", "schema", "strict"] where dict[key] != nil {
                wrapper[key] = dict[key]
            }
            guard wrapper["schema"] != nil else {
                throw ServerRequestError.invalid(
                    message: "json_schema requires a schema",
                    param: "text.format.schema", code: "invalid_value")
            }
            return .object([
                "type": .string("json_schema"),
                "json_schema": .object(wrapper),
            ])
        default:
            throw ServerRequestError.invalid(
                message: "text.format \(type) is not supported; use text, json_object "
                    + "or json_schema",
                param: "text.format", code: "unsupported_value")
        }
    }

    /// The text of a function_call_output's `output`: a string, or a list of
    /// `input_text` parts.
    public static func outputText(_ output: JSONValue?) throws -> String {
        guard let output else { return "" }
        if case .string(let text) = output { return text }
        return try messageText(output, param: "input.output")
    }

    /// Refuse the request features this server cannot honour, before any
    /// mapping. Each refusal names the field and says why, in the API's own
    /// error shape, instead of silently generating something else.
    public static func validateFeatures(_ request: ResponsesAPIRequest) throws {
        if request.background == true {
            throw ServerRequestError.invalid(
                message: "background responses are not supported",
                param: "background", code: "unsupported_value")
        }
        if let topLogprobs = request.topLogprobs, topLogprobs > 0 {
            throw ServerRequestError.invalid(
                message: "logprobs are not supported", param: "top_logprobs",
                code: "unsupported_value")
        }
        if request.prompt != nil {
            throw ServerRequestError.invalid(
                message: "prompt templates are not supported", param: "prompt",
                code: "unsupported_value")
        }
        if request.conversation != nil {
            throw ServerRequestError.invalid(
                message: "conversations are not supported; use previous_response_id",
                param: "conversation", code: "unsupported_value")
        }
        if let include = request.include {
            let supported: Set<String> = [
                "message.output_text.logprobs", "reasoning.encrypted_content",
            ]
            if let bad = include.first(where: { !supported.contains($0) }) {
                throw ServerRequestError.invalid(
                    message: "include value \(bad) is not supported",
                    param: "include", code: "unsupported_value")
            }
        }
    }

    /// The function tools of a request, with namespaced functions flattened
    /// into the list and remembered by namespace. Hosted tool types
    /// (web_search, file_search, code_interpreter, mcp, image_generation,
    /// computer use, shell, apply_patch) have nothing on this server to run
    /// them and are left out; the model never sees them and never calls
    /// them. Codex sends web_search on every turn, so refusing would refuse
    /// Codex.
    package static func functionTools(_ tools: [ResponsesAPIRequest.Tool]?)
        -> (tools: [OpenAITool], namespaces: [String: String])
    {
        var out: [OpenAITool] = []
        var namespaces: [String: String] = [:]
        func add(_ tool: ResponsesAPIRequest.Tool, namespace: String?) {
            guard tool.type == "function", let name = tool.name, !name.isEmpty else { return }
            let parameters: JSONValue =
                tool.parameters
                ?? .object(["type": .string("object"), "properties": .object([:])])
            out.append(
                OpenAITool(
                    type: "function",
                    function: OpenAIFunctionDefinition(
                        name: name, description: tool.description,
                        parameters: parameters)))
            if let namespace { namespaces[name] = namespace }
        }
        for tool in tools ?? [] {
            if tool.type == "namespace" {
                for nested in tool.tools ?? [] { add(nested, namespace: tool.name) }
            } else {
                add(tool, namespace: nil)
            }
        }
        return (out, namespaces)
    }

    /// The conversation a Responses request renders to.
    ///
    /// Instructions and developer guidance become one leading system message —
    /// TinyTitan's chat template requires exactly that and rejects the developer
    /// role — followed by the items in order. Shared by `/v1/responses` and
    /// `/v1/responses/compact`, so the two cannot disagree about what an item
    /// means, and a compacted window replays through exactly the path the
    /// original items did.
    package static func chatMessages(
        items: [ResponsesAPIRequest.Item],
        instructions: String?
    ) throws -> [OpenAIChatMessage] {
        var systemParts: [String] = []
        if let instructions, !instructions.isEmpty {
            systemParts.append(instructions)
        }
        var messages: [OpenAIChatMessage] = []
        for item in items {
            guard let kind = item.resolvedType else {
                throw ServerRequestError.invalid(
                    message: "unsupported input item; cannot determine its type",
                    param: "input", code: "unsupported_input")
            }
            switch kind {
            case "message":
                let role = item.role ?? "user"
                let text = try messageText(item.content)
                if role == "system" || role == "developer" {
                    if !text.isEmpty { systemParts.append(text) }
                } else {
                    messages.append(
                        OpenAIChatMessage(
                            role: role, content: .text(text),
                            toolCalls: nil, toolCallID: nil, name: nil))
                }
            case "function_call":
                guard let callID = item.callID, !callID.isEmpty else {
                    throw ServerRequestError.invalid(
                        message: "function_call item requires call_id",
                        param: "input", code: "invalid_value")
                }
                let function = OpenAIFunctionCall(
                    name: item.name ?? "", arguments: item.arguments ?? "{}")
                messages.append(
                    OpenAIChatMessage(
                        role: "assistant", content: nil,
                        toolCalls: [
                            OpenAIToolCall(id: callID, type: "function", function: function)
                        ],
                        toolCallID: nil, name: nil))
            case "function_call_output":
                messages.append(
                    OpenAIChatMessage(
                        role: "tool", content: .text(try outputText(item.output)),
                        toolCalls: nil, toolCallID: item.callID, name: nil))
            case "compaction":
                // The window `/v1/responses/compact` returned, sent back as the
                // base input of a new response. It becomes standing context, and
                // a payload this server cannot read is refused rather than
                // dropped: silently discarding it would lose the history the
                // client just paid to compact.
                guard let payload = item.encryptedContent else {
                    throw ServerRequestError.invalid(
                        message: "compaction item requires encrypted_content",
                        param: "input", code: "invalid_value")
                }
                systemParts.append(
                    ServerCompaction.replayNote(try ServerCompaction.decode(payload)))
            case "reasoning":
                // A client replaying an earlier turn returns the reasoning
                // item it was given. The model's thoughts are never part of
                // its prompt, so there is nothing to render; accept and skip.
                continue
            case "item_reference":
                throw ServerRequestError.invalid(
                    message: "item_reference \(item.id ?? "") could not be resolved",
                    param: "input", code: "item_not_found")
            default:
                throw ServerRequestError.invalid(
                    message: "unsupported input item type \(kind)",
                    param: "input", code: "unsupported_input")
            }
        }
        var chatMessages = messages
        if !systemParts.isEmpty {
            chatMessages.insert(
                OpenAIChatMessage(
                    role: "system", content: .text(systemParts.joined(separator: "\n\n")),
                    toolCalls: nil, toolCallID: nil, name: nil), at: 0)
        }
        return chatMessages
    }

    /// Build the chat-completions request equivalent to a responses request.
    /// TinyTitan's chat template requires exactly one leading system message and
    /// rejects the developer role, so instructions and developer guidance are
    /// merged into a single opening system message. `priorItems` is the
    /// conversation a `previous_response_id` resolved to; it precedes the
    /// request's own input.
    package static func chatRequest(
        _ request: ResponsesAPIRequest,
        priorItems: [ResponsesAPIRequest.Item] = [],
        inputItems: [ResponsesAPIRequest.Item]? = nil
    ) throws -> OpenAIChatRequest {
        try validateFeatures(request)
        let chatMessages = try chatMessages(
            items: priorItems + (inputItems ?? request.inputItems),
            instructions: request.instructions)
        let tools = functionTools(request.tools).tools
        let responseFormat = try responseFormat(request.text?.format)
        return OpenAIChatRequest(
            model: request.model,
            messages: chatMessages,
            stream: request.stream ?? false,
            streamOptions: nil,
            // Sampling the request omitted stays nil on purpose. Filling it here
            // with the generic `GenerationDefaults` makes the served model's own
            // defaults unreachable: the validator resolves each field as
            // `request.value ?? sampling.value`, where `sampling` is the loaded
            // model's profile. Qwen3.8-Flash-Next's card says temperature 1.0, so
            // a fixed 0.6 here sampled it wrong on this surface while
            // /v1/chat/completions honoured the profile.
            temperature: request.temperature,
            topP: request.topP,
            // Codex and OpenCode omit max_output_tokens; forward nil so the
            // chat validator applies its context-bounded default (no
            // artificial output cap) instead of a fixed token budget.
            maxTokens: request.maxOutputTokens,
            maxCompletionTokens: nil,
            stop: nil,
            seed: nil,
            tools: tools.isEmpty ? nil : tools,
            // The chat validator knows every tool_choice form the server
            // honours ("auto", "none") and refuses the rest by name.
            toolChoice: request.toolChoice,
            // Codex sends parallel_tool_calls=false on every turn. The
            // decoder cannot promise a single call per turn, and refusing
            // would refuse Codex; the value is echoed and not enforced.
            parallelToolCalls: nil,
            topK: request.topK,
            repetitionPenalty: nil,
            n: 1,
            logprobs: nil,
            presencePenalty: request.presencePenalty,
            frequencyPenalty: nil,
            reasoningEffort: request.reasoning?.effort,
            responseFormat: responseFormat)
    }

    /// A finished response's output, in the shape a later request carries it
    /// back as input. This is what `previous_response_id` chains on. The
    /// reasoning item is left out: a replayed one is skipped on the way back
    /// in, because thoughts are never part of a prompt.
    package static func outputAsInput(
        completion: ServerCompletion,
        responseID: String,
        namespaces: [String: String] = [:]
    ) -> [ResponsesAPIRequest.Item] {
        var items: [ResponsesAPIRequest.Item] = []
        let ids = ResponsesAPIBuilder.itemIDs(responseID: responseID, completion: completion)
        if !completion.content.isEmpty {
            items.append(
                ResponsesAPIRequest.Item(
                    type: "message", id: ids.message, status: "completed", role: "assistant",
                    content: .array([
                        .object([
                            "type": .string("output_text"),
                            "text": .string(completion.content),
                            "annotations": .array([]),
                        ])
                    ])))
        }
        for (index, call) in completion.toolCalls.enumerated() {
            items.append(
                ResponsesAPIRequest.Item(
                    type: "function_call", id: ids.calls[index], status: "completed",
                    callID: call.id, name: call.name, arguments: call.argumentsJSON,
                    namespace: namespaces[call.name]))
        }
        return items
    }
}
