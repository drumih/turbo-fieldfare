import Foundation
import TinyTitan
import TinyTitanCore

// The nested input the validator has to walk: a `response_format`, a tool's
// parameter schema and its depth cap, and the message history with its
// tool-call pairing.
//
// Split out of `OpenAIRequestValidator.swift` (2026-10-06) under the
// 500-line-per-file rule as pure code motion. `structuredOutputSchema` was
// already internal, and it kept its own name; the rest of the move is the
// four `private` helpers the cascade in `validate` still calls, and the
// `invalid` constructor they share, became `internal`: `private` is file-scoped
// in Swift, so that is the price of the move. All of it stays module-internal —
// no `public` surface moved, and no behaviour changed.
extension OpenAIRequestValidator {
    /// The compiled schema a `response_format` asks for, or nil for plain text.
    ///
    /// Chat Completions' own spelling is the one parsed. The Responses surface
    /// and the Messages API reshape theirs into it before validation, so the
    /// rule lives in exactly one place: `{"type": "text"}` (the API's own
    /// default) and an unrecognized *shape* stay accepted, a `json_object`
    /// means an object at the top level, and a `json_schema` is compiled by
    /// `JSONSchemaNode` -- which refuses, by name, every keyword a byte-level
    /// grammar cannot promise.
    static func structuredOutputSchema(_ format: JSONValue?) throws -> JSONSchemaNode? {
        guard let format, case .object(let dict) = format,
            case .string(let type)? = dict["type"]
        else {
            return nil
        }
        switch type {
        case "text":
            return nil
        case "json_object":
            return .object(properties: [:], required: [], additional: true)
        case "json_schema":
            guard case .object(let wrapper)? = dict["json_schema"],
                let schema = wrapper["schema"]
            else {
                throw invalid(
                    "json_schema requires json_schema.schema",
                    "response_format.json_schema.schema", "invalid_value")
            }
            do {
                return try JSONSchemaNode.compile(schema)
            } catch let error as JSONSchemaCompileError {
                throw invalid(
                    error.description, "response_format.json_schema.schema",
                    "unsupported_value")
            }
        default:
            throw invalid(
                "response_format \(type) is not supported; use text, json_object or json_schema",
                "response_format", "unsupported_value")
        }
    }

    static func validateTool(_ tool: OpenAITool) throws -> GFTokenizer.FunctionDefinition {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard name.range(of: #"^[A-Za-z0-9_-]{1,64}$"#, options: .regularExpression) != nil else {
            throw invalid(
                "tool name must match [A-Za-z0-9_-]{1,64}",
                "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid(
                "tool parameters must be an object schema",
                "tools", "invalid_tool_schema")
        }
        try validateSchemaKeys(tool.function.parameters)
        let parameters = tool.function.parameters
        guard (try? parameters.jinjaSendableValue()) != nil else {
            throw invalid(
                "tool schema contains a number that cannot be represented exactly",
                "tools", "invalid_tool_schema")
        }
        return GFTokenizer.FunctionDefinition(
            name: name,
            description: tool.function.description ?? "",
            parameters: parameters)
    }

    /// Walk a tool's parameter schema. `depth` bounds the walk: the schema is
    /// network input, and a recursive check with no ceiling of its own is a
    /// check whose limit belongs to whatever parsed the document.
    static func validateSchemaKeys(
        _ schema: JSONValue, depth: Int = 0
    ) throws {
        switch schema {
        case .object(let object):
            try requireSchemaDepth(depth)
            for (schemaKey, value) in object {
                if schemaKey == "properties" {
                    guard case .object(let definitions) = value else {
                        throw invalid(
                            "tool schema properties must be an object",
                            "tools", "invalid_tool_schema")
                    }
                    for (_, definition) in definitions {
                        // ChatML tool-call parameter names are free-form;
                        // only the schema structure itself is validated.
                        try validateSchemaKeys(definition, depth: depth + 1)
                    }
                } else {
                    try validateSchemaKeys(value, depth: depth + 1)
                }
            }
        case .array(let values):
            try requireSchemaDepth(depth)
            for value in values {
                try validateSchemaKeys(value, depth: depth + 1)
            }
        default:
            // Only a container can deepen the walk. Counting a scalar would make
            // this cap mean something other than the same document's schema cap.
            break
        }
    }

    static func requireSchemaDepth(_ depth: Int) throws {
        guard depth <= JSONSchemaNode.maximumNestingDepth else {
            throw invalid(
                "tool schema nests deeper than \(JSONSchemaNode.maximumNestingDepth) levels",
                "tools", "invalid_tool_schema")
        }
    }

    static func validateMessages(
        _ input: [OpenAIChatMessage],
        rules: RequestRules
    ) throws -> [GFTokenizer
        .Message]
    {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        // The thousand-message ceiling is the wire's; a local caller's history
        // is its own business.
        if rules == .wire {
            guard input.count <= 1000 else {
                throw invalid(
                    "message count exceeds maximum of 1000",
                    "messages", "value_too_large")
            }
        }
        var knownCalls: [String: (name: String, resolved: Bool)] = [:]
        var result: [GFTokenizer.Message] = []
        var sawConversationMessage = false
        for message in input {
            guard let role = GFTokenizer.Role(rawValue: message.role) else {
                throw invalid(
                    "unsupported message role \(message.role)",
                    "messages", "invalid_message")
            }
            if role == .system || role == .developer {
                guard !sawConversationMessage else {
                    throw invalid(
                        "system or developer guidance must precede the conversation",
                        "messages", "invalid_message")
                }
            } else {
                sawConversationMessage = true
            }
            let content = try message.content?.textValue()
            let calls: [GFTokenizer.HistoricalToolCall] = try (message.toolCalls ?? []).map {
                call in
                guard role == .assistant, call.type == "function",
                    !call.id.isEmpty, knownCalls[call.id] == nil,
                    call.function.name.range(
                        of: #"^[A-Za-z0-9_-]{1,64}$"#,
                        options: .regularExpression) != nil
                else {
                    throw invalid(
                        "invalid or duplicate historical tool call",
                        "messages", "invalid_tool_call")
                }
                let data = Data(call.function.arguments.utf8)
                let arguments = try JSONDecoder().decode(JSONValue.self, from: data)
                guard arguments.objectValue != nil else {
                    throw invalid(
                        "historical tool arguments must be a JSON object",
                        "messages", "invalid_tool_arguments")
                }
                guard (try? arguments.jinjaSendableValue()) != nil else {
                    throw invalid(
                        "historical tool arguments cannot be represented exactly",
                        "messages",
                        "invalid_tool_arguments")
                }
                knownCalls[call.id] = (call.function.name, false)
                return GFTokenizer.HistoricalToolCall(
                    id: call.id, name: call.function.name, arguments: arguments)
            }
            if role == .tool {
                guard let id = message.toolCallID,
                    let call = knownCalls[id], !call.resolved
                else {
                    throw invalid(
                        "tool result must reference one unresolved call",
                        "messages", "invalid_tool_result")
                }
                knownCalls[id] = (call.name, true)
                guard content != nil else {
                    throw invalid(
                        "tool result content is required",
                        "messages", "invalid_tool_result")
                }
            } else if content == nil && calls.isEmpty {
                throw invalid(
                    "message content is required",
                    "messages", "invalid_message")
            }
            result.append(
                GFTokenizer.Message(
                    role: role,
                    content: content,
                    toolCalls: calls,
                    toolCallID: message.toolCallID,
                    name: message.name))
        }
        // S19: a conversation that ends with an assistant tool call that is
        // never answered by a tool result would resume from an unanswerable
        // state; reject it instead of generating tool-response markup.
        if knownCalls.contains(where: { !$0.value.resolved }) {
            throw invalid(
                "conversation ends with an unresolved tool call",
                "messages", "invalid_tool_call")
        }
        return result
    }
}
