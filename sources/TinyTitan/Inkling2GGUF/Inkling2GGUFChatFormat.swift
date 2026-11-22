import Foundation
import TinyTitanCore

/// Inkling 2's chat format, rendered as the model's bundled `chat_template.jinja`
/// renders it (the GGUF carries the same template, byte for byte):
///
/// - a `tool_declare` system message first, holding the offered tools as
///   sorted, compact JSON;
/// - a `Thinking effort level: <n>` system message before the first
///   non-system message (0 turns thinking off; 0.9 is the template default);
/// - each turn as `<|message_<role>|>[name]<|content_<kind>|>…<|end_message|>`,
///   a tool result's header carrying the tool's name;
/// - an assistant tool call as `<|message_model|><name><|content_invoke_tool_json|>
///   {"name":…,"args":{…}}<|end_message|>`, and every assistant turn closed by
///   `<|content_model_end_sampling|>`.
///
/// The template has no generation prompt: the model opens its own turn and
/// chooses thinking, text or a tool call. `openTextTurn` appends
/// `<|message_model|><|content_text|>` for the cases that must answer in text
/// (thinking off with no tools, or a JSON-constrained answer).
package enum Inkling2GGUFChatFormat {
    /// The template's `effort_map`, keyed by this project's effort levels.
    package static func effortValue(_ effort: ModelReasoningEffort?) -> String {
        switch effort {
        case nil: return "0"
        case .low: return "0.2"
        case .medium: return "0.7"
        case .high: return "0.9"
        case .xhigh: return "0.99"
        }
    }

    package static func render(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        effort: ModelReasoningEffort?,
        openTextTurn: Bool
    ) -> String {
        var out = ""
        if !tools.isEmpty {
            let specs = JSONValue.array(
                tools.map { tool in
                    .object([
                        "description": .string(tool.description),
                        "name": .string(tool.name),
                        "parameters": tool.parameters,
                        "type": .string("function"),
                    ])
                })
            out +=
                "<|message_system|>tool_declare<|content_xml|>\(canonicalJSON(specs))<|end_message|>"
        }
        let effortMessage =
            "<|message_system|><|content_text|>Thinking effort level: "
            + effortValue(effort) + "<|end_message|>"
        var effortEmitted = false
        for message in messages {
            let role = message.role
            if !effortEmitted, role != .system, role != .developer {
                out += effortMessage
                effortEmitted = true
            }
            out += renderTurn(message, history: messages)
        }
        if !effortEmitted { out += effortMessage }
        if openTextTurn { out += "<|message_model|><|content_text|>" }
        return out
    }

    private static func renderTurn(
        _ message: GFTokenizer.Message, history: [GFTokenizer.Message]
    ) -> String {
        var header: String
        switch message.role {
        case .system, .developer: header = "<|message_system|>"
        case .user: header = "<|message_user|>"
        case .assistant: header = "<|message_model|>"
        case .tool: header = "<|message_tool|>"
        }
        if message.role == .tool {
            if let name = message.name, !name.isEmpty {
                header += name
            } else if let id = message.toolCallID, !id.isEmpty,
                let call = history.lazy.filter({ $0.role == .assistant })
                    .flatMap(\.toolCalls).first(where: { $0.id == id })
            {
                header += call.name
            }
        } else if message.role == .assistant, let name = message.name, !name.isEmpty {
            header += name
        }
        let isAssistant = message.role == .assistant
        var out = ""
        if isAssistant, let reasoning = message.reasoning, !reasoning.isEmpty {
            out += header + "<|content_thinking|>" + reasoning + "<|end_message|>"
        }
        if let content = message.content {
            let carriesOther =
                isAssistant && (!message.toolCalls.isEmpty || !(message.reasoning ?? "").isEmpty)
            if !content.isEmpty || !carriesOther {
                out += header + "<|content_text|>" + content + "<|end_message|>"
            }
        }
        if isAssistant {
            for call in message.toolCalls {
                out += "<|message_model|>\(call.name)<|content_invoke_tool_json|>"
                out += "{\"name\":\(canonicalJSON(.string(call.name))),\"args\":"
                out += canonicalJSON(argumentsObject(call.arguments)) + "}<|end_message|>"
            }
            out += "<|content_model_end_sampling|>"
        }
        return out
    }

    /// A call's arguments as the object the template requires; a JSON string
    /// that holds an object (how a client may replay it) is parsed first.
    static func argumentsObject(_ value: JSONValue) -> JSONValue {
        switch value {
        case .object: return value
        case .string(let text):
            if let data = text.data(using: .utf8),
                let parsed = try? JSONDecoder().decode(JSONValue.self, from: data),
                case .object = parsed
            {
                return parsed
            }
            return .object([:])
        default: return .object([:])
        }
    }

    /// Compact JSON with keys sorted by code point and non-ASCII kept as is:
    /// what the template's `tojson(sort_keys=true, separators=(",", ":"))`
    /// produces through transformers' `json.dumps(ensure_ascii=False)`.
    package static func canonicalJSON(_ value: JSONValue) -> String {
        switch value {
        case .null: return "null"
        case .bool(let flag): return flag ? "true" : "false"
        case .integer(let number): return String(number)
        case .unsignedInteger(let number): return String(number)
        case .decimal(let number): return "\(number)"
        case .number(let number):
            guard number.isFinite else { return "null" }
            return "\(number)"
        case .string(let text): return quoted(text)
        case .array(let items): return "[" + items.map(canonicalJSON).joined(separator: ",") + "]"
        case .object(let fields):
            let keys = fields.keys.sorted {
                Array($0.unicodeScalars.map(\.value)).lexicographicallyPrecedes(
                    $1.unicodeScalars.map(\.value))
            }
            return "{"
                + keys.map { quoted($0) + ":" + canonicalJSON(fields[$0] ?? .null) }
                .joined(separator: ",") + "}"
        }
    }

    private static func quoted(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}
