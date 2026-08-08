import Foundation

public enum K3ChatRendererError: Error, CustomStringConvertible {
    /// `normalize_tool_arguments`: an arguments string parsed as JSON that is
    /// not an object.
    case nonObjectToolArguments
    /// A tool message whose name cannot be recovered from `tool`/`name` or
    /// the preceding assistant `toolCalls` by order.
    case unresolvableToolName

    public var description: String {
        switch self {
        case .nonObjectToolArguments:
            return "Kimi K3 tool call arguments must be a JSON object."
        case .unresolvableToolName:
            return "Kimi K3 tool messages need a resolvable tool name: carry `tool`/`name`, "
                + "or match a preceding assistant tool_call by order."
        }
    }
}

/// One rendered unit: `text` plus whether literal special-token strings in it
/// may encode as their special ids. Structural markers are `allowSpecial`;
/// user/tool text never is. Segments are emitted at the same granularity as
/// the reference `encoding_k3.py` so per-segment encoding reproduces the
/// reference token stream exactly — do not coalesce adjacent segments.
public struct K3RenderSegment: Sendable, Equatable {
    public var text: String
    public var allowSpecial: Bool

    public init(text: String, allowSpecial: Bool) {
        self.text = text
        self.allowSpecial = allowSpecial
    }
}

public enum K3ChatRole: String, Sendable, Equatable {
    case system, user, assistant, tool
}

/// Message content: a plain string, or typed text parts (OpenAI-style
/// `[{"type": "text", "text": …}]`). Image parts from the reference
/// implementation are not modeled — K3 here is text-only.
public enum K3ChatContent: Sendable, Equatable {
    case text(String)
    case parts([String])
}

/// Tool-call arguments as delivered by chat APIs: already-parsed objects, or
/// the OpenAI-style JSON string (parsed at render time; an unparseable string
/// renders as a raw `<json type="object">` block, matching
/// `normalize_tool_arguments`).
public enum K3ToolCallArguments: Sendable, Equatable {
    case object(JSONValue)
    case jsonString(String)

    public static var empty: Self { .object(.object([:])) }
}

public struct K3ToolCall: Sendable, Equatable {
    public var id: String?
    public var name: String
    public var arguments: K3ToolCallArguments

    public init(id: String? = nil, name: String, arguments: K3ToolCallArguments = .empty) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

/// One tool spec. Renders as the OpenAI shape
/// `{"function": {"description": …, "name": …, "parameters": …}, "type": "function"}`;
/// key order is irrelevant because rendering deep-sorts, like the reference.
public struct K3ToolDeclaration: Sendable, Equatable {
    public var name: String
    public var description: String
    public var parameters: JSONValue

    public init(name: String, description: String, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    var jsonValue: JSONValue {
        .object([
            "function": .object([
                "description": .string(description),
                "name": .string(name),
                "parameters": parameters,
            ]),
            "type": .string("function"),
        ])
    }
}

public struct K3ChatMessage: Sendable, Equatable {
    public var role: K3ChatRole
    public var content: K3ChatContent?
    /// `name` attribute on the message tag (also a legacy tool-name carrier
    /// on tool messages).
    public var name: String?
    /// Assistant reasoning channel (`reasoning_content`).
    public var reasoningContent: String?
    public var toolCalls: [K3ToolCall]
    /// Explicit tool name on a tool-result message (wins over `name`).
    public var tool: String?
    /// `tool_call_id` on a tool-result message; matched against the preceding
    /// assistant `toolCalls` for re-sorting.
    public var toolCallID: String?
    /// Dynamic tool specs on a system message (renders as a dynamic
    /// tool-declare, replacing the plain system body).
    public var tools: [K3ToolDeclaration]?

    public init(role: K3ChatRole,
                content: K3ChatContent? = nil,
                name: String? = nil,
                reasoningContent: String? = nil,
                toolCalls: [K3ToolCall] = [],
                tool: String? = nil,
                toolCallID: String? = nil,
                tools: [K3ToolDeclaration]? = nil) {
        self.role = role
        self.content = content
        self.name = name
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.tool = tool
        self.toolCallID = toolCallID
        self.tools = tools
    }
}

public enum K3ThinkingEffort: String, Sendable, Equatable {
    case low, high, max
}

public enum K3ToolChoice: Sendable, Equatable {
    case required, none
}

public enum K3ResponseFormat: Sendable, Equatable {
    case jsonObject
    /// The schema (`json_schema.schema` in the reference) carried directly.
    case jsonSchema(JSONValue)
}

public struct K3ChatOptions: Sendable, Equatable {
    public var tools: [K3ToolDeclaration]
    public var thinking: Bool
    public var thinkingEffort: K3ThinkingEffort?
    public var toolChoice: K3ToolChoice?
    public var responseFormat: K3ResponseFormat?
    public var addGenerationPrompt: Bool

    public init(tools: [K3ToolDeclaration] = [],
                thinking: Bool = true,
                thinkingEffort: K3ThinkingEffort? = nil,
                toolChoice: K3ToolChoice? = nil,
                responseFormat: K3ResponseFormat? = nil,
                addGenerationPrompt: Bool = true) {
        self.tools = tools
        self.thinking = thinking
        self.thinkingEffort = thinkingEffort
        self.toolChoice = toolChoice
        self.responseFormat = responseFormat
        self.addGenerationPrompt = addGenerationPrompt
    }
}

/// Swift port of `build_chat_segments` from the official `encoding_k3.py`
/// (Kimi K3 XTML chat encoding). Covers: tool-result re-sorting by
/// `tool_call_id`, deep key-sorted compact JSON for tool/schema dumps,
/// `_xtml_value`/`_xtml_type` argument rendering, assistant think/response
/// channels, static and dynamic tool-declare system messages, internal
/// system messages for thinking-effort / tool-choice / response-format, and
/// the generation prompt.
///
/// Known, deliberate deviations from the Python reference:
/// - Tool-call argument objects and `_xtml_value` JSON enumerate the house
///   `JSONValue` dictionary, whose order is not the JSON insertion order
///   Python preserves. Fixtures pin only single-key arguments.
/// - Image content parts / `image_prompts` are not modeled (text-only port).
public enum K3ChatRenderer {
    private static let openToken = "<|open|>"
    private static let closeToken = "<|close|>"
    private static let sepToken = "<|sep|>"
    private static let endOfMsgToken = "<|end_of_msg|>"

    /// Render a conversation to XTML segments.
    public static func buildSegments(
        messages input: [K3ChatMessage],
        options: K3ChatOptions = .init()
    ) throws -> [K3RenderSegment] {
        // Re-sort tool results by tool_call_id at the lowest layer, like the
        // reference; the helper is side-effect free (messages are values).
        let messages = normalizeToolResultMessages(input)
        var segments: [K3RenderSegment] = []

        if !options.tools.isEmpty {
            segments += renderToolDeclare(options.tools, dynamic: false)
        }

        if options.thinking, let effort = options.thinkingEffort {
            segments += internalSystemMessage(
                "thinking-effort",
                "`thinking_effort` guides on how much to think in your "
                    + "thinking channel (not including the response channel), "
                    + "supported values include `low`, `medium`, `high`, and `max`.\n"
                    + "Now the system is invoked with `thinking_effort=\(effort.rawValue)`.")
        }

        var toolCalls: [K3ToolCall] = []
        var toolIndex = 0

        for message in messages {
            switch message.role {
            case .user:
                segments += openTag("message", messageAttrs("user", name: message.name))
                segments += renderContent(message.content)
                segments += closeTag("message")
                segments += endOfMsg()
            case .system:
                if let tools = message.tools, !tools.isEmpty {
                    segments += renderToolDeclare(tools, dynamic: true)
                } else {
                    segments += openTag("message", messageAttrs("system", name: message.name))
                    segments += renderContent(message.content)
                    segments += closeTag("message")
                    segments += endOfMsg()
                }
            case .tool:
                toolIndex += 1
                var toolName = message.tool ?? message.name
                if toolName == nil, toolIndex <= toolCalls.count {
                    toolName = toolCalls[toolIndex - 1].name
                }
                guard let toolName else {
                    throw K3ChatRendererError.unresolvableToolName
                }
                segments += openTag("message", [
                    ("role", "tool"), ("tool", toolName), ("index", "\(toolIndex)"),
                ])
                segments += renderContent(message.content)
                segments += closeTag("message")
                segments += endOfMsg()
            case .assistant:
                toolCalls = message.toolCalls
                toolIndex = 0
                segments += openTag("message", messageAttrs("assistant", name: message.name))
                segments += try renderAssistant(message, thinking: options.thinking)
                segments += closeTag("message")
                segments += endOfMsg()
            }
        }

        switch options.toolChoice {
        case .required:
            segments += internalSystemMessage(
                "tool-choice",
                "The system is invoked with `tool_choice=required`.\n"
                    + "You MUST call tools in the next message.")
        case .some(.none):
            segments += internalSystemMessage(
                "tool-choice",
                "The system is invoked with `tool_choice=none`.\n"
                    + "You MUST NOT call any tools in the next message.")
        case nil:
            break
        }

        switch options.responseFormat {
        case .jsonObject:
            segments += internalSystemMessage(
                "response-format",
                "The system is invoked with `response_format=json_object`.\n"
                    + "Your response must be raw JSON data without markdown code "
                    + "blocks (```json) or any additional formatting.")
        case .jsonSchema(let schema):
            segments += internalSystemMessage(
                "response-format",
                "The system is invoked with `response_format=json_schema`.\n"
                    + "Your response must be raw JSON data without markdown code "
                    + "blocks (```json) or any additional formatting.\n"
                    + "The JSON data must match the following schema:\n"
                    + "```json\n\(K3JSON.compact(schema))\n```")
        case nil:
            break
        }

        if options.addGenerationPrompt {
            segments += openTag("message", [("role", "assistant")])
            segments += openTag(options.thinking ? "think" : "response")
        }
        return segments
    }

    /// Rendered XTML as a single string (segment texts concatenated).
    public static func render(
        messages: [K3ChatMessage],
        options: K3ChatOptions = .init()
    ) throws -> String {
        try buildSegments(messages: messages, options: options)
            .map(\.text).joined()
    }

    /// Render then encode segment-by-segment, honoring each segment's
    /// `allowSpecial` — the same stream the reference tokenizer produces.
    public static func renderToIDs(
        messages: [K3ChatMessage],
        options: K3ChatOptions = .init(),
        tokenizer: K3Tokenizer
    ) throws -> [Int] {
        var ids: [Int] = []
        for segment in try buildSegments(messages: messages, options: options) {
            ids.append(contentsOf: try tokenizer.encode(
                segment.text, allowSpecial: segment.allowSpecial))
        }
        return ids
    }

    // MARK: - Segment primitives (mirroring encoding_k3.py)

    private static func text(_ text: String) -> [K3RenderSegment] {
        guard !text.isEmpty else { return [] }
        return [K3RenderSegment(text: text, allowSpecial: false)]
    }

    private static func control(_ text: String) -> [K3RenderSegment] {
        guard !text.isEmpty else { return [] }
        return [K3RenderSegment(text: text, allowSpecial: true)]
    }

    private static func escapeAttrValue(_ value: String) -> String {
        value.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }

    private static func attr(_ key: String, _ value: String) -> [K3RenderSegment] {
        text(" \(key)") + text("=\"") + text(escapeAttrValue(value)) + text("\"")
    }

    private static func openTag(_ tag: String, _ attrs: [(String, String)] = []) -> [K3RenderSegment] {
        var segments: [K3RenderSegment] = []
        segments += control(openToken)
        segments += text(tag)
        for (key, value) in attrs {
            segments += attr(key, value)
        }
        segments += control(sepToken)
        return segments
    }

    private static func closeTag(_ tag: String) -> [K3RenderSegment] {
        control(closeToken) + text(tag) + control(sepToken)
    }

    private static func endOfMsg() -> [K3RenderSegment] {
        control(endOfMsgToken)
    }

    private static func messageAttrs(_ role: String, name: String?) -> [(String, String)] {
        var attrs = [("role", role)]
        if let name {
            attrs.append(("name", name))
        }
        return attrs
    }

    private static func renderContent(_ content: K3ChatContent?) -> [K3RenderSegment] {
        guard let content else { return [] }
        switch content {
        case .text(let string):
            return text(string)
        case .parts(let parts):
            var segments: [K3RenderSegment] = []
            for part in parts {
                segments += text(part)
            }
            return segments
        }
    }

    private static func internalSystemMessage(_ type: String, _ body: String) -> [K3RenderSegment] {
        openTag("message", [("role", "system"), ("type", type)])
            + text(body.trimmingCharacters(in: .whitespacesAndNewlines))
            + closeTag("message")
            + endOfMsg()
    }

    private static func renderAssistant(
        _ message: K3ChatMessage,
        thinking: Bool
    ) throws -> [K3RenderSegment] {
        var segments: [K3RenderSegment] = []
        // The think channel is structural: in thinking mode every assistant
        // message carries the tags even with no reasoning content. In
        // non-thinking mode the channel is dropped entirely.
        if thinking {
            segments += openTag("think")
            if let reasoning = message.reasoningContent,
               !reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                segments += text(reasoning)
            }
            segments += closeTag("think")
        }

        segments += openTag("response")
        segments += renderContent(message.content)
        segments += closeTag("response")

        if !message.toolCalls.isEmpty {
            segments += openTag("tools")
            for (index, call) in message.toolCalls.enumerated() {
                segments += openTag("call", [("tool", call.name), ("index", "\(index + 1)")])
                switch try normalizeToolArguments(call.arguments) {
                case .arguments(let object):
                    // Insertion order is not preserved by the JSONValue
                    // dictionary; see the type-level deviation note.
                    for (key, value) in object {
                        segments += openTag("argument", [
                            ("key", key), ("type", K3JSON.xtmlType(of: value)),
                        ])
                        segments += text(K3JSON.xtmlValue(of: value))
                        segments += closeTag("argument")
                    }
                case .rawJSONBlock(let block):
                    segments += openTag("json", [("type", "object")])
                    segments += text(block)
                    segments += closeTag("json")
                }
                segments += closeTag("call")
            }
            segments += closeTag("tools")
        }
        return segments
    }

    private enum K3NormalizedArguments {
        case arguments([String: JSONValue])
        case rawJSONBlock(String)
    }

    /// `normalize_tool_arguments`: empty/blank strings are `{}`, parseable
    /// JSON objects render as argument tags, unparseable strings render as a
    /// raw json block, and parseable non-objects are an error.
    private static func normalizeToolArguments(
        _ arguments: K3ToolCallArguments
    ) throws -> K3NormalizedArguments {
        switch arguments {
        case .object(let value):
            guard case .object(let object) = value else {
                throw K3ChatRendererError.nonObjectToolArguments
            }
            return .arguments(object)
        case .jsonString(let string):
            if string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return .arguments([:])
            }
            guard let data = string.data(using: .utf8),
                  let parsed = try? JSONDecoder().decode(JSONValue.self, from: data) else {
                return .rawJSONBlock(string)
            }
            guard case .object(let object) = parsed else {
                throw K3ChatRendererError.nonObjectToolArguments
            }
            return .arguments(object)
        }
    }

    private static func renderToolDeclare(
        _ tools: [K3ToolDeclaration],
        dynamic: Bool
    ) -> [K3RenderSegment] {
        let json = K3JSON.compact(.array(tools.map(\.jsonValue)))
        let body: String
        if dynamic {
            body = "## New Tools Available\n"
                + "The system dynamically extends the toolset via lazy-loading.\n"
                + "You have access to all existing and extended tools.\n"
                + "Here are the specs for the extended tools.\n\n"
                + "```json\n\(json)\n```"
        } else {
            body = "# Tools\n"
                + "Here are the available tools, described in JSONSchema.\n\n"
                + "```json\n\(json)\n```"
        }
        return openTag("message", [("role", "system"), ("type", "tool-declare")])
            + text(body)
            + closeTag("message")
            + endOfMsg()
    }

    // MARK: - Tool-result re-sorting (normalize_xtml_tool_result_messages)

    /// Map assistant tool-call id → (1-based position, name). The position
    /// mirrors enumeration over `toolCalls` (id-less entries still advance
    /// it); duplicate ids keep their first occurrence.
    private static func toolCallIDIndex(_ toolCalls: [K3ToolCall]) -> [String: (Int, String)] {
        var index: [String: (Int, String)] = [:]
        for (offset, call) in toolCalls.enumerated() {
            guard let id = call.id, index[id] == nil else { continue }
            index[id] = (offset + 1, call.name)
        }
        return index
    }

    /// Re-sort each run of consecutive tool messages into the preceding
    /// assistant `toolCalls` order by `tool_call_id`; matched messages get
    /// `tool` (and any explicit `name`) rewritten to the call's name. A run
    /// that cannot be fully matched is left untouched. Value semantics keep
    /// the caller's array untouched, like the side-effect-free reference.
    private static func normalizeToolResultMessages(
        _ messages: [K3ChatMessage]
    ) -> [K3ChatMessage] {
        var output: [K3ChatMessage] = []
        var currentIndex: [String: (position: Int, name: String)] = [:]
        var i = 0
        let count = messages.count

        while i < count {
            let message = messages[i]

            if message.role == .assistant {
                currentIndex = toolCallIDIndex(message.toolCalls)
                output.append(message)
                i += 1
                continue
            }
            guard message.role == .tool else {
                output.append(message)
                i += 1
                continue
            }

            var run: [(position: Int?, offset: Int, message: K3ChatMessage, name: String?)] = []
            var unresolved = false
            var offset = 0
            while i < count, messages[i].role == .tool {
                let toolMessage = messages[i]
                if let callID = toolMessage.toolCallID,
                   let matched = currentIndex[callID] {
                    run.append((matched.position, offset, toolMessage, matched.name))
                } else {
                    unresolved = true
                    run.append((nil, offset, toolMessage, nil))
                }
                offset += 1
                i += 1
            }

            if unresolved {
                output.append(contentsOf: run.map(\.message))
            } else {
                run.sort { ($0.position ?? 0, $0.offset) < ($1.position ?? 0, $1.offset) }
                for item in run {
                    guard let name = item.name else {
                        output.append(item.message)
                        continue
                    }
                    // The id-matched call is authoritative: align tool (and
                    // any explicit name) with the reordered position.
                    var resolved = item.message
                    resolved.tool = name
                    if resolved.name != nil {
                        resolved.name = name
                    }
                    output.append(resolved)
                }
            }
        }
        return output
    }
}

/// JSON emission matching Python's `json.dumps(ensure_ascii=False)`:
/// non-ASCII passes through, only `"` `\` and control characters are escaped
/// (`\b \t \n \f \r`, otherwise `\u00XX`), `/` is not escaped. Object keys
/// are sorted — the reference deep-sorts every object it compact-dumps
/// (`deep_sort_dict`), and sorting keeps the `_xtml_value` path deterministic
/// where Python would preserve insertion order.
enum K3JSON {
    /// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))` with
    /// sorted keys.
    static func compact(_ value: JSONValue) -> String {
        var out = ""
        write(value, into: &out, itemSeparator: ",", keySeparator: ":")
        return out
    }

    /// `json.dumps(value, ensure_ascii=False)` (default `", "` / `": "`
    /// separators) with sorted keys. Used by `_xtml_value` for non-strings.
    static func spaced(_ value: JSONValue) -> String {
        var out = ""
        write(value, into: &out, itemSeparator: ", ", keySeparator: ": ")
        return out
    }

    /// `_xtml_type`.
    static func xtmlType(of value: JSONValue) -> String {
        switch value {
        case .bool: return "boolean"
        case .null: return "null"
        case .integer, .unsignedInteger, .decimal, .number: return "number"
        case .string: return "string"
        case .object: return "object"
        case .array: return "array"
        }
    }

    /// `_xtml_value`: strings pass through raw; everything else is JSON.
    static func xtmlValue(of value: JSONValue) -> String {
        if case .string(let string) = value {
            return string
        }
        return spaced(value)
    }

    private static func write(
        _ value: JSONValue,
        into out: inout String,
        itemSeparator: String,
        keySeparator: String
    ) {
        switch value {
        case .object(let object):
            out += "{"
            for (index, key) in object.keys.sorted().enumerated() {
                if index > 0 { out += itemSeparator }
                writeString(key, into: &out)
                out += keySeparator
                write(object[key]!, into: &out,
                      itemSeparator: itemSeparator, keySeparator: keySeparator)
            }
            out += "}"
        case .array(let array):
            out += "["
            for (index, item) in array.enumerated() {
                if index > 0 { out += itemSeparator }
                write(item, into: &out,
                      itemSeparator: itemSeparator, keySeparator: keySeparator)
            }
            out += "]"
        case .string(let string):
            writeString(string, into: &out)
        case .integer(let value):
            out += String(value)
        case .unsignedInteger(let value):
            out += String(value)
        case .decimal(let value):
            out += NSDecimalNumber(decimal: value).stringValue
        case .number(let value):
            out += String(value)
        case .bool(let value):
            out += value ? "true" : "false"
        case .null:
            out += "null"
        }
    }

    private static func writeString(_ string: String, into out: inout String) {
        out += "\""
        for scalar in string.unicodeScalars {
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
        out += "\""
    }
}
