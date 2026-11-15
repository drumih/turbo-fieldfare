import Foundation

/// Parses K2-Horizon's tool-call block.
///
/// The checkpoint's template wraps every call in one
/// `<ifm|tool_calls>…</ifm|tool_calls>` block and supports three call bodies,
/// selected by `tool_call_format` (default `xml`):
///
/// - `json`: `<ifm|tool_call>{"name": …, "arguments": {…}}</ifm|tool_call>`
/// - `xml`: the function name, then paired
///   `<ifm|arg_key>K</ifm|arg_key><ifm|arg_value>V</ifm|arg_value>`
/// - `xml_typed`: the same with an `<ifm|arg_type>` between them; the type is
///   informational, because the template already writes arrays and objects as
///   JSON literals and scalars as plain text.
///
/// It is separate from `QwenToolCallParser`: the markers and the body grammar
/// are the checkpoint's, not ChatML's, and accepting one dialect's text as the
/// other would silently produce a call with no arguments.
package struct K2ToolCallParser {
    package enum Format: String, Sendable, CaseIterable {
        case json
        case xml
        case xmlTyped = "xml_typed"
    }

    package static let maximumBytes = 256 * 1024
    package static let callsOpen = "<ifm|tool_calls>"
    package static let callsClose = "</ifm|tool_calls>"
    package static let callOpen = "<ifm|tool_call>"
    package static let callClose = "</ifm|tool_call>"
    package static let keyOpen = "<ifm|arg_key>"
    package static let keyClose = "</ifm|arg_key>"
    package static let typeOpen = "<ifm|arg_type>"
    package static let typeClose = "</ifm|arg_type>"
    package static let valueOpen = "<ifm|arg_value>"
    package static let valueClose = "</ifm|arg_value>"

    /// Parse a complete block. `makeID` runs once per call so a caller can
    /// keep the ids deterministic in tests and unique in a stream.
    package static func parse(
        _ text: String,
        allowedTools: Set<String>,
        format: Format,
        makeID: () -> String = {
            "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
        }
    ) throws -> [ParsedToolCall] {
        guard text.utf8.count <= maximumBytes else {
            throw ToolCallParserError.oversized
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix(callsOpen), trimmed.hasSuffix(callsClose) else {
            throw ToolCallParserError.malformed
        }
        var rest = trimmed.dropFirst(callsOpen.count).dropLast(callsClose.count)

        var calls: [ParsedToolCall] = []
        while let open = rest.range(of: callOpen) {
            guard rest[..<open.lowerBound].allSatisfy(\.isWhitespace) else {
                throw ToolCallParserError.malformed
            }
            let afterOpen = rest[open.upperBound...]
            guard let close = afterOpen.range(of: callClose) else {
                throw ToolCallParserError.malformed
            }
            calls.append(
                try parseCall(
                    String(afterOpen[..<close.lowerBound]),
                    allowedTools: allowedTools, format: format,
                    id: makeID()))
            rest = afterOpen[close.upperBound...]
        }
        guard rest.allSatisfy(\.isWhitespace), !calls.isEmpty else {
            throw ToolCallParserError.malformed
        }
        return calls
    }

    private static func parseCall(
        _ raw: String,
        allowedTools: Set<String>,
        format: Format,
        id: String
    ) throws -> ParsedToolCall {
        let parsed: (String, [String: JSONValue])
        switch format {
        case .json:
            parsed = try parseJSONCall(raw)
        case .xml, .xmlTyped:
            parsed = try parseXMLCall(raw, typed: format == .xmlTyped)
        }
        let (name, arguments) = parsed
        guard isValidFunctionName(name) else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else {
            throw ToolCallParserError.unknownTool(name)
        }
        let value = JSONValue.object(arguments)
        return ParsedToolCall(
            id: id, name: name, arguments: value,
            argumentsJSON: try value.encoded())
    }

    private static func parseJSONCall(
        _ raw: String
    ) throws -> (String, [String: JSONValue]) {
        let body = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = body.data(using: .utf8),
            let object = try? JSONDecoder().decode([String: JSONValue].self, from: data),
            let name = object["name"],
            case .string(let function) = name
        else {
            throw ToolCallParserError.malformed
        }
        guard case .object(let fields) = object["arguments"] ?? .object([:]) else {
            throw ToolCallParserError.malformed
        }
        return (function, fields)
    }

    private static func parseXMLCall(
        _ raw: String,
        typed: Bool
    ) throws -> (String, [String: JSONValue]) {
        var body = Substring(raw.trimmingCharacters(in: .whitespacesAndNewlines))
        guard let end = body.firstIndex(where: { $0 == "\n" || $0 == "\r" || $0 == "<" })
        else {
            return (String(body), [:])
        }
        let name = body[..<end].trimmingCharacters(in: .whitespaces)
        body = body[end...]
        var arguments: [String: JSONValue] = [:]
        while true {
            body = body.drop(while: { $0.isWhitespace })
            if body.isEmpty { break }
            let key = try tagBody(&body, open: keyOpen, close: keyClose)
            if typed {
                body = body.drop(while: { $0.isWhitespace })
                _ = try tagBody(&body, open: typeOpen, close: typeClose)
            }
            body = body.drop(while: { $0.isWhitespace })
            let value = try tagBody(&body, open: valueOpen, close: valueClose)
            let keyText = key.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !keyText.isEmpty else { throw ToolCallParserError.malformed }
            arguments[keyText] = typedValue(value)
        }
        guard !name.isEmpty else { throw ToolCallParserError.malformed }
        return (String(name), arguments)
    }

    /// Consume one `<open>…</close>` pair. The returned text has the one
    /// newline the template writes around a value trimmed from its edges; a
    /// string value's own inner newlines are left alone.
    private static func tagBody(
        _ body: inout Substring,
        open: String,
        close: String
    ) throws -> String {
        guard body.hasPrefix(open) else { throw ToolCallParserError.malformed }
        body = body.dropFirst(open.count)
        guard let end = body.range(of: close) else {
            throw ToolCallParserError.malformed
        }
        let value = body[..<end.lowerBound]
        body = body[end.upperBound...]
        return value.trimmingCharacters(in: CharacterSet(charactersIn: "\r\n"))
    }

    private static func typedValue(_ raw: String) -> JSONValue {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8),
            let value = try? JSONDecoder().decode(JSONValue.self, from: data)
        else {
            return .string(raw)
        }
        switch value {
        case .object, .array, .integer, .unsignedInteger, .decimal, .number,
            .bool, .null:
            return value
        case .string:
            // The template writes strings unquoted, so a value that decodes
            // as a JSON string was written with quotes on purpose; keep it.
            return .string(raw)
        }
    }

    /// Tool names are the template's function names: an identifier-like word
    /// with `.`/`-` allowed, as `QwenToolCallParser` accepts.
    private static func isValidFunctionName(_ name: String) -> Bool {
        guard !name.isEmpty, name.count <= 128 else { return false }
        return name.allSatisfy { character in
            character.isLetter || character.isNumber
                || character == "_" || character == "." || character == "-"
        }
    }
}
