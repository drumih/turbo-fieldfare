import Foundation

/// Parses the Qwen ChatML tool-call body — the text BETWEEN the `<tool_call>`
/// and `</tool_call>` special tokens:
///
///     \n<function=NAME>\n
///     <parameter=KEY>\nVALUE\n</parameter>\n
///     ...
///     </function>\n
///
/// Each VALUE is everything between the newline following the parameter open
/// tag and the newline before `</parameter>`; multi-line values are allowed.
/// Values that parse as structural JSON (object / array / number / bool /
/// null) become typed `JSONValue`s; everything else is kept as a raw string,
/// mirroring the template's asymmetric serialization (strings pass through
/// unquoted, non-strings via `tojson`).
public struct QwenToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
        var body = Substring(text)
        trimOuterWhitespace(&body)

        let name = try functionName(&body)
        guard isValidFunctionName(name) else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else {
            throw ToolCallParserError.unknownTool(name)
        }

        var arguments: [String: JSONValue] = [:]
        while !body.hasPrefix("</function>") {
            let (key, value) = try parameter(&body)
            arguments[key] = value
        }
        body.removeFirst("</function>".count)
        trimOuterWhitespace(&body)
        guard body.isEmpty else { throw ToolCallParserError.malformed }

        let argumentsValue = JSONValue.object(arguments)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: argumentsValue,
                              argumentsJSON: try argumentsValue.encoded())
    }

    private func trimOuterWhitespace(_ body: inout Substring) {
        while let first = body.first, first.isWhitespace { body.removeFirst() }
        while let last = body.last, last.isWhitespace { body.removeLast() }
    }

    private func functionName(_ body: inout Substring) throws -> String {
        guard body.hasPrefix("<function=") else {
            throw ToolCallParserError.malformed
        }
        body.removeFirst("<function=".count)
        guard let close = body.firstIndex(of: ">") else {
            throw ToolCallParserError.malformed
        }
        let name = String(body[..<close])
        body = body[body.index(after: close)...]
        guard body.first == "\n" else { throw ToolCallParserError.malformed }
        body.removeFirst()
        return name
    }

    private func parameter(_ body: inout Substring) throws -> (String, JSONValue) {
        guard body.hasPrefix("<parameter=") else {
            throw ToolCallParserError.malformed
        }
        body.removeFirst("<parameter=".count)
        guard let close = body.firstIndex(of: ">") else {
            throw ToolCallParserError.malformed
        }
        let key = String(body[..<close])
        guard !key.isEmpty, !key.contains("\n"), !key.contains("<") else {
            throw ToolCallParserError.malformed
        }
        body = body[body.index(after: close)...]
        guard body.first == "\n" else { throw ToolCallParserError.malformed }
        body.removeFirst()
        // VALUE runs to the newline before `</parameter>`; the newline that
        // opened it may double as that closer for an empty value.
        guard let closeRange = body.range(of: "\n</parameter>\n")
                ?? emptyValueCloseRange(body) else {
            throw ToolCallParserError.malformed
        }
        let value = String(body[..<closeRange.lowerBound])
        body = body[closeRange.upperBound...]
        return (key, parsedValue(value))
    }

    /// Handles `<parameter=k>\n</parameter>\n` where the single newline both
    /// opens the value and precedes the close tag (empty value, no blank line).
    private func emptyValueCloseRange(_ body: Substring) -> Range<Substring.Index>? {
        guard body.hasPrefix("</parameter>\n") else { return nil }
        return body.startIndex..<body.index(body.startIndex,
                                            offsetBy: "</parameter>\n".count)
    }

    private func isValidFunctionName(_ name: String) -> Bool {
        name.range(of: "^[A-Za-z0-9_-]{1,64}$", options: .regularExpression) != nil
    }

    private func parsedValue(_ raw: String) -> JSONValue {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let first = trimmed.first,
              "{[-0123456789tfn".contains(first),
              let value = try? JSONDecoder().decode(JSONValue.self,
                                                    from: Data(trimmed.utf8)) else {
            return .string(raw)
        }
        // Quoted strings stay raw text: the template writes string arguments
        // unquoted, so literal quotes belong to the argument itself.
        if case .string = value { return .string(raw) }
        return value
    }
}
