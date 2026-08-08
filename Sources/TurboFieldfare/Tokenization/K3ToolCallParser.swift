import Foundation

/// The parsed shape of one generated K3 assistant turn.
public struct K3ParsedAssistantMessage: Equatable, Sendable {
    /// Think-channel text. `nil` only when parsing in non-thinking mode;
    /// an empty string means the model opened and closed an empty think
    /// channel.
    public let reasoningContent: String?
    /// Response-channel text (may be empty when the turn is pure tool calls).
    public let content: String
    /// Completed tool calls, in `<|open|>call` order. Truncated trailing
    /// calls are dropped; `malformed` signals that happened.
    public let toolCalls: [K3ToolCall]
    /// True when the input was truncated mid-structure or deviated from the
    /// expected channel grammar. Partial results are still returned.
    public let malformed: Bool

    public init(reasoningContent: String?, content: String,
                toolCalls: [K3ToolCall], malformed: Bool) {
        self.reasoningContent = reasoningContent
        self.content = content
        self.toolCalls = toolCalls
        self.malformed = malformed
    }
}

/// Parser for generated Kimi K3 XTML, the inverse of `K3ChatRenderer`'s
/// assistant channel rendering. Input is the decoded text the model produced
/// after the generation prompt (`<|open|>think<|sep|>` in thinking mode,
/// `<|open|>response<|sep|>` otherwise):
///
///     <reasoning><|close|>think<|sep|><|open|>response<|sep|><content>
///     <|close|>response<|sep|>[<|open|>tools<|sep|><calls><|close|>tools<|sep|>]
///     <|close|>message<|sep|>
///
/// Calls carry `<|open|>call tool="name" index="i"<|sep|>` followed by
/// `<|open|>argument key="k" type="t"<|sep|>value<|close|>argument<|sep|>`
/// entries or one raw `<|open|>json type="object"<|sep|>…<|close|>json<|sep|>`
/// block, then `<|close|>call<|sep|>`.
///
/// The parser never throws: truncation or grammar deviations return the
/// completed prefix with `malformed == true`, mirroring how the house
/// `GemmaToolCallParser` surfaces malformed output (it throws; streaming K3
/// generation needs partials instead). Attribute values are unescaped
/// (`&quot;`, `&amp;`); non-string argument values are JSON-parsed per their
/// `type` attribute, falling back to the raw string on failure.
public enum K3ToolCallParser {

    public static func parse(
        _ text: String,
        thinking: Bool = true
    ) -> K3ParsedAssistantMessage {
        var parser = Parser(lex(text), thinking: thinking)
        return parser.run()
    }

    // MARK: - Lexer

    private enum Marker {
        case open   // <|open|>
        case close  // <|close|>
        case sep    // <|sep|>
    }

    private enum Lexeme: Equatable {
        case marker(Marker)
        case text(String)
    }

    /// Split into marker/text lexemes. The three marker literals are fixed
    /// strings; the earliest match wins at every step.
    private static func lex(_ text: String) -> [Lexeme] {
        let literals: [(String, Marker)] = [
            ("<|open|>", .open),
            ("<|close|>", .close),
            ("<|sep|>", .sep),
        ]
        var lexemes: [Lexeme] = []
        var rest = text[...]
        while !rest.isEmpty {
            var nearest: (Range<String.Index>, Marker)?
            for (literal, marker) in literals {
                guard let range = rest.range(of: literal) else { continue }
                if let current = nearest, range.lowerBound >= current.0.lowerBound {
                    continue
                }
                nearest = (range, marker)
            }
            guard let (range, marker) = nearest else {
                lexemes.append(.text(String(rest)))
                break
            }
            if range.lowerBound > rest.startIndex {
                lexemes.append(.text(String(rest[..<range.lowerBound])))
            }
            lexemes.append(.marker(marker))
            rest = rest[range.upperBound...]
        }
        return lexemes
    }

    // MARK: - State machine

    private enum State {
        case think
        case expectResponse
        case response
        case afterResponse
        case tools
        case call(name: String, arguments: [String: JSONValue], jsonBlock: String?)
        case argument(name: String, type: String,
                      call: (name: String, arguments: [String: JSONValue], jsonBlock: String?))
        case jsonBlock(call: (name: String, arguments: [String: JSONValue], jsonBlock: String?))
        case afterTools
        case done
    }

    private struct Parser {
        private let lexemes: [Lexeme]
        private let thinking: Bool
        private var index = 0
        private var state: State
        private var reasoning = ""
        private var content = ""
        private var calls: [K3ToolCall] = []
        private var malformed = false

        init(_ lexemes: [Lexeme], thinking: Bool) {
            self.lexemes = lexemes
            self.thinking = thinking
            self.state = thinking ? .think : .response
        }

        mutating func run() -> K3ParsedAssistantMessage {
            while index < lexemes.count {
                step()
            }
            switch state {
            case .done:
                break
            case .response:
                // A turn that ends inside the response channel is only
                // malformed when it is empty-structured: the response text
                // itself is complete as far as it goes; flag truncation.
                malformed = true
            case .think, .expectResponse, .afterResponse, .tools, .afterTools,
                 .call, .argument, .jsonBlock:
                malformed = true
            }
            return K3ParsedAssistantMessage(
                reasoningContent: thinking ? reasoning : nil,
                content: content,
                toolCalls: calls,
                malformed: malformed)
        }

        private mutating func step() {
            switch state {
            case .think:
                // Reasoning text until <|close|>think<|sep|>.
                if let text = takeText() {
                    reasoning += text
                    return
                }
                if takeCloseTag("think") {
                    state = .expectResponse
                    return
                }
                markMalformedSkipping()

            case .expectResponse:
                if takeOpenTag("response") != nil {
                    state = .response
                    return
                }
                // Tolerate a missing response channel.
                if takeOpenTag("tools") != nil {
                    malformed = true
                    state = .tools
                    return
                }
                if takeCloseTag("message") {
                    malformed = true
                    state = .done
                    return
                }
                markMalformedSkipping()

            case .response:
                if let text = takeText() {
                    content += text
                    return
                }
                if takeCloseTag("response") {
                    state = .afterResponse
                    return
                }
                markMalformedSkipping()

            case .afterResponse:
                if takeOpenTag("tools") != nil {
                    state = .tools
                    return
                }
                if takeCloseTag("message") {
                    state = .done
                    return
                }
                markMalformedSkipping()

            case .tools:
                if let attrs = takeOpenTag("call") {
                    guard let name = attrs["tool"], !name.isEmpty else {
                        malformed = true
                        state = .tools
                        return
                    }
                    state = .call(name: name, arguments: [:], jsonBlock: nil)
                    return
                }
                if takeCloseTag("tools") {
                    state = .afterTools
                    return
                }
                markMalformedSkipping()

            case .call(let name, let arguments, let jsonBlock):
                if let attrs = takeOpenTag("argument") {
                    guard let key = attrs["key"], !key.isEmpty else {
                        malformed = true
                        state = .call(name: name, arguments: arguments, jsonBlock: jsonBlock)
                        return
                    }
                    state = .argument(name: key,
                                      type: attrs["type"] ?? "string",
                                      call: (name, arguments, jsonBlock))
                    return
                }
                if takeOpenTag("json") != nil {
                    state = .jsonBlock(call: (name, arguments, jsonBlock))
                    return
                }
                if takeCloseTag("call") {
                    calls.append(Self.makeCall(name: name,
                                               arguments: arguments,
                                               jsonBlock: jsonBlock,
                                               malformed: &malformed))
                    state = .tools
                    return
                }
                markMalformedSkipping()

            case .argument(let argName, let type, let call):
                // The value is the text up to <|close|>argument<|sep|>.
                var value = ""
                while let text = takeText() {
                    value += text
                }
                guard takeCloseTag("argument") else {
                    malformed = true
                    state = .tools
                    return
                }
                var arguments = call.arguments
                arguments[argName] = typedValue(value, type: type)
                state = .call(name: call.name, arguments: arguments,
                              jsonBlock: call.jsonBlock)
                return

            case .jsonBlock(let call):
                var block = ""
                while let text = takeText() {
                    block += text
                }
                guard takeCloseTag("json") else {
                    malformed = true
                    state = .tools
                    return
                }
                state = .call(name: call.name, arguments: call.arguments,
                              jsonBlock: block)
                return

            case .afterTools:
                if takeCloseTag("message") {
                    state = .done
                    return
                }
                markMalformedSkipping()

            case .done:
                index = lexemes.count
            }
        }

        // MARK: Lexeme helpers

        private mutating func takeText() -> String? {
            guard index < lexemes.count, case .text(let text) = lexemes[index] else {
                return nil
            }
            index += 1
            return text
        }

        /// Match `<|close|>tag<|sep|>`.
        private mutating func takeCloseTag(_ tag: String) -> Bool {
            guard index + 2 < lexemes.count,
                  case .marker(.close) = lexemes[index],
                  case .text(let text) = lexemes[index + 1],
                  case .marker(.sep) = lexemes[index + 2] else {
                return false
            }
            let parsed = parseTagText(text)
            guard parsed.tag == tag, parsed.attrs.isEmpty else { return false }
            index += 3
            return true
        }

        /// Match `<|open|>tag [attrs]<|sep|>`, returning the parsed attrs.
        private mutating func takeOpenTag(_ tag: String) -> [String: String]? {
            guard index + 2 < lexemes.count,
                  case .marker(.open) = lexemes[index],
                  case .text(let text) = lexemes[index + 1],
                  case .marker(.sep) = lexemes[index + 2] else {
                return nil
            }
            let parsed = parseTagText(text)
            guard parsed.tag == tag else { return nil }
            index += 3
            return parsed.attrs
        }

        private mutating func markMalformedSkipping() {
            malformed = true
            index += 1
        }

        private static func makeCall(
            name: String,
            arguments: [String: JSONValue],
            jsonBlock: String?,
            malformed: inout Bool
        ) -> K3ToolCall {
            if let jsonBlock {
                // A raw json block mirrors the renderer's unparseable-string
                // path; a well-formed model emits one that parses as an object.
                if let parsed = try? JSONDecoder().decode(
                    JSONValue.self, from: Data(jsonBlock.utf8)),
                   case .object = parsed {
                    return K3ToolCall(name: name, arguments: .object(parsed))
                }
                malformed = true
                return K3ToolCall(name: name, arguments: .jsonString(jsonBlock))
            }
            return K3ToolCall(name: name, arguments: .object(.object(arguments)))
        }

        /// `type` drives coercion: strings pass through raw; everything else
        /// is JSON (`_xtml_value` emitted `json.dumps`), with a raw-string
        /// fallback for model garbage.
        private func typedValue(_ raw: String, type: String) -> JSONValue {
            if type == "string" {
                return .string(raw)
            }
            if let parsed = try? JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8)) {
                return parsed
            }
            return .string(raw)
        }

        /// Parse `tag key="value" …` — values are attribute-escaped
        /// (`_escape_attr_value` wrote `&amp;` then `&quot;`; unescape in
        /// reverse).
        private func parseTagText(_ text: String) -> (tag: String, attrs: [String: String]) {
            var scalars = Substring(text)
            let tag: String
            if let space = scalars.firstIndex(of: " ") {
                tag = String(scalars[..<space])
                scalars = scalars[scalars.index(after: space)...]
            } else {
                return (String(scalars), [:])
            }
            var attrs: [String: String] = [:]
            while !scalars.isEmpty {
                guard let equal = scalars.firstIndex(of: "="),
                      equal > scalars.startIndex else { break }
                let valueStart = scalars.index(after: equal)
                guard valueStart < scalars.endIndex,
                      scalars[valueStart] == "\"",
                      let close = scalars[valueStart...].dropFirst().firstIndex(of: "\"")
                else { break }
                let key = String(scalars[..<equal])
                let raw = String(scalars[scalars.index(after: valueStart)..<close])
                attrs[key] = raw
                    .replacingOccurrences(of: "&quot;", with: "\"")
                    .replacingOccurrences(of: "&amp;", with: "&")
                let rest = scalars.index(after: close)
                guard rest < scalars.endIndex else {
                    scalars = ""
                    break
                }
                scalars = scalars[rest...]
                while scalars.first == " " { scalars = scalars.dropFirst() }
            }
            return (tag, attrs)
        }
    }
}
