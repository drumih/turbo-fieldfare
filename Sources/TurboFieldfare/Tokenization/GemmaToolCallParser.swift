import Foundation

public struct ParsedToolCall: Equatable, Sendable {
    public let id: String
    public let name: String
    public let arguments: JSONValue
    public let argumentsJSON: String

    public init(id: String,
                name: String,
                arguments: JSONValue,
                argumentsJSON: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.argumentsJSON = argumentsJSON
    }
}

public enum GemmaToolCallParserError: Error, Equatable {
    case malformed
    case unknownTool(String)
    case oversized
}

public struct GemmaToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    public static func isRepresentableObjectKey(_ key: String) -> Bool {
        !key.isEmpty && key.allSatisfy {
            $0.isLetter || $0.isNumber || "_-.$".contains($0)
        }
    }

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= Self.maximumBytes else {
            throw GemmaToolCallParserError.oversized
        }
        var parser = Parser(text)
        try parser.consume("call:")
        let name = try parser.identifier()
        guard allowedTools.contains(name) else {
            throw GemmaToolCallParserError.unknownTool(name)
        }
        let arguments = try parser.object()
        parser.skipWhitespace()
        guard parser.isAtEnd else { throw GemmaToolCallParserError.malformed }
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: arguments,
                              argumentsJSON: try arguments.encoded())
    }
}

private struct Parser {
    private let characters: [Character]
    private var index = 0

    init(_ text: String) {
        characters = Array(text)
    }

    var isAtEnd: Bool { index == characters.count }

    mutating func skipWhitespace() {
        while index < characters.count, characters[index].isWhitespace { index += 1 }
    }

    mutating func consume(_ literal: String) throws {
        skipWhitespace()
        let value = Array(literal)
        guard index + value.count <= characters.count,
              Array(characters[index..<(index + value.count)]) == value else {
            throw GemmaToolCallParserError.malformed
        }
        index += value.count
    }

    mutating func identifier() throws -> String {
        skipWhitespace()
        let start = index
        while index < characters.count {
            let character = characters[index]
            // Matches OpenAIToolName.isValid, which the server already accepts:
            // hyphens are legal in OpenAI tool names, and the chat template
            // renders the name verbatim into the call, so a narrower identifier
            // alphabet makes every hyphenated tool undecodable.
            guard character.isLetter || character.isNumber
                    || character == "_" || character == "-" else { break }
            index += 1
        }
        guard index > start else { throw GemmaToolCallParserError.malformed }
        return String(characters[start..<index])
    }

    mutating func object() throws -> JSONValue {
        try consume("{")
        var result: [String: JSONValue] = [:]
        skipWhitespace()
        if take("}") { return .object(result) }
        while true {
            let key = try objectKey()
            try consume(":")
            result[key] = try value()
            skipWhitespace()
            if take("}") { return .object(result) }
            try consume(",")
        }
    }

    mutating func value() throws -> JSONValue {
        skipWhitespace()
        if starts(with: "<|\"|>") { return .string(try gemmaString()) }
        if starts(with: "\"") { return .string(try jsonString()) }
        if starts(with: "{") { return try object() }
        if starts(with: "[") { return try array() }
        if takeWord("true") { return .bool(true) }
        if takeWord("false") { return .bool(false) }
        if takeWord("null") { return .null }
        return try number()
    }

    mutating func objectKey() throws -> String {
        skipWhitespace()
        let start = index
        while index < characters.count {
            let character = characters[index]
            guard character.isLetter
                    || character.isNumber
                    || "_-.$".contains(character) else {
                break
            }
            index += 1
        }
        guard index > start else { throw GemmaToolCallParserError.malformed }
        return String(characters[start..<index])
    }

    mutating func array() throws -> JSONValue {
        try consume("[")
        var result: [JSONValue] = []
        skipWhitespace()
        if take("]") { return .array(result) }
        while true {
            result.append(try value())
            skipWhitespace()
            if take("]") { return .array(result) }
            try consume(",")
        }
    }

    mutating func gemmaString() throws -> String {
        try consume("<|\"|>")
        var result = ""
        while !isAtEnd {
            if starts(with: "<|\"|>") {
                try consume("<|\"|>")
                return result
            }
            if characters[index] == "\\", index + 1 < characters.count {
                index += 1
                result += try escapedFragment()
            } else {
                result.append(characters[index])
                index += 1
            }
        }
        throw GemmaToolCallParserError.malformed
    }

    mutating func jsonString() throws -> String {
        try consume("\"")
        var result = ""
        while !isAtEnd {
            // `take` skips whitespace before matching, which silently swallowed
            // every space, tab and newline inside the string: `"/tmp/a b c"`
            // decoded as `/tmp/abc`. Inside a string no character is a
            // delimiter to skip past.
            let character = characters[index]
            if character == "\"" {
                index += 1
                return result
            }
            index += 1
            if character == "\\" {
                result += try escapedFragment()
            } else {
                result.append(character)
            }
        }
        throw GemmaToolCallParserError.malformed
    }

    mutating func escapedFragment() throws -> String {
        guard index < characters.count else { throw GemmaToolCallParserError.malformed }
        let escape = characters[index]
        index += 1
        switch escape {
        case "\"": return "\""
        case "\\": return "\\"
        case "/": return "/"
        case "b": return "\u{8}"
        case "f": return "\u{c}"
        case "n": return "\n"
        case "r": return "\r"
        case "t": return "\t"
        case "u":
            let first = try unicodeCodeUnit()
            let scalar: UInt32
            if (0xD800...0xDBFF).contains(first) {
                guard index + 2 <= characters.count,
                      characters[index] == "\\",
                      characters[index + 1] == "u" else {
                    throw GemmaToolCallParserError.malformed
                }
                index += 2
                let second = try unicodeCodeUnit()
                guard (0xDC00...0xDFFF).contains(second) else {
                    throw GemmaToolCallParserError.malformed
                }
                scalar = 0x10000
                    + (UInt32(first - 0xD800) << 10)
                    + UInt32(second - 0xDC00)
            } else {
                guard !(0xDC00...0xDFFF).contains(first) else {
                    throw GemmaToolCallParserError.malformed
                }
                scalar = UInt32(first)
            }
            guard let unicode = UnicodeScalar(scalar) else {
                throw GemmaToolCallParserError.malformed
            }
            return String(unicode)
        default: throw GemmaToolCallParserError.malformed
        }
    }

    mutating func unicodeCodeUnit() throws -> UInt16 {
        guard index + 4 <= characters.count else {
            throw GemmaToolCallParserError.malformed
        }
        var value: UInt16 = 0
        for _ in 0..<4 {
            guard let digit = characters[index].hexDigitValue else {
                throw GemmaToolCallParserError.malformed
            }
            value = value * 16 + UInt16(digit)
            index += 1
        }
        return value
    }

    mutating func number() throws -> JSONValue {
        let start = index
        while index < characters.count,
              "-+0123456789.eE".contains(characters[index]) {
            index += 1
        }
        guard index > start else {
            throw GemmaToolCallParserError.malformed
        }
        let literal = String(characters[start..<index])
        guard literal.range(
            of: #"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"#,
            options: .regularExpression) != nil else {
            throw GemmaToolCallParserError.malformed
        }
        if !literal.contains(where: { ".eE".contains($0) }) {
            if let value = Int64(literal) { return .integer(value) }
            if let value = UInt64(literal) { return .unsignedInteger(value) }
        }
        guard let value = Decimal(
            string: literal,
            locale: Locale(identifier: "en_US_POSIX")),
              !value.isNaN else {
            throw GemmaToolCallParserError.malformed
        }
        return .decimal(value)
    }

    func starts(with literal: String) -> Bool {
        let value = Array(literal)
        return index + value.count <= characters.count
            && Array(characters[index..<(index + value.count)]) == value
    }

    mutating func take(_ literal: Character) -> Bool {
        skipWhitespace()
        guard index < characters.count, characters[index] == literal else { return false }
        index += 1
        return true
    }

    mutating func takeWord(_ literal: String) -> Bool {
        guard starts(with: literal) else { return false }
        index += literal.count
        return true
    }
}
