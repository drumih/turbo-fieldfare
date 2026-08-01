import Foundation

public indirect enum JSONValue: Codable, Equatable, Sendable {
    case object([String: JSONValue])
    case array([JSONValue])
    case string(String)
    case integer(Int64)
    case unsignedInteger(UInt64)
    case decimal(Decimal)
    case number(Double)
    case bool(Bool)
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int64.self) {
            self = .integer(value)
        } else if let value = try? container.decode(UInt64.self) {
            self = .unsignedInteger(value)
        } else if let value = try? container.decode(Decimal.self) {
            self = .decimal(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .unsignedInteger(let value): try container.encode(value)
        case .decimal(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    public func jinjaSendableValue() throws -> any Sendable {
        switch self {
        case .object(let value):
            return try value.mapValues { try $0.jinjaSendableValue() }
        case .array(let value):
            return try value.map { try $0.jinjaSendableValue() }
        case .string(let value):
            return value
        case .integer(let value):
            guard let value = Int(exactly: value) else {
                throw ToolCallParserError.malformed
            }
            return value
        case .unsignedInteger(let value):
            guard let value = Int(exactly: value) else {
                throw ToolCallParserError.malformed
            }
            return value
        case .decimal(let value):
            let text = NSDecimalNumber(decimal: value).stringValue
            guard let double = Double(text),
                  double.isFinite,
                  let roundTrip = Decimal(
                    string: String(double),
                    locale: Locale(identifier: "en_US_POSIX")),
                  roundTrip == value else {
                throw ToolCallParserError.malformed
            }
            return double
        case .number(let value):
            return value
        case .bool(let value):
            return value
        case .null:
            return Optional<String>.none as String?
        }
    }

    public var objectValue: [String: JSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    public func encoded(sortedKeys: Bool = true) throws -> String {
        let encoder = JSONEncoder()
        if sortedKeys { encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes] }
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }

    public func gemmaToolArgumentBody() throws -> String {
        guard case .object(let value) = self else {
            throw ToolCallParserError.malformed
        }
        return try value.keys.sorted().map { key in
            guard GemmaToolCallParser.isRepresentableObjectKey(key) else {
                throw ToolCallParserError.malformed
            }
            return "\(key):\(try value[key]!.gemmaToolValue())"
        }.joined(separator: ",")
    }

    private func gemmaToolValue() throws -> String {
        switch self {
        case .object:
            return "{\(try gemmaToolArgumentBody())}"
        case .array(let value):
            return "[\(try value.map { try $0.gemmaToolValue() }.joined(separator: ","))]"
        case .string(let value):
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.withoutEscapingSlashes]
            return String(decoding: try encoder.encode(value), as: UTF8.self)
        case .integer(let value):
            return String(value)
        case .unsignedInteger(let value):
            return String(value)
        case .decimal(let value):
            return NSDecimalNumber(decimal: value).stringValue
        case .number(let value):
            guard value.isFinite else {
                throw ToolCallParserError.malformed
            }
            return String(value)
        case .bool(let value):
            return value ? "true" : "false"
        case .null:
            return "null"
        }
    }
}
