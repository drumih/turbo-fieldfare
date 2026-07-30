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
                throw GemmaToolCallParserError.malformed
            }
            return value
        case .unsignedInteger(let value):
            guard let value = Int(exactly: value) else {
                throw GemmaToolCallParserError.malformed
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
                throw GemmaToolCallParserError.malformed
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

    /// Rewrites this JSON Schema fragment so every property the Gemma tool
    /// template renders carries a concrete scalar `type`. The template evaluates
    /// `value['type'] | upper` unconditionally for each property, so union
    /// (`anyOf`/`oneOf`/`allOf`), array-typed (`"type": ["string", "null"]`), or
    /// type-less properties would abort rendering. Unions collapse to their first
    /// branch that resolves to a concrete type (merging in the parent's sibling
    /// keys, e.g. `description`); an array `type` takes its first non-null member;
    /// a type-less node defaults to `object` when it carries `properties`,
    /// otherwise `string`. Every nested `properties` value and `items` schema is
    /// normalized recursively. Non-object values are returned unchanged.
    public func gemmaSchemaNormalized() -> JSONValue {
        guard case .object(var object) = self else { return self }

        if !object.hasScalarStringType,
           case .array(let members)? = object["type"],
           let concrete = members.firstNonNullTypeName {
            object["type"] = .string(concrete)
        }
        if !object.hasScalarStringType {
            for keyword in Self.unionKeywords {
                guard case .array(let branches)? = object[keyword] else { continue }
                return Self.collapsedUnion(parent: object, branches: branches)
            }
        }
        if !object.hasScalarStringType {
            object["type"] = object["properties"] != nil
                ? .string("object")
                : .string("string")
        }

        if case .object(let properties)? = object["properties"] {
            object["properties"] = .object(
                properties.mapValues { $0.gemmaSchemaNormalized() })
        }
        switch object["items"] {
        case .object?:
            object["items"] = object["items"]?.gemmaSchemaNormalized()
        case .array(let items)?:
            object["items"] = .array(items.map { $0.gemmaSchemaNormalized() })
        default:
            break
        }
        return .object(object)
    }

    private static let unionKeywords = ["anyOf", "oneOf", "allOf"]

    private static func collapsedUnion(parent: [String: JSONValue],
                                       branches: [JSONValue]) -> JSONValue {
        for branch in branches {
            guard case .object(var merged) = branch.gemmaSchemaNormalized(),
                  merged.hasScalarStringType else { continue }
            for (key, value) in parent where !unionKeywords.contains(key) {
                if merged[key] == nil { merged[key] = value }
            }
            return .object(merged)
        }
        var fallback = parent
        for keyword in unionKeywords { fallback[keyword] = nil }
        fallback["type"] = .string("string")
        return .object(fallback)
    }

    public func encoded(sortedKeys: Bool = true) throws -> String {
        let encoder = JSONEncoder()
        if sortedKeys { encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes] }
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }

    public func gemmaToolArgumentBody() throws -> String {
        guard case .object(let value) = self else {
            throw GemmaToolCallParserError.malformed
        }
        return try value.keys.sorted().map { key in
            guard GemmaToolCallParser.isRepresentableObjectKey(key) else {
                throw GemmaToolCallParserError.malformed
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
                throw GemmaToolCallParserError.malformed
            }
            return String(value)
        case .bool(let value):
            return value ? "true" : "false"
        case .null:
            return "null"
        }
    }
}

private extension Dictionary where Key == String, Value == JSONValue {
    /// True when `type` is present as a single JSON string, which is the only
    /// shape the Gemma template can feed to its `| upper` filter.
    var hasScalarStringType: Bool {
        if case .string? = self["type"] { return true }
        return false
    }
}

private extension Array where Element == JSONValue {
    /// The first `type` member that names a concrete (non-`"null"`) type, used to
    /// flatten `"type": ["string", "null"]` into a single scalar type.
    var firstNonNullTypeName: String? {
        for case .string(let name) in self where name != "null" { return name }
        return nil
    }
}
