import Foundation

/// The JSON Schema keywords this server compiles into a grammar.
///
/// Structured output here is a *subset* on purpose. A keyword this compiler
/// cannot turn into a byte-level guarantee is refused by name, at request time,
/// in the API's own error shape -- never accepted and then quietly ignored,
/// which is the failure mode that makes a "strict" schema worse than none: the
/// client validates against a promise the server never made.
///
/// Supported: `type` (a name or a list of names), `properties`, `required`,
/// `additionalProperties` (a boolean), `items` (one schema), `enum` and
/// `const` with string or integer values. Annotations (`title`, `description`,
/// `default`, `examples`, `$comment`, `$schema`, `$id`, `x-*`) are accepted and
/// ignored, because they constrain nothing.
///
/// Refused: `$ref`/`$defs`, `allOf`, `anyOf`, `oneOf`, `not`, `if`/`then`/
/// `else`, `patternProperties`, `propertyNames`, `unevaluatedProperties`,
/// `dependencies`, `pattern`, `format`, `minimum`/`maximum`/`multipleOf`,
/// `minLength`/`maxLength`, `minItems`/`maxItems`/`uniqueItems`, `contains`,
/// `prefixItems`, `additionalProperties` given as a schema, and `enum` values
/// that are not strings or integers.
public enum JSONSchemaCompileError: Error, Equatable, CustomStringConvertible {
    /// A keyword outside the supported subset, at a dotted path within the
    /// schema (e.g. `properties.name.pattern`).
    case unsupported(keyword: String, at: String)
    /// The schema is structurally wrong rather than unsupported (a `type` that
    /// is not a name, `properties` that is not an object, ...).
    case malformed(String, at: String)
    /// `required` names a property that can never be generated.
    case unsatisfiable(String)
    /// An `enum`/`const` value cannot be matched by its canonical spelling, so
    /// a grammar cannot promise it (a string needing escapes, a fractional or
    /// non-finite number).
    case unmatchableLiteral(String)

    public var description: String {
        switch self {
        case .unsupported(let keyword, let at):
            return "JSON Schema keyword '\(keyword)' at \(at) is not supported by this "
                + "server's grammar; supported keywords are type, properties, required, "
                + "additionalProperties, items, enum and const"
        case .malformed(let detail, let at):
            return "JSON Schema at \(at) is malformed: \(detail)"
        case .unsatisfiable(let detail):
            return "JSON Schema cannot be satisfied: \(detail)"
        case .unmatchableLiteral(let detail):
            return "JSON Schema literal \(detail) cannot be matched by a grammar; "
                + "enum and const values must be strings without escapes, or integers"
        }
    }
}

/// The JSON types a schema may name.
public enum JSONScalarKind: String, Sendable, Hashable, CaseIterable {
    case object
    case array
    case string
    case number
    /// `integer` is `number` without a fraction or exponent, which is what the
    /// schema asks for and what a grammar can actually enforce.
    case integer
    case boolean
    case null
}

/// A compiled schema position: what may appear where.
public indirect enum JSONSchemaNode: Sendable, Hashable {
    /// The deepest nesting any walk over a schema document may reach.
    ///
    /// `compile` walks the schema recursively, and until now the only bound on
    /// what it could be handed was Foundation's own JSON parser, which was
    /// measured on this toolchain (Swift 6.4) to accept 512 nesting levels and
    /// throw at 513 -- an error this server reports as "malformed JSON request".
    /// That bound is real but undocumented, so the depth guard here is the one
    /// this project owns; sixty-four is well inside the parse limit and well
    /// outside anything a client writes by hand.
    public static let maximumNestingDepth = 64

    /// Any JSON value (no `type`, or `true` as a schema).
    case any
    case scalar(Set<JSONScalarKind>)
    case object(properties: [String: JSONSchemaNode], required: [String], additional: Bool)
    case array(items: JSONSchemaNode?)
    /// Exact literal spellings, already quoted/encoded (`"a"`, `1`, `true`,
    /// `null`). Every one is matched byte for byte.
    case enumeration([String])

    /// Keywords that are pure annotations: accepting them changes nothing.
    static let annotations: Set<String> = [
        "title", "description", "default", "examples", "$comment", "$schema",
        "$id", "id", "deprecated", "readOnly", "writeOnly",
    ]

    /// Keywords that would constrain, but not in a way this grammar implements.
    static let refused: Set<String> = [
        "$ref", "$defs", "definitions", "allOf", "anyOf", "oneOf", "not",
        "if", "then", "else", "patternProperties", "propertyNames",
        "unevaluatedProperties", "unevaluatedItems", "dependencies",
        "dependentSchemas", "dependentRequired", "pattern", "format",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf", "minLength", "maxLength", "minItems", "maxItems",
        "uniqueItems", "contains", "minContains", "maxContains",
        "prefixItems", "minProperties", "maxProperties",
    ]

    /// Compile a schema document. `at` is the dotted path used in errors, and
    /// `depth` is the nesting level this call sits at, so the walk carries its
    /// own ceiling rather than inheriting one from the parser.
    public static func compile(
        _ value: JSONValue,
        at path: String = "$",
        depth: Int = 0
    ) throws -> JSONSchemaNode {
        guard depth <= maximumNestingDepth else {
            throw JSONSchemaCompileError.malformed(
                "a schema may not nest deeper than \(maximumNestingDepth) levels", at: path)
        }
        // A boolean schema is the JSON Schema shorthand: `true` allows
        // anything, `false` allows nothing.
        if case .bool(let allowed) = value {
            guard allowed else {
                throw JSONSchemaCompileError.unsatisfiable(
                    "the schema at \(path) is `false`, which permits no value at all")
            }
            return .any
        }
        guard case .object(let keywords) = value else {
            throw JSONSchemaCompileError.malformed(
                "a schema must be an object or a boolean", at: path)
        }
        for keyword in keywords.keys.sorted() {
            if refused.contains(keyword) {
                throw JSONSchemaCompileError.unsupported(keyword: keyword, at: path)
            }
            if annotations.contains(keyword) { continue }
            switch keyword {
            case "type", "properties", "required", "additionalProperties",
                "items", "enum", "const":
                continue
            default:
                // An unknown keyword is refused rather than ignored: a schema
                // written against a newer draft must not silently lose a
                // constraint. Vendor annotations (`x-...`) are the exception.
                if keyword.hasPrefix("x-") { continue }
                throw JSONSchemaCompileError.unsupported(keyword: keyword, at: path)
            }
        }
        if let constant = keywords["const"] {
            try refuseStructureBesideLiteral(keywords, at: path)
            return try enumerationNode([constant], at: path)
        }
        if let values = keywords["enum"] {
            try refuseStructureBesideLiteral(keywords, at: path)
            guard case .array(let entries) = values, !entries.isEmpty else {
                throw JSONSchemaCompileError.malformed("enum must be a non-empty array", at: path)
            }
            return try enumerationNode(entries, at: path)
        }
        let declared: Set<JSONScalarKind>? =
            keywords["type"] == nil
            ? nil : try types(keywords["type"], at: path)
        if let properties = keywords["properties"] {
            guard case .object(let entries) = properties else {
                throw JSONSchemaCompileError.malformed("properties must be an object", at: path)
            }
            try requireContainer(declared, .object, at: path)
            var compiled: [String: JSONSchemaNode] = [:]
            for (name, schema) in entries {
                // A property name is matched as a quoted literal, so a name
                // needing escapes has several legal spellings and cannot be
                // promised byte for byte.
                guard canonicalLiteral(.string(name)) != nil else {
                    throw JSONSchemaCompileError.unmatchableLiteral(
                        "property name '\(name)' at \(path) (it needs escaping)")
                }
                compiled[name] = try compile(
                    schema, at: "\(path).properties.\(name)", depth: depth + 1)
            }
            let required = try requiredNames(keywords["required"], at: path)
            let additional = try additionalAllowed(keywords["additionalProperties"], at: path)
            if !additional {
                for name in required where compiled[name] == nil {
                    throw JSONSchemaCompileError.unsatisfiable(
                        "required property '\(name)' at \(path) is not in properties and "
                            + "additionalProperties is false, so it can never be generated")
                }
            }
            // `properties` implies an object even when `type` was omitted,
            // which is how most hand-written schemas are shaped.
            return .object(properties: compiled, required: required, additional: additional)
        }
        if let items = keywords["items"] {
            try requireContainer(declared, .array, at: path)
            return .array(items: try compile(items, at: "\(path).items", depth: depth + 1))
        }
        // `required` and `additionalProperties` constrain an object even when
        // `properties` is absent, and neither can be honoured for a schema that
        // says nothing else about the object.
        if keywords["additionalProperties"] != nil || keywords["required"] != nil {
            guard declared == [.object] else {
                throw JSONSchemaCompileError.malformed(
                    "required/additionalProperties need type object", at: path)
            }
            _ = try additionalAllowed(keywords["additionalProperties"], at: path)
            let required = try requiredNames(keywords["required"], at: path)
            guard required.isEmpty else {
                throw JSONSchemaCompileError.unsupported(
                    keyword: "required without properties", at: path)
            }
        }
        guard let declared else { return .any }
        if declared == [.object] { return .object(properties: [:], required: [], additional: true) }
        if declared == [.array] { return .array(items: nil) }
        // A union that mixes a container with anything else would need the
        // grammar to keep two shapes alive at once. A union of scalars is
        // fine -- `["string", "null"]` is the common nullable spelling, and
        // the first byte already decides which one it is.
        if declared.contains(.object) || declared.contains(.array) {
            throw JSONSchemaCompileError.unsupported(
                keyword: "type (a union including object or array)", at: path)
        }
        return .scalar(declared)
    }

    /// `enum`/`const` decide the value on their own; a structural keyword
    /// beside them would be silently ignored, which is exactly the failure this
    /// compiler refuses everywhere else.
    static func refuseStructureBesideLiteral(
        _ keywords: [String: JSONValue],
        at path: String
    ) throws {
        for structural in ["properties", "required", "additionalProperties", "items"]
        where keywords[structural] != nil {
            throw JSONSchemaCompileError.unsupported(
                keyword: "\(structural) beside enum/const", at: path)
        }
    }

    /// A schema that declares `properties` is an object schema, and one that
    /// declares `items` is an array schema; a `type` that says otherwise is a
    /// contradiction, not something to guess about.
    static func requireContainer(
        _ declared: Set<JSONScalarKind>?,
        _ kind: JSONScalarKind, at path: String
    ) throws {
        guard let declared else { return }
        guard declared == [kind] else {
            throw JSONSchemaCompileError.malformed(
                "\(kind == .object ? "properties" : "items") needs type \(kind.rawValue), "
                    + "but type is \(declared.map(\.rawValue).sorted().joined(separator: "|"))",
                at: path)
        }
    }

    /// `type`: a name or a list of names. An omitted `type` constrains nothing.
    static func types(_ value: JSONValue?, at path: String) throws -> Set<JSONScalarKind> {
        guard let value else { return Set(JSONScalarKind.allCases) }
        func kind(_ name: String) throws -> JSONScalarKind {
            guard let kind = JSONScalarKind(rawValue: name) else {
                throw JSONSchemaCompileError.malformed("unknown type '\(name)'", at: path)
            }
            return kind
        }
        if case .string(let name) = value { return [try kind(name)] }
        if case .array(let names) = value {
            var kinds: Set<JSONScalarKind> = []
            for name in names {
                guard case .string(let text) = name else {
                    throw JSONSchemaCompileError.malformed("type entries must be strings", at: path)
                }
                kinds.insert(try kind(text))
            }
            guard !kinds.isEmpty else {
                throw JSONSchemaCompileError.malformed("type must not be an empty array", at: path)
            }
            return kinds
        }
        throw JSONSchemaCompileError.malformed("type must be a string or an array", at: path)
    }

    static func requiredNames(_ value: JSONValue?, at path: String) throws -> [String] {
        guard let value else { return [] }
        guard case .array(let names) = value else {
            throw JSONSchemaCompileError.malformed("required must be an array", at: path)
        }
        return try names.map {
            guard case .string(let name) = $0 else {
                throw JSONSchemaCompileError.malformed("required entries must be strings", at: path)
            }
            return name
        }
    }

    static func additionalAllowed(_ value: JSONValue?, at path: String) throws -> Bool {
        guard let value else { return true }
        if case .bool(let allowed) = value { return allowed }
        throw JSONSchemaCompileError.unsupported(
            keyword: "additionalProperties (as a schema)", at: path)
    }

    static func enumerationNode(_ values: [JSONValue], at path: String) throws
        -> JSONSchemaNode
    {
        var literals: [String] = []
        for value in values {
            guard let literal = canonicalLiteral(value) else {
                throw JSONSchemaCompileError.unmatchableLiteral("\(value) at \(path)")
            }
            literals.append(literal)
        }
        // `"a"` and `"ab"` would need the grammar to hold two states at once
        // (stop here, or keep going); refusing is honest, and no real schema
        // enumerates a value and its own prefix.
        for (index, literal) in literals.enumerated() {
            for (other, candidate) in literals.enumerated() where other != index {
                if candidate.hasPrefix(literal) {
                    throw JSONSchemaCompileError.unsatisfiable(
                        "enum values \(literal) and \(candidate) at \(path) share a prefix, "
                            + "which this grammar cannot match unambiguously")
                }
            }
        }
        return .enumeration(literals)
    }

    /// The exact bytes a model would have to emit for this value, or nil when
    /// the value cannot be matched byte for byte.
    static func canonicalLiteral(_ value: JSONValue) -> String? {
        switch value {
        case .string(let text):
            // Escapes are refused rather than encoded: the model may legally
            // spell the same string several ways, and a grammar that allowed
            // only one of them would be a surprise.
            guard !text.contains("\""), !text.contains("\\"),
                !text.unicodeScalars.contains(where: { $0.value < 0x20 })
            else {
                return nil
            }
            return "\"\(text)\""
        case .integer(let number):
            return String(number)
        case .unsignedInteger(let number):
            return String(number)
        case .decimal(let number):
            let text = NSDecimalNumber(decimal: number).stringValue
            return text.contains(".") ? nil : text
        case .number(let number):
            guard number.isFinite, number == number.rounded() else { return nil }
            return String(Int64(number))
        case .bool(let flag):
            return flag ? "true" : "false"
        case .null:
            return "null"
        case .object, .array:
            return nil
        }
    }
}

extension JSONSchemaNode {
    /// Whether some value can be written at this position. A value node the
    /// compiler produced is always satisfiable -- an unsatisfiable schema is
    /// refused when it is compiled -- so this is the check that keeps that
    /// promise honest rather than an assumption the grammar relies on.
    var canProduceValue: Bool {
        switch self {
        case .enumeration(let literals):
            return !literals.isEmpty
        case .object(let properties, let required, let additional):
            return additional || required.allSatisfy { properties[$0] != nil }
        case .array, .scalar, .any:
            return true
        }
    }
}
