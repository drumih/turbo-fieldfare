import TurboFieldfare

enum GemmaToolSchema {
    private static let types: Set<String> = [
        "array", "boolean", "integer", "null", "number", "object", "string",
    ]
    private static let annotations: Set<String> = [
        "$comment", "default", "deprecated", "description", "examples", "readOnly",
        "title", "writeOnly",
    ]

    static func adapted(_ schema: JSONValue, toolName: String) throws -> JSONValue {
        let value = try adapt(schema, toolName: toolName, path: "parameters")
        guard value.objectValue?["type"] == .string("object"),
              value.objectValue?["nullable"] != .bool(true) else {
            throw invalid(toolName, "parameters", "top-level type must be a non-null object")
        }
        return value
    }

    private static func adapt(_ schema: JSONValue,
                              toolName: String,
                              path: String) throws -> JSONValue {
        guard case .object(var object) = schema else {
            throw invalid(toolName, path, "schema must be an object")
        }
        if object["allOf"] != nil {
            throw invalid(toolName, path, "allOf is not representable by the Gemma template")
        }
        if object["anyOf"] != nil || object["oneOf"] != nil {
            object = try adaptNullableUnion(object, toolName: toolName, path: path)
        }

        let type: String
        switch object["type"] {
        case .string(let name):
            type = try validatedType(name, toolName: toolName, path: path)
        case .array(let values):
            type = try adaptNullableTypeArray(
                values, object: &object, toolName: toolName, path: path)
        case nil:
            throw invalid(toolName, path, "type is required")
        default:
            throw invalid(toolName, path, "type must be a string or a nullable type array")
        }

        if let properties = object["properties"] {
            guard type == "object", case .object(let definitions) = properties else {
                throw invalid(toolName, path, "properties requires an object type and object value")
            }
            var adapted: [String: JSONValue] = [:]
            adapted.reserveCapacity(definitions.count)
            for key in definitions.keys.sorted() {
                adapted[key] = try adapt(
                    definitions[key]!,
                    toolName: toolName,
                    path: "\(path).properties.\(key)")
            }
            object["properties"] = .object(adapted)
        }

        if let items = object["items"] {
            guard type == "array" else {
                throw invalid(toolName, path, "items requires an array type")
            }
            guard case .object = items else {
                throw invalid(toolName, "\(path).items", "tuple and boolean item schemas are unsupported")
            }
            object["items"] = try adapt(
                items, toolName: toolName, path: "\(path).items")
        }
        return .object(object)
    }

    private static func adaptNullableTypeArray(
        _ values: [JSONValue],
        object: inout [String: JSONValue],
        toolName: String,
        path: String
    ) throws -> String {
        let names = try values.map { value -> String in
            guard case .string(let name) = value else {
                throw invalid(toolName, path, "type array must contain only strings")
            }
            return try validatedType(name, toolName: toolName, path: path)
        }
        let unique = Set(names)
        let concrete = unique.subtracting(["null"])
        guard unique.count == 2, unique.contains("null"), concrete.count == 1,
              let type = concrete.first else {
            throw invalid(toolName, path, "only one concrete type plus null is representable")
        }
        if let nullable = object["nullable"], nullable != .bool(true) {
            throw invalid(toolName, path, "nullable union conflicts with nullable keyword")
        }
        object["type"] = .string(type)
        object["nullable"] = .bool(true)
        return type
    }

    private static func adaptNullableUnion(
        _ source: [String: JSONValue],
        toolName: String,
        path: String
    ) throws -> [String: JSONValue] {
        let keyword = source["anyOf"] != nil ? "anyOf" : "oneOf"
        guard !(source["anyOf"] != nil && source["oneOf"] != nil),
              case .array(let branches)? = source[keyword], branches.count == 2 else {
            throw invalid(toolName, path, "only a two-branch nullable union is representable")
        }
        let nullIndexes = branches.indices.filter { index in
            branches[index] == .object(["type": .string("null")])
        }
        guard nullIndexes.count == 1 else {
            throw invalid(toolName, path, "union must contain one exact null branch")
        }
        let concreteIndex = nullIndexes[0] == 0 ? 1 : 0
        guard case .object(let concreteSource) = branches[concreteIndex] else {
            throw invalid(toolName, path, "non-null union branch must be an object schema")
        }
        let concreteValue = try adapt(
            .object(concreteSource), toolName: toolName, path: path)
        guard case .object(let concrete) = concreteValue,
              concrete["type"] != .string("null") else {
            throw invalid(toolName, path, "union requires one non-null typed branch")
        }
        if keyword == "oneOf", concrete["nullable"] == .bool(true) {
            throw invalid(toolName, path, "oneOf branches overlap on null")
        }

        var result = source
        result[keyword] = nil
        if let nullable = result["nullable"], nullable != .bool(true) {
            throw invalid(toolName, path, "nullable union conflicts with nullable keyword")
        }
        for (key, value) in concrete {
            if let existing = result[key], existing != value,
               !annotations.contains(key) {
                throw invalid(toolName, path, "union branch conflicts with sibling keyword \(key)")
            }
            if result[key] == nil || annotations.contains(key) {
                result[key] = result[key] ?? value
            }
        }
        result["nullable"] = .bool(true)
        return result
    }

    private static func validatedType(_ name: String,
                                      toolName: String,
                                      path: String) throws -> String {
        guard types.contains(name) else {
            throw invalid(toolName, path, "unknown JSON Schema type \(name)")
        }
        return name
    }

    private static func invalid(_ toolName: String,
                                _ path: String,
                                _ reason: String) -> ServerRequestError {
        .invalid(message: "tool \(toolName) schema at \(path): \(reason)",
                 param: "tools",
                 code: "invalid_tool_schema")
    }
}
