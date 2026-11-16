/// A parsed tool call and the parser's error type, shared by the 7.0 engine
/// variants and the server backend contract (one spelling across the boundary).
public struct ParsedToolCall: Equatable, Sendable {
    public let id: String
    public let name: String
    public let arguments: JSONValue
    public let argumentsJSON: String

    public init(
        id: String,
        name: String,
        arguments: JSONValue,
        argumentsJSON: String
    ) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.argumentsJSON = argumentsJSON
    }
}

public enum ToolCallParserError: Error, Equatable {
    case malformed
    case unknownTool(String)
    case oversized
}
