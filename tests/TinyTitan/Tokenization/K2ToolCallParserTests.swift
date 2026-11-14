import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// K2-Horizon's three tool-call bodies. The checkpoint writes one wrapper
/// around one or more calls; the body is JSON, tag-paired XML, or typed XML,
/// and the three are distinct grammars rather than whitespace variants.
@Suite("K2 tool-call parser")
struct K2ToolCallParserTests {
    private static let tools: Set<String> = ["read_file", "shell.exec", "search"]
    private static let id = "call_test"

    private static func json(_ body: String) -> String {
        "<ifm|tool_calls><ifm|tool_call>\(body)</ifm|tool_call></ifm|tool_calls>"
    }

    @Test func jsonCallCarriesTypedArguments() throws {
        let text = Self.json(
            #"{"name":"read_file","arguments":{"path":"a.txt","lines":12,"raw":false}}"#)
        let calls = try K2ToolCallParser.parse(
            text, allowedTools: Self.tools, format: .json,
            makeID: { Self.id })
        #expect(calls.count == 1)
        #expect(calls[0].name == "read_file")
        #expect(calls[0].argumentsJSON == #"{"lines":12,"path":"a.txt","raw":false}"#)
    }

    @Test func jsonCallWithNoArgumentsIsAnEmptyObject() throws {
        let text = Self.json(#"{"name":"read_file"}"#)
        let calls = try K2ToolCallParser.parse(
            text, allowedTools: Self.tools, format: .json,
            makeID: { Self.id })
        #expect(calls[0].arguments == .object([:]))
    }

    @Test func xmlCallPairsEachKeyWithItsValue() throws {
        let text = """
            <ifm|tool_calls>
            <ifm|tool_call>read_file
            <ifm|arg_key>path</ifm|arg_key>
            <ifm|arg_value>notes/today.md</ifm|arg_value>
            <ifm|arg_key>lines</ifm|arg_key>
            <ifm|arg_value>42</ifm|arg_value>
            <ifm|arg_key>flags</ifm|arg_key>
            <ifm|arg_value>["fast"]</ifm|arg_value>
            </ifm|tool_call>
            </ifm|tool_calls>
            """
        let calls = try K2ToolCallParser.parse(
            text, allowedTools: Self.tools, format: .xml,
            makeID: { Self.id })
        #expect(calls.count == 1)
        #expect(calls[0].name == "read_file")
        #expect(calls[0].argumentsJSON == #"{"flags":["fast"],"lines":42,"path":"notes/today.md"}"#)
    }

    @Test func typedXMLAcceptsTheTypeTagAndKeepsTheValue() throws {
        let text = """
            <ifm|tool_calls><ifm|tool_call>shell.exec
            <ifm|arg_key>command</ifm|arg_key>
            <ifm|arg_type>string</ifm|arg_type>
            <ifm|arg_value>ls -la</ifm|arg_value>
            <ifm|arg_key>timeout</ifm|arg_key>
            <ifm|arg_type>integer</ifm|arg_type>
            <ifm|arg_value>30</ifm|arg_value>
            </ifm|tool_call></ifm|tool_calls>
            """
        let calls = try K2ToolCallParser.parse(
            text, allowedTools: Self.tools, format: .xmlTyped,
            makeID: { Self.id })
        #expect(calls[0].name == "shell.exec")
        #expect(calls[0].argumentsJSON == #"{"command":"ls -la","timeout":30}"#)
    }

    @Test func oneWrapperCanCarrySeveralCalls() throws {
        let text =
            Self.json(
                #"{"name":"read_file","arguments":{"path":"a"}}"#)
            + Self.json(#"{"name":"search","arguments":{"q":"b"}}"#)
        // Two wrappers is not one block, so the parser refuses it; the
        // multi-call form is one wrapper with two call bodies.
        #expect(throws: ToolCallParserError.malformed) {
            _ = try K2ToolCallParser.parse(
                text, allowedTools: Self.tools, format: .json,
                makeID: { Self.id })
        }
        let oneWrapper = """
            <ifm|tool_calls>
            <ifm|tool_call>{"name":"read_file","arguments":{"path":"a"}}</ifm|tool_call>
            <ifm|tool_call>{"name":"search","arguments":{"q":"b"}}</ifm|tool_call>
            </ifm|tool_calls>
            """
        let calls = try K2ToolCallParser.parse(
            oneWrapper, allowedTools: Self.tools, format: .json,
            makeID: { Self.id })
        #expect(calls.map(\.name) == ["read_file", "search"])
    }

    @Test func unknownToolIsRefused() {
        let text = Self.json(#"{"name":"delete_everything","arguments":{}}"#)
        #expect(throws: ToolCallParserError.unknownTool("delete_everything")) {
            _ = try K2ToolCallParser.parse(
                text, allowedTools: Self.tools, format: .json,
                makeID: { Self.id })
        }
    }

    @Test func malformedBodiesAreRefused() {
        for text in [
            "<ifm|tool_call>read_file</ifm|tool_call>",
            "<ifm|tool_calls><ifm|tool_call>read_file",
            "<ifm|tool_calls><ifm|tool_call>{not json}</ifm|tool_call></ifm|tool_calls>",
            "<ifm|tool_calls></ifm|tool_calls>",
        ] {
            #expect(throws: ToolCallParserError.malformed, "\(text)") {
                _ = try K2ToolCallParser.parse(
                    text, allowedTools: Self.tools, format: .json,
                    makeID: { Self.id })
            }
        }
    }

    @Test func oversizedBlockIsRefused() {
        let padding = String(repeating: " ", count: K2ToolCallParser.maximumBytes + 1)
        #expect(throws: ToolCallParserError.oversized) {
            _ = try K2ToolCallParser.parse(
                padding, allowedTools: Self.tools, format: .xml,
                makeID: { Self.id })
        }
    }
}
