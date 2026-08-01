import Foundation
import Testing
@testable import TurboFieldfare

@Suite("Qwen tool call parser")
struct QwenToolCallParserTests {
    private let parser = QwenToolCallParser()
    private let tools: Set<String> = ["get_weather", "run_query", "no_args"]

    private func parse(_ body: String,
                       allowedTools: Set<String>? = nil) throws -> ParsedToolCall {
        try parser.parse(body, allowedTools: allowedTools ?? tools, id: "call_test")
    }

    @Test("Happy path with a single string parameter")
    func singleStringParameter() throws {
        let call = try parse("""

        <function=get_weather>
        <parameter=city>
        Paris
        </parameter>
        </function>

        """)
        #expect(call.name == "get_weather")
        #expect(call.arguments == .object(["city": .string("Paris")]))
        #expect(call.argumentsJSON == #"{"city":"Paris"}"#)
        #expect(call.id == "call_test")
    }

    @Test("JSON values become typed, non-JSON stays string")
    func jsonAndStringValues() throws {
        let call = try parse("""

        <function=run_query>
        <parameter=limit>
        25
        </parameter>
        <parameter=filters>
        {"active": true, "tags": ["a", "b"]}
        </parameter>
        <parameter=ratio>
        0.5
        </parameter>
        <parameter=verbose>
        true
        </parameter>
        <parameter=cursor>
        null
        </parameter>
        <parameter=query>
        SELECT * FROM t
        </parameter>
        </function>

        """)
        #expect(call.arguments == .object([
            "limit": .integer(25),
            "filters": .object([
                "active": .bool(true),
                "tags": .array([.string("a"), .string("b")]),
            ]),
            "ratio": .decimal(Decimal(string: "0.5", locale: Locale(identifier: "en_US_POSIX"))!),
            "verbose": .bool(true),
            "cursor": .null,
            "query": .string("SELECT * FROM t"),
        ]))
    }

    @Test("Multi-line values are preserved verbatim")
    func multiLineValue() throws {
        let call = try parse("""

        <function=run_query>
        <parameter=script>
        line one
          line two
        line three
        </parameter>
        </function>

        """)
        #expect(call.arguments == .object([
            "script": .string("line one\n  line two\nline three"),
        ]))
    }

    @Test("Quoted JSON strings keep their literal quotes")
    func quotedStringStaysRaw() throws {
        let call = try parse("""

        <function=run_query>
        <parameter=q>
        "hello"
        </parameter>
        </function>

        """)
        #expect(call.arguments == .object(["q": .string("\"hello\"")]))
    }

    @Test("Zero-parameter call parses to an empty object")
    func noParameters() throws {
        let call = try parse("\n<function=no_args>\n</function>\n")
        #expect(call.name == "no_args")
        #expect(call.arguments == .object([:]))
        #expect(call.argumentsJSON == "{}")
    }

    @Test("Empty parameter value parses to an empty string")
    func emptyValue() throws {
        let call = try parse("""

        <function=run_query>
        <parameter=q>

        </parameter>
        </function>

        """)
        #expect(call.arguments == .object(["q": .string("")]))
    }

    @Test("Unknown tools fail closed")
    func unknownTool() {
        #expect(throws: ToolCallParserError.unknownTool("secret_tool")) {
            _ = try parse("\n<function=secret_tool>\n</function>\n")
        }
    }

    @Test("Invalid function names are malformed even if allowed", arguments: [
        "a b", "café", "x/y", "", String(repeating: "a", count: 65),
    ])
    func invalidFunctionName(_ name: String) {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse("\n<function=\(name)>\n</function>\n",
                          allowedTools: [name])
        }
    }

    @Test("Malformed bodies are rejected", arguments: [
        "just text",
        "<function=get_weather></function>",
        "\n<function=get_weather>\n<parameter=city>\nParis\n</function>\n",
        "\n<function=get_weather>\n<parameter=>\nx\n</parameter>\n</function>\n",
        "\n<function=get_weather>\n</function>\ntrailing junk",
        "\n<function=get_weather>\n</function>\n<function=get_weather>\n</function>\n",
        "",
    ])
    func malformedBodies(_ body: String) {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try parse(body)
        }
    }

    @Test("Oversized bodies are rejected before parsing")
    func oversized() {
        let body = "\n<function=run_query>\n<parameter=q>\n"
            + String(repeating: "x", count: QwenToolCallParser.maximumBytes)
            + "\n</parameter>\n</function>\n"
        #expect(throws: ToolCallParserError.oversized) {
            _ = try parse(body)
        }
    }

    @Test("Duplicate parameter keys keep the last value")
    func duplicateKeysLastWins() throws {
        let call = try parse("""

        <function=run_query>
        <parameter=q>
        first
        </parameter>
        <parameter=q>
        second
        </parameter>
        </function>

        """)
        #expect(call.arguments == .object(["q": .string("second")]))
    }
}
