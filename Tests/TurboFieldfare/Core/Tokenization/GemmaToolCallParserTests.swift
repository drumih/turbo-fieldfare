import Testing
import Foundation
@testable import TurboFieldfare

/// Parser-level regressions for the dialect the Gemma 4 chat template renders:
/// hyphenated tool names (legal per `OpenAIToolName.isValid`) and string
/// arguments whose interior whitespace must survive verbatim.
@Suite struct GemmaToolCallParserTests {

    private func parse(_ text: String, tools: Set<String>) throws -> ParsedToolCall {
        try GemmaToolCallParser().parse(text, allowedTools: tools, id: "call_test")
    }

    @Test func hyphenatedToolNameDecodes() throws {
        let call = try parse(#"call:get-weather{city:<|"|>Rosario<|"|>}"#,
                             tools: ["get-weather"])
        #expect(call.name == "get-weather")
        #expect(call.arguments == .object(["city": .string("Rosario")]))
    }

    @Test func hyphenatedNameIsNotTruncatedAtTheHyphen() throws {
        // Before the fix the identifier stopped at `-`, so a declared `read`
        // matched `read-file` and the leftover `-file` failed as arguments.
        #expect(throws: GemmaToolCallParserError.unknownTool("read-file")) {
            try parse(#"call:read-file{path:<|"|>/a<|"|>}"#, tools: ["read"])
        }
        let call = try parse(#"call:read-file{path:<|"|>/a<|"|>}"#,
                             tools: ["read", "read-file"])
        #expect(call.name == "read-file")
    }

    @Test func mixedSeparatorNameDecodes() throws {
        let call = try parse(#"call:web_search-v2{q:<|"|>x<|"|>}"#,
                             tools: ["web_search-v2"])
        #expect(call.name == "web_search-v2")
    }

    @Test func gemmaStringPreservesWhitespaceAndPunctuation() throws {
        let call = try parse(#"call:f{s:<|"|>a , b 's c<|"|>}"#, tools: ["f"])
        #expect(call.arguments == .object(["s": .string("a , b 's c")]))
    }

    /// The `"`-delimited path used to run every character through a
    /// whitespace-skipping helper, so interior spaces vanished from the decoded
    /// argument while the request still looked successful.
    @Test func quotedStringPreservesInteriorWhitespace() throws {
        let call = try parse(#"call:f{path:"/tmp/a b c"}"#, tools: ["f"])
        #expect(call.arguments == .object(["path": .string("/tmp/a b c")]))
    }

    @Test func quotedStringPreservesTabsNewlinesAndEscapes() throws {
        let call = try parse("call:f{s:\" a\tb\nc \",t:\"x \\n y\"}", tools: ["f"])
        #expect(call.arguments == .object(["s": .string(" a\tb\nc "),
                                           "t": .string("x \n y")]))
    }

    @Test func quotedEmptyStringStillDecodes() throws {
        let call = try parse(#"call:f{s:""}"#, tools: ["f"])
        #expect(call.arguments == .object(["s": .string("")]))
    }
}
