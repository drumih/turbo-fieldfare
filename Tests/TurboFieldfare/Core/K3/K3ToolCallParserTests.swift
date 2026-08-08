import Foundation
import Testing
@testable import TurboFieldfare

/// `K3ToolCallParser` over handcrafted generated XTML, including the exact
/// call framing the `tool_loop` golden fixture renders (see
/// `Fixtures/k3/xtml_fixtures.json`).
@Suite struct K3ToolCallParserTests {

    /// The `tool_loop` fixture's assistant turn as generated text (the stream
    /// starts after the generation prompt's `<|open|>think<|sep|>`).
    private static let toolLoopTurn = """
        Need two weather calls.<|close|>think<|sep|><|open|>response<|sep|>Let me check both.<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|><|open|>argument key="city" type="string"<|sep|>Seoul<|close|>argument<|sep|><|close|>call<|sep|><|open|>call tool="get_weather" index="2"<|sep|><|open|>argument key="city" type="string"<|sep|>Busan<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>
        """

    @Test func toolLoopFixtureShape() {
        let parsed = K3ToolCallParser.parse(Self.toolLoopTurn)
        #expect(!parsed.malformed)
        #expect(parsed.reasoningContent == "Need two weather calls.")
        #expect(parsed.content == "Let me check both.")
        #expect(parsed.toolCalls == [
            K3ToolCall(name: "get_weather",
                       arguments: .object(.object(["city": .string("Seoul")]))),
            K3ToolCall(name: "get_weather",
                       arguments: .object(.object(["city": .string("Busan")]))),
        ])
    }

    @Test func plainThinkingTurnWithoutTools() {
        let parsed = K3ToolCallParser.parse(
            "User greeted me casually.<|close|>think<|sep|>"
                + "<|open|>response<|sep|>Hello! How can I help?<|close|>response<|sep|>"
                + "<|close|>message<|sep|>")
        #expect(!parsed.malformed)
        #expect(parsed.reasoningContent == "User greeted me casually.")
        #expect(parsed.content == "Hello! How can I help?")
        #expect(parsed.toolCalls.isEmpty)
    }

    @Test func nonThinkingTurnStartsInResponse() {
        let parsed = K3ToolCallParser.parse(
            "Just the answer.<|close|>response<|sep|><|close|>message<|sep|>",
            thinking: false)
        #expect(!parsed.malformed)
        #expect(parsed.reasoningContent == nil)
        #expect(parsed.content == "Just the answer.")
        #expect(parsed.toolCalls.isEmpty)
    }

    @Test func emptyThinkAndResponseChannels() {
        let parsed = K3ToolCallParser.parse(
            "<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>"
                + "<|close|>message<|sep|>")
        #expect(!parsed.malformed)
        #expect(parsed.reasoningContent == "")
        #expect(parsed.content == "")
    }

    @Test func argumentTypesAreCoercedFromJSON() {
        let parsed = K3ToolCallParser.parse(
            "r<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>"
                + "<|open|>tools<|sep|><|open|>call tool=\"fn\" index=\"1\"<|sep|>"
                + "<|open|>argument key=\"count\" type=\"number\"<|sep|>3<|close|>argument<|sep|>"
                + "<|open|>argument key=\"ratio\" type=\"number\"<|sep|>2.5<|close|>argument<|sep|>"
                + "<|open|>argument key=\"ok\" type=\"boolean\"<|sep|>true<|close|>argument<|sep|>"
                + "<|open|>argument key=\"tags\" type=\"array\"<|sep|>[\"a\", \"b\"]<|close|>argument<|sep|>"
                + "<|open|>argument key=\"opts\" type=\"object\"<|sep|>{\"x\": null}<|close|>argument<|sep|>"
                + "<|open|>argument key=\"missing\" type=\"null\"<|sep|>null<|close|>argument<|sep|>"
                + "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>")
        #expect(!parsed.malformed)
        guard case .object(let object) = parsed.toolCalls.first?.arguments,
              case .object(let arguments) = object else {
            Issue.record("expected object arguments")
            return
        }
        #expect(arguments["count"] == .integer(3))
        #expect(arguments["ratio"] == .decimal(Decimal(2.5)))
        #expect(arguments["ok"] == .bool(true))
        #expect(arguments["tags"] == .array([.string("a"), .string("b")]))
        #expect(arguments["opts"] == .object(["x": .null]))
        #expect(arguments["missing"] == .null)
    }

    @Test func stringArgumentsKeepJSONLikeText() {
        let parsed = K3ToolCallParser.parse(
            "<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>"
                + "<|open|>tools<|sep|><|open|>call tool=\"fn\" index=\"1\"<|sep|>"
                + "<|open|>argument key=\"verbatim\" type=\"string\"<|sep|>true<|close|>argument<|sep|>"
                + "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>")
        guard case .object(let object) = parsed.toolCalls.first?.arguments,
              case .object(let arguments) = object else {
            Issue.record("expected object arguments")
            return
        }
        #expect(arguments["verbatim"] == .string("true"))
    }

    @Test func rawJSONBlockArgument() {
        let parsed = K3ToolCallParser.parse(
            "<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>"
                + "<|open|>tools<|sep|><|open|>call tool=\"fn\" index=\"1\"<|sep|>"
                + "<|open|>json type=\"object\"<|sep|>{\"city\": \"Seoul\"}<|close|>json<|sep|>"
                + "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>")
        #expect(!parsed.malformed)
        #expect(parsed.toolCalls == [
            K3ToolCall(name: "fn",
                       arguments: .object(.object(["city": .string("Seoul")]))),
        ])
    }

    @Test func unparseableJSONBlockFlagsMalformed() {
        let parsed = K3ToolCallParser.parse(
            "<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>"
                + "<|open|>tools<|sep|><|open|>call tool=\"fn\" index=\"1\"<|sep|>"
                + "<|open|>json type=\"object\"<|sep|>not json<|close|>json<|sep|>"
                + "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>")
        #expect(parsed.malformed)
        #expect(parsed.toolCalls == [
            K3ToolCall(name: "fn", arguments: .jsonString("not json")),
        ])
    }

    @Test func attributeEscapesAreUnescaped() {
        let parsed = K3ToolCallParser.parse(
            "<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>"
                + "<|open|>tools<|sep|><|open|>call tool=\"a&amp;b&quot;c\" index=\"1\"<|sep|>"
                + "<|open|>argument key=\"k&amp;&quot;\" type=\"string\"<|sep|>v<|close|>argument<|sep|>"
                + "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>")
        #expect(!parsed.malformed)
        #expect(parsed.toolCalls.first?.name == "a&b\"c")
        guard case .object(let object) = parsed.toolCalls.first?.arguments,
              case .object(let arguments) = object else {
            Issue.record("expected object arguments")
            return
        }
        #expect(arguments["k&\""] == .string("v"))
    }

    @Test func truncationMidArgumentReturnsPartials() {
        let truncated = """
            Need a call.<|close|>think<|sep|><|open|>response<|sep|>Checking.<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|><|open|>argument key="city" type="string"<|sep|>Seo
            """
        let parsed = K3ToolCallParser.parse(truncated)
        #expect(parsed.malformed)
        #expect(parsed.reasoningContent == "Need a call.")
        #expect(parsed.content == "Checking.")
        // The incomplete call is dropped; the completed prefix survives.
        #expect(parsed.toolCalls.isEmpty)
    }

    @Test func truncationAfterFirstCallKeepsCompletedCalls() {
        let truncated = """
            r<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|><|open|>argument key="city" type="string"<|sep|>Seoul<|close|>argument<|sep|><|close|>call<|sep|><|open|>call tool="get_weather" index="2"<|sep|><|open|>argument key="ci
            """
        let parsed = K3ToolCallParser.parse(truncated)
        #expect(parsed.malformed)
        #expect(parsed.toolCalls == [
            K3ToolCall(name: "get_weather",
                       arguments: .object(.object(["city": .string("Seoul")]))),
        ])
    }

    @Test func truncationMidThinkReturnsPartialReasoning() {
        let parsed = K3ToolCallParser.parse("Still thinking about")
        #expect(parsed.malformed)
        #expect(parsed.reasoningContent == "Still thinking about")
        #expect(parsed.content == "")
        #expect(parsed.toolCalls.isEmpty)
    }

    @Test func truncationMidResponseIsFlagged() {
        let parsed = K3ToolCallParser.parse(
            "r<|close|>think<|sep|><|open|>response<|sep|>partial ans")
        #expect(parsed.malformed)
        #expect(parsed.content == "partial ans")
    }

    @Test func strayTextBetweenChannelsFlagsMalformed() {
        let parsed = K3ToolCallParser.parse(
            "r<|close|>think<|sep|>garbage<|open|>response<|sep|>c"
                + "<|close|>response<|sep|><|close|>message<|sep|>")
        #expect(parsed.malformed)
        #expect(parsed.content == "c")
    }

    @Test func missingToolNameFlagsMalformedAndSkipsCall() {
        let parsed = K3ToolCallParser.parse(
            "r<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>"
                + "<|open|>tools<|sep|><|open|>call index=\"1\"<|sep|>"
                + "<|close|>call<|sep|>"
                + "<|open|>call tool=\"fn\" index=\"2\"<|sep|><|close|>call<|sep|>"
                + "<|close|>tools<|sep|><|close|>message<|sep|>")
        #expect(parsed.malformed)
        #expect(parsed.toolCalls == [K3ToolCall(name: "fn", arguments: .object(.object([:])))])
    }

    @Test func callWithoutArgumentsYieldsEmptyObject() {
        let parsed = K3ToolCallParser.parse(
            "r<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>"
                + "<|open|>tools<|sep|><|open|>call tool=\"ping\" index=\"1\"<|sep|>"
                + "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>")
        #expect(!parsed.malformed)
        #expect(parsed.toolCalls == [
            K3ToolCall(name: "ping", arguments: .object(.object([:]))),
        ])
    }
}
