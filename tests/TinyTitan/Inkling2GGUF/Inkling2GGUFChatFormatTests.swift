import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// Model-free: Inkling's chat template as the server renders it, and the
/// parser that splits generated tokens into text, thinking and tool calls.
@Suite struct Inkling2GGUFChatFormatTests {
    typealias Message = GFTokenizer.Message

    let weather = GFTokenizer.FunctionDefinition(
        name: "get_weather", description: "Look up the weather",
        parameters: .object([
            "type": .string("object"),
            "properties": .object(["city": .object(["type": .string("string")])]),
            "required": .array([.string("city")]),
        ]))

    @Test func plainTurnWithThinkingOffOpensATextMessage() {
        let text = Inkling2GGUFChatFormat.render(
            messages: [
                Message(role: .system, content: "Be brief."), Message(role: .user, content: "Hi"),
            ],
            tools: [], effort: nil, openTextTurn: true)
        #expect(
            text
                == "<|message_system|><|content_text|>Be brief.<|end_message|>"
                + "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                + "<|message_user|><|content_text|>Hi<|end_message|>"
                + "<|message_model|><|content_text|>")
    }

    /// The bytes the bundled `chat_template.jinja` renders for this
    /// conversation, read off the template: tool declaration first, sorted
    /// compact JSON, the call as `name` + `args`, the tool result headed by
    /// the tool's name, and no generation prompt.
    @Test func toolConversationMatchesTheBundledTemplate() {
        let call = GFTokenizer.HistoricalToolCall(
            id: "call_1", name: "get_weather",
            arguments: .object(["unit": .string("C"), "city": .string("Köln")]))
        let text = Inkling2GGUFChatFormat.render(
            messages: [
                Message(role: .user, content: "Weather in Köln?"),
                Message(role: .assistant, content: nil, toolCalls: [call]),
                Message(role: .tool, content: "{\"temp\":21}", toolCallID: "call_1"),
            ],
            tools: [weather], effort: .high, openTextTurn: false)
        #expect(
            text
                == "<|message_system|>tool_declare<|content_xml|>"
                + #"[{"description":"Look up the weather","name":"get_weather","#
                + #""parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"#
                + #""type":"object"},"type":"function"}]<|end_message|>"#
                + "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                + "<|message_user|><|content_text|>Weather in Köln?<|end_message|>"
                + "<|message_model|>get_weather<|content_invoke_tool_json|>"
                + #"{"name":"get_weather","args":{"city":"Köln","unit":"C"}}<|end_message|>"#
                + "<|content_model_end_sampling|>"
                + #"<|message_tool|>get_weather<|content_text|>{"temp":21}<|end_message|>"#)
    }

    @Test func assistantReasoningAndTextRenderInTemplateOrder() {
        let text = Inkling2GGUFChatFormat.render(
            messages: [
                Message(role: .user, content: "2+2?"),
                Message(role: .assistant, content: "4", reasoning: "Add them."),
            ],
            tools: [], effort: .low, openTextTurn: false)
        #expect(
            text.hasSuffix(
                "<|message_model|><|content_thinking|>Add them.<|end_message|>"
                    + "<|message_model|><|content_text|>4<|end_message|>"
                    + "<|content_model_end_sampling|>"))
        #expect(text.contains("Thinking effort level: 0.2"))
    }

    @Test func canonicalJSONEscapesLikePython() {
        #expect(
            Inkling2GGUFChatFormat.canonicalJSON(
                .object(["b": .integer(1), "a": .string("q\"\n\u{01}é"), "c": .number(0.5)]))
                == #"{"a":"q\"\n\u0001é","b":1,"c":0.5}"#)
    }

    // MARK: - Output parser

    /// A toy vocabulary: control ids 100-106, words 1-9.
    static let vocabulary = Inkling2GGUFOutputParser.Vocabulary(
        messageModel: 100, contentText: 101, contentThinking: 102, invokeToolJSON: 103,
        invokeToolText: 104, endMessage: 105, foreignTurn: [106],
        control: Set(100...106))
    static let words: [Int: String] = [
        1: "Hello", 2: " world", 3: "let me think", 4: "get_weather",
        5: #"{"name":"get_weather","args":{"city":"Köln"}}"#, 6: "rm_rf",
        7: #"{"name":"rm_rf","args":{}}"#, 8: "\u{FFFD}",
    ]

    func parser(tools: Set<String> = ["get_weather"], startInText: Bool = false)
        -> Inkling2GGUFOutputParser
    {
        Inkling2GGUFOutputParser(
            vocabulary: Self.vocabulary,
            decode: { $0.compactMap { Self.words[$0] }.joined() },
            offeredTools: tools, startInText: startInText)
    }

    func run(_ tokens: [Int], _ parser: inout Inkling2GGUFOutputParser)
        -> [StructuredAssistantEvent]
    {
        tokens.flatMap { parser.consume($0) } + parser.finish()
    }

    @Test func splitsThinkingTextAndToolCalls() throws {
        var parser = parser()
        let events = run([100, 102, 3, 105, 100, 101, 1, 2, 105, 100, 4, 103, 5, 105], &parser)
        #expect(events.count == 4)
        #expect(events[0] == .reasoning("let me think"))
        #expect(events[1] == .content("Hello"))
        #expect(events[2] == .content(" world"))
        guard case .toolCall(let call) = events[3] else {
            Issue.record("expected a tool call, got \(events[3])")
            return
        }
        #expect(call.name == "get_weather")
        #expect(call.arguments == .object(["city": .string("Köln")]))
        #expect(call.argumentsJSON == #"{"city":"Köln"}"#)
        #expect(call.id.hasPrefix("call_"))
    }

    @Test func aCallToAToolNobodyOfferedStaysVisibleAsText() {
        var parser = parser()
        let events = run([100, 6, 103, 7, 105], &parser)
        #expect(events == [.content(#"{"name":"rm_rf","args":{}}"#)])
    }

    @Test func aPromptThatOpenedTextStreamsFromTheFirstToken() {
        var parser = parser(startInText: true)
        #expect(run([1, 2, 105], &parser) == [.content("Hello"), .content(" world")])
    }

    @Test func aForeignTurnEndsTheAnswer() {
        var parser = parser(startInText: true)
        #expect(parser.consume(1) == [.content("Hello")])
        #expect(parser.consume(106).isEmpty)
        #expect(parser.ended)
    }

    @Test func aHalfSpelledCharacterWaitsForTheNextToken() {
        var parser = parser(startInText: true)
        #expect(parser.consume(8).isEmpty)
        #expect(parser.finish() == [.content("\u{FFFD}")])
    }
}
