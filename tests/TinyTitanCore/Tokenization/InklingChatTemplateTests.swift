import Foundation
import Testing

@testable import TinyTitanCore

/// Model-free: Inkling's chat template as the server renders it (relocated from
/// `Inkling2GGUFChatFormatTests` under TT-127).
@Suite struct InklingChatTemplateTests {
    typealias Message = GFTokenizer.Message

    let weather = GFTokenizer.FunctionDefinition(
        name: "get_weather", description: "Look up the weather",
        parameters: .object([
            "type": .string("object"),
            "properties": .object(["city": .object(["type": .string("string")])]),
            "required": .array([.string("city")]),
        ]))

    @Test func plainTurnWithThinkingOffOpensATextMessage() {
        let text = InklingChatTemplate.render(
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

    @Test func toolConversationMatchesTheBundledTemplate() {
        let call = GFTokenizer.HistoricalToolCall(
            id: "call_1", name: "get_weather",
            arguments: .object(["unit": .string("C"), "city": .string("Köln")]))
        let text = InklingChatTemplate.render(
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
        let text = InklingChatTemplate.render(
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
            InklingChatTemplate.canonicalJSON(
                .object(["b": .integer(1), "a": .string("q\"\n\u{01}é"), "c": .number(0.5)]))
                == #"{"a":"q\"\n\u0001é","b":1,"c":0.5}"#)
    }
}
