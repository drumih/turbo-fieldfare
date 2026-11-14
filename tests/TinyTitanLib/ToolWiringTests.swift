import Foundation
import Testing
import TinyTitan
import TinyTitanCore

@testable import TinyTitanLib

/// Tools, from the facade's types to the ones the validator reads.
///
/// Both directions of a tool loop have a mapping to get wrong: offering a tool
/// means parsing the caller's schema text, and feeding a result back means
/// replaying the assistant turn that asked, because a result is matched to the
/// call it answers.
@Suite struct ToolWiringTests {
    private let schema =
        #"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#

    @Test func anOfferedToolBecomesAWireTool() throws {
        let wire = try #require(
            try Session.wireTools([
                ToolDefinition(
                    name: "get_weather", description: "Look up the weather.",
                    parametersJSON: schema)
            ]))
        #expect(wire.count == 1)
        #expect(wire[0].type == "function")
        #expect(wire[0].function.name == "get_weather")
        #expect(wire[0].function.description == "Look up the weather.")
        // Parsed out of the text the facade carries, not passed through as one.
        #expect(wire[0].function.parameters.objectValue != nil)
    }

    @Test func anEmptyDescriptionIsOmittedAndNoToolsIsNil() throws {
        let wire = try #require(
            try Session.wireTools([ToolDefinition(name: "t", parametersJSON: "{}")]))
        #expect(wire[0].function.description == nil)
        #expect(try Session.wireTools([]) == nil)
    }

    /// A schema that is not JSON at all is the caller's mistake, and it is
    /// reported as one rather than reaching the validator as a mystery.
    @Test func parametersThatAreNotJSONAreTheCallersMistake() {
        #expect(throws: TinyTitanError.self) {
            _ = try Session.wireTools([ToolDefinition(name: "t", parametersJSON: "{not json")])
        }
    }

    @Test func aConversationCarriesItsToolCallsAndResults() {
        let call = ToolCall(id: "call_1", name: "get_weather", argumentsJSON: #"{"city":"Paris"}"#)
        let wire = Session.wireMessages(
            system: "Be terse.",
            messages: [
                ChatMessage(role: .user, content: "weather?"),
                ChatMessage(role: .assistant, content: "", toolCalls: [call]),
                ChatMessage(role: .tool, content: "18C", toolCallID: "call_1"),
            ])

        #expect(wire.count == 4)  // the system prompt, then the three turns
        #expect(wire[0].role == "system")
        #expect(wire[1].toolCalls == nil)
        #expect(wire[2].toolCalls?.first?.id == "call_1")
        #expect(wire[2].toolCalls?.first?.type == "function")
        #expect(wire[2].toolCalls?.first?.function.name == "get_weather")
        // The arguments stay the model's own JSON text.
        #expect(wire[2].toolCalls?.first?.function.arguments == #"{"city":"Paris"}"#)
        #expect(wire[3].role == "tool")
        #expect(wire[3].toolCallID == "call_1")
    }

    /// The pair the validator insists on: a tool result that names a call the
    /// conversation actually made validates, and one that names nothing does
    /// not. This is the half an embedder gets wrong when it drops the assistant
    /// turn from the history.
    @Test func aToolResultNeedsTheCallItAnswers() throws {
        let call = ToolCall(id: "call_1", name: "get_weather", argumentsJSON: "{}")
        let matched = OpenAIChatRequest(
            model: "test-model",
            messages: Session.wireMessages(
                system: nil,
                messages: [
                    ChatMessage(role: .user, content: "weather?"),
                    ChatMessage(role: .assistant, content: "", toolCalls: [call]),
                    ChatMessage(role: .tool, content: "18C", toolCallID: "call_1"),
                ]))
        _ = try OpenAIRequestValidator.validate(matched, modelID: "test-model", rules: .local)

        let orphaned = OpenAIChatRequest(
            model: "test-model",
            messages: Session.wireMessages(
                system: nil,
                messages: [
                    ChatMessage(role: .user, content: "weather?"),
                    ChatMessage(role: .tool, content: "18C", toolCallID: "call_1"),
                ]))
        #expect(throws: ServerRequestError.self) {
            _ = try OpenAIRequestValidator.validate(
                orphaned, modelID: "test-model", rules: .local)
        }
    }
}
