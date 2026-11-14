import Foundation
import Testing
import TinyTitan
import TinyTitanCore

@testable import TinyTitanLib

/// What the facade reports from the orchestrator's event stream.
///
/// One mapper serves both entry points — a conversation and a raw completion —
/// so they cannot drift apart in what they tell a caller. Everything the
/// orchestrator produces has a case here now; the `nil` arm is what a future
/// event this surface cannot express would use.
@Suite struct FacadeEventMappingTests {
    @Test func contentBecomesAToken() {
        guard case .token(let text)? = Session.facadeEvent(.content("hello")) else {
            Issue.record("content should map to a token event")
            return
        }
        #expect(text == "hello")
    }

    @Test func thePromptEventIsForwardedWithItsCounts() {
        guard
            case .promptProcessed(let tokens, let cachedTokens)? = Session.facadeEvent(
                .promptProcessed(tokens: 19, cachedTokens: 4))
        else {
            Issue.record("the prompt event should survive the mapping")
            return
        }
        #expect(tokens == 19)
        #expect(cachedTokens == 4)
    }

    /// The thought text is carried as its own event, not dropped: a caller that
    /// wants to show it beside the answer can, and one that judges the answer
    /// never reads it.
    @Test func reasoningIsCarriedAsItsOwnEvent() {
        guard case .reasoning(let text)? = Session.facadeEvent(.reasoning("thinking…")) else {
            Issue.record("reasoning should map to its own event")
            return
        }
        #expect(text == "thinking…")
    }

    /// A tool call crosses the boundary without the runtime's JSON value type:
    /// the facade hands out the arguments as the model's own JSON text.
    @Test func aToolCallCrossesAsTheFacadeType() {
        let parsed = ParsedToolCall(
            id: "call_1",
            name: "get_weather",
            arguments: .object(["city": .string("Paris")]),
            argumentsJSON: #"{"city":"Paris"}"#)
        guard case .toolCall(let call)? = Session.facadeEvent(.toolCall(parsed)) else {
            Issue.record("a tool call should map to the facade's own type")
            return
        }
        #expect(call.id == "call_1")
        #expect(call.name == "get_weather")
        #expect(call.argumentsJSON == #"{"city":"Paris"}"#)
    }
}
