import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLib
import TinyTitanCore

/// Which request limits apply to whom.
///
/// The structural rules are shared — a message needs content, guidance precedes
/// the conversation — but the wire's *caps* are OpenAI's numbers, written to
/// protect the engine from a third party's request. A caller inside the process
/// is not a third party, and the CLI never had those caps before it moved onto
/// the facade: an extra stop string was a regression, not a policy.
@Suite struct RequestRulesTests {
    private func request(stops: [String] = [], messages: Int = 1) -> OpenAIChatRequest {
        OpenAIChatRequest(
            model: "test-model",
            messages: (0..<messages).map { _ in
                OpenAIChatMessage(
                    role: "user", content: .text("hello"),
                    toolCalls: nil, toolCallID: nil, name: nil)
            },
            stop: stops.isEmpty ? nil : .many(stops))
    }

    @Test func theWireRulesEnforceTheWireCaps() {
        #expect(throws: ServerRequestError.self) {
            _ = try OpenAIRequestValidator.validate(
                request(stops: ["a", "b", "c", "d", "e"]), modelID: "test-model", rules: .wire)
        }
        #expect(throws: ServerRequestError.self) {
            _ = try OpenAIRequestValidator.validate(
                request(messages: 1_001), modelID: "test-model", rules: .wire)
        }
    }

    @Test func aLocalCallerIsNotHeldToThem() throws {
        _ = try OpenAIRequestValidator.validate(
            request(stops: ["a", "b", "c", "d", "e"]), modelID: "test-model", rules: .local)

        let many = try OpenAIRequestValidator.validate(
            request(messages: 1_001), modelID: "test-model", rules: .local)
        #expect(many.messages.count == 1_001)
    }

    /// The rules that are not caps still apply to everyone: an empty message and
    /// an unknown role are malformed whichever caller sent them.
    @Test func theStructuralRulesApplyToALocalCallerToo() {
        #expect(throws: ServerRequestError.self) {
            let empty = OpenAIChatRequest(
                model: "test-model",
                messages: [
                    OpenAIChatMessage(
                        role: "user", content: nil, toolCalls: nil, toolCallID: nil, name: nil)
                ])
            _ = try OpenAIRequestValidator.validate(empty, modelID: "test-model", rules: .local)
        }
        #expect(throws: ServerRequestError.self) {
            let unknown = OpenAIChatRequest(
                model: "test-model",
                messages: [
                    OpenAIChatMessage(
                        role: "wizard", content: .text("hi"),
                        toolCalls: nil, toolCallID: nil, name: nil)
                ])
            _ = try OpenAIRequestValidator.validate(unknown, modelID: "test-model", rules: .local)
        }
    }
}
