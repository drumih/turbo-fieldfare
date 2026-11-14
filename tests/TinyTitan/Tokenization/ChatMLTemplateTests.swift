import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// ChatML (Qwen) dialect coverage against a synthetic tokenizer fixture: a
/// byte-level BPE vocab plus the nine ChatML special tokens at their real
/// Qwen3.6 IDs, with the model's `chat_template.jinja` alongside.
@Suite("ChatML template")
struct ChatMLTemplateTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load(from: Self.fixtureFolder())
    }

    static func fixtureFolder() throws -> URL {
        try #require(
            Bundle.module.url(
                forResource: "ChatMLTokenizer",
                withExtension: nil,
                subdirectory: "Fixtures"))
    }

    private typealias Message = GFTokenizer.Message

    @Test("Special-token IDs match the Qwen3.6 contract")
    func specialTokenIDs() {
        #expect(tok.endOfTurnID == 248046)
        #expect(tok.eosID == 248044)
        #expect(tok.toolCallStartID == 248058)
        #expect(tok.toolCallEndID == 248059)
        #expect(tok.toolResponseID == 248066)
        #expect(tok.toolResponseEndID == 248067)
        #expect(tok.thinkStartID == 248068)
        #expect(tok.thinkEndID == 248069)
    }

    @Test("Stop tokens are im_end and endoftext only")
    func stopTokens() {
        #expect(tok.stopTokenIDs == [tok.endOfTurnID, tok.eosID])
        #expect(tok.stopTokenIDs.count == 2)
    }

    @Test("Logits vocab is the model's padded row count")
    func vocabSize() {
        #expect(tok.vocabSize == 248_320)
    }

    @Test("Encode never prepends a BOS")
    func noBOS() {
        let with = tok.encode("hi", addBOS: true)
        let without = tok.encode("hi", addBOS: false)
        #expect(with == without)
    }

    @Test("Single user turn renders the exact ChatML string")
    func singleUserTurn() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(
            p == "<|im_start|>user\nHi<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test("Thinking mode leaves the generation think block open")
    func thinkingModeGenerationPrompt() async throws {
        let thinking = try await GFTokenizer.load(
            from: Self.fixtureFolder(), thinkingMode: .on)
        let prompt = try thinking.applyChatTemplate([
            Message(role: .user, content: "Hi")
        ])
        #expect(thinking.thinkingMode == .on)
        #expect(prompt.hasSuffix("<|im_start|>assistant\n<think>\n"))
        #expect(!prompt.hasSuffix("<think>\n\n</think>\n\n"))
    }

    @Test("Environment compatibility resolves only the documented binary modes")
    func thinkingModeEnvironmentCompatibility() {
        #expect(ModelThinkingMode.resolved(environment: [:]) == .off)
        #expect(
            ModelThinkingMode.resolved(
                environment: ["TINYTITAN_THINKING_MODE": "on"]) == .on)
        #expect(
            ModelThinkingMode.resolved(
                environment: ["TINYTITAN_THINKING_MODE": "medium"]) == .off)
    }

    @Test("Multi-turn renders roles verbatim with assistant unrenamed")
    func multiTurn() throws {
        let p = try tok.applyChatTemplate([
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "A"),
            Message(role: .assistant, content: "B"),
            Message(role: .user, content: "C"),
        ])
        #expect(
            p == "<|im_start|>system\nBe terse.<|im_end|>\n"
                + "<|im_start|>user\nA<|im_end|>\n"
                + "<|im_start|>assistant\nB<|im_end|>\n"
                + "<|im_start|>user\nC<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test("Message content is trimmed before rendering")
    func contentTrimming() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "  Hi \n")])
        #expect(p.contains("<|im_start|>user\nHi<|im_end|>\n"))
    }

    /// OpenAI's `developer` role is the documented successor of `system`, and
    /// this template defines only system/user/assistant/tool. Handing the name
    /// through raised the template's `Unexpected message role.`, which the Chat
    /// Completions surface answered as HTTP 500 — for a role pi-ai sends on
    /// every reasoning request.
    @Test("A developer message renders as the system turn it stands for")
    func developerRoleIsSystem() throws {
        let asDeveloper = try tok.applyChatTemplate([
            Message(role: .developer, content: "Be terse."),
            Message(role: .user, content: "Hi"),
        ])
        let asSystem = try tok.applyChatTemplate([
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Hi"),
        ])
        #expect(asDeveloper == asSystem)
        #expect(
            asDeveloper.hasPrefix("<|im_start|>system\nBe terse.<|im_end|>\n"),
            "the developer turn did not render as system: \(asDeveloper)")
    }

    @Test("The Jinja tool path renders a developer message as system too")
    func developerRoleThroughJinja() throws {
        let messages = [
            Message(role: .developer, content: "Be helpful."),
            Message(role: .user, content: "Hi"),
        ]
        let asDeveloper = try tok.encodeToolChat(messages: messages, tools: [])
        let asSystem = try tok.encodeToolChat(
            messages: [
                Message(role: .system, content: "Be helpful."),
                Message(role: .user, content: "Hi"),
            ],
            tools: [])
        #expect(asDeveloper == asSystem)
        let text = tok.decode(asDeveloper, skipSpecialTokens: false)
        #expect(
            text.hasPrefix("<|im_start|>system\nBe helpful."),
            "the developer turn did not render as system: \(text)")
    }

    @Test("System message after a user turn is rejected")
    func misplacedSystemTurn() {
        #expect(throws: GFTokenizerError.self) {
            _ = try tok.applyChatTemplate([
                Message(role: .user, content: "Hi"),
                Message(role: .system, content: "Too late"),
            ])
        }
    }

    @Test("Prompt encodes turn boundaries to the special IDs")
    func encodesToSpecialIDs() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        let ids = tok.encode(p, addBOS: false)
        #expect(
            ids.first == 248045, "expected <|im_start|> first, got \(String(describing: ids.first))"
        )
        #expect(ids.contains(tok.endOfTurnID))
        #expect(ids.contains(tok.thinkStartID ?? -1))
        #expect(ids.contains(tok.thinkEndID ?? -1))
        #expect(tok.decode(ids, skipSpecialTokens: false) == p)
    }

    @Test("Text continuation bridges from im_end into the next user turn")
    func textContinuation() throws {
        let ids = tok.encodeTextContinuation(userContent: " Next \n")
        #expect(ids.first == tok.endOfTurnID)
        let text = tok.decode(ids, skipSpecialTokens: false)
        #expect(
            text == "<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test("Tool-result KV continuation is unsupported for chatml")
    func toolResultContinuationUnsupported() {
        #expect(throws: GFTokenizerError.self) {
            _ = try tok.encodeToolResultContinuation(
                cachedMessages: [Message(role: .user, content: "Hi")],
                assistant: Message(
                    role: .assistant, content: nil,
                    toolCalls: [
                        .init(id: "call_1", name: "lookup", arguments: .object([:]))
                    ]),
                incomingMessages: [Message(role: .user, content: "Hi")],
                tools: [])
        }
    }

    @Test("Tool chat renders the bundled Jinja template with thinking disabled")
    func toolChatRendersJinja() throws {
        let ids = try tok.encodeToolChat(
            messages: [
                Message(role: .system, content: "Be helpful."),
                Message(role: .user, content: "Weather in Paris?"),
            ],
            tools: [
                .init(
                    name: "get_weather",
                    description: "Look up weather",
                    parameters: .object([
                        "type": .string("object"),
                        "properties": .object([
                            "city": .object(["type": .string("string")])
                        ]),
                    ]))
            ])
        let text = tok.decode(ids, skipSpecialTokens: false)
        #expect(text.hasPrefix("<|im_start|>system\n# Tools"))
        #expect(text.contains("get_weather"))
        #expect(text.contains("Be helpful."))
        #expect(text.contains("<|im_start|>user\nWeather in Paris?<|im_end|>\n"))
        let suffix = String(text.suffix(80))
        #expect(
            text.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "expected enable_thinking=false generation prompt, got suffix: \(suffix)")
    }

    /// A historical assistant turn renders its answer, not its reasoning.
    ///
    /// The bundled template takes everything after `</think>` as the content and
    /// wraps it back in `<think>…</think>` only for a turn *after* the last real
    /// user query — the turn being continued. The manual renderer emitted the raw
    /// content, so every previous turn's whole chain of thought stayed in the
    /// prompt: context the template exists to remove, in a shape the model was not
    /// trained on.
    @Test("Historical assistant reasoning is stripped, the continued turn keeps it")
    func assistantReasoningHandling() throws {
        let answer = Message(
            role: .assistant,
            content: "<think>\nlet me think\n</think>\n\nthe answer")

        // History: the assistant turn sits before the last user query.
        let history = try tok.applyChatTemplate([
            Message(role: .user, content: "first"),
            answer,
            Message(role: .user, content: "second"),
        ])
        #expect(
            !history.contains("let me think"),
            "a historical turn kept its reasoning: \(history)")
        #expect(
            history.contains("<|im_start|>assistant\nthe answer<|im_end|>"),
            "the answer is not rendered on its own: \(history)")

        // Continuation: the assistant turn is after the last user query, which is
        // the one case the template keeps the reasoning for.
        let continued = try tok.applyChatTemplate([
            Message(role: .user, content: "first"),
            answer,
        ])
        #expect(
            continued.contains("<think>\nlet me think\n</think>\n\nthe answer"),
            "the continued turn should keep its reasoning: \(continued)")
    }

    @Test("Tool chat uses the same explicit thinking mode as text chat")
    func thinkingToolChatRendersJinja() async throws {
        let thinking = try await GFTokenizer.load(
            from: Self.fixtureFolder(), thinkingMode: .on)
        let ids = try thinking.encodeToolChat(
            messages: [Message(role: .user, content: "Weather?")],
            tools: [
                .init(
                    name: "weather", description: "Look up weather",
                    parameters: .object(["type": .string("object")]))
            ])
        let text = thinking.decode(ids, skipSpecialTokens: false)
        #expect(text.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }
}
