import Foundation
import Testing

@testable import TinyTitan
import TinyTitanCore

/// Reasoning-effort rendering against the pinned upstream Qwen3.8-Flash-Next
/// `chat_template.jinja` (fetched 2026-08-27), paired with the synthetic
/// byte-level BPE fixture vocab. The template owns the effort wording; these
/// tests hold the runtime to it: the instruction lands at the head of the
/// system block, `medium` injects nothing, and thinking off disables effort.
@Suite("Qwen3.8 effort template")
struct Qwen38TemplateTests {
    private typealias Message = GFTokenizer.Message

    private static let xhighInstruction =
        "Reasoning effort is set to xhigh. Please think carefully through the "
        + "task, validate key assumptions, consider plausible alternatives, and "
        + "prioritize correctness, consistency, and clarity in the final answer."
    private static let lowInstruction =
        "Reasoning effort is set to low. Keep your thinking brief and focused, "
        + "moving directly to the conclusion without unnecessary elaboration."

    static func fixtureFolder() throws -> URL {
        try #require(
            Bundle.module.url(
                forResource: "Qwen38ChatMLTokenizer",
                withExtension: nil,
                subdirectory: "Fixtures"))
    }

    private static func load(
        thinkingMode: ModelThinkingMode,
        effort: ModelReasoningEffort?
    ) async throws -> GFTokenizer {
        try await GFTokenizer.load(
            from: fixtureFolder(),
            thinkingMode: thinkingMode,
            reasoningEffort: effort)
    }

    @Test("Thinking on defaults to the template's xhigh instruction")
    func defaultEffortInjectsXhigh() async throws {
        let tok = try await Self.load(thinkingMode: .on, effort: nil)
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(
            p == "<|im_start|>system\n" + Self.xhighInstruction + "<|im_end|>\n"
                + "<|im_start|>user\nHi<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n")
    }

    @Test("Explicit low effort injects the low instruction")
    func lowEffort() async throws {
        let tok = try await Self.load(thinkingMode: .on, effort: .low)
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(
            p == "<|im_start|>system\n" + Self.lowInstruction + "<|im_end|>\n"
                + "<|im_start|>user\nHi<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n")
    }

    @Test("Medium effort injects no instruction")
    func mediumEffort() async throws {
        let tok = try await Self.load(thinkingMode: .on, effort: .medium)
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(
            p == "<|im_start|>user\nHi<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n")
    }

    @Test("The instruction opens an existing system block")
    func effortPrependsToSystemMessage() async throws {
        let tok = try await Self.load(thinkingMode: .on, effort: .low)
        let p = try tok.applyChatTemplate([
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Hi"),
        ])
        #expect(
            p == "<|im_start|>system\n" + Self.lowInstruction + "\n\nBe terse.<|im_end|>\n"
                + "<|im_start|>user\nHi<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n")
    }

    /// A leading `developer` message *is* the conversation's system block
    /// (`Role.templateRole`), so the effort instruction folds into it rather
    /// than opening a synthetic block in front of it — the same shape a
    /// `system` first message produces, byte for byte.
    @Test("A leading developer message takes the instruction like system")
    func effortPrependsToDeveloperMessage() async throws {
        let tok = try await Self.load(thinkingMode: .on, effort: .low)
        let p = try tok.applyChatTemplate([
            Message(role: .developer, content: "Be terse."),
            Message(role: .user, content: "Hi"),
        ])
        #expect(
            p == "<|im_start|>system\n" + Self.lowInstruction + "\n\nBe terse.<|im_end|>\n"
                + "<|im_start|>user\nHi<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n")
    }

    @Test("Thinking off renders the closed think block and no instruction")
    func thinkingOff() async throws {
        let tok = try await Self.load(thinkingMode: .off, effort: nil)
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(
            p == "<|im_start|>user\nHi<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test("An effort passed with thinking off is stored as none")
    func effortClearedWhileOff() async throws {
        let tok = try await Self.load(thinkingMode: .off, effort: .low)
        #expect(tok.reasoningEffort == nil)
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(
            p == "<|im_start|>user\nHi<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test("The binary Qwen3.6 template ignores a reasoning effort")
    func binaryTemplateUnchanged() async throws {
        let folder = try ChatMLTemplateTests.fixtureFolder()
        let plain = try await GFTokenizer.load(from: folder, thinkingMode: .on)
        let effortful = try await GFTokenizer.load(
            from: folder,
            thinkingMode: .on,
            reasoningEffort: .xhigh)
        let messages = [Message(role: .user, content: "Hi")]
        #expect(
            try effortful.applyChatTemplate(messages)
                == plain.applyChatTemplate(messages))
    }
}
