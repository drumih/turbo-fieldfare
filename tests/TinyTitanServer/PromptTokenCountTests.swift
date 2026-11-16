import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

/// `POST /v1/messages/count_tokens` promises "the same encoding generation uses,
/// minus the generation", and it was not: it encoded `request.messages` and
/// `request.tools` verbatim while `preparePrompt` applies the CLI strip and then
/// prepends the concise-mode system prompt.
///
/// The two errors point in opposite directions — inflated for the
/// `<model>-fast` alias and whenever `TINYTITAN_STRIP_CLI_PROMPT` is set,
/// under-reported in concise mode — which is why neither showed up as a single
/// suspicious number. A client that budgets or compacts on this number gets it
/// wrong in whichever direction its configuration happens to be.
@Suite("Prompt token counting")
struct PromptTokenCountTests {
    private func request(
        system: String? = "You are a coding agent with a long preamble.",
        strip: Bool = false
    ) -> ValidatedChatRequest {
        var messages: [GFTokenizer.Message] = []
        if let system {
            messages.append(GFTokenizer.Message(role: .system, content: system))
        }
        messages.append(GFTokenizer.Message(role: .user, content: "hello there"))
        return ValidatedChatRequest(
            messages: messages,
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 4, temperature: 0),
            maximumCompletionTokens: 4,
            stripCLIPrompt: strip)
    }

    private func count(
        _ request: ValidatedChatRequest,
        concise: String? = nil
    ) async throws -> Int {
        let tokenizer = try await GFTokenizer.load(
            from: try TokenizerFixture.folder(),
            thinkingMode: .off)
        return try ServerModelSession.promptTokenCount(
            request,
            tokenizer: tokenizer,
            concisePrompt: concise)
    }

    @Test func theStripReducesTheCount() async throws {
        let plain = try await count(request())
        let stripped = try await count(request(strip: true))
        #expect(
            stripped < plain,
            Comment(
                rawValue:
                    "the CLI strip is not reflected: \(stripped) stripped vs \(plain) plain. "
                    + "The count must be the render generation would use."))
    }

    @Test func theConcisePromptIncreasesTheCount() async throws {
        let plain = try await count(request())
        let concise = try await count(request(), concise: ConcisePrompt.standard)
        #expect(
            concise > plain,
            Comment(
                rawValue:
                    "the concise system prompt is not counted: \(concise) vs \(plain). "
                    + "Generation appends it, so the count has to include it."))
    }

    /// Without a system message there is nothing for the strip to remove, so the
    /// two renders are the same and the count must not move. This is the control:
    /// it rules out "the count changed because something else did".
    @Test func nothingToStripMeansNoDifference() async throws {
        let plain = try await count(request(system: nil))
        let stripped = try await count(request(system: nil, strip: true))
        #expect(stripped == plain)
    }
}
