import Foundation
import TinyTitan
import TinyTitanCore
import TinyTitanLib

/// Single-model server backend for an Unsloth Inkling-2-Mini GGUF
/// (`UD-Q2_K_XL`, `UD-Q4_K_XL`) on the native decoder, through
/// `Inkling2GGUFSession` — the generation path the CLI's GGUF route uses.
///
/// Text only: tools are not offered to the model (no tool calls are
/// returned), and stop strings are not applied; generation stops on the
/// GGUF's end-of-sequence/turn tokens or the token budget.
package actor Inkling2GGUFServerBackend: ServerInferenceBackend {
    /// The session holds the decoder's Metal state; the actor serialises
    /// every request onto it.
    private let session: Inkling2GGUFSession
    package nonisolated let maximumContext: Int
    package nonisolated let samplingDefaults: SamplingProfile = .house

    /// The context the decoder's KV ring and caches are sized for: the
    /// server's `--max-context`, capped as the CLI caps it.
    package static let contextCap = 16_384

    package init(gguf: URL, maxContext: Int) async throws {
        let context = min(maxContext, Self.contextCap)
        session = try await Inkling2GGUFSession(gguf: gguf, maxContext: context)
        maximumContext = context
    }

    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let promptIDs =
            request.renderedPromptIDs.map { $0.map(Int.init) }
            ?? session.tokenizer.encode(Self.render(request.messages))
        onEvent(.promptProcessed(tokens: promptIDs.count, cachedTokens: 0))
        let configuration = request.generationConfig
        let sampler = CPUSampler(
            temperature: configuration.temperature, topP: configuration.topP ?? 1,
            topK: configuration.topK ?? 0, presencePenalty: configuration.presencePenalty,
            repetitionPenalty: configuration.repetitionPenalty, seed: configuration.seed)
        let result = try session.generate(
            promptIDs: promptIDs, maxTokens: max(configuration.maxNewTokens, 1),
            sampler: sampler
        ) { delta in onEvent(.content(delta)) }
        return ServerCompletion(
            content: result.text, toolCalls: [],
            finishReason: result.stopReason == "stop" ? "stop" : "length",
            usage: OpenAIUsage(
                promptTokens: promptIDs.count, completionTokens: result.tokens.count,
                totalTokens: promptIDs.count + result.tokens.count))
    }

    package func tokenizeRawPrompt(_ text: String) async throws -> [Int32] {
        session.tokenizer.encode(text).map(Int32.init)
    }

    /// The request's messages in Inkling's message format.
    static func render(_ messages: [GFTokenizer.Message]) -> String {
        Inkling2GGUFSession.render(
            messages.map { message in
                let role: Inkling2GGUFSession.Role
                switch message.role {
                case .system, .developer: role = .system
                case .user: role = .user
                case .assistant: role = .assistant
                case .tool: role = .tool
                }
                return (role, message.content ?? "")
            })
    }
}
