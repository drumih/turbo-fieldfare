/// Token accounting for a completion, shared by the 7.0 server backend contract
/// so usage has one wire shape independent of the engine.
package struct OpenAIUsage: Codable, Equatable, Sendable {
    package struct PromptTokensDetails: Codable, Equatable, Sendable {
        package let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }

        package init(cachedTokens: Int) {
            self.cachedTokens = cachedTokens
        }
    }

    /// How many of `completion_tokens` the model spent thinking.
    ///
    /// The clients that route reasoning separately (llama.cpp's patched servers,
    /// the coding harnesses built on them) read this to bill and to budget; the
    /// runtime already knows the split, because the decoder puts every token in
    /// one channel or the other. Kept as its own object rather than folded into
    /// `completion_tokens` so a client can see both.
    package struct CompletionTokensDetails: Codable, Equatable, Sendable {
        package let reasoningTokens: Int

        enum CodingKeys: String, CodingKey {
            case reasoningTokens = "reasoning_tokens"
        }

        package init(reasoningTokens: Int) {
            self.reasoningTokens = reasoningTokens
        }
    }

    package let promptTokens: Int
    package let completionTokens: Int
    package let totalTokens: Int
    package let promptTokensDetails: PromptTokensDetails
    package let completionTokensDetails: CompletionTokensDetails

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
        case completionTokensDetails = "completion_tokens_details"
    }

    package init(
        promptTokens: Int,
        completionTokens: Int,
        totalTokens: Int,
        cachedTokens: Int = 0,
        reasoningTokens: Int = 0
    ) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: cachedTokens)
        self.completionTokensDetails = CompletionTokensDetails(
            reasoningTokens: reasoningTokens)
    }
}
