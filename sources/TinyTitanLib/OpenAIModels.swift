import Foundation
import TinyTitan
import TinyTitanCore

package struct OpenAIChatRequest: Codable, Equatable, Sendable {
    package let model: String
    package let messages: [OpenAIChatMessage]
    package let stream: Bool?
    package let streamOptions: OpenAIStreamOptions?
    package let temperature: Float?
    package let topP: Float?
    package let maxTokens: Int?
    package let maxCompletionTokens: Int?
    package let stop: OpenAIStop?
    package let seed: UInt64?
    package let tools: [OpenAITool]?
    package let toolChoice: JSONValue?
    package let parallelToolCalls: Bool?
    package let topK: Int?
    package let repetitionPenalty: Float?
    package let n: Int?
    package let logprobs: Bool?
    package let presencePenalty: Float?
    package let frequencyPenalty: Float?
    /// Decoded so a non-zero value can be refused: the sampler has no min-p
    /// filter, and ignoring the field would sample as if it were zero while
    /// the client believes it asked for one.
    package let minP: Float?
    /// Requested reasoning-effort level. Validated against the served model
    /// family's chat template and the server's load-time profile.
    package let reasoningEffort: String?
    /// The same controls in the template-kwargs dialect (see above).
    package let chatTemplateKwargs: OpenAIChatTemplateKwargs?
    /// llama.cpp's hard per-request thinking-token budget.
    ///
    /// Decoded, never refused, and not enforced: this runtime bounds thinking by
    /// the level a template renders rather than by a token count, and a client
    /// that sends this also sends the level it wants. Refusing a field the
    /// runtime does not implement would break that client for the rest of the
    /// session, which is the failure the effort mapping already exists to avoid.
    package let reasoningBudgetTokens: Int?
    /// The requested output format. Decoded so a structured-output request can
    /// be *refused* rather than silently answered as prose: a client that asks
    /// for JSON and gets unconstrained text is worse off than one told no.
    /// `{"type": "text"}`, the API's own default, is accepted.
    package let responseFormat: JSONValue?
    /// Stream the prompt's reading progress before the first token.
    ///
    /// An opt-in because the chunks are an extension of OpenAI's shape: a
    /// client that did not ask for them must not receive unknown events. When
    /// set, each prefill chunk emits a `chat.completion.chunk` with empty
    /// `choices` and a `prompt_progress` object, which strict clients ignore
    /// exactly as they ignore the usage-only chunk.
    package let returnProgress: Bool?

    /// Explicit, with the two thinking-control extras defaulted, so the protocol
    /// mappers that build a chat request from their own shapes keep compiling
    /// unchanged. A `let` with an inline default would have been skipped by the
    /// synthesised decoder, which is how the budget silently decoded to nil.
    package init(
        model: String,
        messages: [OpenAIChatMessage],
        stream: Bool? = nil,
        streamOptions: OpenAIStreamOptions? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        maxTokens: Int? = nil,
        maxCompletionTokens: Int? = nil,
        stop: OpenAIStop? = nil,
        seed: UInt64? = nil,
        tools: [OpenAITool]? = nil,
        toolChoice: JSONValue? = nil,
        parallelToolCalls: Bool? = nil,
        topK: Int? = nil,
        repetitionPenalty: Float? = nil,
        n: Int? = nil,
        logprobs: Bool? = nil,
        presencePenalty: Float? = nil,
        frequencyPenalty: Float? = nil,
        minP: Float? = nil,
        reasoningEffort: String? = nil,
        chatTemplateKwargs: OpenAIChatTemplateKwargs? = nil,
        reasoningBudgetTokens: Int? = nil,
        responseFormat: JSONValue? = nil,
        returnProgress: Bool? = nil
    ) {
        self.model = model
        self.messages = messages
        self.stream = stream
        self.streamOptions = streamOptions
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.stop = stop
        self.seed = seed
        self.tools = tools
        self.toolChoice = toolChoice
        self.parallelToolCalls = parallelToolCalls
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.n = n
        self.logprobs = logprobs
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.minP = minP
        self.reasoningEffort = reasoningEffort
        self.chatTemplateKwargs = chatTemplateKwargs
        self.reasoningBudgetTokens = reasoningBudgetTokens
        self.responseFormat = responseFormat
        self.returnProgress = returnProgress
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, stop, seed, tools, n, logprobs
        case streamOptions = "stream_options"
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case minP = "min_p"
        case reasoningEffort = "reasoning_effort"
        case chatTemplateKwargs = "chat_template_kwargs"
        case reasoningBudgetTokens = "reasoning_budget_tokens"
        case responseFormat = "response_format"
        case returnProgress = "return_progress"
    }
}

/// The load-time reasoning configuration the HTTP layer validates requests
/// against: the served family's template capability plus the flags the model
/// was loaded with. Effort is a load-time control because it changes the
/// rendered prompt, so a request may only confirm the active level, never
/// switch it.
package struct ServerReasoningProfile: Sendable, Equatable {
    package let family: ModelFamily
    package let thinkingMode: ModelThinkingMode
    package let reasoningEffort: ModelReasoningEffort?

    package init(
        family: ModelFamily,
        thinkingMode: ModelThinkingMode,
        reasoningEffort: ModelReasoningEffort?
    ) {
        self.family = family
        self.thinkingMode = thinkingMode
        self.reasoningEffort = reasoningEffort
    }

    /// The compatible Qwen3.5-MoE baseline: binary thinking, off.
    package static let `default` = ServerReasoningProfile(
        family: .qwen36, thinkingMode: .off, reasoningEffort: nil)

    /// The effort the template actually applies under this profile; nil for
    /// binary families and while thinking is off.
    package var effectiveEffort: ModelReasoningEffort? {
        family.effectiveReasoningEffort(
            thinkingMode: thinkingMode,
            effort: reasoningEffort)
    }
}

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

package struct OpenAIModelList: Codable, Equatable, Sendable {
    package struct Model: Codable, Equatable, Sendable {
        package let id: String
        package let object: String
        /// Model creation time. Omitted when unknown rather than lying with a
        /// fabricated epoch (S30).
        package let created: Int?
        package let ownedBy: String

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }

        package init(id: String, object: String, created: Int?, ownedBy: String) {
            self.id = id
            self.object = object
            self.created = created
            self.ownedBy = ownedBy
        }
    }

    package let object: String
    package let data: [Model]

    // Declared rather than left to the memberwise initializer, which is
    // internal and so invisible to the server target that lists models
    // (2026-10-02, phase A1 of `docs/plan-embedded-library.md`).
    package init(object: String, data: [Model]) {
        self.object = object
        self.data = data
    }
}
