import TinyTitanCore
import Foundation
import TinyTitan

/// Request validation results: the server's request errors and the validated
/// shape a handler receives.
///
/// Split out of `OpenAIModels.swift` (2026-09-28) under the 500-line-per-file
/// rule (Task 8 of the cleanup runbook) as pure code motion.
package enum ServerRequestError: Error, Equatable, Sendable {
    case invalid(message: String, param: String?, code: String)
    case unknownModel
    case queueFull
    /// A well-formed request for something this backend cannot do at all
    /// (a token count without a tokenizer, for instance): 501, not 400.
    case unsupportedOperation(String)
    /// A stored response named by `previous_response_id` or a path that
    /// does not exist: 404.
    case notFound(message: String, param: String?)

    package var envelope: OpenAIErrorEnvelope {
        switch self {
        case .invalid(let message, let param, let code):
            OpenAIErrorEnvelope(message: message, param: param, code: code)
        case .unknownModel:
            OpenAIErrorEnvelope(
                message: "requested model is not available",
                param: "model", code: "model_not_found")
        case .queueFull:
            OpenAIErrorEnvelope(
                message: "generation queue is full",
                code: "queue_full",
                type: "rate_limit_error")
        case .unsupportedOperation(let operation):
            OpenAIErrorEnvelope(
                message: "\(operation) is not supported by this backend",
                code: "unsupported_operation",
                type: "server_error")
        case .notFound(let message, let param):
            OpenAIErrorEnvelope(message: message, param: param, code: "not_found")
        }
    }

    /// The HTTP status each error maps to, shared by every API surface.
    package var httpStatus: Int {
        switch self {
        case .invalid: 400
        case .unknownModel, .notFound: 404
        case .queueFull: 429
        case .unsupportedOperation: 501
        }
    }
}

package struct ValidatedChatRequest: Sendable {
    package let messages: [GFTokenizer.Message]
    package let tools: [GFTokenizer.FunctionDefinition]
    package let stream: Bool
    package let includeUsage: Bool
    package let generationConfig: GenerationConfig
    package let maximumCompletionTokens: Int
    /// Set when the request named the "<model>-fast" alias: the CLI-strip
    /// heuristic runs for this request regardless of TINYTITAN_STRIP_CLI_PROMPT.
    package let stripCLIPrompt: Bool
    /// Memory workspace named by the X-TinyTitan-Workspace header, when the
    /// server allows a request to choose one. Nil means the workspace the
    /// server was launched with.
    package let workspace: String?
    /// True for a generation the engine asked for itself -- memory
    /// consolidation is the only one today. Watchdogs do not police these
    /// (B6): their prompts are repetitive by construction and their answers
    /// are meant to be terse, which is the shape the detectors hunt, and no
    /// person is waiting on the result.
    package let isEngineInternal: Bool
    /// The catalog id the request was validated against. The routing backend
    /// loads it; a single-model backend serves what it has and ignores it.
    /// Nil for the engine's own requests, which run on whatever is resident.
    package let model: String?
    /// Request fields that asked for something the served model cannot do and
    /// were answered by the nearest thing it can. Empty on the common path.
    ///
    /// These are never errors: a coding agent that names a reasoning level
    /// this project never defined keeps working, and the server says what it
    /// applied in its log. Additive and defaulted so every existing caller is
    /// unchanged.
    package let reasoningNotes: [String]
    /// The thinking mode and effort this request should actually render at.
    ///
    /// `nil` means "whatever the model was loaded with", which is the common
    /// path and keeps the session's own tokenizer. A value that differs from
    /// the loaded one is a mid-session switch: the session resolves a
    /// tokenizer for it instead of reusing its own, so a coding agent can
    /// turn thinking off (or change effort) inside a live session.
    ///
    /// Carried on the request rather than passed alongside it because the
    /// prompt cache keys on this value: two levels render different prompts,
    /// and a cached KV range from one must never be spliced onto the other.
    package let reasoning: RequestReasoning?
    /// The compiled JSON schema this request must produce, when it asked for
    /// structured output. Nil is free text -- the only value every caller but
    /// the three JSON spellings passes.
    ///
    /// The schema is compiled here, during validation, so an unsupported
    /// keyword is a 400 before a model is touched rather than a failure in the
    /// middle of a generation.
    package let jsonSchema: JSONSchemaNode?
    /// Prompt tokens a raw completion was already rendered to, or nil for the
    /// chat path.
    ///
    /// A raw prompt never goes through `applyChatTemplate` and is never
    /// matched against the prompt cache: the caller tokenized its own text, so
    /// there is no message list to re-render a tail from. Carried on the
    /// request so the one orchestrator can serve both, rather than a second
    /// decode path existing beside it.
    package let renderedPromptIDs: [Int32]?
    /// Stream the prompt's reading progress before the first token.
    package let returnProgress: Bool

    package init(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        stream: Bool,
        includeUsage: Bool,
        generationConfig: GenerationConfig,
        maximumCompletionTokens: Int,
        stripCLIPrompt: Bool = false,
        workspace: String? = nil,
        isEngineInternal: Bool = false,
        model: String? = nil,
        reasoningNotes: [String] = [],
        reasoning: RequestReasoning? = nil,
        jsonSchema: JSONSchemaNode? = nil,
        renderedPromptIDs: [Int32]? = nil,
        returnProgress: Bool = false
    ) {
        self.messages = messages
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.stripCLIPrompt = stripCLIPrompt
        self.workspace = workspace
        self.isEngineInternal = isEngineInternal
        self.model = model
        self.reasoningNotes = reasoningNotes
        self.reasoning = reasoning
        self.jsonSchema = jsonSchema
        self.renderedPromptIDs = renderedPromptIDs
        self.returnProgress = returnProgress
    }

    /// Every derived request is built through here.
    ///
    /// The three public helpers below used to rebuild the struct field by
    /// field, which silently dropped any field added later -- `jsonSchema` was
    /// lost that way the moment it existed, so a request that asked for
    /// structured output validated, then generated free text. One builder that
    /// carries every unmentioned field makes that impossible to repeat.
    private func copy(
        messages: [GFTokenizer.Message]? = nil,
        tools: [GFTokenizer.FunctionDefinition]? = nil,
        generationConfig: GenerationConfig? = nil,
        stripCLIPrompt: Bool? = nil,
        workspace: String?? = nil,
        isEngineInternal: Bool? = nil,
        model: String?? = nil
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages ?? self.messages,
            tools: tools ?? self.tools,
            stream: stream,
            includeUsage: includeUsage,
            generationConfig: generationConfig ?? self.generationConfig,
            maximumCompletionTokens: maximumCompletionTokens,
            stripCLIPrompt: stripCLIPrompt ?? self.stripCLIPrompt,
            workspace: workspace ?? self.workspace,
            isEngineInternal: isEngineInternal ?? self.isEngineInternal,
            model: model ?? self.model,
            reasoningNotes: reasoningNotes,
            reasoning: reasoning,
            jsonSchema: jsonSchema,
            renderedPromptIDs: renderedPromptIDs,
            returnProgress: returnProgress)
    }

    /// The same request with its generation config replaced.
    ///
    /// Used by the facade to honour an option the wire cannot spell (top-k
    /// off), never by the HTTP surfaces.
    package func withGenerationConfig(
        _ config: GenerationConfig
    ) -> ValidatedChatRequest {
        copy(generationConfig: config)
    }

    /// The post-strip view of this request: the same request carrying the
    /// messages and tools that were actually encoded into the prompt.
    ///
    /// The prompt cache must key on this view, not the raw request. Its
    /// entries describe a KV range that was prefilled from the filtered
    /// messages, and its continuation paths re-render the tail with the same
    /// template -- so matching on the raw messages would splice an unfiltered
    /// tail onto a filtered prefix (see `ServerPromptCache`).
    package func replacingMessages(
        _ messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) -> ValidatedChatRequest {
        copy(messages: messages, tools: tools)
    }

    /// The memory workspace this request names, from the X-TinyTitan-Workspace
    /// header. Nil takes the server's launch-time workspace, which is the
    /// usual case: one server, one checkout.
    package func withWorkspace(_ workspace: String?) -> ValidatedChatRequest {
        copy(workspace: .some(workspace))
    }

    /// The same request, bound to the catalog model it was validated for.
    package func withModel(_ model: String) -> ValidatedChatRequest {
        copy(model: .some(model))
    }
}
