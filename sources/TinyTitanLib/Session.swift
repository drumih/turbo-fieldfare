// One conversation over a loaded model.
//
// Part of the supported facade (phase A1 of `docs/plan-embedded-library.md`,
// §4). It is a thin, honest wrapper: `respond` builds the same validated
// request the chat surface builds, hands it to the moved orchestrator
// (`ServerModelSession.generate`), and maps the events and the completion back
// into kit-native shapes. Nothing about prompt rendering, caching or sampling
// is reimplemented here.
//
// Phase A1 gaps, reported rather than papered over:
//
//   * The orchestrator's callback carries reasoning text and tool calls that
//     this surface's `GenerationEvent`/`GenerationSummary` cannot express yet,
//     so they are dropped. Closing that is additive.
//   * Request validation failures are rethrown as the validator's own error;
//     `TinyTitanError` has no case for a malformed request.
import Foundation
import TinyTitan
import TinyTitanCore

public actor Session {
    private let system: String?
    /// Weak on purpose: `Engine` owns the model, so `unload()` really releases
    /// it and this session then reports `engineShutDown`.
    private weak var modelSession: ServerModelSession?
    private let descriptor: ModelDescriptor
    private let reasoningProfile: ServerReasoningProfile
    private var currentTask: Task<GenerationSummary, any Error>?

    init(
        system: String?,
        modelSession: ServerModelSession?,
        descriptor: ModelDescriptor,
        reasoningProfile: ServerReasoningProfile
    ) {
        self.system = system
        self.modelSession = modelSession
        self.descriptor = descriptor
        self.reasoningProfile = reasoningProfile
    }

    /// Generates one answer and streams its visible text through `onEvent`.
    ///
    /// The returned summary is the same one the `finished` event carries. One
    /// generation per session is the contract; a second call waits on the
    /// orchestrator's slot pool.
    public func respond(
        to messages: [ChatMessage],
        tools: [ToolDefinition] = [],
        options: GenerationOptions = .init(),
        onEvent: @escaping @Sendable (GenerationEvent) -> Void
    ) async throws -> GenerationSummary {
        // The work runs in its own task so `cancel()` has something to cancel:
        // the actor is free while this call awaits the result, which is what
        // lets `cancel()` in. The orchestrator already stops on task
        // cancellation, the same way the server stops a disconnected client.
        try await runOneShot {
            try await self.generate(
                messages: messages, tools: tools, options: options, onEvent: onEvent)
        }
    }

    /// Generates one answer for either kind of prompt.
    ///
    /// `.messages` is exactly the overload above. `.raw` tokenizes the text
    /// with BOS, prefills it without the chat template, and streams whatever
    /// the model writes verbatim — no reasoning split, no tool parsing — which
    /// is the behaviour the CLI's `--prompt` mode has always had.
    public func respond(
        to prompt: Prompt,
        options: GenerationOptions = .init(),
        onEvent: @escaping @Sendable (GenerationEvent) -> Void
    ) async throws -> GenerationSummary {
        switch prompt {
        case .messages(let messages):
            return try await respond(to: messages, options: options, onEvent: onEvent)
        case .raw(let text):
            return try await runOneShot {
                try await self.generateRaw(
                    text: text, options: options, onEvent: onEvent)
            }
        }
    }

    /// Stops the generation in flight at its next token boundary.
    ///
    /// Cancels the task that is running `generate`, which is how the server
    /// stops a generation whose client has gone; the KV and expert state stay
    /// usable, as they do there. A no-op when nothing is running.
    public func cancel() async {
        currentTask?.cancel()
    }

    /// Runs one generation in its own task so `cancel()` can reach it.
    ///
    /// The task is unstructured, so cancelling the caller does not cancel it by
    /// itself: awaiting `Task.value` in a cancelled task returns normally. The
    /// CLI's SIGINT handler cancels exactly the caller task, so cancellation is
    /// forwarded here -- otherwise `Ctrl-C` left the generation decoding to its
    /// token limit (AUD2-035).
    private func runOneShot(
        _ body: @escaping @Sendable () async throws -> GenerationSummary
    ) async throws -> GenerationSummary {
        let work = Task { try await body() }
        currentTask = work
        defer { currentTask = nil }
        do {
            return try await withTaskCancellationHandler {
                try await work.value
            } onCancel: {
                work.cancel()
            }
        } catch is CancellationError {
            throw TinyTitanError.cancelled
        }
    }

    private func generate(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        options: GenerationOptions,
        onEvent: @escaping @Sendable (GenerationEvent) -> Void
    ) async throws -> GenerationSummary {
        guard let modelSession else { throw TinyTitanError.engineShutDown }
        let validated = try validate(
            messages: messages, tools: tools, options: options, modelSession: modelSession)
        let completion: ServerCompletion
        do {
            completion = try await modelSession.generate(validated) { event in
                if let mapped = Session.facadeEvent(event) { onEvent(mapped) }
            }
        } catch let error as GeneratorError {
            switch error {
            case .contextOverflow(let prompt, _, let maxContext):
                throw TinyTitanError.contextWindowExceeded(
                    prompt: prompt, window: maxContext)
            default:
                throw error
            }
        } catch let error as ServerRequestError {
            throw Self.mapContextOverflow(
                error, window: modelSession.maximumContext)
        } catch is CancellationError {
            throw TinyTitanError.cancelled
        }
        let summary = GenerationSummary(
            completion: completion, cancelled: Task.isCancelled)
        onEvent(.finished(summary))
        return summary
    }

    /// A raw completion: the caller's text, tokenized with BOS and prefilled
    /// without the chat template.
    private func generateRaw(
        text: String,
        options: GenerationOptions,
        onEvent: @escaping @Sendable (GenerationEvent) -> Void
    ) async throws -> GenerationSummary {
        guard let modelSession else { throw TinyTitanError.engineShutDown }
        let promptIDs = await modelSession.tokenizeRawPrompt(text)
        // The pre-facade CLI refused an over-long prompt before loading the
        // model; the count is known here, so it keeps that exact answer.
        guard promptIDs.count < modelSession.maximumContext else {
            throw TinyTitanError.contextWindowExceeded(
                prompt: promptIDs.count, window: modelSession.maximumContext)
        }
        let config = GenerationConfig(
            maxNewTokens: options.maxTokens,
            temperature: Float(options.temperature),
            // Zero is the CLI's spelling of "top-k off", which the engine
            // expresses as a missing k. See `GenerationOptions.topK`.
            topK: options.topK == 0 ? nil : options.topK,
            topP: Float(options.topP),
            presencePenalty: Float(options.presencePenalty),
            minP: GenerationDefaults.minP,
            repetitionPenalty: Float(options.repetitionPenalty),
            seed: options.seed,
            stopStrings: options.stop,
            extraStopTokens: [])
        let request = ValidatedChatRequest(
            messages: [],
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: config,
            maximumCompletionTokens: options.maxTokens,
            renderedPromptIDs: promptIDs)
        let completion: ServerCompletion
        do {
            completion = try await modelSession.generate(request) { event in
                // A raw completion has no decoder, so reasoning and tool calls
                // are never produced for it; the mapper drops them anyway.
                if let mapped = Session.facadeEvent(event) { onEvent(mapped) }
            }
        } catch let error as GeneratorError {
            switch error {
            case .contextOverflow(let prompt, _, let maxContext):
                throw TinyTitanError.contextWindowExceeded(
                    prompt: prompt, window: maxContext)
            default:
                throw error
            }
        } catch let error as ServerRequestError {
            throw Self.mapContextOverflow(
                error, window: modelSession.maximumContext)
        } catch is CancellationError {
            throw TinyTitanError.cancelled
        }
        let summary = GenerationSummary(
            completion: completion, cancelled: Task.isCancelled)
        onEvent(.finished(summary))
        return summary
    }

    /// The validator's over-long-prompt refusal as the facade's typed case.
    ///
    /// The prompt's token count is not in the error, and the chat path cannot
    /// count without a tokenizer; reporting zero says "the window, not a
    /// number" rather than inventing one. Everything else is rethrown
    /// unchanged, exactly as before.
    private static func mapContextOverflow(
        _ error: ServerRequestError,
        window: Int
    ) -> any Error {
        if case .invalid(_, _, let code) = error, code == "context_length_exceeded" {
            return TinyTitanError.contextWindowExceeded(prompt: 0, window: window)
        }
        return error
    }

    /// The facade's view of one orchestrator event, or `nil` for the ones this
    /// surface cannot express.
    ///
    /// One function for both entry points, so a raw completion and a
    /// conversation cannot drift apart in what they report. Nothing returns
    /// `nil` today: reasoning and tool calls are carried now, and the `nil` arm
    /// is what a future event the facade cannot express would use.
    package static func facadeEvent(_ event: ServerInferenceEvent) -> GenerationEvent? {
        switch event {
        case .content(let text):
            return .token(text)
        case .promptProcessed(let tokens, let cachedTokens):
            return .promptProcessed(tokens: tokens, cachedTokens: cachedTokens)
        case .promptProgress:
            // The embedded facade has no streaming-progress surface; its
            // callers get the promptProcessed event when the prompt is read.
            return nil
        case .reasoning(let text):
            return .reasoning(text)
        case .toolCall(let call):
            return .toolCall(ToolCall(parsed: call))
        }
    }

    private func validate(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        options: GenerationOptions,
        modelSession: ServerModelSession
    ) throws -> ValidatedChatRequest {
        let request = OpenAIChatRequest(
            model: descriptor.id,
            messages: Self.wireMessages(system: system, messages: messages),
            temperature: Float(options.temperature),
            topP: Float(options.topP),
            maxTokens: options.maxTokens,
            stop: options.stop.isEmpty ? nil : .many(options.stop),
            seed: options.seed,
            tools: try Self.wireTools(tools),
            // The validator requires 1...256 and fills a missing k from the
            // model's row, so "off" is passed as missing and then restored on
            // the validated config below.
            topK: options.topK == 0 ? nil : options.topK,
            repetitionPenalty: Float(options.repetitionPenalty),
            presencePenalty: Float(options.presencePenalty))
        var validated = try OpenAIRequestValidator.validate(
            request,
            modelID: descriptor.id,
            maxContext: modelSession.maximumContext,
            reasoningProfile: reasoningProfile,
            sampling: modelSession.samplingDefaults,
            // The wire's caps are for a third party's request; this caller is
            // inside the process and was never held to them before the CLI
            // moved onto the facade.
            rules: .local
        )
        .withModel(descriptor.id)
        if options.topK == 0 {
            var config = validated.generationConfig
            config.topK = nil
            validated = validated.withGenerationConfig(config)
        }
        return validated
    }

    /// The wire's view of a conversation: the session's system prompt first,
    /// then the caller's turns with their tool calls and results intact.
    ///
    /// `package` and pure so the mapping can be tested without a model; it is
    /// the only place a `ChatMessage` becomes something the validator reads.
    package static func wireMessages(
        system: String?,
        messages: [ChatMessage]
    ) -> [OpenAIChatMessage] {
        var result: [OpenAIChatMessage] = []
        if let system, !system.isEmpty {
            result.append(
                OpenAIChatMessage(
                    role: "system", content: .text(system),
                    toolCalls: nil, toolCallID: nil, name: nil))
        }
        result.append(
            contentsOf: messages.map { message in
                OpenAIChatMessage(
                    role: message.role.openAIRole,
                    content: .text(message.content),
                    toolCalls: message.toolCalls.isEmpty
                        ? nil
                        : message.toolCalls.map(Self.wireToolCall),
                    toolCallID: message.toolCallID,
                    name: nil)
            })
        return result
    }

    /// The wire's view of a call the model made: the arguments stay the model's
    /// own JSON text, which is how this type carries them.
    private static func wireToolCall(_ call: ToolCall) -> OpenAIToolCall {
        OpenAIToolCall(
            id: call.id, type: "function",
            function: OpenAIFunctionCall(name: call.name, arguments: call.argumentsJSON))
    }

    /// The wire's view of the tools the caller offered, parsing each schema out
    /// of the text the facade carries.
    ///
    /// A schema that is not JSON at all is the caller's mistake and is reported
    /// as one; everything else about a tool — the name's shape, that the schema
    /// is an object, that its keys and numbers can be rendered — is the
    /// validator's, which is where the engine's own rules already live.
    /// `package` and pure for the same reason as `wireMessages`: the schema
    /// parse is the part a caller can get wrong, so it is worth a test.
    package static func wireTools(_ tools: [ToolDefinition]) throws -> [OpenAITool]? {
        guard !tools.isEmpty else { return nil }
        return try tools.map { tool in
            guard let data = tool.parametersJSON.data(using: .utf8),
                let parameters = try? JSONDecoder().decode(JSONValue.self, from: data)
            else {
                throw TinyTitanError.invalidToolDefinition(
                    tool: tool.name, detail: "parameters are not JSON")
            }
            return OpenAITool(
                type: "function",
                function: OpenAIFunctionDefinition(
                    name: tool.name,
                    description: tool.description.isEmpty ? nil : tool.description,
                    parameters: parameters))
        }
    }
}
