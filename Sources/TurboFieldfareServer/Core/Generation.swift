import Foundation

/// OpenAI-facing finish reasons; mapped from the runtime's `StopReason`.
public enum FinishReason: String, Sendable, Equatable {
    case stop
    case length
}

/// Per-request sampling knobs after defaults are applied. Defaults mirror the
/// README/CLI: temperature 0.2, topP 0.95 (topK is a runtime constant, not an
/// OpenAI field).
public struct SamplingParams: Sendable, Equatable {
    public var maxTokens: Int?
    public var temperature: Float
    public var topP: Float
    public var seed: UInt64?
    public var stop: [String]

    public init(maxTokens: Int? = nil,
                temperature: Float = 0.2,
                topP: Float = 0.95,
                seed: UInt64? = nil,
                stop: [String] = []) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.seed = seed
        self.stop = stop
    }

    public init(maxTokens: Int?,
                temperature: Float?,
                topP: Float?,
                seed: Int?,
                stop: StopValue?) {
        self.init(maxTokens: maxTokens,
                  temperature: temperature ?? 0.2,
                  topP: topP ?? 0.95,
                  seed: seed.map { UInt64(bitPattern: Int64($0)) },
                  stop: stop?.values ?? [])
    }
}

/// Final, Metal-free result of one generation.
public struct GenerationOutcome: Sendable, Equatable {
    public var text: String
    public var promptTokens: Int
    public var completionTokens: Int
    public var finishReason: FinishReason

    public init(text: String, promptTokens: Int, completionTokens: Int, finishReason: FinishReason) {
        self.text = text
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.finishReason = finishReason
    }

    public var usage: Usage {
        Usage(promptTokens: promptTokens, completionTokens: completionTokens)
    }
}

/// A validated, encode-ready prompt source.
public enum GenerationInput: Sendable, Equatable {
    /// Chat messages with roles already restricted to system/user/assistant;
    /// the tokenizer's chat template renders `assistant` as `model`.
    case chat([ChatMessage])
    /// Raw completion prompt; encoded verbatim with BOS.
    case raw(String)
}

/// One generation that passed validation and is ready to run.
public struct PreparedGeneration: Sendable, Equatable {
    public var input: GenerationInput
    public var sampling: SamplingParams
    public var effectiveMaxTokens: Int
    public var promptTokens: Int

    public init(input: GenerationInput,
                sampling: SamplingParams,
                effectiveMaxTokens: Int,
                promptTokens: Int) {
        self.input = input
        self.sampling = sampling
        self.effectiveMaxTokens = effectiveMaxTokens
        self.promptTokens = promptTokens
    }
}

/// Errors surfaced to the HTTP layer as OpenAI-style 400s.
public enum GenerationRequestError: Error, Equatable, CustomStringConvertible {
    case emptyMessages
    case emptyPrompt
    case unsupportedRole(String)
    case unsupportedParameter(String)
    case contextOverflow(prompt: Int, maxTokens: Int, maxContext: Int)

    public var description: String {
        switch self {
        case .emptyMessages:
            return "messages must contain at least one message"
        case .emptyPrompt:
            return "prompt produced no tokens"
        case .unsupportedRole(let role):
            return "unsupported role '\(role)'; supported roles are system, user, assistant"
        case .unsupportedParameter(let detail):
            return detail
        case .contextOverflow(let prompt, let maxTokens, let maxContext):
            return "context overflow: prompt (\(prompt) tokens) + max_tokens (\(maxTokens)) exceeds max context (\(maxContext))"
        }
    }
}

/// Seam between the HTTP-facing session and the runtime. The real
/// implementation wraps `runRawCompletion`; tests substitute a fake that
/// emits canned tokens. Kept free of Metal/runtime types so the whole
/// route/SSE stack is testable without a GPU.
public protocol TokenGenerating: Sendable {
    /// Token count the input encodes to (BOS included where applicable).
    func tokenCount(for input: GenerationInput) throws -> Int
    /// Run one generation. `onDelta` fires per streamed text piece; the
    /// returned outcome carries the full text and stop metadata.
    func generate(_ input: GenerationInput,
                  sampling: SamplingParams,
                  effectiveMaxTokens: Int,
                  onDelta: (@Sendable (String) -> Void)?) async throws -> GenerationOutcome
}

/// Owns the single-in-flight runtime. All generation funnels through this
/// actor, so requests execute serially in FIFO order — the contract the
/// underlying `RawCompletionScratch` requires.
public actor GenerationSession {
    public let modelID: String
    public let maxContext: Int
    /// Unix timestamp reported in `/v1/models`.
    public let created: Int
    private let generator: any TokenGenerating

    public init(modelID: String,
                maxContext: Int,
                created: Int = Int(Date().timeIntervalSince1970),
                generator: any TokenGenerating) {
        self.modelID = modelID
        self.maxContext = maxContext
        self.created = created
        self.generator = generator
    }

    /// Validate + budget a chat request. Split from `generate` so streaming
    /// handlers can fail with a 400 before the response head is written.
    public func prepareChat(messages: [ChatMessage],
                            sampling: SamplingParams) throws -> PreparedGeneration {
        try prepare(input: .chat(Self.validateMessages(messages)), sampling: sampling)
    }

    /// Validate + budget a raw-completion request.
    public func prepareCompletion(prompt: String,
                                  sampling: SamplingParams) throws -> PreparedGeneration {
        try prepare(input: .raw(prompt), sampling: sampling)
    }

    /// Run a prepared generation. Actor isolation alone does NOT serialize
    /// this: the method suspends on the external `generator.generate` await,
    /// which would let a second request enter. An explicit FIFO slot closes
    /// that reentrancy hole — at most one generation holds the slot, queued
    /// waiters are resumed in arrival order, and a cancelled waiter is
    /// removed from the queue instead of running.
    public func generate(_ prepared: PreparedGeneration,
                         onDelta: (@Sendable (String) -> Void)? = nil) async throws -> GenerationOutcome {
        try await acquireSlot()
        // A waiter cancelled in the same instant the slot was handed over must
        // not start a generation; check before touching the runtime.
        try Task.checkCancellation()
        defer { releaseSlot() }
        return try await generator.generate(prepared.input,
                                            sampling: prepared.sampling,
                                            effectiveMaxTokens: prepared.effectiveMaxTokens,
                                            onDelta: onDelta)
    }

    // MARK: - Single-in-flight slot

    // s5d:debt(ceiling="disconnect while parked here is detected only when the generation's first SSE write fails — Hummingbird 2 does not cancel the responder task on socket close", trigger="hummingbird exposes channel-close cancellation, or a watchdog is added")
    // s5d:debt(ceiling="waiter queue is unbounded — a flood parks one connection+continuation each", trigger="server is exposed beyond localhost or a queue cap (503) is requested")
    private var slotBusy = false
    private var waiterOrder: [UUID] = []
    private var waiters: [UUID: CheckedContinuation<Void, any Error>] = [:]

    private func acquireSlot() async throws {
        if !slotBusy {
            slotBusy = true
            return
        }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, any Error>) in
                waiters[id] = cont
                waiterOrder.append(id)
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        // Resumed by releaseSlot: the slot is handed over, slotBusy stays true.
    }

    private func cancelWaiter(_ id: UUID) {
        guard let cont = waiters.removeValue(forKey: id) else { return }
        waiterOrder.removeAll { $0 == id }
        cont.resume(throwing: CancellationError())
    }

    private func releaseSlot() {
        while let nextID = waiterOrder.first {
            waiterOrder.removeFirst()
            if let cont = waiters.removeValue(forKey: nextID) {
                cont.resume()
                return
            }
        }
        slotBusy = false
    }

    /// Convenience for non-streaming callers: prepare + generate in one call.
    public func chat(messages: [ChatMessage],
                     sampling: SamplingParams,
                     onDelta: (@Sendable (String) -> Void)? = nil) async throws -> GenerationOutcome {
        try await generate(prepareChat(messages: messages, sampling: sampling), onDelta: onDelta)
    }

    public func complete(prompt: String,
                         sampling: SamplingParams,
                         onDelta: (@Sendable (String) -> Void)? = nil) async throws -> GenerationOutcome {
        try await generate(prepareCompletion(prompt: prompt, sampling: sampling), onDelta: onDelta)
    }

    private func prepare(input: GenerationInput,
                         sampling: SamplingParams) throws -> PreparedGeneration {
        try Self.validateSampling(sampling)
        let promptTokens = try generator.tokenCount(for: input)
        let effectiveMax = try Self.effectiveMaxTokens(requested: sampling.maxTokens,
                                                       promptTokens: promptTokens,
                                                       maxContext: maxContext)
        return PreparedGeneration(input: input,
                                  sampling: sampling,
                                  effectiveMaxTokens: effectiveMax,
                                  promptTokens: promptTokens)
    }

    /// Numeric field validation; failures surface as OpenAI-style 400s instead
    /// of undefined runtime behavior (e.g. maxNewTokens: -5 reaching Metal).
    public static func validateSampling(_ sampling: SamplingParams) throws {
        if let maxTokens = sampling.maxTokens, maxTokens <= 0 {
            throw GenerationRequestError.unsupportedParameter("max_tokens must be a positive integer")
        }
        guard sampling.temperature >= 0 else {
            throw GenerationRequestError.unsupportedParameter("temperature must be >= 0")
        }
        guard sampling.topP > 0, sampling.topP <= 1 else {
            throw GenerationRequestError.unsupportedParameter("top_p must be in (0, 1]")
        }
    }

    /// Restricts roles to the chat template's supported set. OpenAI
    /// `assistant` needs no remapping: the template renders it as `model`.
    public static func validateMessages(_ messages: [ChatMessage]) throws -> [ChatMessage] {
        guard !messages.isEmpty else { throw GenerationRequestError.emptyMessages }
        for message in messages {
            switch message.role {
            case "system", "user", "assistant":
                break
            default:
                throw GenerationRequestError.unsupportedRole(message.role)
            }
        }
        return messages
    }

    /// Picks the decode budget. An explicit `max_tokens` that would overflow
    /// the context is a client error (400); an absent one clamps to whatever
    /// the context has left, mirroring the CLI.
    public static func effectiveMaxTokens(requested: Int?,
                                          promptTokens: Int,
                                          maxContext: Int) throws -> Int {
        guard promptTokens > 0 else { throw GenerationRequestError.emptyPrompt }
        if let requested {
            guard promptTokens + requested <= maxContext else {
                throw GenerationRequestError.contextOverflow(prompt: promptTokens,
                                                             maxTokens: requested,
                                                             maxContext: maxContext)
            }
            return requested
        }
        let remaining = maxContext - promptTokens
        guard remaining > 0 else {
            throw GenerationRequestError.contextOverflow(prompt: promptTokens,
                                                         maxTokens: 0,
                                                         maxContext: maxContext)
        }
        return remaining
    }
}
