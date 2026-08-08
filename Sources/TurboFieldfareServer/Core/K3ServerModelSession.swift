import Foundation
import TurboFieldfare

/// K3 inference backend for the OpenAI-compatible server, the v2-bundle
/// counterpart of `ServerModelSession`.
///
/// The validated request's `k3` payload already carries renderer-ready
/// messages and options, so generation is: render to ids, run the engine,
/// incrementally detokenize and parse the XTML channels (think → response →
/// tools), and report content/reasoning deltas plus completed tool calls.
///
/// K3 uses an engine-bound single-prefix cache. A hit requires exact token
/// identity with the snapshot's state-backed prefix; any mismatch discards
/// the snapshot and replays the full prompt. The cache owns recurrent KDA,
/// conv, active MLA, last-logit, and routing-predictor state rather than
/// pretending that an MLA-only KV cache is sufficient.
///
/// `K3Engine.generate` is synchronous and single-in-flight; running it inside
/// this actor serializes K3 generations the same way the house runner's
/// scratch buffers do. Mid-generation cancellation is not possible at this
/// layer (the engine has no cancel hook); a disconnected client's result is
/// discarded after the run.
struct K3ServerPromptCache {
    private(set) var snapshot: K3PrefixSnapshot?

    mutating func invalidate() {
        snapshot = nil
    }

    mutating func publish(_ snapshot: K3PrefixSnapshot) {
        self.snapshot = snapshot
    }

    func match(renderedPromptIDs: [Int32]) -> K3PrefixSnapshot? {
        guard let snapshot,
              Self.isExactPrefix(snapshot.tokenIDs, of: renderedPromptIDs) else {
            return nil
        }
        return snapshot
    }

    static func isExactPrefix(_ cached: [Int32], of rendered: [Int32]) -> Bool {
        !cached.isEmpty
            && cached.count <= rendered.count
            && rendered.prefix(cached.count).elementsEqual(cached)
    }
}

public actor K3ServerModelSession: ServerInferenceBackend {
    private let engine: K3Engine
    private let tokenizer: K3Tokenizer
    private let maxContext: Int
    private let prefillMode: K3PrefillMode
    private let promptCacheMode: ServerPromptCacheMode
    private var promptCache = K3ServerPromptCache()

    public static func load(
        bundleURL: URL,
        maxContext: Int,
        prefillMode: K3PrefillMode = .chunked(chunkTokens: 32),
        promptCacheMode: ServerPromptCacheMode = .singlePrefix,
        integrityPolicy: ModelIntegrityPolicy = .fullSha256
    ) throws -> K3ServerModelSession {
        let tokenizer = try K3Tokenizer(vocabURL: bundleURL
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
            .appendingPathComponent("tiktoken.model"))
        let engine = try K3Engine.load(bundleURL: bundleURL,
                                       maxContext: maxContext,
                                       integrityPolicy: integrityPolicy)
        return K3ServerModelSession(engine: engine,
                                    tokenizer: tokenizer,
                                    maxContext: maxContext,
                                    prefillMode: prefillMode,
                                    promptCacheMode: promptCacheMode)
    }

    private init(engine: K3Engine,
                 tokenizer: K3Tokenizer,
                 maxContext: Int,
                 prefillMode: K3PrefillMode,
                 promptCacheMode: ServerPromptCacheMode) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.maxContext = maxContext
        self.prefillMode = prefillMode
        self.promptCacheMode = promptCacheMode
    }

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        var completed = false
        defer {
            if !completed {
                promptCache.invalidate()
                engine.reset()
            }
        }
        guard let k3 = request.k3 else {
            throw ServerRequestError.invalid(
                message: "request was not validated for the K3 pipeline",
                param: nil,
                code: "internal_error")
        }
        let promptIDs: [Int32]
        do {
            promptIDs = try K3ChatRenderer.renderToIDs(
                messages: k3.messages, options: k3.options, tokenizer: tokenizer)
                .map(Int32.init)
        } catch let error as K3ChatRendererError {
            throw ServerRequestError.invalid(
                message: "\(error)", param: "messages", code: "invalid_message")
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        let maxNew = min(request.maximumCompletionTokens, maxContext - promptIDs.count)
        var config = request.generationConfig
        config.maxNewTokens = maxNew
        config.stopStrings = []

        let prefixSnapshot: K3PrefixSnapshot?
        if promptCacheMode == .singlePrefix {
            prefixSnapshot = promptCache.match(renderedPromptIDs: promptIDs)
            if prefixSnapshot == nil { promptCache.invalidate() }
        } else {
            promptCache.invalidate()
            prefixSnapshot = nil
        }

        try Task.checkCancellation()
        let thinking = k3.options.thinking
        var detokenizer = K3Detokenizer(tokenizer: tokenizer)
        var text = ""
        var emittedReasoning = 0
        var emittedContent = 0

        func emitDeltas(from parsed: K3ParsedAssistantMessage) {
            if let reasoning = parsed.reasoningContent,
               reasoning.count > emittedReasoning {
                let start = reasoning.index(reasoning.startIndex, offsetBy: emittedReasoning)
                emittedReasoning = reasoning.count
                let delta = String(reasoning[start...])
                if !delta.isEmpty { onEvent(.reasoning(delta)) }
            }
            if parsed.content.count > emittedContent {
                let start = parsed.content.index(
                    parsed.content.startIndex, offsetBy: emittedContent)
                emittedContent = parsed.content.count
                let delta = String(parsed.content[start...])
                if !delta.isEmpty { onEvent(.content(delta)) }
            }
        }

        let stats = try engine.generate(
            promptTokens: promptIDs,
            config: config,
            maxNew: maxNew,
            prefillMode: prefillMode,
            prefixSnapshot: prefixSnapshot) { token in
                text += detokenizer.push(token)
                emitDeltas(from: K3ToolCallParser.parse(text, thinking: thinking))
            }
        text += detokenizer.flush()
        let parsed = K3ToolCallParser.parse(text, thinking: thinking)
        emitDeltas(from: parsed)
        try Task.checkCancellation()

        let calls: [ParsedToolCall] = parsed.toolCalls.map { call in
            let arguments: JSONValue
            let argumentsJSON: String
            switch call.arguments {
            case .object(let value):
                arguments = value
                argumentsJSON = (try? value.encoded()) ?? "{}"
            case .jsonString:
                // The parser only produces this for an unparseable raw json
                // block; OpenAI clients need a JSON object string regardless.
                arguments = .object([:])
                argumentsJSON = "{}"
            }
            return ParsedToolCall(id: Self.newCallID(),
                                  name: call.name,
                                  arguments: arguments,
                                  argumentsJSON: argumentsJSON)
        }
        for call in calls {
            onEvent(.toolCall(call))
        }
        let reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if stats.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        if promptCacheMode == .singlePrefix,
           !parsed.malformed,
           stats.position == stats.stateBackedTokenIDs.count,
           stats.uncommittedBoundaryTokenIDs.count == 1,
           let snapshot = try? engine.capturePrefixSnapshot(
                tokenIDs: stats.stateBackedTokenIDs) {
            promptCache.publish(snapshot)
        } else {
            promptCache.invalidate()
        }
        completed = true
        return ServerCompletion(
            content: parsed.content,
            toolCalls: calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: stats.prefillTokens,
                               completionTokens: stats.newTokens,
                               totalTokens: stats.prefillTokens + stats.newTokens,
                               cachedTokens: stats.cachedPromptTokens),
            reasoningContent: parsed.reasoningContent)
    }

    /// Same id shape the house `StructuredAssistantDecoder` synthesizes.
    private static func newCallID() -> String {
        "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
    }
}
