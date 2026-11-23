import Foundation
import TinyTitan
import TinyTitanCore
import TinyTitanLib

/// Single-model server backend for an Unsloth Inkling-2-Mini GGUF
/// (`UD-Q2_K_XL`, `UD-Q4_K_XL`) on the native decoder, through
/// `Inkling2GGUFSession` — the decode loop the CLI's GGUF route uses.
///
/// Text in, text out, with the rest of the API the other families serve:
/// tool calls in Inkling's own format (`Inkling2GGUFChatFormat`), thinking at
/// the requested effort level, stop strings, `response_format` JSON schemas,
/// and the sampling fields the validator accepts (temperature, top-k, top-p,
/// penalties, seed). Images and audio are not accepted.
package actor Inkling2GGUFServerBackend: ServerInferenceBackend, PromptTokenCounting {
    /// The session holds the decoder's Metal state; the actor serialises
    /// every request onto it.
    private let session: Inkling2GGUFSession
    private let vocabulary: Inkling2GGUFOutputParser.Vocabulary
    /// The reasoning a request gets when it names none: the server's
    /// `--reasoning` level.
    private let defaultReasoning: RequestReasoning
    private var jsonTokenTable: JSONTokenTable?
    package nonisolated let maximumContext: Int
    package nonisolated let samplingDefaults: SamplingProfile = .house

    /// The context the decoder's KV ring and caches are sized for: the
    /// server's `--max-context`, capped as the CLI caps it.
    package static let contextCap = Inkling2GGUFSession.modelMaxContext

    /// Inkling's thinking control: off, or one of the template's effort
    /// levels, `high` (0.9) being the template's own default.
    package static let reasoningControl = ReasoningControl.thinkingWithEffortLevels(
        defaultEffort: .high, supported: [.low, .medium, .high, .xhigh])

    package init(gguf: URL, maxContext: Int, reasoning: ReasoningLevel = .off) async throws {
        let context = min(maxContext, Self.contextCap)
        session = try await Inkling2GGUFSession(gguf: gguf, maxContext: context)
        vocabulary = try Inkling2GGUFOutputParser.Vocabulary(tokenizer: session.tokenizer)
        maximumContext = session.maximumContext
        defaultReasoning = Self.requestReasoning(for: reasoning)
    }

    /// A server-wide level as the reasoning a request carries.
    package static func requestReasoning(for level: ReasoningLevel) -> RequestReasoning {
        guard level != .off else { return RequestReasoning(thinkingMode: .off, effort: nil) }
        let effort =
            level.effort
            ?? (level == .max ? .xhigh : level == .minimal ? .low : .high)
        return RequestReasoning(thinkingMode: .on, effort: effort)
    }

    /// The rendered prompt, and whether it opened a text message itself.
    struct Prompt {
        let ids: [Int]
        let opensText: Bool
    }

    func prompt(for request: ValidatedChatRequest) -> Prompt {
        if let raw = request.renderedPromptIDs {
            return Prompt(ids: raw.map(Int.init), opensText: true)
        }
        let reasoning = request.reasoning ?? defaultReasoning
        // A grammar constrains every token, so a JSON answer starts in text;
        // so does a turn with thinking off and nothing to call. Otherwise the
        // model opens its own turn, as the template intends.
        let opensText =
            request.jsonSchema != nil
            || (!reasoning.thinkingMode.isEnabled && request.tools.isEmpty)
        let effort =
            request.jsonSchema == nil && reasoning.thinkingMode.isEnabled
            ? (reasoning.effort ?? .high) : nil
        let text = Inkling2GGUFChatFormat.render(
            messages: request.messages, tools: request.tools, effort: effort,
            openTextTurn: opensText)
        return Prompt(ids: session.tokenizer.encode(text), opensText: opensText)
    }

    package func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        prompt(for: request).ids.count
    }

    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let prompt = prompt(for: request)
        guard !prompt.ids.isEmpty else {
            throw ServerRequestError.invalid(
                message: "the prompt tokenized to nothing", param: "messages",
                code: "invalid_prompt")
        }
        guard prompt.ids.count < maximumContext else {
            throw ServerRequestError.invalid(
                message:
                    "prompt is \(prompt.ids.count) tokens and the model's context is \(maximumContext)",
                param: "messages", code: "context_length_exceeded")
        }
        onEvent(.promptProcessed(tokens: prompt.ids.count, cachedTokens: 0))
        let configuration = request.generationConfig
        let sampler = CPUSampler(
            temperature: configuration.temperature, topP: configuration.topP ?? 1,
            topK: configuration.topK ?? 0, presencePenalty: configuration.presencePenalty,
            repetitionPenalty: configuration.repetitionPenalty, seed: configuration.seed)
        let constraint = request.jsonSchema.map { node in
            JSONConstraint(table: jsonTable(), node: node, vocab: session.tokenizer.vocabularySize)
        }
        let shape = Self.logitShaper(constraint: constraint)
        let tokenizer = session.tokenizer
        var parser = Inkling2GGUFOutputParser(
            vocabulary: vocabulary, decode: { tokenizer.decode($0) },
            offeredTools: Set(request.tools.map(\.name)), startInText: prompt.opensText)
        var output = AssistantOutput(stops: configuration.stopStrings, onEvent: onEvent)
        // A JSON answer is one text message: its end is the turn's end.
        let stopIDs =
            constraint == nil
            ? tokenizer.endOfTurnIDs : tokenizer.endOfTurnIDs.union([vocabulary.endMessage])
        let run = try session.decode(
            promptIDs: prompt.ids, maxTokens: max(request.maximumCompletionTokens, 1),
            sampler: sampler, stopIDs: stopIDs, shapeLogits: shape
        ) { token in
            if let constraint, !constraint.observe(Int32(token)) {
                throw GeneratorError.constrainedDecodeViolation(id: Int32(token))
            }
            let events = parser.consume(token)
            output.publish(events)
            return !parser.ended && !output.isStopped
        }
        output.publish(parser.finish(), isToken: false)
        output.finish()
        let reason: String
        if !output.calls.isEmpty {
            reason = "tool_calls"
        } else if run.stopReason == "stop" {
            reason = "stop"
        } else {
            reason = "length"
        }
        return ServerCompletion(
            content: output.content, toolCalls: output.calls, finishReason: reason,
            usage: OpenAIUsage(
                promptTokens: prompt.ids.count, completionTokens: run.tokens.count,
                totalTokens: prompt.ids.count + run.tokens.count, cachedTokens: 0,
                reasoningTokens: output.reasoningTokens),
            stopSequence: output.matchedStop, reasoning: output.reasoning,
            prefillSeconds: run.prefillSeconds, decodeSeconds: run.decodeSeconds)
    }

    package func tokenizeRawPrompt(_ text: String) async throws -> [Int32] {
        session.tokenizer.encode(text).map(Int32.init)
    }

    private func jsonTable() -> JSONTokenTable {
        if let jsonTokenTable { return jsonTokenTable }
        let made = session.tokenizer.jsonTokenTable(
            stopTokens: session.tokenizer.endOfTurnIDs.union([vocabulary.endMessage]))
        jsonTokenTable = made
        return made
    }

    /// The grammar's mask, or nil when the request has none (which keeps
    /// greedy decoding on the GPU argmax): every token the grammar no longer
    /// accepts is floored before sampling.
    static func logitShaper(
        constraint: JSONConstraint?
    ) -> ((inout [Float]) throws -> Void)? {
        guard let constraint else { return nil }
        return { logits in
            let mask = constraint.allowedMask()
            guard !mask.isEmpty else { throw GeneratorError.constrainedDecodeStalled }
            mask.apply(toLogits: &logits)
        }
    }
}
