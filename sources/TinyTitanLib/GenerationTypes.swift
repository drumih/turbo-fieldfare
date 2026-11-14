// The generation surface an embedder calls: messages in, events and a summary
// out.
//
// Part of the supported facade (phase A1 of `docs/plan-embedded-library.md`,
// §4). Every type here is kit-native: the shapes are the engine's, but no
// engine type appears in a signature. The wrapper in `Session` maps to and
// from `OpenAIChatRequest`/`ServerInferenceEvent` internally.
import TinyTitan
import TinyTitanCore

/// One turn of a conversation, in the shape the chat template expects.
public struct ChatMessage: Sendable, Equatable {
    public enum Role: Sendable, Equatable {
        case system
        /// OpenAI's successor to `system`; the shipped templates normalise it
        /// to `system` when they render.
        case developer
        case user
        case assistant
        case tool

        /// The wire spelling the validator's role table is keyed on.
        var openAIRole: String {
            switch self {
            case .system: "system"
            case .developer: "developer"
            case .user: "user"
            case .assistant: "assistant"
            case .tool: "tool"
            }
        }

        /// The tokenizer's own role, for a render done outside a request.
        var tokenizerRole: GFTokenizer.Role {
            switch self {
            case .system: .system
            case .developer: .developer
            case .user: .user
            case .assistant: .assistant
            case .tool: .tool
            }
        }
    }

    public var role: Role
    public var content: String
    /// The calls an assistant turn made, in order.
    ///
    /// A caller feeding a tool loop back in has to replay the assistant turn
    /// that asked for the tool, because a result is matched to the call it
    /// answers — the template renders the pair, and the validator refuses a
    /// result that names no open call.
    public var toolCalls: [ToolCall]
    /// For a `tool` message: the id of the call this is the result of.
    public var toolCallID: String?

    public init(
        role: Role,
        content: String,
        toolCalls: [ToolCall] = [],
        toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }
}

/// A tool the model may ask to call.
///
/// The parameters are JSON Schema **as text**, for the same reason a `ToolCall`'s
/// arguments are: the engine parses them into a value type of its own, and that
/// representation is not part of the facade's promise. A caller that prefers to
/// build the schema in Swift spells it as a string literal or serializes its own
/// type into one.
public struct ToolDefinition: Sendable, Equatable {
    public let name: String
    public let description: String
    public let parametersJSON: String

    public init(name: String, description: String = "", parametersJSON: String) {
        self.name = name
        self.description = description
        self.parametersJSON = parametersJSON
    }
}

/// What to generate for: a chat conversation the template renders, or a raw
/// completion the caller has already spelled as text.
///
/// `raw` skips the chat template entirely — the text is tokenized with BOS and
/// prefilled verbatim, which is what the CLI's `--prompt` mode has always been.
/// A raw completion is not routed through the reasoning/tool decoder either:
/// whatever the model writes is the answer, think markers included.
public enum Prompt: Sendable {
    case messages([ChatMessage])
    case raw(String)
}

/// Sampling and length controls for one generation.
///
/// Every field is sent explicitly, so the served model's profile only fills
/// what this type cannot express. `Engine.samplingDefaults` (or
/// `SamplingDefaults.forInstall(at:thinkingMode:)`) is how a caller fills a
/// field it wants to leave to the model.
public struct GenerationOptions: Sendable {
    public var maxTokens: Int
    public var temperature: Double
    public var topP: Double
    /// Top-k truncation. `0` turns it off — the engine's "no k" — which is
    /// only legal alongside `topP == 1`, because this sampler has no
    /// full-vocabulary nucleus path.
    public var topK: Int
    public var repetitionPenalty: Double
    /// OpenAI presence penalty: subtracted once per distinct id already in the
    /// history. Zero is neutral.
    public var presencePenalty: Double
    /// `nil` draws from system entropy, as the server does.
    public var seed: UInt64?
    /// Strings that end the answer when they appear in the visible text.
    public var stop: [String]

    public init(
        maxTokens: Int = 512,
        temperature: Double = 0.6,
        topP: Double = 0.95,
        topK: Int = 20,
        repetitionPenalty: Double = 1.0,
        presencePenalty: Double = 0,
        seed: UInt64? = nil,
        stop: [String] = []
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.stop = stop
    }
}

/// Why generation ended, named rather than spelled as the wire's string.
///
/// `toolCalls` is carried because the orchestrator produces it (the model
/// asked for a tool); the §4 sketch listed only the first three cases plus
/// cancellation, and dropping the distinction would have reported a tool call
/// as an ordinary end of turn.
public enum GenerationStopReason: Sendable, Equatable {
    case endOfText
    case length
    case stopString
    case toolCalls
    case cancelled
}

/// Why the engine's decode loop stopped, in the engine's own vocabulary.
///
/// Distinct from `GenerationStopReason`, which is the wire's view of the same
/// end (`endOfTurn` and `eos` both become OpenAI's `stop`): a caller that
/// reports or measures the loop itself needs the loop's answer.
public enum DecodeStopReason: String, Sendable, Equatable {
    case eos
    case endOfTurn
    case maxTokens
    case stopString
    case toolCalls
    /// The caller's external stop signal, before any configured stop string.
    case external

    init(engine: StopReason?) {
        switch engine {
        case .eos: self = .eos
        case .endOfTurn: self = .endOfTurn
        case .maxTokens: self = .maxTokens
        case .stopString: self = .stopString
        case .toolCalls: self = .toolCalls
        case .external: self = .external
        case nil: self = .eos
        }
    }
}

/// A tool the model asked to call, in the facade's own vocabulary.
///
/// The arguments travel as the JSON text the model emitted. The runtime parses
/// them into its own value type, and the facade deliberately does not hand that
/// out: a caller that wants structure parses `argumentsJSON`, and nothing about
/// the engine's JSON representation becomes part of the promise.
public struct ToolCall: Sendable, Equatable {
    public let id: String
    public let name: String
    public let argumentsJSON: String

    public init(id: String, name: String, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
    }

    init(parsed call: ParsedToolCall) {
        self.init(id: call.id, name: call.name, argumentsJSON: call.argumentsJSON)
    }
}

/// What a finished generation produced.
public struct GenerationSummary: Sendable, Equatable {
    /// The visible answer, with any client stop string removed.
    public let text: String
    /// Everything the model thought, in order; empty with thinking off.
    /// `completionTokens` already counts these tokens.
    public let reasoning: String
    /// The tools the model asked for, in the order it asked. Empty unless the
    /// request offered tools — which the facade cannot do yet, so today this is
    /// always empty and the type is here so the gap is a field rather than a
    /// missing case. See TT-046.
    public let toolCalls: [ToolCall]
    public let promptTokens: Int
    public let completionTokens: Int
    public let stopReason: GenerationStopReason
    /// Wall time the engine spent prefilling this request's prompt.
    public let prefillSeconds: Double
    /// Wall time the engine spent decoding, excluding prefill.
    public let decodeSeconds: Double
    /// The decode loop's own stop reason.
    public let decodeStopReason: DecodeStopReason
}

/// One step of a streaming generation.
///
/// `promptProcessed` arrives once, before any token: `tokens` is the prompt's
/// length and `cachedTokens` the part of it the prompt cache already held, so a
/// caller can tell "still prefilling" from "content is coming". A fully cached
/// prompt emits it too — there is nothing to prefill, but the prompt is read.
///
/// `reasoning` is the thought text from inside the model's think block, kept
/// apart from `token` so a caller can put it where its clients look for it and
/// so nothing that judges the answer reads it.
public enum GenerationEvent: Sendable {
    case promptProcessed(tokens: Int, cachedTokens: Int)
    case token(String)
    case reasoning(String)
    case toolCall(ToolCall)
    case finished(GenerationSummary)
}

extension GenerationSummary {
    /// The server reports a wire reason string; this is the one place it
    /// becomes a case.
    init(completion: ServerCompletion, cancelled: Bool) {
        self.init(
            text: completion.content,
            reasoning: completion.reasoning,
            toolCalls: completion.toolCalls.map(ToolCall.init(parsed:)),
            promptTokens: completion.usage.promptTokens,
            completionTokens: completion.usage.completionTokens,
            stopReason: cancelled
                ? .cancelled
                : GenerationStopReason(
                    finishReason: completion.finishReason,
                    stopSequence: completion.stopSequence),
            prefillSeconds: completion.prefillSeconds,
            decodeSeconds: completion.decodeSeconds,
            decodeStopReason: DecodeStopReason(engine: completion.engineStopReason))
    }
}

extension GenerationStopReason {
    init(finishReason: String, stopSequence: String?) {
        switch finishReason {
        case "length": self = .length
        case "tool_calls": self = .toolCalls
        default: self = stopSequence == nil ? .endOfText : .stopString
        }
    }
}
