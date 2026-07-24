import Foundation

/// Anthropic Messages API wire types. Unknown request fields are ignored by
/// the default `Decodable` synthesis, matching the tolerance of the OpenAI
/// surface; fields we explicitly do not support (tools, thinking, ...) are
/// detected by key presence and rejected in the route layer.

public struct AnthropicTextBlock: Codable, Sendable, Equatable {
    public var type: String
    public var text: String

    enum CodingKeys: String, CodingKey {
        case type
        case text
    }

    public init(text: String) {
        self.type = "text"
        self.text = text
    }

    /// Request-side decoding accepts only `text` blocks; anything else
    /// (image, tool_use, ...) fails the request decode and surfaces as a 400.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        guard type == "text" else {
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "only text content blocks are supported")
        }
        self.type = type
        self.text = try container.decode(String.self, forKey: .text)
    }
}

/// Anthropic `content`/`system` accept either a plain string or an array of
/// text blocks; both collapse to a single string for the chat template.
public enum AnthropicContent: Decodable, Sendable, Equatable {
    case text(String)
    case blocks([AnthropicTextBlock])

    public var text: String {
        switch self {
        case .text(let value): return value
        case .blocks(let blocks): return blocks.map(\.text).joined()
        }
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .text(value)
        } else if let blocks = try? container.decode([AnthropicTextBlock].self) {
            self = .blocks(blocks)
        } else {
            throw DecodingError.typeMismatch(
                AnthropicContent.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "content must be a string or an array of text blocks"))
        }
    }
}

public struct AnthropicMessage: Decodable, Sendable {
    public var role: String
    public var content: AnthropicContent

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        content = try container.decodeIfPresent(AnthropicContent.self, forKey: .content) ?? .text("")
    }

    enum CodingKeys: String, CodingKey {
        case role
        case content
    }
}

public struct AnthropicMessagesRequest: Decodable, Sendable {
    public var model: String?
    public var maxTokens: Int?
    public var messages: [AnthropicMessage]
    public var system: AnthropicContent?
    public var temperature: Float?
    public var topP: Float?
    public var stopSequences: [String]?
    public var stream: Bool?
    public var n: Int?
    /// Key-presence markers for parameters we reject rather than ignore.
    public var hasTools: Bool
    public var hasToolChoice: Bool
    public var hasThinking: Bool
    public var hasMetadata: Bool

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case system
        case temperature
        case stream
        case n
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case stopSequences = "stop_sequences"
        case tools
        case toolChoice = "tool_choice"
        case thinking
        case metadata
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        messages = try container.decodeIfPresent([AnthropicMessage].self, forKey: .messages) ?? []
        system = try container.decodeIfPresent(AnthropicContent.self, forKey: .system)
        temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        stopSequences = try container.decodeIfPresent([String].self, forKey: .stopSequences)
        stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        n = try container.decodeIfPresent(Int.self, forKey: .n)
        hasTools = container.contains(.tools)
        hasToolChoice = container.contains(.toolChoice)
        hasThinking = container.contains(.thinking)
        hasMetadata = container.contains(.metadata)
    }
}

public struct AnthropicUsage: Encodable, Sendable {
    public var inputTokens: Int
    public var outputTokens: Int

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }

    public init(inputTokens: Int, outputTokens: Int) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
    }
}

/// Non-streaming response, also embedded (with empty content) in the
/// streaming `message_start` event.
public struct AnthropicMessageResponse: Encodable, Sendable {
    public var id: String
    public var type: String
    public var role: String
    public var content: [AnthropicTextBlock]
    public var model: String
    public var stopReason: String?
    public var stopSequence: String?
    public var usage: AnthropicUsage

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case role
        case content
        case model
        case stopReason = "stop_reason"
        case stopSequence = "stop_sequence"
        case usage
    }

    public init(id: String,
                model: String,
                content: [AnthropicTextBlock],
                stopReason: String?,
                usage: AnthropicUsage) {
        self.id = id
        self.type = "message"
        self.role = "assistant"
        self.content = content
        self.model = model
        self.stopReason = stopReason
        self.stopSequence = nil
        self.usage = usage
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(type, forKey: .type)
        try container.encode(role, forKey: .role)
        try container.encode(content, forKey: .content)
        try container.encode(model, forKey: .model)
        // Anthropic emits explicit nulls while the turn is in flight; mirror.
        try container.encode(stopReason, forKey: .stopReason)
        try container.encode(stopSequence, forKey: .stopSequence)
        try container.encode(usage, forKey: .usage)
    }
}

public struct AnthropicMessageStart: Encodable, Sendable {
    public var type: String
    public var message: AnthropicMessageResponse

    public init(message: AnthropicMessageResponse) {
        self.type = "message_start"
        self.message = message
    }
}

public struct AnthropicContentBlockStart: Encodable, Sendable {
    public var type: String
    public var index: Int
    public var contentBlock: AnthropicTextBlock

    enum CodingKeys: String, CodingKey {
        case type
        case index
        case contentBlock = "content_block"
    }

    public init(index: Int, contentBlock: AnthropicTextBlock) {
        self.type = "content_block_start"
        self.index = index
        self.contentBlock = contentBlock
    }
}

public struct AnthropicTextDelta: Encodable, Sendable {
    public var type: String
    public var text: String

    public init(text: String) {
        self.type = "text_delta"
        self.text = text
    }
}

public struct AnthropicContentBlockDelta: Encodable, Sendable {
    public var type: String
    public var index: Int
    public var delta: AnthropicTextDelta

    public init(index: Int, delta: AnthropicTextDelta) {
        self.type = "content_block_delta"
        self.index = index
        self.delta = delta
    }
}

public struct AnthropicContentBlockStop: Encodable, Sendable {
    public var type: String
    public var index: Int

    public init(index: Int) {
        self.type = "content_block_stop"
        self.index = index
    }
}

public struct AnthropicMessageDeltaBody: Encodable, Sendable {
    public var stopReason: String?
    public var stopSequence: String?

    enum CodingKeys: String, CodingKey {
        case stopReason = "stop_reason"
        case stopSequence = "stop_sequence"
    }

    public init(stopReason: String?, stopSequence: String? = nil) {
        self.stopReason = stopReason
        self.stopSequence = stopSequence
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(stopReason, forKey: .stopReason)
        try container.encode(stopSequence, forKey: .stopSequence)
    }
}

public struct AnthropicOutputUsage: Encodable, Sendable {
    public var outputTokens: Int

    enum CodingKeys: String, CodingKey {
        case outputTokens = "output_tokens"
    }

    public init(outputTokens: Int) {
        self.outputTokens = outputTokens
    }
}

public struct AnthropicMessageDelta: Encodable, Sendable {
    public var type: String
    public var delta: AnthropicMessageDeltaBody
    public var usage: AnthropicOutputUsage

    public init(delta: AnthropicMessageDeltaBody, usage: AnthropicOutputUsage) {
        self.type = "message_delta"
        self.delta = delta
        self.usage = usage
    }
}

public struct AnthropicMessageStop: Encodable, Sendable {
    public var type: String

    public init() {
        self.type = "message_stop"
    }
}

public struct AnthropicErrorBody: Encodable, Sendable {
    public var type: String
    public var message: String

    public init(type: String, message: String) {
        self.type = type
        self.message = message
    }
}

public struct AnthropicErrorResponse: Encodable, Sendable {
    public var type: String
    public var error: AnthropicErrorBody

    public init(type: String = "invalid_request_error", message: String) {
        self.type = "error"
        self.error = AnthropicErrorBody(type: type, message: message)
    }
}

extension FinishReason {
    /// Anthropic stop reasons: OpenAI `stop` maps to `end_turn`, `length` to
    /// `max_tokens`.
    public var anthropicStopReason: String {
        switch self {
        case .stop: return "end_turn"
        case .length: return "max_tokens"
        }
    }
}
