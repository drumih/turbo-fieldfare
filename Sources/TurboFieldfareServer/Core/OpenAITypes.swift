import Foundation

/// OpenAI-compatible wire types for the LM Studio-style HTTP API. Unknown
/// request fields are ignored by the default `Decodable` synthesis, which
/// keeps the server tolerant of clients that send extra knobs.

public struct ChatMessage: Codable, Sendable, Equatable {
    public var role: String
    public var content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct StreamOptions: Codable, Sendable, Equatable {
    public var includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }

    public init(includeUsage: Bool? = nil) {
        self.includeUsage = includeUsage
    }
}

/// OpenAI `stop` accepts either a single string or an array of strings.
public enum StopValue: Codable, Sendable, Equatable {
    case single(String)
    case multiple([String])

    public var values: [String] {
        switch self {
        case .single(let value): return [value]
        case .multiple(let values): return values
        }
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .single(value)
        } else if let values = try? container.decode([String].self) {
            self = .multiple(values)
        } else {
            throw DecodingError.typeMismatch(
                StopValue.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "stop must be a string or an array of strings"))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .single(let value): try container.encode(value)
        case .multiple(let values): try container.encode(values)
        }
    }
}

public struct ChatCompletionRequest: Decodable, Sendable {
    public var model: String?
    public var messages: [ChatMessage]
    public var temperature: Float?
    public var topP: Float?
    public var maxTokens: Int?
    public var stop: StopValue?
    public var seed: Int?
    public var stream: Bool?
    public var streamOptions: StreamOptions?
    public var n: Int?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case temperature
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case stop
        case seed
        case stream
        case streamOptions = "stream_options"
        case n
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        messages = try container.decodeIfPresent([ChatMessage].self, forKey: .messages) ?? []
        temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        // `max_completion_tokens` is the newer alias; `max_tokens` wins when
        // both are present.
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
            ?? container.decodeIfPresent(Int.self, forKey: .maxCompletionTokens)
        stop = try container.decodeIfPresent(StopValue.self, forKey: .stop)
        seed = try container.decodeIfPresent(Int.self, forKey: .seed)
        stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        streamOptions = try container.decodeIfPresent(StreamOptions.self, forKey: .streamOptions)
        n = try container.decodeIfPresent(Int.self, forKey: .n)
    }

    public init(model: String? = nil,
                messages: [ChatMessage],
                temperature: Float? = nil,
                topP: Float? = nil,
                maxTokens: Int? = nil,
                stop: StopValue? = nil,
                seed: Int? = nil,
                stream: Bool? = nil,
                streamOptions: StreamOptions? = nil,
                n: Int? = nil) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.stop = stop
        self.seed = seed
        self.stream = stream
        self.streamOptions = streamOptions
        self.n = n
    }
}

public struct CompletionRequest: Decodable, Sendable {
    public var model: String?
    public var prompt: String
    public var temperature: Float?
    public var topP: Float?
    public var maxTokens: Int?
    public var stop: StopValue?
    public var seed: Int?
    public var stream: Bool?
    public var streamOptions: StreamOptions?
    public var n: Int?

    enum CodingKeys: String, CodingKey {
        case model
        case prompt
        case temperature
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case stop
        case seed
        case stream
        case streamOptions = "stream_options"
        case n
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        prompt = try container.decodeIfPresent(String.self, forKey: .prompt) ?? ""
        temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
            ?? container.decodeIfPresent(Int.self, forKey: .maxCompletionTokens)
        stop = try container.decodeIfPresent(StopValue.self, forKey: .stop)
        seed = try container.decodeIfPresent(Int.self, forKey: .seed)
        stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        streamOptions = try container.decodeIfPresent(StreamOptions.self, forKey: .streamOptions)
        n = try container.decodeIfPresent(Int.self, forKey: .n)
    }

    public init(model: String? = nil,
                prompt: String,
                temperature: Float? = nil,
                topP: Float? = nil,
                maxTokens: Int? = nil,
                stop: StopValue? = nil,
                seed: Int? = nil,
                stream: Bool? = nil,
                streamOptions: StreamOptions? = nil,
                n: Int? = nil) {
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.stop = stop
        self.seed = seed
        self.stream = stream
        self.streamOptions = streamOptions
        self.n = n
    }
}

public struct Usage: Codable, Sendable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = promptTokens + completionTokens
    }
}

public struct HealthResponse: Encodable, Sendable {
    public var status: String

    public init(status: String = "ok") {
        self.status = status
    }
}

public struct ModelObject: Encodable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var ownedBy: String

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case ownedBy = "owned_by"
    }

    public init(id: String, created: Int, ownedBy: String = "local") {
        self.id = id
        self.object = "model"
        self.created = created
        self.ownedBy = ownedBy
    }
}

public struct ModelList: Encodable, Sendable {
    public var object: String
    public var data: [ModelObject]

    public init(models: [ModelObject]) {
        self.object = "list"
        self.data = models
    }
}

public struct ChatChoiceMessage: Encodable, Sendable {
    public var role: String
    public var content: String

    public init(content: String) {
        self.role = "assistant"
        self.content = content
    }
}

public struct ChatChoice: Encodable, Sendable {
    public var index: Int
    public var message: ChatChoiceMessage
    public var finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index
        case message
        case finishReason = "finish_reason"
    }

    public init(index: Int, message: ChatChoiceMessage, finishReason: String?) {
        self.index = index
        self.message = message
        self.finishReason = finishReason
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(index, forKey: .index)
        try container.encode(message, forKey: .message)
        // OpenAI emits an explicit null for in-flight choices; mirror that.
        try container.encode(finishReason, forKey: .finishReason)
    }
}

public struct ChatCompletionResponse: Encodable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [ChatChoice]
    public var usage: Usage

    public init(id: String, created: Int, model: String, choices: [ChatChoice], usage: Usage) {
        self.id = id
        self.object = "chat.completion"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct ChatChunkDelta: Encodable, Sendable {
    public var role: String?
    public var content: String?

    public init(role: String? = nil, content: String? = nil) {
        self.role = role
        self.content = content
    }
}

public struct ChatChunkChoice: Encodable, Sendable {
    public var index: Int
    public var delta: ChatChunkDelta
    public var finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index
        case delta
        case finishReason = "finish_reason"
    }

    public init(index: Int, delta: ChatChunkDelta, finishReason: String?) {
        self.index = index
        self.delta = delta
        self.finishReason = finishReason
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(index, forKey: .index)
        try container.encode(delta, forKey: .delta)
        try container.encode(finishReason, forKey: .finishReason)
    }
}

public struct ChatCompletionChunk: Encodable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [ChatChunkChoice]
    public var usage: Usage?

    public init(id: String,
                created: Int,
                model: String,
                choices: [ChatChunkChoice],
                usage: Usage? = nil) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct CompletionChoice: Encodable, Sendable {
    public var index: Int
    public var text: String
    public var finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index
        case text
        case finishReason = "finish_reason"
    }

    public init(index: Int, text: String, finishReason: String?) {
        self.index = index
        self.text = text
        self.finishReason = finishReason
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(index, forKey: .index)
        try container.encode(text, forKey: .text)
        try container.encode(finishReason, forKey: .finishReason)
    }
}

public struct CompletionResponse: Encodable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [CompletionChoice]
    public var usage: Usage

    public init(id: String, created: Int, model: String, choices: [CompletionChoice], usage: Usage) {
        self.id = id
        self.object = "text_completion"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct CompletionChunk: Encodable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [CompletionChoice]
    public var usage: Usage?

    public init(id: String,
                created: Int,
                model: String,
                choices: [CompletionChoice],
                usage: Usage? = nil) {
        self.id = id
        self.object = "text_completion"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct OpenAIErrorBody: Encodable, Sendable {
    public var message: String
    public var type: String
    public var code: String?

    public init(message: String, type: String = "invalid_request_error", code: String? = nil) {
        self.message = message
        self.type = type
        self.code = code
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(message, forKey: .message)
        try container.encode(type, forKey: .type)
        try container.encode(code, forKey: .code)
    }

    enum CodingKeys: String, CodingKey {
        case message
        case type
        case code
    }
}

public struct OpenAIErrorResponse: Encodable, Sendable {
    public var error: OpenAIErrorBody

    public init(message: String, type: String = "invalid_request_error", code: String? = nil) {
        self.error = OpenAIErrorBody(message: message, type: type, code: code)
    }
}
