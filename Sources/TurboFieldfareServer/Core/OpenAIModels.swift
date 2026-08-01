import Foundation
import TurboFieldfare

public struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String
    }

    public let error: Detail

    public init(message: String, param: String? = nil, code: String) {
        error = Detail(message: message,
                       type: "invalid_request_error",
                       param: param,
                       code: code)
    }
}

public struct OpenAITextPart: Codable, Equatable, Sendable {
    public let type: String
    public let text: String?
}

public enum OpenAIMessageContent: Codable, Equatable, Sendable {
    case text(String)
    case parts([OpenAITextPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAITextPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .parts(let parts): try container.encode(parts)
        }
    }

    func textValue() throws -> String {
        switch self {
        case .text(let text):
            return text
        case .parts(let parts):
            guard parts.allSatisfy({ $0.type == "text" && $0.text != nil }) else {
                throw ServerRequestError.invalid(
                    message: "only text content parts are supported",
                    param: "messages",
                    code: "unsupported_content")
            }
            return parts.compactMap(\.text).joined()
        }
    }
}

public struct OpenAIFunctionCall: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: String
}

public struct OpenAIToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let function: OpenAIFunctionCall
}

public struct OpenAIChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: OpenAIMessageContent?
    public let toolCalls: [OpenAIToolCall]?
    public let toolCallID: String?
    public let name: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

public struct OpenAIFunctionDefinition: Codable, Equatable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue
}

public struct OpenAITool: Codable, Equatable, Sendable {
    public let type: String
    public let function: OpenAIFunctionDefinition
}

public enum OpenAIStop: Codable, Equatable, Sendable {
    case one(String)
    case many([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let one = try? container.decode(String.self) {
            self = .one(one)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .one(let value): try container.encode(value)
        case .many(let value): try container.encode(value)
        }
    }

    var values: [String] {
        switch self {
        case .one(let value): [value]
        case .many(let value): value
        }
    }
}

public struct OpenAIStreamOptions: Codable, Equatable, Sendable {
    public let includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

public struct OpenAIChatRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let stream: Bool?
    public let streamOptions: OpenAIStreamOptions?
    public let temperature: Float?
    public let topP: Float?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let stop: OpenAIStop?
    public let seed: UInt64?
    public let tools: [OpenAITool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let n: Int?
    public let logprobs: Bool?
    public let presencePenalty: Float?
    public let frequencyPenalty: Float?

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
    }
}

public struct OpenAIUsage: Codable, Equatable, Sendable {
    public struct PromptTokensDetails: Codable, Equatable, Sendable {
        public let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }

        public init(cachedTokens: Int) {
            self.cachedTokens = cachedTokens
        }
    }

    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let promptTokensDetails: PromptTokensDetails

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
    }

    public init(promptTokens: Int,
                completionTokens: Int,
                totalTokens: Int,
                cachedTokens: Int = 0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: cachedTokens)
    }
}

public struct OpenAIModelList: Codable, Equatable, Sendable {
    public struct Model: Codable, Equatable, Sendable {
        public let id: String
        public let object: String
        public let created: Int
        public let ownedBy: String

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }
    }

    public let object: String
    public let data: [Model]
}

public enum ServerRequestError: Error, Equatable, Sendable {
    case invalid(message: String, param: String?, code: String)
    case unknownModel
    case queueFull

    public var envelope: OpenAIErrorEnvelope {
        switch self {
        case .invalid(let message, let param, let code):
            OpenAIErrorEnvelope(message: message, param: param, code: code)
        case .unknownModel:
            OpenAIErrorEnvelope(message: "requested model is not available",
                                param: "model", code: "model_not_found")
        case .queueFull:
            OpenAIErrorEnvelope(message: "generation queue is full",
                                code: "queue_full")
        }
    }
}

public struct ValidatedChatRequest: Sendable {
    public let messages: [GFTokenizer.Message]
    public let tools: [GFTokenizer.FunctionDefinition]
    public let stream: Bool
    public let includeUsage: Bool
    public let generationConfig: GenerationConfig
    public let maximumCompletionTokens: Int
}

public enum OpenAIRequestValidator {
    public static func validate(_ request: OpenAIChatRequest,
                                modelID: String,
                                dialect: ChatDialect = .gemma) throws -> ValidatedChatRequest {
        guard request.model == modelID else { throw ServerRequestError.unknownModel }
        guard request.n == nil || request.n == 1 else {
            throw invalid("only n=1 is supported", "n", "unsupported_value")
        }
        guard request.logprobs != true else {
            throw invalid("logprobs are not supported", "logprobs", "unsupported_value")
        }
        guard request.presencePenalty == nil || request.presencePenalty == 0 else {
            throw invalid("presence_penalty must be zero", "presence_penalty", "unsupported_value")
        }
        guard request.frequencyPenalty == nil || request.frequencyPenalty == 0 else {
            throw invalid("frequency_penalty must be zero", "frequency_penalty", "unsupported_value")
        }
        guard request.parallelToolCalls != false else {
            throw invalid("parallel_tool_calls=false is not supported",
                          "parallel_tool_calls", "unsupported_value")
        }

        let temperature = request.temperature ?? 0.2
        guard temperature >= 0, temperature <= 2 else {
            throw invalid("temperature must be between 0 and 2",
                          "temperature", "invalid_value")
        }
        let topP = request.topP ?? 0.95
        guard topP > 0, topP <= 1 else {
            throw invalid("top_p must be greater than 0 and at most 1",
                          "top_p", "invalid_value")
        }
        let topK = request.topK ?? 64
        guard (1...256).contains(topK) else {
            throw invalid("top_k must be between 1 and 256", "top_k", "invalid_value")
        }
        let repetitionPenalty = request.repetitionPenalty ?? 1
        guard repetitionPenalty > 0 else {
            throw invalid("repetition_penalty must be positive",
                          "repetition_penalty", "invalid_value")
        }
        let maximum = request.maxCompletionTokens ?? request.maxTokens ?? 4096
        guard maximum > 0 else {
            throw invalid("maximum completion tokens must be positive",
                          request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                          "invalid_value")
        }

        let includeTools: Bool
        switch request.toolChoice {
        case nil, .some(.string("auto")):
            includeTools = true
        case .some(.string("none")):
            includeTools = false
        case .some(.string("required")):
            throw invalid("tool_choice=required is not supported",
                          "tool_choice", "unsupported_value")
        default:
            throw invalid("named tool choices are not supported",
                          "tool_choice", "unsupported_value")
        }

        let tools = try (includeTools ? request.tools ?? [] : []).map {
            try validateTool($0, dialect: dialect)
        }
        let messages = try validateMessages(request.messages, dialect: dialect)
        let config = GenerationConfig(maxNewTokens: maximum,
                                      temperature: temperature,
                                      topK: topK,
                                      topP: topP,
                                      repetitionPenalty: repetitionPenalty,
                                      seed: request.seed,
                                      stopStrings: request.stop?.values ?? [])
        return ValidatedChatRequest(messages: messages,
                                    tools: tools,
                                    stream: request.stream ?? false,
                                    includeUsage: request.streamOptions?.includeUsage ?? false,
                                    generationConfig: config,
                                    maximumCompletionTokens: maximum)
    }

    private static func validateTool(_ tool: OpenAITool,
                                     dialect: ChatDialect) throws -> GFTokenizer.FunctionDefinition {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard name.range(of: #"^[A-Za-z0-9_]{1,64}$"#, options: .regularExpression) != nil else {
            throw invalid("tool name must match [A-Za-z0-9_]{1,64}",
                          "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid("tool parameters must be an object schema",
                          "tools", "invalid_tool_schema")
        }
        try validateSchemaKeys(tool.function.parameters, dialect: dialect)
        guard (try? tool.function.parameters.jinjaSendableValue()) != nil else {
            throw invalid("tool schema contains a number that cannot be represented exactly",
                          "tools", "invalid_tool_schema")
        }
        return GFTokenizer.FunctionDefinition(name: name,
                                              description: tool.function.description ?? "",
                                              parameters: tool.function.parameters)
    }

    private static func validateSchemaKeys(_ schema: JSONValue,
                                           dialect: ChatDialect) throws {
        switch schema {
        case .object(let object):
            for (schemaKey, value) in object {
                if schemaKey == "properties" {
                    guard case .object(let definitions) = value else {
                        throw invalid("tool schema properties must be an object",
                                      "tools", "invalid_tool_schema")
                    }
                    for (key, definition) in definitions {
                        // Gemma's tool-call DSL cannot round-trip arbitrary
                        // parameter names; ChatML tool calls are free-form.
                        guard dialect == .chatml
                                || GemmaToolCallParser.isRepresentableObjectKey(key) else {
                            throw invalid(
                                "tool parameter names may contain only letters, numbers, _, -, ., and $",
                                "tools",
                                "invalid_tool_schema")
                        }
                        try validateSchemaKeys(definition, dialect: dialect)
                    }
                } else {
                    try validateSchemaKeys(value, dialect: dialect)
                }
            }
        case .array(let values):
            for value in values {
                try validateSchemaKeys(value, dialect: dialect)
            }
        default:
            break
        }
    }

    private static func validateMessages(_ input: [OpenAIChatMessage],
                                         dialect: ChatDialect) throws -> [GFTokenizer.Message] {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        var knownCalls: [String: (name: String, resolved: Bool)] = [:]
        var result: [GFTokenizer.Message] = []
        var sawConversationMessage = false
        for message in input {
            guard let role = GFTokenizer.Role(rawValue: message.role) else {
                throw invalid("unsupported message role \(message.role)",
                              "messages", "invalid_message")
            }
            if role == .system || role == .developer {
                guard !sawConversationMessage else {
                    throw invalid("system or developer guidance must precede the conversation",
                                  "messages", "invalid_message")
                }
            } else {
                sawConversationMessage = true
            }
            let content = try message.content?.textValue()
            let calls: [GFTokenizer.HistoricalToolCall] = try (message.toolCalls ?? []).map { call in
                guard role == .assistant, call.type == "function",
                      !call.id.isEmpty, knownCalls[call.id] == nil,
                      call.function.name.range(
                        of: #"^[A-Za-z0-9_]{1,64}$"#,
                        options: .regularExpression) != nil else {
                    throw invalid("invalid or duplicate historical tool call",
                                  "messages", "invalid_tool_call")
                }
                let data = Data(call.function.arguments.utf8)
                let arguments = try JSONDecoder().decode(JSONValue.self, from: data)
                guard arguments.objectValue != nil else {
                    throw invalid("historical tool arguments must be a JSON object",
                                  "messages", "invalid_tool_arguments")
                }
                guard dialect == .chatml
                        || (try? arguments.gemmaToolArgumentBody()) != nil,
                      (try? arguments.jinjaSendableValue()) != nil else {
                    throw invalid(
                        "historical tool arguments cannot be represented exactly",
                        "messages",
                        "invalid_tool_arguments")
                }
                knownCalls[call.id] = (call.function.name, false)
                return GFTokenizer.HistoricalToolCall(
                    id: call.id, name: call.function.name, arguments: arguments)
            }
            if role == .tool {
                guard let id = message.toolCallID,
                      let call = knownCalls[id], !call.resolved else {
                    throw invalid("tool result must reference one unresolved call",
                                  "messages", "invalid_tool_result")
                }
                knownCalls[id] = (call.name, true)
                guard content != nil else {
                    throw invalid("tool result content is required",
                                  "messages", "invalid_tool_result")
                }
            } else if content == nil && calls.isEmpty {
                throw invalid("message content is required",
                              "messages", "invalid_message")
            }
            result.append(GFTokenizer.Message(role: role,
                                              content: content,
                                              toolCalls: calls,
                                              toolCallID: message.toolCallID,
                                              name: message.name))
        }
        return result
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}
