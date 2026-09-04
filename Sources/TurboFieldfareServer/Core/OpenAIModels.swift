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

    public init(message: String, param: String? = nil, code: String,
                type: String = "invalid_request_error") {
        error = Detail(message: message,
                       type: type,
                       param: param,
                       code: code)
    }
}

public struct OpenAIImageURL: Codable, Equatable, Sendable {
    public let url: String
    public let detail: String?
}

public struct OpenAIContentPart: Codable, Equatable, Sendable {
    public let type: String
    public let text: String?
    public let imageURL: OpenAIImageURL?

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageURL = "image_url"
    }
}

public typealias OpenAITextPart = OpenAIContentPart

public enum OpenAIMessageContent: Codable, Equatable, Sendable {
    case text(String)
    case parts([OpenAIContentPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAIContentPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .parts(let parts): try container.encode(parts)
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

/// Bounds what a rejection quotes back. The body cap is 5 MiB, so a caller must
/// not be able to have an arbitrary slice of its own request echoed back;
/// `String(reflecting:)` also escapes a value that embeds a quote, which would
/// otherwise read as two values.
private func boundedQuoted(_ text: String, maxLength: Int) -> String {
    String(reflecting: bounded(text, maxLength: maxLength))
}

/// The unquoted form, for an error's `param`. A `param` names a field rather
/// than quoting prose, but it needs the same bound: an unknown key is echoed
/// back out of the request body, and the 5 MiB body cap is the only other
/// limit on how long that key can be.
///
/// The bound is in UTF-8 bytes, never Characters: one Character can carry
/// megabytes of combining marks, so a Character-counted prefix of a key like
/// "a" followed by a million U+0301 is the whole key. Cutting between scalars
/// may split a grapheme, which is harmless in a diagnostic.
private func bounded(_ text: String, maxLength: Int) -> String {
    var bytes = 0
    var head = String.UnicodeScalarView()
    for scalar in text.unicodeScalars {
        bytes += scalar.utf8.count
        if bytes > maxLength {
            return String(head) + "..."
        }
        head.append(scalar)
    }
    return text
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
    /// Kept as raw JSON. Only `type` is ever read, because the rest of a
    /// `response_format` describes a schema this server cannot honour, and a
    /// value of any other shape has to reach the validator as a request error
    /// rather than reading as malformed JSON.
    public let responseFormat: JSONValue?

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
        case responseFormat = "response_format"
    }

    /// Top-level keys accepted and ignored because they are caller-side
    /// bookkeeping that cannot change what the model generates. Every other
    /// undeclared key is a 400, so a misspelled option cannot silently
    /// generate under settings the caller did not ask for. This set is the
    /// only place the list lives; the OpenAI server doc quotes it.
    static let toleratedKeys: Set<String> = [
        "user",
        "store",
        "metadata",
        "service_tier",
        "prompt_cache_key",
        "safety_identifier",
    ]

    /// Real OpenAI parameters this server cannot honour. They are refused as
    /// unsupported rather than unknown, so a caller sending a parameter that
    /// exists is not told it looks like a typo. Refused here, before the typed
    /// decode, so the answer is the same beside a mistyped declared field as
    /// alone; a refusal left to the validator would lose to whatever
    /// DecodingError the typed decode raised first.
    static let unsupportedKeys: Set<String> = [
        "logit_bias",
        "top_logprobs",
        "reasoning_effort",
        "verbosity",
        "modalities",
        "audio",
        "prediction",
        "web_search_options",
        "functions",
        "function_call",
    ]

    /// Refusals that can point at the supported replacement. Every other
    /// member of `unsupportedKeys` is refused with the plain form.
    static let unsupportedKeyMessages: [String: String] = [
        "functions": "legacy functions are not supported; use tools",
        "function_call": "legacy function_call is not supported; use tools and tool_choice",
    ]

    /// Reads the request object's keys as written, which the `CodingKeys`
    /// container cannot: that one reports only the keys it declares, so an
    /// undeclared key is discarded before validation could see it.
    private struct AnyKey: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }

        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    private static let maximumNamedKeyLength = 64
    private static let maximumNamedKeys = 8

    private static func renderedKeyNames(_ names: [String]) -> String {
        let shown = names.prefix(maximumNamedKeys).map {
            boundedQuoted($0, maxLength: maximumNamedKeyLength)
        }
        let listed = shown.joined(separator: ", ")
        let remaining = names.count - shown.count
        return remaining > 0 ? "\(listed), and \(remaining) more" : listed
    }

    public init(from decoder: any Decoder) throws {
        // The key sweep runs before any typed decode, so a misspelled key is
        // named as itself rather than answered with whatever DecodingError
        // another field happens to raise first.
        let anyKeys = try decoder.container(keyedBy: AnyKey.self)
        // A key set to null asks for nothing: openai-python sends an unset
        // option as an explicit `null`, and the declared keys already read null
        // as absent through `decodeIfPresent`.
        let written = try anyKeys.allKeys
            .filter { try !anyKeys.decodeNil(forKey: $0) }
            .map(\.stringValue)
        // Sorted throughout so the answer never depends on the order the keys
        // arrived in.
        if let unsupported = written.filter(Self.unsupportedKeys.contains).sorted().first {
            throw ServerRequestError.invalid(
                message: Self.unsupportedKeyMessages[unsupported]
                    ?? "\(unsupported) is not supported",
                param: unsupported,
                code: "unsupported_value")
        }
        let unknown = written
            .filter { CodingKeys(stringValue: $0) == nil && !Self.toleratedKeys.contains($0) }
            .sorted()
        if let first = unknown.first {
            throw ServerRequestError.invalid(
                message: "unrecognized request field\(unknown.count == 1 ? "" : "s") "
                    + Self.renderedKeyNames(unknown),
                param: bounded(first, maxLength: Self.maximumNamedKeyLength),
                code: "unknown_parameter")
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        messages = try container.decode([OpenAIChatMessage].self, forKey: .messages)
        stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        streamOptions = try container.decodeIfPresent(
            OpenAIStreamOptions.self, forKey: .streamOptions)
        temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        maxCompletionTokens = try container.decodeIfPresent(
            Int.self, forKey: .maxCompletionTokens)
        stop = try container.decodeIfPresent(OpenAIStop.self, forKey: .stop)
        seed = try container.decodeIfPresent(UInt64.self, forKey: .seed)
        tools = try container.decodeIfPresent([OpenAITool].self, forKey: .tools)
        toolChoice = try container.decodeIfPresent(JSONValue.self, forKey: .toolChoice)
        parallelToolCalls = try container.decodeIfPresent(
            Bool.self, forKey: .parallelToolCalls)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        repetitionPenalty = try container.decodeIfPresent(
            Float.self, forKey: .repetitionPenalty)
        n = try container.decodeIfPresent(Int.self, forKey: .n)
        logprobs = try container.decodeIfPresent(Bool.self, forKey: .logprobs)
        presencePenalty = try container.decodeIfPresent(
            Float.self, forKey: .presencePenalty)
        frequencyPenalty = try container.decodeIfPresent(
            Float.self, forKey: .frequencyPenalty)
        responseFormat = try container.decodeIfPresent(
            JSONValue.self, forKey: .responseFormat)
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
        public let capabilities: [String]?

        enum CodingKeys: String, CodingKey {
            case id, object, created, capabilities
            case ownedBy = "owned_by"
        }

        public init(id: String, object: String, created: Int,
                    ownedBy: String, capabilities: [String]? = nil) {
            self.id = id
            self.object = object
            self.created = created
            self.ownedBy = ownedBy
            self.capabilities = capabilities
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
    public let multimodalMessages: [MultimodalMessage]?
    public let imageFiles: [UUID: URL]
    /// Content SHA-256 of each message's images, in order, aligned with
    /// `messages`. Staged image UUIDs are fresh per request, so they cannot
    /// identify an image across turns; the content hash can. Empty for
    /// text-only requests.
    public let imageIdentities: [[String]]
    public let tools: [GFTokenizer.FunctionDefinition]
    public let stream: Bool
    public let includeUsage: Bool
    public let generationConfig: GenerationConfig
    public let maximumCompletionTokens: Int
    /// Every staging directory this request's image files live in. The parser
    /// and the validator's store each stage under their own lease, and a
    /// request may carry files from both, so dropping either would delete
    /// files the other path staged before `generate` reads them.
    fileprivate let attachmentLeases: [ServerAttachmentLease]

    public init(
        messages: [GFTokenizer.Message],
        multimodalMessages: [MultimodalMessage]? = nil,
        imageFiles: [UUID: URL] = [:],
        imageIdentities: [[String]] = [],
        tools: [GFTokenizer.FunctionDefinition],
        stream: Bool,
        includeUsage: Bool,
        generationConfig: GenerationConfig,
        maximumCompletionTokens: Int
    ) {
        self.messages = messages
        self.multimodalMessages = multimodalMessages
        self.imageFiles = imageFiles
        self.imageIdentities = imageIdentities
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.attachmentLeases = []
    }

    fileprivate init(
        messages: [GFTokenizer.Message],
        multimodalMessages: [MultimodalMessage]?,
        imageFiles: [UUID: URL],
        imageIdentities: [[String]],
        tools: [GFTokenizer.FunctionDefinition],
        stream: Bool,
        includeUsage: Bool,
        generationConfig: GenerationConfig,
        maximumCompletionTokens: Int,
        attachmentLeases: [ServerAttachmentLease]
    ) {
        self.messages = messages
        self.multimodalMessages = multimodalMessages
        self.imageFiles = imageFiles
        self.imageIdentities = imageIdentities
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.attachmentLeases = attachmentLeases
    }
}

private enum OpenAIToolName {
    static let maximumLength = 64

    static func isValid(_ name: String) -> Bool {
        let bytes = name.utf8
        guard !bytes.isEmpty, bytes.count <= maximumLength else { return false }
        return bytes.allSatisfy { byte in
            switch byte {
            case 45, 48...57, 65...90, 95, 97...122:
                true
            default:
                false
            }
        }
    }

    static func validationMessage(for name: String) -> String {
        "tool name \(boundedQuoted(name, maxLength: maximumLength)) must contain 1 to 64 ASCII letters, numbers, underscores, or hyphens"
    }
}

public enum OpenAIRequestValidator {
    public static func validate(_ request: OpenAIChatRequest,
                                modelID: String) throws -> ValidatedChatRequest {
        try validate(
            request,
            modelID: modelID,
            preStagedImages: [:],
            attachmentLease: nil)
    }

    static func validate(_ request: OpenAIChatRequest,
                         modelID: String,
                         preStagedImages: [String: ServerStagedImage],
                         attachmentLease: ServerAttachmentLease?) throws -> ValidatedChatRequest {
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
        switch request.responseFormat {
        case nil:
            break
        case .object(let fields)?:
            switch fields["type"] {
            case nil, .null?:
                throw invalid("response_format.type is required",
                              "response_format", "invalid_value")
            case .string(let type)?:
                switch type {
                case "text":
                    break
                case "json_object", "json_schema":
                    throw invalid("structured output is not supported",
                                  "response_format", "unsupported_value")
                default:
                    throw invalid(
                        "response_format type \(boundedQuoted(type, maxLength: 64)) "
                            + "is not recognized",
                        "response_format", "invalid_value")
                }
            default:
                throw invalid("response_format.type must be a string",
                              "response_format", "invalid_value")
            }
        default:
            throw invalid(#"response_format must be an object such as {"type": "text"}"#,
                          "response_format", "invalid_value")
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

        let tools = try (includeTools ? request.tools ?? [] : []).map(validateTool)
        let validatedMessages = try validateMessages(
            request.messages,
            preStagedImages: preStagedImages,
            attachmentLease: attachmentLease)
        let config = GenerationConfig(maxNewTokens: maximum,
                                      temperature: temperature,
                                      topK: topK,
                                      topP: topP,
                                      repetitionPenalty: repetitionPenalty,
                                      seed: request.seed,
                                      stopStrings: request.stop?.values ?? [])
        return ValidatedChatRequest(messages: validatedMessages.messages,
                                    multimodalMessages: validatedMessages.multimodal,
                                    imageFiles: validatedMessages.imageFiles,
                                    imageIdentities: validatedMessages.imageIdentities,
                                    tools: tools,
                                    stream: request.stream ?? false,
                                    includeUsage: request.streamOptions?.includeUsage ?? false,
                                    generationConfig: config,
                                    maximumCompletionTokens: maximum,
                                    attachmentLeases: validatedMessages.leases)
    }

    private static func validateTool(_ tool: OpenAITool) throws -> GFTokenizer.FunctionDefinition {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard OpenAIToolName.isValid(name) else {
            throw invalid(OpenAIToolName.validationMessage(for: name),
                          "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid("tool parameters must be an object schema",
                          "tools", "invalid_tool_schema")
        }
        try validateSchemaKeys(tool.function.parameters)
        let parameters = try GemmaToolSchema.adapted(
            tool.function.parameters, toolName: name)
        do {
            guard (try? parameters.jinjaSendableValue()) != nil else {
                throw invalid("tool schema contains a number that cannot be represented exactly",
                              "tools", "invalid_tool_schema")
            }
        } catch {
            // Carry the underlying cause: "cannot be represented exactly" alone
            // does not say which value, and this surfaces to a remote client.
            throw invalid(
                "tool schema cannot be represented exactly: \(error)",
                "tools", "invalid_tool_schema")
        }
        return GFTokenizer.FunctionDefinition(name: name,
                                              description: tool.function.description ?? "",
                                              parameters: parameters)
    }

    private static func validateSchemaKeys(_ schema: JSONValue) throws {
        switch schema {
        case .object(let object):
            for (schemaKey, value) in object {
                if schemaKey == "properties" {
                    guard case .object(let definitions) = value else {
                        throw invalid("tool schema properties must be an object",
                                      "tools", "invalid_tool_schema")
                    }
                    for (key, definition) in definitions {
                        guard GemmaToolCallParser.isRepresentableObjectKey(key) else {
                            throw invalid(
                                "tool parameter names may contain only letters, numbers, _, -, ., and $",
                                "tools",
                                "invalid_tool_schema")
                        }
                        try validateSchemaKeys(definition)
                    }
                } else {
                    try validateSchemaKeys(value)
                }
            }
        case .array(let values):
            for value in values {
                try validateSchemaKeys(value)
            }
        default:
            break
        }
    }

    private struct ValidatedMessages {
        let messages: [GFTokenizer.Message]
        let multimodal: [MultimodalMessage]?
        let imageFiles: [UUID: URL]
        let imageIdentities: [[String]]
        let leases: [ServerAttachmentLease]
    }

    private static func validateMessages(
        _ input: [OpenAIChatMessage],
        preStagedImages: [String: ServerStagedImage],
        attachmentLease: ServerAttachmentLease?
    ) throws -> ValidatedMessages {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        var knownCalls: [String: (name: String, resolved: Bool)] = [:]
        var result: [GFTokenizer.Message] = []
        var multimodal: [MultimodalMessage] = []
        var imageFiles: [UUID: URL] = [:]
        var imageIdentities: [[String]] = []
        var messageIdentities: [String] = []
        var store: ServerAttachmentStore?
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
            var orderedContent: [MultimodalContentPart] = []
            let content: String?
            switch message.content {
            case nil:
                content = nil
            case .text(let text):
                content = text
                orderedContent = [.text(text)]
            case .parts(let parts):
                var joined = ""
                for part in parts {
                    switch part.type {
                    case "text":
                        guard let text = part.text else {
                            throw invalid("text content part requires text",
                                          "messages", "invalid_message")
                        }
                        joined += text
                        orderedContent.append(.text(text))
                    case "image_url":
                        guard role == .user else {
                            throw invalid("image_url is supported only in user messages",
                                          "messages", "unsupported_content")
                        }
                        guard let image = part.imageURL else {
                            throw invalid("image_url content part requires an image_url object",
                                          "messages", "invalid_message")
                        }
                        guard image.detail == nil || image.detail == "auto" else {
                            throw invalid("image detail must be absent or auto",
                                          "messages", "unsupported_value")
                        }
                        let staged: ServerStagedImage
                        let prefix = "turbofieldfare-attachment:"
                        if image.url.hasPrefix(prefix) {
                            let token = String(image.url.dropFirst(prefix.count))
                            guard let existing = preStagedImages[token] else {
                                throw invalid("image attachment lease is missing",
                                              "messages", "invalid_image")
                            }
                            staged = existing
                        } else {
                            if store == nil { store = try ServerAttachmentStore() }
                            staged = try store!.stage(dataURL: image.url)
                        }
                        imageFiles[staged.id] = staged.fileURL
                        messageIdentities.append(staged.sha256)
                        orderedContent.append(.image(id: staged.id))
                    default:
                        throw invalid("unsupported content part \(part.type)",
                                      "messages", "unsupported_content")
                    }
                }
                content = joined
            }
            // The semantic bound on images is the context budget, checked
            // against the model's actual context in `ServerModelSession.prepare`
            // where the per-image token cost is known. Staging enforces its own
            // per-file, per-request, and count resource caps; no further count
            // check belongs here.

            let calls: [GFTokenizer.HistoricalToolCall] = try (message.toolCalls ?? []).map { call in
                guard role == .assistant, call.type == "function",
                      !call.id.isEmpty, knownCalls[call.id] == nil else {
                    throw invalid("invalid or duplicate historical tool call",
                                  "messages", "invalid_tool_call")
                }
                guard OpenAIToolName.isValid(call.function.name) else {
                    throw invalid(OpenAIToolName.validationMessage(for: call.function.name),
                                  "messages", "invalid_tool_call")
                }
                let data = Data(call.function.arguments.utf8)
                let arguments = try JSONDecoder().decode(JSONValue.self, from: data)
                guard arguments.objectValue != nil else {
                    throw invalid("historical tool arguments must be a JSON object",
                                  "messages", "invalid_tool_arguments")
                }
                do {
                    guard (try? arguments.gemmaToolArgumentBody()) != nil,
                          (try? arguments.jinjaSendableValue()) != nil else {
                        throw invalid(
                            "historical tool arguments cannot be represented exactly",
                            "messages", "invalid_message")
                    }
                } catch {
                    throw invalid(
                        "historical tool arguments cannot be represented exactly: "
                            + "\(error)",
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
            if orderedContent.isEmpty, let content {
                orderedContent = [.text(content)]
            }
            multimodal.append(MultimodalMessage(
                role: role,
                content: orderedContent,
                toolCalls: calls,
                toolCallID: message.toolCallID,
                name: message.name))
            imageIdentities.append(messageIdentities)
            messageIdentities = []
        }
        return ValidatedMessages(
            messages: result,
            multimodal: imageFiles.isEmpty ? nil : multimodal,
            imageFiles: imageFiles,
            imageIdentities: imageFiles.isEmpty ? [] : imageIdentities,
            leases: [attachmentLease, store?.lease].compactMap { $0 })
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}
