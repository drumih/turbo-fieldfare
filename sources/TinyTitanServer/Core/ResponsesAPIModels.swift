import Foundation
import TinyTitan
import TinyTitanCore

// MARK: - Responses API request decoding

/// The OpenAI Responses API request body (`POST /v1/responses`). Every
/// documented field is decoded so that a client's request never fails for
/// naming something this server merely ignores; the mapper below decides
/// which of them it can honour, echo, or must refuse. Text-only: image and
/// file inputs are rejected with a spec-shaped error.
public struct ResponsesAPIRequest: Decodable, Sendable {
    public struct Item: Codable, Sendable, Equatable {
        /// Item kind. Optional: some clients (e.g. OpenCode) omit it and rely
        /// on role+content / call_id+output to convey the kind.
        public let type: String?
        public let id: String?
        public let status: String?
        public let role: String?
        /// String content or an array of parts ({type: input_text, text}).
        public let content: JSONValue?
        public let callID: String?
        public let name: String?
        public let arguments: String?
        /// A function_call_output's result: a string or an array of parts.
        public let output: JSONValue?
        /// Reasoning items round-tripped by a client; carried, never read.
        public let summary: JSONValue?
        public let encryptedContent: String?
        /// The namespace a function call belongs to, when its tool was
        /// declared inside a `namespace` tool.
        public let namespace: String?

        enum CodingKeys: String, CodingKey {
            case type, id, status, role, content, name, arguments, output, summary, namespace
            case callID = "call_id"
            case encryptedContent = "encrypted_content"
        }

        public init(
            type: String?, id: String? = nil, status: String? = nil,
            role: String? = nil, content: JSONValue? = nil,
            callID: String? = nil, name: String? = nil,
            arguments: String? = nil, output: JSONValue? = nil,
            summary: JSONValue? = nil, encryptedContent: String? = nil,
            namespace: String? = nil
        ) {
            self.type = type
            self.id = id
            self.status = status
            self.role = role
            self.content = content
            self.callID = callID
            self.name = name
            self.arguments = arguments
            self.output = output
            self.summary = summary
            self.encryptedContent = encryptedContent
            self.namespace = namespace
        }

        /// Resolved item kind: the explicit `type`, or inferred from the
        /// present fields when the client omits it.
        public var resolvedType: String? {
            if let type { return type }
            if role != nil && content != nil { return "message" }
            if callID != nil && output != nil { return "function_call_output" }
            if name != nil && arguments != nil { return "function_call" }
            return nil
        }
    }

    public struct Tool: Codable, Sendable, Equatable {
        public let type: String
        public let name: String?
        public let description: String?
        public let parameters: JSONValue?
        public let strict: Bool?
        /// A `namespace` tool groups function tools under a name; the model
        /// calls the functions, and the call carries the namespace back.
        public let tools: [Tool]?

        public init(
            type: String, name: String?, description: String?,
            parameters: JSONValue?, strict: Bool?, tools: [Tool]? = nil
        ) {
            self.type = type
            self.name = name
            self.description = description
            self.parameters = parameters
            self.strict = strict
            self.tools = tools
        }
    }

    /// The Responses API's nested reasoning options; only `effort` is read.
    public struct Reasoning: Codable, Equatable, Sendable {
        public let effort: String?
        public let summary: String?
    }

    public struct TextConfig: Decodable, Sendable {
        public let format: JSONValue?
        public let verbosity: String?
    }

    public struct StreamOptions: Decodable, Sendable {
        public let includeObfuscation: Bool?

        enum CodingKeys: String, CodingKey {
            case includeObfuscation = "include_obfuscation"
        }
    }

    /// `input` is either a plain string (one user message) or a list of items.
    public enum Input: Decodable, Sendable {
        case text(String)
        case items([Item])

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let text = try? container.decode(String.self) {
                self = .text(text)
            } else {
                self = .items(try container.decode([Item].self))
            }
        }

        public var items: [Item] {
            switch self {
            case .text(let text):
                [Item(type: "message", role: "user", content: .string(text))]
            case .items(let items):
                items
            }
        }
    }

    public let model: String
    public let instructions: String?
    public let input: Input?
    public let tools: [Tool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let maxOutputTokens: Int?
    public let maxToolCalls: Int?
    public let temperature: Float?
    public let topP: Float?
    public let topK: Int?
    public let topLogprobs: Int?
    public let presencePenalty: Float?
    public let stream: Bool?
    public let streamOptions: StreamOptions?
    public let store: Bool?
    public let background: Bool?
    public let reasoning: Reasoning?
    public let previousResponseID: String?
    public let metadata: JSONValue?
    public let user: String?
    public let safetyIdentifier: String?
    public let promptCacheKey: String?
    public let serviceTier: String?
    public let truncation: String?
    public let text: TextConfig?
    public let include: [String]?
    public let prompt: JSONValue?
    public let conversation: JSONValue?

    enum CodingKeys: String, CodingKey {
        case model, instructions, input, tools, stream, store, temperature, reasoning
        case metadata, user, truncation, text, include, prompt, conversation, background
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case maxOutputTokens = "max_output_tokens"
        case maxToolCalls = "max_tool_calls"
        case topP = "top_p"
        case topK = "top_k"
        case topLogprobs = "top_logprobs"
        case presencePenalty = "presence_penalty"
        case streamOptions = "stream_options"
        case previousResponseID = "previous_response_id"
        case safetyIdentifier = "safety_identifier"
        case promptCacheKey = "prompt_cache_key"
        case serviceTier = "service_tier"
    }

    /// The input as items, however the client wrote it.
    public var inputItems: [Item] { input?.items ?? [] }

    /// Whether the finished response should be kept for `previous_response_id`
    /// and `GET /v1/responses/{id}`. The API's default is true.
    public var stores: Bool { store ?? true }
}

/// `POST /v1/responses/compact`.
///
/// Deliberately not `ResponsesAPIRequest`: compaction takes a conversation and
/// returns another window, not a turn. The fields that shape a turn — tools,
/// `store`, `stream`, `previous_response_id`, `max_output_tokens` — have no
/// meaning here, so they are not modelled and a client that sends them is
/// ignored rather than misread.
public struct CompactionRequest: Decodable, Sendable {
    /// Required by the spec; optional in the shape so a missing one is refused
    /// with a named parameter rather than as malformed JSON.
    public let model: String?
    public let instructions: String?
    public let input: ResponsesAPIRequest.Input?
    public let promptCacheKey: String?
    /// This server's own ceiling for the compacted note, in tokens. Absent
    /// leaves `ServerCompaction.targetTokens` to choose one from the context.
    /// Present but not positive is refused by name — the default is for the
    /// caller who asked for none, not for the one who set a number.
    public let maxCompactionTokens: Int?

    enum CodingKeys: String, CodingKey {
        case model, instructions, input
        case promptCacheKey = "prompt_cache_key"
        case maxCompactionTokens = "max_compaction_tokens"
    }

    public var inputItems: [ResponsesAPIRequest.Item] { input?.items ?? [] }
}
