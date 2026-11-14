import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

// MARK: - Error envelope

/// The Anthropic Messages API error body: `{"type":"error","error":{...}}`.
public struct AnthropicErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let type: String
        public let message: String
    }

    public let type: String
    public let error: Detail
    public let requestID: String?

    enum CodingKeys: String, CodingKey {
        case type, error
        case requestID = "request_id"
    }

    public init(type: String, message: String, requestID: String? = nil) {
        self.type = "error"
        self.error = Detail(type: type, message: message)
        self.requestID = requestID
    }

    /// The Anthropic rendering of a server request error. The validator
    /// speaks OpenAI (message, param, code); the Anthropic API names the
    /// field inside the message, so the two are folded together here.
    package static func from(_ error: ServerRequestError, requestID: String? = nil)
        -> AnthropicErrorEnvelope
    {
        switch error {
        case .invalid(let message, let param, _):
            return AnthropicErrorEnvelope(
                type: "invalid_request_error",
                message: param.map { "\($0): \(message)" } ?? message,
                requestID: requestID)
        case .unknownModel:
            return AnthropicErrorEnvelope(
                type: "not_found_error",
                message: "model: requested model is not available on this server",
                requestID: requestID)
        case .queueFull:
            return AnthropicErrorEnvelope(
                type: "overloaded_error", message: "Overloaded", requestID: requestID)
        case .unsupportedOperation(let operation):
            return AnthropicErrorEnvelope(
                type: "api_error",
                message: "\(operation) is not supported by this backend",
                requestID: requestID)
        case .notFound(let message, _):
            return AnthropicErrorEnvelope(
                type: "not_found_error", message: message, requestID: requestID)
        }
    }

    /// HTTP status for an Anthropic error type.
    public var httpStatus: Int {
        switch error.type {
        case "invalid_request_error": 400
        case "authentication_error": 401
        case "permission_error": 403
        case "not_found_error": 404
        case "request_too_large": 413
        case "rate_limit_error": 429
        case "overloaded_error": 529
        default: 500
        }
    }
}

// MARK: - Request decoding

/// `POST /v1/messages`. Content is kept as `JSONValue` because the block
/// grammar is wide and mostly refused; the mapper reads what it honours and
/// names what it cannot.
public struct AnthropicMessagesRequest: Decodable, Sendable {
    public struct Message: Decodable, Sendable {
        public let role: String
        public let content: JSONValue
    }

    public let model: String
    public let messages: [Message]
    public let maxTokens: Int?
    public let system: JSONValue?
    public let metadata: JSONValue?
    public let stopSequences: [String]?
    public let stream: Bool?
    public let temperature: Float?
    public let topK: Int?
    public let topP: Float?
    public let tools: [JSONValue]?
    public let toolChoice: JSONValue?
    public let thinking: JSONValue?
    public let serviceTier: String?
    public let outputConfig: JSONValue?
    public let outputFormat: JSONValue?
    public let container: JSONValue?
    public let mcpServers: JSONValue?
    public let contextManagement: JSONValue?

    enum CodingKeys: String, CodingKey {
        case model, messages, system, metadata, stream, temperature, tools, thinking, container
        case maxTokens = "max_tokens"
        case stopSequences = "stop_sequences"
        case topK = "top_k"
        case topP = "top_p"
        case toolChoice = "tool_choice"
        case serviceTier = "service_tier"
        case outputConfig = "output_config"
        case outputFormat = "output_format"
        case mcpServers = "mcp_servers"
        case contextManagement = "context_management"
    }
}

/// `POST /v1/messages/count_tokens` accepts the same body without
/// `max_tokens`, `stream` and the sampling controls.
public struct AnthropicCountTokensRequest: Decodable, Sendable {
    public let model: String
    public let messages: [AnthropicMessagesRequest.Message]
    public let system: JSONValue?
    public let tools: [JSONValue]?
    public let toolChoice: JSONValue?
    public let thinking: JSONValue?

    enum CodingKeys: String, CodingKey {
        case model, messages, system, tools, thinking
        case toolChoice = "tool_choice"
    }
}
