import Foundation
import TinyTitan
import TinyTitanCore

/// The OpenAI-compatible wire types: the error envelope, message content and
/// its parts, tool and function definitions, stop sequences, stream options and
/// the template kwargs block.
///
/// Split out of `OpenAIModels.swift` (2026-09-28) under the 500-line-per-file
/// rule (Task 8 of the cleanup runbook) as pure code motion: the declarations
/// are unchanged and remain top-level types of this module.
package struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    package struct Detail: Codable, Equatable, Sendable {
        package let message: String
        package let type: String
        package let param: String?
        package let code: String
    }

    package let error: Detail

    package init(
        message: String, param: String? = nil, code: String, type: String = "invalid_request_error"
    ) {
        error = Detail(
            message: message,
            type: type,
            param: param,
            code: code)
    }
}

package struct OpenAITextPart: Codable, Equatable, Sendable {
    package let type: String
    package let text: String?
}

package enum OpenAIMessageContent: Codable, Equatable, Sendable {
    case text(String)
    case parts([OpenAITextPart])

    package init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAITextPart].self))
        }
    }

    package func encode(to encoder: Encoder) throws {
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

package struct OpenAIFunctionCall: Codable, Equatable, Sendable {
    package let name: String
    package let arguments: String

    // Declared rather than left to the memberwise initializer: that one is
    // internal, so the server target could not build a tool call after this
    // type moved into the kit (2026-10-02, phase A1 of
    // `docs/plan-embedded-library.md`).
    package init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

package struct OpenAIToolCall: Codable, Equatable, Sendable {
    package let id: String
    package let type: String
    package let function: OpenAIFunctionCall

    package init(id: String, type: String, function: OpenAIFunctionCall) {
        self.id = id
        self.type = type
        self.function = function
    }
}

package struct OpenAIChatMessage: Codable, Equatable, Sendable {
    package let role: String
    package let content: OpenAIMessageContent?
    package let toolCalls: [OpenAIToolCall]?
    package let toolCallID: String?
    package let name: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }

    package init(
        role: String,
        content: OpenAIMessageContent?,
        toolCalls: [OpenAIToolCall]?,
        toolCallID: String?,
        name: String?
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.name = name
    }
}

package struct OpenAIFunctionDefinition: Codable, Equatable, Sendable {
    package let name: String
    package let description: String?
    package let parameters: JSONValue

    package init(name: String, description: String?, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

package struct OpenAITool: Codable, Equatable, Sendable {
    package let type: String
    package let function: OpenAIFunctionDefinition

    package init(type: String, function: OpenAIFunctionDefinition) {
        self.type = type
        self.function = function
    }
}

package enum OpenAIStop: Codable, Equatable, Sendable {
    case one(String)
    case many([String])

    package init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let one = try? container.decode(String.self) {
            self = .one(one)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    package func encode(to encoder: Encoder) throws {
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

package struct OpenAIStreamOptions: Codable, Equatable, Sendable {
    package let includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

/// The thinking controls as Qwen's chat templates take them.
///
/// llama.cpp, vLLM and TabbyAPI clients send these inside a
/// `chat_template_kwargs` object rather than as the top-level
/// `reasoning_effort`, and `enable_thinking: false` there is the only way those
/// clients turn thinking off for one request. A server that read only the
/// top-level field honoured the *level* beside this object and dropped the
/// switch, which is the case that matters: a client that forces thinking off
/// for its summarization calls (so the model's own thinking cannot eat the
/// output cap and truncate the summary) had the fix silently lost.
package struct OpenAIChatTemplateKwargs: Codable, Equatable, Sendable {
    package let enableThinking: Bool?
    package let reasoningEffort: String?

    enum CodingKeys: String, CodingKey {
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
    }

    package init(enableThinking: Bool? = nil, reasoningEffort: String? = nil) {
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
    }
}
