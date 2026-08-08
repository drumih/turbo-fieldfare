import Foundation
import TurboFieldfare

/// The K3-shaped payload of a `ValidatedChatRequest`: renderer-ready messages
/// plus renderer options (tools, thinking, thinking effort, tool choice).
public struct K3ValidatedChat: Sendable, Equatable {
    public let messages: [K3ChatMessage]
    public let options: K3ChatOptions

    public init(messages: [K3ChatMessage], options: K3ChatOptions) {
        self.messages = messages
        self.options = options
    }
}

/// OpenAI → K3 request mapping, the K3 counterpart of
/// `OpenAIRequestValidator`. Numeric sampling checks mirror the house
/// validator; the differences are the K3 feature surface:
///
/// - `reasoning_effort` maps to the renderer's thinking effort
///   (`low`/`high`/`max` — the values the reference `encoding_k3.py`
///   accepts; note there is no `medium`).
/// - `tool_choice` accepts `required` and `none` in addition to `auto`;
///   named tool selection stays unsupported.
/// - `stop` substrings are rejected: `K3Engine` does not evaluate stop
///   strings, so accepting them would silently mis-serve the request.
/// - `developer` maps to `system` (K3 XTML has no developer role).
/// - Tool messages may omit `tool_call_id` (the renderer falls back to call
///   order); tool declarations are validated minimally (function type,
///   non-empty name, object parameters schema).
public enum K3RequestValidator {
    public static func validate(_ request: OpenAIChatRequest,
                                modelID: String) throws -> ValidatedChatRequest {
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
        guard request.stop == nil else {
            throw invalid("stop substrings are not supported for this model",
                          "stop", "unsupported_value")
        }

        // The published K3 generation profile is deterministic unless the
        // client explicitly opts into sampling.
        let temperature = request.temperature ?? 0
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

        let thinkingEffort: K3ThinkingEffort?
        if let effort = request.reasoningEffort {
            guard let parsed = K3ThinkingEffort(rawValue: effort) else {
                throw invalid("reasoning_effort must be low, high, or max",
                              "reasoning_effort", "unsupported_value")
            }
            thinkingEffort = parsed
        } else {
            thinkingEffort = nil
        }

        let includeTools: Bool
        let toolChoice: K3ToolChoice?
        switch request.toolChoice {
        case nil, .some(.string("auto")):
            includeTools = true
            toolChoice = nil
        case .some(.string("none")):
            includeTools = false
            toolChoice = K3ToolChoice.none
        case .some(.string("required")):
            includeTools = true
            toolChoice = .required
        default:
            throw invalid("named tool choices are not supported",
                          "tool_choice", "unsupported_value")
        }

        let tools = try (includeTools ? request.tools ?? [] : []).map(validateTool)
        let messages = try validateMessages(request.messages)
        let options = K3ChatOptions(
            tools: tools,
            thinking: true,
            thinkingEffort: thinkingEffort,
            toolChoice: toolChoice)
        let config = GenerationConfig(maxNewTokens: maximum,
                                      temperature: temperature,
                                      topK: topK,
                                      topP: topP,
                                      repetitionPenalty: repetitionPenalty,
                                      seed: request.seed)
        return ValidatedChatRequest(messages: [],
                                    tools: [],
                                    stream: request.stream ?? false,
                                    includeUsage: request.streamOptions?.includeUsage ?? false,
                                    generationConfig: config,
                                    maximumCompletionTokens: maximum,
                                    k3: K3ValidatedChat(messages: messages, options: options))
    }

    private static func validateTool(_ tool: OpenAITool) throws -> K3ToolDeclaration {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard !name.isEmpty else {
            throw invalid("tool name must not be empty", "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid("tool parameters must be an object schema",
                          "tools", "invalid_tool_schema")
        }
        return K3ToolDeclaration(name: name,
                                 description: tool.function.description ?? "",
                                 parameters: tool.function.parameters)
    }

    private static func validateMessages(_ input: [OpenAIChatMessage]) throws -> [K3ChatMessage] {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        var knownCallIDs = Set<String>()
        var result: [K3ChatMessage] = []
        for message in input {
            let role: K3ChatRole
            switch message.role {
            case "system", "developer":
                role = .system
            case "user":
                role = .user
            case "assistant":
                role = .assistant
            case "tool":
                role = .tool
            default:
                throw invalid("unsupported message role \(message.role)",
                              "messages", "invalid_message")
            }

            let content: K3ChatContent?
            switch message.content {
            case nil:
                content = nil
            case .text(let text):
                content = .text(text)
            case .parts(let parts):
                guard parts.allSatisfy({ $0.type == "text" && $0.text != nil }) else {
                    throw invalid("only text content parts are supported",
                                  "messages", "unsupported_content")
                }
                content = .parts(parts.compactMap(\.text))
            }

            let calls: [K3ToolCall] = try (message.toolCalls ?? []).map { call in
                guard role == .assistant, call.type == "function",
                      !call.id.isEmpty, knownCallIDs.insert(call.id).inserted else {
                    throw invalid("invalid or duplicate historical tool call",
                                  "messages", "invalid_tool_call")
                }
                guard !call.function.name.isEmpty else {
                    throw invalid("tool call name must not be empty",
                                  "messages", "invalid_tool_call")
                }
                // Renderer normalizes the string at render time; validate the
                // object shape here so a bad request fails as HTTP 400.
                let parsed = try? JSONDecoder().decode(
                    JSONValue.self, from: Data(call.function.arguments.utf8))
                guard parsed?.objectValue != nil else {
                    throw invalid("historical tool arguments must be a JSON object",
                                  "messages", "invalid_tool_arguments")
                }
                return K3ToolCall(id: call.id,
                                  name: call.function.name,
                                  arguments: .jsonString(call.function.arguments))
            }

            if role == .tool, content == nil {
                throw invalid("tool result content is required",
                              "messages", "invalid_tool_result")
            }
            if role != .tool, content == nil, calls.isEmpty {
                throw invalid("message content is required",
                              "messages", "invalid_message")
            }
            result.append(K3ChatMessage(
                role: role,
                content: content,
                name: role == .tool ? nil : message.name,
                reasoningContent: role == .assistant ? message.reasoningContent : nil,
                toolCalls: calls,
                tool: role == .tool ? message.name : nil,
                toolCallID: role == .tool ? message.toolCallID : nil))
        }
        return result
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}
