import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

/// Building the Responses API objects, and the JSONValue bridging they need.
///
/// Split out of `ResponsesAPIModels.swift` (2026-09-28) under the 500-line-per-
/// file rule (Task 8 of the cleanup runbook) as pure code motion.
public enum ResponsesAPIBuilder {
    public static func responseID() -> String {
        "resp_" + hex()
    }

    static func hex() -> String {
        UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
    }

    /// The output item ids of a response, derived from the response id so
    /// that the streamed `output_item.added` and the final object agree.
    package static func itemIDs(
        responseID: String,
        completion: ServerCompletion
    ) -> (message: String, calls: [String]) {
        let suffix = String(responseID.dropFirst("resp_".count))
        return (
            "msg_" + suffix,
            completion.toolCalls.indices.map { "fc_" + suffix + String($0) }
        )
    }

    public static func messageItemID(responseID: String) -> String {
        "msg_" + String(responseID.dropFirst("resp_".count))
    }

    public static func functionCallItemID(responseID: String, index: Int) -> String {
        "fc_" + String(responseID.dropFirst("resp_".count)) + String(index)
    }

    /// Numbered like function calls: a model that thinks again after it has
    /// started answering produces a second reasoning item in the stream.
    public static func reasoningItemID(responseID: String, index: Int) -> String {
        "rs_" + String(responseID.dropFirst("resp_".count)) + String(index)
    }

    /// One streaming event. `sequence_number` is the client's ordering key;
    /// the handler owns the counter.
    public static func event(
        _ type: String, sequence: Int,
        _ fields: [String: Any]
    ) -> [String: Any] {
        var object = fields
        object["type"] = type
        object["sequence_number"] = sequence
        return object
    }

    package static func responseObject(
        id: String,
        created: Int,
        model: String,
        status: String,
        output: [[String: Any]],
        usage: OpenAIUsage?,
        echo: ResponsesAPIEcho,
        incompleteReason: String? = nil,
        error: (code: String, message: String)? = nil,
        completedAt: Int? = nil
    ) -> [String: Any] {
        var object: [String: Any] = [
            "id": id,
            "object": "response",
            "created_at": created,
            "completed_at": completedAt.map { $0 as Any } ?? NSNull(),
            "status": status,
            "background": false,
            "error": error.map { ["code": $0.code, "message": $0.message] as Any } ?? NSNull(),
            "incomplete_details": incompleteReason.map { ["reason": $0] as Any } ?? NSNull(),
            "instructions": echo.instructions.map { $0 as Any } ?? NSNull(),
            "max_output_tokens": echo.maxOutputTokens.map { $0 as Any } ?? NSNull(),
            "max_tool_calls": echo.maxToolCalls.map { $0 as Any } ?? NSNull(),
            "model": model,
            "output": output,
            "parallel_tool_calls": echo.parallelToolCalls,
            "previous_response_id": echo.previousResponseID.map { $0 as Any } ?? NSNull(),
            "prompt_cache_key": echo.promptCacheKey.map { $0 as Any } ?? NSNull(),
            "reasoning": [
                "effort": echo.reasoningEffort.map { $0 as Any } ?? NSNull(),
                "summary": echo.reasoningSummary.map { $0 as Any } ?? NSNull(),
            ],
            "safety_identifier": echo.safetyIdentifier.map { $0 as Any } ?? NSNull(),
            "service_tier": echo.serviceTier,
            "store": echo.store,
            "temperature": echo.temperature,
            "text": ["format": ["type": "text"], "verbosity": echo.textVerbosity],
            "tool_choice": echo.toolChoice.foundationObject(),
            "tools": echo.tools.map(toolObject),
            "top_logprobs": 0,
            "top_p": echo.topP,
            "presence_penalty": echo.presencePenalty,
            "frequency_penalty": echo.frequencyPenalty,
            "truncation": echo.truncation,
            "usage": NSNull(),
            "user": echo.user.map { $0 as Any } ?? NSNull(),
            "metadata": echo.metadata.foundationObject(),
        ]
        if let usage {
            object["usage"] = usageObject(usage)
        }
        return object
    }

    package static func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        [
            "input_tokens": usage.promptTokens,
            "input_tokens_details": [
                "cached_tokens": usage.promptTokensDetails.cachedTokens,
                "cache_write_tokens": 0,
            ],
            "output_tokens": usage.completionTokens,
            "output_tokens_details": [
                "reasoning_tokens": usage.completionTokensDetails.reasoningTokens
            ],
            "total_tokens": usage.totalTokens,
        ]
    }

    /// `POST /v1/responses/compact` — the compacted input window.
    ///
    /// The spec's shape, and its semantics: this is a value the client sends back
    /// as the base `input` of a new response, not a response id to continue from.
    /// `output` holds the caller's instructions verbatim (the spec asks
    /// compaction to preserve system prompts) followed by the `compaction` item
    /// that carries the note.
    package static func compactResource(
        id: String,
        created: Int,
        output: [[String: Any]],
        usage: OpenAIUsage
    ) -> [String: Any] {
        [
            "id": id,
            "object": "response.compaction",
            "created_at": created,
            "output": output,
            "usage": usageObject(usage),
        ]
    }

    /// The item that carries a compaction note. `encrypted_content` is required
    /// by the schema; what this server puts in it is documented in
    /// `docs/server-api.md`.
    public static func compactionItem(
        id: String,
        encryptedContent: String,
        createdBy: String
    ) -> [String: Any] {
        [
            "id": id,
            "type": "compaction",
            "encrypted_content": encryptedContent,
            "created_by": createdBy,
        ]
    }

    static func toolObject(_ tool: ResponsesAPIRequest.Tool) -> [String: Any] {
        var object: [String: Any] = ["type": tool.type]
        if let name = tool.name { object["name"] = name }
        if tool.type == "function" || tool.type == "custom" {
            object["description"] = tool.description.map { $0 as Any } ?? NSNull()
            object["parameters"] = tool.parameters?.foundationObject() ?? NSNull()
            object["strict"] = tool.strict ?? false
        }
        if let nested = tool.tools {
            object["tools"] = nested.map(toolObject)
        }
        return object
    }

    public static func outputTextPart(_ text: String) -> [String: Any] {
        ["type": "output_text", "text": text, "annotations": [], "logprobs": []]
    }

    public static func messageItem(
        id: String,
        role: String,
        text: String,
        status: String
    ) -> [String: Any] {
        [
            "id": id, "type": "message", "role": role, "status": status,
            "content": [outputTextPart(text)],
        ]
    }

    public static func functionCallItem(
        id: String,
        name: String,
        arguments: String,
        callID: String,
        status: String,
        namespace: String? = nil
    ) -> [String: Any] {
        var item: [String: Any] = [
            "id": id, "type": "function_call", "status": status,
            "name": name, "arguments": arguments, "call_id": callID,
        ]
        if let namespace { item["namespace"] = namespace }
        return item
    }

    public static func summaryTextPart(_ text: String) -> [String: Any] {
        ["type": "summary_text", "text": text]
    }

    /// The model's thoughts as a reasoning item. They are the whole thought,
    /// not a summary of it, but `summary` is the part of a reasoning item
    /// that Responses clients read and show; the item's other fields carry
    /// hosted-model state (encrypted content) this server has none of.
    public static func reasoningItem(id: String, text: String) -> [String: Any] {
        ["id": id, "type": "reasoning", "summary": [summaryTextPart(text)]]
    }

    /// Output items for a completed generation: the reasoning when there is
    /// any, then the message, then calls.
    package static func outputItems(
        completion: ServerCompletion,
        responseID: String,
        namespaces: [String: String] = [:]
    ) -> [[String: Any]] {
        let ids = itemIDs(responseID: responseID, completion: completion)
        var output: [[String: Any]] = []
        if !completion.reasoning.isEmpty {
            output.append(
                reasoningItem(
                    id: reasoningItemID(responseID: responseID, index: 0),
                    text: completion.reasoning))
        }
        if !completion.content.isEmpty || completion.toolCalls.isEmpty {
            output.append(
                messageItem(
                    id: ids.message, role: "assistant",
                    text: completion.content, status: "completed"))
        }
        for (index, call) in completion.toolCalls.enumerated() {
            output.append(
                functionCallItem(
                    id: ids.calls[index], name: call.name,
                    arguments: call.argumentsJSON, callID: call.id, status: "completed",
                    namespace: namespaces[call.name]))
        }
        return output
    }

    /// The terminal status of a generation: "incomplete" when the output
    /// cap ended it, which the API reports with `incomplete_details`.
    package static func terminalStatus(for completion: ServerCompletion) -> (
        status: String, reason: String?
    ) {
        completion.finishReason == "length"
            ? ("incomplete", "max_output_tokens")
            : ("completed", nil)
    }

    /// `GET /v1/responses/{id}/input_items`.
    public static func inputItemsList(_ items: [ResponsesAPIRequest.Item]) throws -> Data {
        let encoder = JSONEncoder()
        let data = try items.map { try JSONSerialization.jsonObject(with: encoder.encode($0)) }
        let ids = items.compactMap(\.id)
        let object: [String: Any] = [
            "object": "list",
            "data": data,
            "first_id": ids.first.map { $0 as Any } ?? NSNull(),
            "last_id": ids.last.map { $0 as Any } ?? NSNull(),
            "has_more": false,
        ]
        return try JSONSerialization.data(withJSONObject: object)
    }
}

// MARK: - JSONValue bridging

extension JSONValue {
    /// The Foundation object `JSONSerialization` writes for this value.
    public func foundationObject() -> Any {
        switch self {
        case .object(let value): value.mapValues { $0.foundationObject() }
        case .array(let value): value.map { $0.foundationObject() }
        case .string(let value): value
        case .integer(let value): value
        case .unsignedInteger(let value): value
        case .decimal(let value): value
        case .number(let value): value
        case .bool(let value): value
        case .null: NSNull()
        }
    }
}
