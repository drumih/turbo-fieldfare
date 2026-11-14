import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

// The Anthropic response builders.
//
// Split out of `AnthropicModels.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion.
// MARK: - Response builders

public enum AnthropicBuilder {
    /// The API version this server implements. Sent back on every response.
    public static let version = "2023-06-01"

    public static func messageID() -> String {
        "msg_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
    }

    public static func requestID() -> String {
        "req_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
    }

    /// Anthropic's stop_reason for a completion: tool calls first (a turn
    /// that called a tool ends with tool_use whatever else it said), then the
    /// output cap, then a matched stop string, then a natural end.
    package static func stopReason(for completion: ServerCompletion) -> (
        reason: String, sequence: String?
    ) {
        if !completion.toolCalls.isEmpty { return ("tool_use", nil) }
        if completion.finishReason == "length" { return ("max_tokens", nil) }
        if let stop = completion.stopSequence { return ("stop_sequence", stop) }
        return ("end_turn", nil)
    }

    public static func textBlock(_ text: String) -> [String: Any] {
        ["type": "text", "text": text]
    }

    public static func toolUseBlock(_ call: ParsedToolCall) -> [String: Any] {
        [
            "type": "tool_use", "id": call.id, "name": call.name,
            "input": call.arguments.foundationObject(),
        ]
    }

    /// The signature on every thinking block this server returns: empty.
    ///
    /// Anthropic's signature is an opaque token its API uses to verify a
    /// thought it is handed back. Here there is nothing to verify -- replayed
    /// thinking is dropped on the way in (see `chatMessages`) -- so nothing
    /// is signed. The field is still sent, because the SDKs type it as a
    /// required string, accumulate `signature_delta` into it, and replay the
    /// block verbatim; an empty string satisfies all three, where a made-up
    /// token would only pretend to be verifiable.
    public static let thinkingSignature = ""

    public static func thinkingBlock(_ thinking: String) -> [String: Any] {
        ["type": "thinking", "thinking": thinking, "signature": thinkingSignature]
    }

    /// Content blocks of a completion: the thinking (when there is any),
    /// then the text (when there is any) and one tool_use block per call. An
    /// empty completion is one empty text block, never an empty content
    /// array, and a turn that only thought still carries that text block.
    package static func contentBlocks(_ completion: ServerCompletion) -> [[String: Any]] {
        var blocks: [[String: Any]] = []
        if !completion.reasoning.isEmpty {
            blocks.append(thinkingBlock(completion.reasoning))
        }
        if !completion.content.isEmpty || completion.toolCalls.isEmpty {
            blocks.append(textBlock(completion.content))
        }
        blocks += completion.toolCalls.map(toolUseBlock)
        return blocks
    }

    /// Anthropic counts cache reads apart from input_tokens: the two sum to
    /// what OpenAI reports as prompt_tokens.
    package static func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        let cached = usage.promptTokensDetails.cachedTokens
        return [
            "input_tokens": max(usage.promptTokens - cached, 0),
            "output_tokens": usage.completionTokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cached,
            "service_tier": "standard",
        ]
    }

    public static func messageObject(
        id: String,
        model: String,
        content: [[String: Any]],
        stopReason: String?,
        stopSequence: String?,
        usage: [String: Any]
    ) -> [String: Any] {
        [
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stopReason.map { $0 as Any } ?? NSNull(),
            "stop_sequence": stopSequence.map { $0 as Any } ?? NSNull(),
            "usage": usage,
        ]
    }

    /// `GET /v1/models` in the Anthropic shape.
    public static func modelList(ids: [String]) -> [String: Any] {
        modelList(models: ids.map { (id: $0, displayName: $0) })
    }

    /// The same list where each model has a human name of its own, as
    /// catalog models do.
    public static func modelList(models: [(id: String, displayName: String)]) -> [String: Any] {
        [
            "data": models.map { modelObject(id: $0.id, displayName: $0.displayName) },
            "has_more": false,
            "first_id": models.first.map { $0.id as Any } ?? NSNull(),
            "last_id": models.last.map { $0.id as Any } ?? NSNull(),
        ]
    }

    public static func modelObject(id: String, displayName: String? = nil) -> [String: Any] {
        [
            "type": "model", "id": id, "display_name": displayName ?? id,
            "created_at": ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: 0)),
        ]
    }
}
