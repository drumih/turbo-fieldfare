import Foundation
import Tokenizers

/// A model's chat-formatting and structured-output dialect (TPP/1.1, TT-132).
///
/// Core ships three reference implementations (`ChatMLDialect`, `K2Dialect`,
/// `InklingDialect`). A plugin declares its dialect via
/// `EnginePlugin.chatDialect(for:)` or supplies its own conformer without
/// touching host switch tables.
public protocol ChatDialect: Sendable {
    /// Stable short identifier (e.g. `"chatml"`, `"k2"`, `"inkling"`).
    var identifier: String { get }
    /// Whether the prompt cache may splice a ChatML multi-turn continuation
    /// bridge onto a cached assistant turn.
    var supportsChatMLContinuationBridge: Bool { get }
    /// Whether thought-channel close markers are detected by literal text even
    /// when the tokenizer emits no distinct end-of-thought token.
    var closesThoughtOnMarkerText: Bool { get }
    /// Whether assistant streams use the structured Inkling message parser
    /// instead of the ChatML/K2 marker-pair state machine.
    var usesStructuredInklingParser: Bool { get }

    /// Tokens that may end a constrained structured (JSON-schema) answer.
    func documentEndTokenIDs(for tokenizer: GFTokenizer) -> Set<Int32>
    /// Message boundary markers for prompt-cache interior checkpoints.
    func promptMessageMarkers(for tokenizer: GFTokenizer) -> PromptMessageMarkers
    /// Encodes a chat prompt (with optional tools and structured-answer framing).
    func encodePrompt(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        usesToolTemplate: Bool,
        structuredAnswer: Bool,
        tokenizer: GFTokenizer
    ) throws -> [Int32]
    /// Parses a single buffered tool-call payload between the dialect's
    /// tool-call start and end markers.
    func parseToolCall(
        _ text: String,
        allowedTools: Set<String>,
        makeID: () -> String
    ) throws -> ParsedToolCall
}

/// Reference ChatML dialect (`<|im_start|>`, `<think>`, `<tool_call>`).
public struct ChatMLDialect: ChatDialect, Equatable, Sendable {
    public init() {}

    public var identifier: String { "chatml" }
    public var supportsChatMLContinuationBridge: Bool { true }
    public var closesThoughtOnMarkerText: Bool { false }
    public var usesStructuredInklingParser: Bool { false }

    public func documentEndTokenIDs(for tokenizer: GFTokenizer) -> Set<Int32> {
        tokenizer.stopTokenIDs
    }

    public func promptMessageMarkers(for tokenizer: GFTokenizer) -> PromptMessageMarkers {
        .chatML(imStartID: tokenizer.imStartID)
    }

    public func encodePrompt(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        usesToolTemplate: Bool,
        structuredAnswer: Bool,
        tokenizer: GFTokenizer
    ) throws -> [Int32] {
        if usesToolTemplate {
            return try tokenizer.encodeToolChat(messages: messages, tools: tools)
        }
        let rendered = try tokenizer.applyChatTemplate(messages)
        return tokenizer.encode(rendered, addBOS: false)
    }

    public func parseToolCall(
        _ text: String,
        allowedTools: Set<String>,
        makeID: () -> String
    ) throws -> ParsedToolCall {
        try QwenToolCallParser().parse(text, allowedTools: allowedTools, id: makeID())
    }
}

/// Reference K2 dialect (`<|ifm|im_start|>`, `<ifm|think>`, `<ifm|tool_call>`).
public struct K2Dialect: ChatDialect, Equatable, Sendable {
    public init() {}

    public var identifier: String { "k2" }
    public var supportsChatMLContinuationBridge: Bool { true }
    public var closesThoughtOnMarkerText: Bool { true }
    public var usesStructuredInklingParser: Bool { false }

    public func documentEndTokenIDs(for tokenizer: GFTokenizer) -> Set<Int32> {
        tokenizer.stopTokenIDs
    }

    public func promptMessageMarkers(for tokenizer: GFTokenizer) -> PromptMessageMarkers {
        .chatML(imStartID: tokenizer.imStartID)
    }

    public func encodePrompt(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        usesToolTemplate: Bool,
        structuredAnswer: Bool,
        tokenizer: GFTokenizer
    ) throws -> [Int32] {
        if usesToolTemplate {
            return try tokenizer.encodeToolChat(messages: messages, tools: tools)
        }
        let rendered = try tokenizer.applyChatTemplate(messages)
        return tokenizer.encode(rendered, addBOS: false)
    }

    public func parseToolCall(
        _ text: String,
        allowedTools: Set<String>,
        makeID: () -> String
    ) throws -> ParsedToolCall {
        let wrapped =
            "<ifm|tool_calls><ifm|tool_call>\(text)</ifm|tool_call></ifm|tool_calls>"
        let calls = try K2ToolCallParser.parse(
            wrapped, allowedTools: allowedTools, format: .xml, makeID: makeID)
        guard calls.count == 1 else {
            throw ToolCallParserError.malformed
        }
        return calls[0]
    }
}

/// Reference Inkling 2 structured-message dialect (`<|message_model|>`,
/// `<|content_text|>`, `<|content_thinking|>`, `<|content_invoke_tool_json|>`).
public struct InklingDialect: ChatDialect, Equatable, Sendable {
    public init() {}

    public var identifier: String { "inkling" }
    public var supportsChatMLContinuationBridge: Bool { false }
    public var closesThoughtOnMarkerText: Bool { false }
    public var usesStructuredInklingParser: Bool { true }

    public func documentEndTokenIDs(for tokenizer: GFTokenizer) -> Set<Int32> {
        tokenizer.stopTokenIDs.union([tokenizer.toolCallEndID])
    }

    public func promptMessageMarkers(for tokenizer: GFTokenizer) -> PromptMessageMarkers {
        let ids = [
            "<|message_user|>", "<|message_system|>", "<|message_model|>", "<|message_tool|>",
        ]
        .compactMap { tokenizer.tokenID(for: $0) }
        let header = ["<|message_model|>", "<|content_text|>"]
            .compactMap { tokenizer.tokenID(for: $0) }
        return PromptMessageMarkers(
            startIDs: Set(ids), headerSuffix: header.count == 2 ? header : [])
    }

    public func encodePrompt(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        usesToolTemplate: Bool,
        structuredAnswer: Bool,
        tokenizer: GFTokenizer
    ) throws -> [Int32] {
        tokenizer.encodeInklingPrompt(
            messages: messages, tools: usesToolTemplate ? tools : [],
            structuredAnswer: structuredAnswer)
    }

    public func parseToolCall(
        _ text: String,
        allowedTools: Set<String>,
        makeID: () -> String
    ) throws -> ParsedToolCall {
        // Inkling streams go through `InklingAssistantParser` before the
        // marker-pair path is consulted.
        throw ToolCallParserError.malformed
    }
}

extension GFTokenizer {
    /// The `ChatDialect` value for this tokenizer.
    public var chatDialect: any ChatDialect {
        switch dialect {
        case .chatML: ChatMLDialect()
        case .k2: K2Dialect()
        case .inkling: InklingDialect()
        }
    }

    /// Looks up a special-token ID by its literal spelling.
    package func tokenID(for token: String) -> Int32? {
        tokenizer.convertTokenToId(token).map(Int32.init)
    }
}
