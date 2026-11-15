import Foundation
import Tokenizers
import TinyTitanCore

/// The control-token dialect a tokenizer speaks.
public enum TokenizerDialect: String, Sendable, Equatable {
    /// Qwen/Ornith ChatML: `<|im_start|>`, `<think>`, `<tool_call>`.
    case chatML
    /// K2-Horizon: `<|ifm|im_start|>`, `<ifm|think>`, `<ifm|tool_call>`.
    case k2
}

/// Tokenizer wrapper for the compatible Qwen3.5-MoE ChatML model family.
///
/// Loads tokenizer sidecars in a completed `.ssdai/tokenizer/` directory.
/// Exposes typed accessors for the IDs the generator actually needs (BOS / EOS /
/// pad / end-of-turn) and adapts encode/decode to Int32 to match the buffer
/// types kernels consume.
///
/// TinyTitan owns the minimal chat framing because the upstream
/// `tokenizer_config.json` has no `chat_template`. Literal control-token text in
/// user content is accepted as a trusted-input research-runtime limitation.
/// unchecked-invariant: immutable after `load`. The stored token ids and the
/// underlying swift-transformers tokenizer are never mutated afterwards, so
/// concurrent encode/decode calls only read.
public struct GFTokenizer: @unchecked Sendable {
    /// `<|im_start|>`: the token every ChatML message and the generation
    /// prompt open with. Interior-checkpoint boundaries are located by it.
    public let imStartID: Int32
    /// Nominal BOS. This is `<|endoftext|>` (the config's unused
    /// `bos_token_id`); it is never prepended — see `encode(_:addBOS:)`.
    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let endOfTurnID: Int32
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseID: Int32
    public let toolResponseEndID: Int32
    /// Alias of the `<think>` / `</think>` markers.
    public let channelStartID: Int32
    public let channelEndID: Int32
    /// ChatML `<think>` / `</think>` special-token IDs.
    public let thinkStartID: Int32?
    public let thinkEndID: Int32?
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int
    public let dialect: TokenizerDialect
    /// The literal spellings of the markers, for the streaming decoder's
    /// boundary check. The ids alone cannot answer "what text ended this
    /// delta" when the two dialects spell the same slot differently.
    let thinkStartMark: String
    let thinkEndMark: String
    let toolCallStartMark: String
    let toolCallEndMark: String
    /// The outer `<ifm|tool_calls>` wrapper: present for K2, nil elsewhere.
    public let toolCallsStartID: Int32?
    public let toolCallsEndID: Int32?
    let toolCallsStartMark: String
    let toolCallsEndMark: String

    /// Tokens that stand for a control marker rather than for text: the ChatML
    /// turn and tool barriers, the channel markers, and every stop token.
    ///
    /// Anything that reads a token as *bytes* -- the JSON grammar's token table
    /// is the one caller -- must skip these. Their `convertIdToToken` text is
    /// the marker's spelling, not a byte string the model emitted, and a
    /// grammar that treated `<|im_end|>` as those nine characters would allow a
    /// marker inside a JSON string, where the streaming decoder would then
    /// swallow it as a barrier.
    public var nonByteTokenIDs: Set<Int32> {
        var ids: Set<Int32> = [
            bosID, eosID, padID, endOfTurnID,
            toolCallStartID, toolCallEndID, toolResponseID, toolResponseEndID,
            channelStartID, channelEndID,
        ]
        if let thinkStartID { ids.insert(thinkStartID) }
        if let thinkEndID { ids.insert(thinkEndID) }
        if let toolCallsStartID { ids.insert(toolCallsStartID) }
        if let toolCallsEndID { ids.insert(toolCallsEndID) }
        ids.formUnion(stopTokenIDs)
        return ids
    }
    public let thinkingMode: ModelThinkingMode
    /// Requested effort override for effort-aware templates; nil means the
    /// template's own default. Cleared when thinking is off because every
    /// supported template ignores effort without thinking.
    public let reasoningEffort: ModelReasoningEffort?
    /// The instruction the bundled template injects at the head of the system
    /// block for the active thinking/effort context (Qwen3.8-style templates
    /// inject an effort sentence; binary templates inject nothing). Derived
    /// from the template so it owns the wording; the manual ChatML renderer
    /// mirrors it.
    let effortSystemInstruction: String?

    /// Generation-prompt suffix appended after the last message: derived from
    /// the tokenizer's bundled `chat_template.jinja`
    /// (`add_generation_prompt` with thinking disabled) when available,
    /// falling back to the pinned constant otherwise (R6).
    let generationSuffix: String

    @usableFromInline
    let tokenizer: any Tokenizer
    let byteLevelDecoderConfiguration: GFByteLevelDecoderConfiguration

    public init(
        tokenizer: any Tokenizer,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil
    ) throws {
        try self.init(
            tokenizer: tokenizer,
            byteLevelDecoderConfiguration: .knownChatMLTokens(tokenizer: tokenizer),
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort)
    }

    init(
        tokenizer: any Tokenizer,
        byteLevelDecoderConfiguration: GFByteLevelDecoderConfiguration,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil
    ) throws {
        self.tokenizer = tokenizer
        self.byteLevelDecoderConfiguration = byteLevelDecoderConfiguration

        let resolved = try Self.resolveSpecialTokens(
            tokenizer, reasoningEffort: reasoningEffort)
        try Self.validateStreamingDecoder(
            byteLevelDecoderConfiguration,
            tokenizer: tokenizer,
            resolved: resolved)
        self.imStartID = resolved.imStartID
        self.bosID = resolved.bosID
        self.eosID = resolved.eosID
        self.padID = resolved.padID
        self.endOfTurnID = resolved.endOfTurnID
        self.toolCallStartID = resolved.toolCallStartID
        self.toolCallEndID = resolved.toolCallEndID
        self.toolResponseID = resolved.toolResponseID
        self.toolResponseEndID = resolved.toolResponseEndID
        self.channelStartID = resolved.channelStartID
        self.channelEndID = resolved.channelEndID
        self.thinkStartID = resolved.thinkStartID
        self.thinkEndID = resolved.thinkEndID
        self.stopTokenIDs = resolved.stopTokenIDs
        self.vocabSize = resolved.vocabSize
        self.dialect = resolved.dialect
        self.thinkStartMark = resolved.thinkStartMark
        self.thinkEndMark = resolved.thinkEndMark
        self.toolCallStartMark = resolved.toolCallStartMark
        self.toolCallEndMark = resolved.toolCallEndMark
        self.toolCallsStartID = resolved.toolCallsStartID
        self.toolCallsEndID = resolved.toolCallsEndID
        self.toolCallsStartMark = resolved.toolCallsStartMark
        self.toolCallsEndMark = resolved.toolCallsEndMark
        self.thinkingMode = thinkingMode
        // Every supported template ignores effort while thinking is off, so
        // an off-mode tokenizer stores none rather than an inert value.
        let activeEffort = thinkingMode.isEnabled ? reasoningEffort : nil
        self.reasoningEffort = activeEffort
        let context = Self.templateContext(
            thinkingEnabled: thinkingMode.isEnabled,
            reasoningEffort: activeEffort)
        self.generationSuffix = Self.deriveGenerationSuffix(
            tokenizer, thinkingEnabled: thinkingMode.isEnabled, context: context)
        self.effortSystemInstruction =
            thinkingMode.isEnabled
            ? Self.deriveEffortSystemInstruction(tokenizer, context: context)
            : nil
    }

    private var templateContext: [String: any Sendable] {
        var context = Self.templateContext(
            thinkingEnabled: thinkingMode.isEnabled,
            reasoningEffort: reasoningEffort)
        if dialect == .k2 {
            // The checkpoint's markdown tool renderer compares schema nodes
            // with `is sameas true`, and swift-jinja's `sameas` goes through
            // `compare(to:)`, which throws for a mapping ("Cannot compare
            // values of different types ({type: string} and true)"). JSON
            // presentation is a mode the template itself supports and renders
            // through `tojson`, so a tool prompt can be built at all.
            context["tool_presentation_format"] = "json"
        }
        return context
    }

    /// Encode UTF-8 text to token IDs.
    ///
    /// ChatML has no BOS, so `addBOS` is a no-op; BOS is never prepended.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    /// Decode token IDs to text. `skipSpecialTokens` strips BOS/EOS/turn markers from the output.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - Chat template

    public enum Role: String, Codable, Sendable {
        case system, developer, user, assistant, tool

        /// The name a chat template is handed for this role.
        ///
        /// OpenAI's `developer` role is the documented successor of `system`,
        /// and every chat template this project ships (Qwen 3.5/3.6/3.8,
        /// AgentWorld, Ornith, KAT) defines only `system`, `user`, `assistant`
        /// and `tool`. Handing the name through raised the template's own
        /// `raise_exception('Unexpected message role.')`, which the Chat
        /// Completions surface answered as HTTP 500 — for a role the OpenAI API
        /// defines and pi-ai sends on every reasoning request. The protocol
        /// model keeps the role it was given; only the render is normalized.
        public var templateRole: String {
            self == .developer ? Role.system.rawValue : rawValue
        }
    }
    public struct HistoricalToolCall: Codable, Sendable, Equatable {
        public let id: String
        public let name: String
        public let arguments: JSONValue

        public init(id: String, name: String, arguments: JSONValue) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct FunctionDefinition: Codable, Sendable, Equatable {
        public let name: String
        public let description: String
        public let parameters: JSONValue

        public init(name: String, description: String, parameters: JSONValue) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    public struct Message: Codable, Sendable, Equatable {
        public let role: Role
        public let content: String?
        public let toolCalls: [HistoricalToolCall]
        public let toolCallID: String?
        public let name: String?
        /// A historical assistant turn's reasoning, when the caller kept it.
        /// The K2 template refuses an assistant message without a thinking
        /// field, so this is what a replayed turn carries; nil renders an
        /// empty block (the turn had closed it).
        public let reasoning: String?

        public init(role: Role, content: String, reasoning: String? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = []
            self.toolCallID = nil
            self.name = nil
            self.reasoning = reasoning
        }

        public init(
            role: Role,
            content: String?,
            toolCalls: [HistoricalToolCall] = [],
            toolCallID: String? = nil,
            name: String? = nil,
            reasoning: String? = nil
        ) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
            self.reasoning = reasoning
        }
    }

    /// Text-only, no-tool rendering of the pinned checkpoint's bundled
    /// `chat_template.jinja`, with thinking disabled. Keeping this narrow makes
    /// unsupported tool/media behavior explicit instead of approximating it.
    static let imStartMark = "<|im_start|>"
    static let imEndMark = "<|im_end|>"
    /// Generation prompt with thinking disabled, matching the Jinja template's
    /// `add_generation_prompt` + `enable_thinking=false` branch. Used only
    /// when the tokenizer has no chat template or template rendering fails
    /// (R6); `generationSuffix` carries the template-derived value otherwise.
    private static let fallbackChatMLGenerationSuffix =
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    /// Same as `fallbackChatMLGenerationSuffix`, but for thinking mode ON:
    /// the template's `enable_thinking=true` branch leaves the `<think>`
    /// block open so the model must reason before answering.
    private static let fallbackChatMLGenerationSuffixThinking =
        "<|im_start|>assistant\n<think>\n"

    /// Derive the generation-prompt suffix from the tokenizer's bundled
    /// `chat_template.jinja` (`add_generation_prompt` with thinking per
    /// `thinkingEnabled`), falling back to the pinned constant when no
    /// template is available or rendering fails. The probe renders one empty
    /// user turn both with and without the generation prompt; the generation
    /// prompt is appended after the message loop, so the suffix is the
    /// token-level difference between the two renders.
    private static func deriveGenerationSuffix(
        _ tokenizer: any Tokenizer,
        thinkingEnabled: Bool,
        context: [String: any Sendable]
    ) -> String {
        let fallback =
            thinkingEnabled
            ? Self.fallbackChatMLGenerationSuffixThinking
            : Self.fallbackChatMLGenerationSuffix
        guard tokenizer.hasChatTemplate else {
            return fallback
        }
        let probe: [Tokenizers.Message] = [["role": "user", "content": ""]]
        do {
            let withPrompt = try tokenizer.applyChatTemplate(
                messages: probe,
                chatTemplate: nil,
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: [],
                additionalContext: context)
            let withoutPrompt = try tokenizer.applyChatTemplate(
                messages: probe,
                chatTemplate: nil,
                addGenerationPrompt: false,
                truncation: false,
                maxLength: nil,
                tools: [],
                additionalContext: context)
            guard withPrompt.count > withoutPrompt.count else {
                return fallback
            }
            // The generation prompt is appended after the message loop, so the
            // with-prompt render is the without-prompt render plus the suffix.
            let suffixIDs = Array(withPrompt[withoutPrompt.count...])
            return tokenizer.decode(tokens: suffixIDs, skipSpecialTokens: false)
        } catch {
            return fallback
        }
    }

    /// Derive the instruction the bundled template injects at the head of the
    /// system block for the active context. The probe renders one user turn
    /// with no system message: effort-aware templates open the render with a
    /// synthetic system block holding only the instruction, while binary
    /// templates render no leading system block at all (nil).
    private static func deriveEffortSystemInstruction(
        _ tokenizer: any Tokenizer,
        context: [String: any Sendable]
    ) -> String? {
        guard tokenizer.hasChatTemplate else { return nil }
        let probe: [Tokenizers.Message] = [["role": "user", "content": "x"]]
        guard
            let ids = try? tokenizer.applyChatTemplate(
                messages: probe,
                chatTemplate: nil,
                addGenerationPrompt: false,
                truncation: false,
                maxLength: nil,
                tools: [],
                additionalContext: context)
        else { return nil }
        let text = tokenizer.decode(tokens: ids, skipSpecialTokens: false)
        let blockStart = Self.imStartMark + "system\n"
        guard text.hasPrefix(blockStart),
            let blockEnd = text.range(of: Self.imEndMark)
        else { return nil }
        let instruction = String(
            text[text.index(text.startIndex, offsetBy: blockStart.count)..<blockEnd.lowerBound])
        return instruction.isEmpty ? nil : instruction
    }

    /// The bundled template's assistant split:
    /// `content.split('</think>')[0] … split('<think>')[-1]` for the reasoning
    /// and `content.split('</think>')[-1].lstrip('\n')` for the answer. The
    /// reasoning is whitespace-trimmed either way (`reasoning_content|trim`).
    static func splitThinking(_ content: String) -> (reasoning: String, answer: String) {
        guard let firstClose = content.range(of: "</think>") else { return ("", content) }
        let beforeFirst = String(content[content.startIndex..<firstClose.lowerBound])
        let afterLast =
            content.range(of: "</think>", options: .backwards)
            .map { String(content[$0.upperBound...]) } ?? content
        let reasoning = (beforeFirst.components(separatedBy: "<think>").last ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (reasoning, String(afterLast.drop(while: { $0 == "\n" })))
    }

    /// The field name the K2 template reads for a historical assistant turn's
    /// thinking. It selects the tag by the effort the tokenizer was loaded
    /// with, matching the generation prompt the same template writes.
    private var k2ThinkKey: String {
        switch reasoningEffort ?? .high {
        case .medium: return "think_fast"
        case .low: return "think_faster"
        case .high, .xhigh: return "think"
        }
    }

    public func encodeToolChat(
        messages: [Message],
        tools: [FunctionDefinition]
    ) throws -> [Int32] {
        guard tokenizer.hasChatTemplate else {
            throw GFTokenizerError.missingToolTemplate
        }
        let upstreamMessages: [Tokenizers.Message] = try messages.map { message in
            var value: Tokenizers.Message = [
                "role": message.role.templateRole,
                "content": message.content,
            ]
            if dialect == .k2, message.role == .assistant {
                // The K2 template raises unless an assistant turn names the
                // block its thinking lived in, so a replayed turn has to carry
                // one: the caller's reasoning when it kept it, an empty block
                // when it did not (the turn closed it before answering).
                value[k2ThinkKey] = message.reasoning ?? ""
            }
            if !message.toolCalls.isEmpty {
                value["tool_calls"] = try message.toolCalls.map { call -> [String: any Sendable] in
                    [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": try call.arguments.jinjaSendableValue(),
                        ] as [String: any Sendable],
                    ]
                }
            }
            if let toolCallID = message.toolCallID { value["tool_call_id"] = toolCallID }
            if let name = message.name { value["name"] = name }
            return value
        }
        let upstreamTools: [ToolSpec] = try tools.map { tool in
            [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": try tool.parameters.jinjaSendableValue(),
                ] as [String: any Sendable],
            ]
        }
        return try tokenizer.applyChatTemplate(
            messages: upstreamMessages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: upstreamTools,
            additionalContext: templateContext
        ).map(Int32.init)
    }

    public func encodeTextContinuation(userContent: String) -> [Int32] {
        // The template trims user content (`render_content(...)|trim`), so the
        // continuation bridge mirrors it; see `chatMLChatTemplate`.
        let content = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        return [endOfTurnID]
            + encode(
                "\n\(Self.imStartMark)user\n\(content)\(Self.imEndMark)\n"
                    + generationSuffix,
                addBOS: false)
    }

    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition]
    ) throws -> [Int32] {
        // The ChatML template's `<think>` stripping depends on each assistant
        // turn's position relative to the last user query, so a re-rendered
        // prefix is not guaranteed to be a token prefix of the full render.
        // Callers (ServerPromptCache) fall back to prefix matching; the
        // tool-result KV continuation is unsupported for ChatML.
        throw GFTokenizerError.unsupportedForDialect("tool-result KV continuation")
    }
}
