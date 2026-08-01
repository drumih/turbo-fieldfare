import Foundation
import Tokenizers

public enum GFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate
    case unsupportedForDialect(String)

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        case .unsupportedForDialect(let operation):
            return "operation is not supported for this tokenizer's chat dialect: \(operation)"
        }
    }
}

/// Chat framing dialect, resolved from the loaded tokenizer's special tokens.
///
/// `.chatml` is detected by the presence of the `<|im_end|>` special token
/// (Qwen-style ChatML); everything else uses the Gemma 4 contract.
public enum ChatDialect: String, Sendable {
    case gemma
    case chatml
}

/// Tokenizer wrapper for the supported model families (Gemma 4 and ChatML/Qwen).
///
/// Prefers tokenizer sidecars in a completed `.gturbo/tokenizer/` directory,
/// then falls back to the IT variant's Hugging Face Hub tokenizer cache. Exposes
/// typed accessors for the IDs the generator actually needs (BOS / EOS / pad /
/// end-of-turn) and adapts encode/decode to Int32 to match the buffer types
/// kernels consume.
///
/// TurboFieldfare owns the minimal chat framing because the upstream
/// `tokenizer_config.json` has no `chat_template`. Literal control-token text in
/// user content is accepted as a trusted-input research-runtime limitation.
public struct GFTokenizer: @unchecked Sendable {
    public static let modelID = "google/gemma-4-26B-A4B-it"
    public static let chatTemplateIdentity = "gemma4-it-text-no-tools-v1"
    public static let toolChatTemplateIdentity = "gemma4-it-tools-jinja-v1"

    public let dialect: ChatDialect
    /// Nominal BOS. For ChatML this is `<|endoftext|>` (the config's unused
    /// `bos_token_id`); it is never prepended — see `encode(_:addBOS:)`.
    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let endOfTurnID: Int32
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseID: Int32
    public let toolResponseEndID: Int32
    /// For ChatML these alias the `<think>` / `</think>` markers, the dialect's
    /// closest analog of Gemma's hidden-channel delimiters.
    public let channelStartID: Int32
    public let channelEndID: Int32
    /// ChatML `<think>` / `</think>` special-token IDs; nil for Gemma.
    public let thinkStartID: Int32?
    public let thinkEndID: Int32?
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int

    /// BOS actually prepended by `encode(_:addBOS:)`; nil for dialects that
    /// never use a BOS prefix (ChatML).
    private let bosPrefixID: Int32?

    @usableFromInline
    let tokenizer: any Tokenizer

    public static func load() async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.pretrained(modelID))
    }

    public static func load(from folder: URL) async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.local(folder.standardizedFileURL.path))
    }

    public static func load(forModelDirectory modelDirectory: URL,
                            environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> GFTokenizer {
        if let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) {
            return try await load(from: folder)
        }
        return try await load()
    }

    public static func tokenizerFolder(forModelDirectory modelDirectory: URL,
                                       environment: [String: String] = ProcessInfo.processInfo.environment,
                                       fileManager: FileManager = .default) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }

        guard let override = environment["TURBO_FIELDFARE_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    static func loadUncached(pretrained modelID: String = Self.modelID) async throws -> GFTokenizer {
        let underlying = try await AutoTokenizer.from(pretrained: modelID)
        return try GFTokenizer(tokenizer: underlying)
    }

    static func loadUncached(from folder: URL) async throws -> GFTokenizer {
        let underlying = try await AutoTokenizer.from(modelFolder: folder)
        return try GFTokenizer(tokenizer: underlying)
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    public init(tokenizer: any Tokenizer) throws {
        self.tokenizer = tokenizer

        let dialect: ChatDialect =
            Self.specialTokenID(tokenizer, Self.imEndMark) != nil ? .chatml : .gemma
        let resolved = dialect == .chatml
            ? try Self.resolveChatMLTokens(tokenizer)
            : try Self.resolveGemmaTokens(tokenizer)

        self.dialect = dialect
        self.bosID = resolved.bosID
        self.bosPrefixID = resolved.bosPrefixID
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
    }

    private struct ResolvedSpecialTokens {
        let bosID: Int32
        let bosPrefixID: Int32?
        let eosID: Int32
        let padID: Int32
        let endOfTurnID: Int32
        let toolCallStartID: Int32
        let toolCallEndID: Int32
        let toolResponseID: Int32
        let toolResponseEndID: Int32
        let channelStartID: Int32
        let channelEndID: Int32
        let thinkStartID: Int32?
        let thinkEndID: Int32?
        let stopTokenIDs: Set<Int32>
        let vocabSize: Int
    }

    private static func resolveGemmaTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        guard let bos = tokenizer.bosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<bos>")
        }
        guard let eos = tokenizer.eosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<eos>")
        }
        guard let pad = tokenizer.convertTokenToId("<pad>") else {
            throw GFTokenizerError.missingSpecialToken("<pad>")
        }
        guard let eot = tokenizer.convertTokenToId("<turn|>") else {
            throw GFTokenizerError.missingSpecialToken("<turn|>")
        }
        guard let toolResponse = tokenizer.convertTokenToId("<|tool_response>") else {
            throw GFTokenizerError.missingSpecialToken("<|tool_response>")
        }
        guard let toolCallStart = tokenizer.convertTokenToId("<|tool_call>"),
              let toolCallEnd = tokenizer.convertTokenToId("<tool_call|>"),
              let toolResponseEnd = tokenizer.convertTokenToId("<tool_response|>"),
              let channelStart = tokenizer.convertTokenToId("<|channel>"),
              let channelEnd = tokenizer.convertTokenToId("<channel|>") else {
            throw GFTokenizerError.missingSpecialToken("Gemma tool/channel markers")
        }
        return ResolvedSpecialTokens(
            bosID: Int32(bos),
            bosPrefixID: Int32(bos),
            eosID: Int32(eos),
            padID: Int32(pad),
            endOfTurnID: Int32(eot),
            toolCallStartID: Int32(toolCallStart),
            toolCallEndID: Int32(toolCallEnd),
            toolResponseID: Int32(toolResponse),
            toolResponseEndID: Int32(toolResponseEnd),
            channelStartID: Int32(channelStart),
            channelEndID: Int32(channelEnd),
            thinkStartID: nil,
            thinkEndID: nil,
            stopTokenIDs: [Int32(eos), Int32(eot), Int32(toolResponse)],
            vocabSize: 262_144)
    }

    /// Resolves a token string to its ID, rejecting the unk-token fallback
    /// some tokenizers substitute for out-of-vocabulary strings.
    private static func specialTokenID(_ tokenizer: any Tokenizer, _ token: String) -> Int? {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else { return nil }
        return id
    }

    private static func resolveChatMLTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw GFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        // `<|im_start|>` is required even though no stored property holds it;
        // template rendering relies on the tokenizer recognizing its text.
        _ = try id(Self.imStartMark)
        let imEnd = try id(Self.imEndMark)
        let endOfText = try id("<|endoftext|>")
        let toolCallStart = try id("<tool_call>")
        let toolCallEnd = try id("</tool_call>")
        let toolResponse = try id("<tool_response>")
        let toolResponseEnd = try id("</tool_response>")
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        return ResolvedSpecialTokens(
            bosID: endOfText,
            bosPrefixID: nil,
            eosID: endOfText,
            padID: endOfText,
            endOfTurnID: imEnd,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            stopTokenIDs: [imEnd, endOfText],
            // The model's padded embedding/lm_head row count, not the
            // tokenizer's actual vocab (248 077) — logits buffers use this.
            vocabSize: 248_320)
    }

    /// Encode UTF-8 text to token IDs. `addBOS = true` prepends `<bos>`.
    ///
    /// The library's `addSpecialTokens: true` flag is a no-op for the Gemma 4 IT
    /// tokenizer (its config has `add_bos_token = false`; BOS is expected to come
    /// from the chat template). We prepend manually so the kernel-facing API stays
    /// the same regardless of upstream defaults. ChatML has no BOS, so `addBOS`
    /// is a no-op for that dialect.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        let base = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        guard addBOS, let bosPrefixID else { return base }
        return [bosPrefixID] + base
    }

    /// Decode token IDs to text. `skipSpecialTokens` strips BOS/EOS/turn markers from the output.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - Chat template

    public enum Role: String, Sendable { case system, developer, user, assistant, tool }
    public struct HistoricalToolCall: Sendable, Equatable {
        public let id: String
        public let name: String
        public let arguments: JSONValue

        public init(id: String, name: String, arguments: JSONValue) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct FunctionDefinition: Sendable, Equatable {
        public let name: String
        public let description: String
        public let parameters: JSONValue

        public init(name: String, description: String, parameters: JSONValue) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    public struct Message: Sendable, Equatable {
        public let role: Role
        public let content: String?
        public let toolCalls: [HistoricalToolCall]
        public let toolCallID: String?
        public let name: String?

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
            self.toolCalls = []
            self.toolCallID = nil
            self.name = nil
        }

        public init(role: Role,
                    content: String?,
                    toolCalls: [HistoricalToolCall] = [],
                    toolCallID: String? = nil,
                    name: String? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
        }
    }

    /// Text-only, no-tool rendering of the pinned checkpoint's bundled
    /// `chat_template.jinja`, with thinking disabled. Keeping this narrow makes
    /// unsupported tool/media behavior explicit instead of approximating it.
    private static let turnOpen    = "<|turn>"
    private static let turnClose   = "<turn|>"
    private static let bosMark     = "<bos>"
    private static let imStartMark = "<|im_start|>"
    private static let imEndMark   = "<|im_end|>"
    /// Generation prompt with thinking disabled, matching the Jinja template's
    /// `add_generation_prompt` + `enable_thinking=false` branch.
    private static let chatMLGenerationSuffix =
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        switch dialect {
        case .gemma: return try gemmaChatTemplate(messages)
        case .chatml: return try chatMLChatTemplate(messages)
        }
    }

    private func gemmaChatTemplate(_ messages: [Message]) throws -> String {
        var s = Self.bosMark
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            let role = message.role == .assistant ? "model" : message.role.rawValue
            s += Self.turnOpen + role + "\n" + content + Self.turnClose + "\n"
        }
        s += Self.turnOpen + "model\n<|channel>thought\n<channel|>"
        return s
    }

    private func chatMLChatTemplate(_ messages: [Message]) throws -> String {
        var s = ""
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            s += Self.imStartMark + message.role.rawValue + "\n" + content + Self.imEndMark + "\n"
        }
        s += Self.chatMLGenerationSuffix
        return s
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition]) throws -> [Int32] {
        guard tokenizer.hasChatTemplate else {
            throw GFTokenizerError.missingToolTemplate
        }
        let upstreamMessages: [Tokenizers.Message] = try messages.map { message in
            var value: Tokenizers.Message = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
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
            additionalContext: ["enable_thinking": false]
        ).map(Int32.init)
    }

    public func encodeTextContinuation(userContent: String) -> [Int32] {
        let content = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        switch dialect {
        case .gemma:
            return [endOfTurnID] + encode(
                "\n\(Self.turnOpen)user\n\(content)\(Self.turnClose)\n"
                    + "\(Self.turnOpen)model\n<|channel>thought\n<channel|>",
                addBOS: false)
        case .chatml:
            return [endOfTurnID] + encode(
                "\n\(Self.imStartMark)user\n\(content)\(Self.imEndMark)\n"
                    + Self.chatMLGenerationSuffix,
                addBOS: false)
        }
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
        // Callers (ServerPromptCache) fall back to prefix matching.
        guard dialect == .gemma else {
            throw GFTokenizerError.unsupportedForDialect("tool-result KV continuation")
        }
        let prefix = try encodeToolChat(
            messages: cachedMessages + [assistant],
            tools: tools)
        let full = try encodeToolChat(messages: incomingMessages, tools: tools)
        let callCount = assistant.toolCalls.count
        let starts = prefix.indices.filter { prefix[$0] == toolCallStartID }
        guard callCount > 0, starts.count >= callCount,
              let callEnd = prefix.lastIndex(of: toolCallEndID) else {
            throw GFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is missing")
        }
        let callStart = starts[starts.count - callCount]
        let callSequence = Array(prefix[callStart...callEnd])
        let matches = full.subsequenceStartIndices(matching: callSequence)
        guard matches.count == 1 else {
            throw GFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is ambiguous")
        }
        let suffixStart = matches[0] + callSequence.count
        let suffix = Array(full[suffixStart...])
        guard suffix.first == toolResponseID else {
            throw GFTokenizerError.invalidChatTemplate(
                "tool-result continuation does not begin at the KV boundary")
        }
        return suffix
    }
}

private extension Array where Element: Equatable {
    func subsequenceStartIndices(matching needle: [Element]) -> [Int] {
        guard !needle.isEmpty, needle.count <= count else { return [] }
        return indices.dropLast(needle.count - 1).filter { start in
            self[start..<(start + needle.count)].elementsEqual(needle)
        }
    }
}

private enum GFTokenizerLoadSource: Hashable {
    case pretrained(String)
    case local(String)
}

private actor GFTokenizerLoadCoordinator {
    static let shared = GFTokenizerLoadCoordinator()

    private var tasks: [GFTokenizerLoadSource: Task<GFTokenizer, Error>] = [:]

    func load(_ source: GFTokenizerLoadSource) async throws -> GFTokenizer {
        if let task = tasks[source] {
            return try await task.value
        }

        // Keep the CPU-heavy tokenizer build off the coordinator actor; callers
        // share the task result instead of owning its cancellation.
        let task = Task.detached(priority: .userInitiated) { () throws -> GFTokenizer in
            switch source {
            case .pretrained(let modelID):
                return try await GFTokenizer.loadUncached(pretrained: modelID)
            case .local(let path):
                return try await GFTokenizer.loadUncached(from: URL(fileURLWithPath: path))
            }
        }
        tasks[source] = task

        do {
            return try await task.value
        } catch {
            tasks[source] = nil
            throw error
        }
    }
}
