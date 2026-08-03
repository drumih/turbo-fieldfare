import Foundation
import Tokenizers

public enum GFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        }
    }
}

/// Gemma 4 tokenizer wrapper.
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

    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let endOfTurnID: Int32
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseID: Int32
    public let toolResponseEndID: Int32
    public let channelStartID: Int32
    public let channelEndID: Int32
    public let beginningOfImageID: Int32
    public let imageID: Int32
    public let endOfImageID: Int32
    public let stopTokenIDs: Set<Int32>
    public let suppressedGenerationTokenIDs: Set<Int32>
    public let vocabSize: Int

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
        guard let beginningOfImage = tokenizer.convertTokenToId("<|image>"),
              let image = tokenizer.convertTokenToId("<|image|>"),
              let endOfImage = tokenizer.convertTokenToId("<image|>") else {
            throw GFTokenizerError.missingSpecialToken("Gemma image markers")
        }

        self.bosID = Int32(bos)
        self.eosID = Int32(eos)
        self.padID = Int32(pad)
        self.endOfTurnID = Int32(eot)
        self.toolCallStartID = Int32(toolCallStart)
        self.toolCallEndID = Int32(toolCallEnd)
        self.toolResponseID = Int32(toolResponse)
        self.toolResponseEndID = Int32(toolResponseEnd)
        self.channelStartID = Int32(channelStart)
        self.channelEndID = Int32(channelEnd)
        self.beginningOfImageID = Int32(beginningOfImage)
        self.imageID = Int32(image)
        self.endOfImageID = Int32(endOfImage)
        self.stopTokenIDs = [self.eosID, self.endOfTurnID, self.toolResponseID]
        self.suppressedGenerationTokenIDs = [
            self.padID,
            self.bosID,
            self.beginningOfImageID,
            self.imageID,
            self.endOfImageID,
        ]
        self.vocabSize = 262_144
    }

    /// Encode UTF-8 text to token IDs. `addBOS = true` prepends `<bos>`.
    ///
    /// The library's `addSpecialTokens: true` flag is a no-op for the Gemma 4 IT
    /// tokenizer (its config has `add_bos_token = false`; BOS is expected to come
    /// from the chat template). We prepend manually so the kernel-facing API stays
    /// the same regardless of upstream defaults.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        let base = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        return addBOS ? [bosID] + base : base
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
        /// Number of projected soft tokens for each image, in content order.
        /// The app currently supplies at most one image per user turn.
        public let imageTokenCounts: [Int]
        public let toolCalls: [HistoricalToolCall]
        public let toolCallID: String?
        public let name: String?

        public init(role: Role, content: String, imageTokenCounts: [Int] = []) {
            self.role = role
            self.content = content
            self.imageTokenCounts = imageTokenCounts
            self.toolCalls = []
            self.toolCallID = nil
            self.name = nil
        }

        public init(role: Role,
                    content: String?,
                    imageTokenCounts: [Int] = [],
                    toolCalls: [HistoricalToolCall] = [],
                    toolCallID: String? = nil,
                    name: String? = nil) {
            self.role = role
            self.content = content
            self.imageTokenCounts = imageTokenCounts
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
        }
    }

    /// Text-only, no-tool rendering of the pinned IT checkpoint's bundled
    /// `chat_template.jinja`, with thinking disabled. Keeping this narrow makes
    /// unsupported tool/media behavior explicit instead of approximating it.
    private static let turnOpen    = "<|turn>"
    private static let turnClose   = "<turn|>"
    private static let bosMark     = "<bos>"
    private static let beginningOfImageMark = "<|image>"
    private static let imageMark = "<|image|>"
    private static let endOfImageMark = "<image|>"

    public struct PreparedChatPrompt: Sendable, Equatable {
        public let tokenIDs: [Int32]
        /// Ranges occupied by `<|image|>` placeholder tokens. Boundary tokens
        /// remain ordinary text embeddings, matching the reference processor.
        public let imageSpans: [Range<Int>]

        public init(tokenIDs: [Int32], imageSpans: [Range<Int>]) {
            self.tokenIDs = tokenIDs
            self.imageSpans = imageSpans
        }
    }

    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        var s = Self.bosMark
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            if !message.imageTokenCounts.isEmpty && message.role != .user {
                throw GFTokenizerError.invalidChatTemplate(
                    "images are only supported in user messages")
            }
            guard message.imageTokenCounts.allSatisfy({ (1...280).contains($0) }) else {
                throw GFTokenizerError.invalidChatTemplate(
                    "each image must produce between 1 and 280 soft tokens")
            }
            let role = message.role == .assistant ? "model" : message.role.rawValue
            let images = message.imageTokenCounts.map { count in
                Self.beginningOfImageMark
                    + String(repeating: Self.imageMark, count: count)
                    + Self.endOfImageMark
            }.joined()
            s += Self.turnOpen + role + "\n" + images + content + Self.turnClose + "\n"
        }
        s += Self.turnOpen + "model\n<|channel>thought\n<channel|>"
        return s
    }

    /// Render and tokenize a chat while retaining the exact placeholder rows
    /// that must be overwritten by projected image features during prefill.
    public func prepareChatPrompt(_ messages: [Message]) throws -> PreparedChatPrompt {
        let expectedCounts = messages.flatMap(\.imageTokenCounts)
        let rendered = try applyChatTemplate(messages)
        let tokenIDs = encode(rendered, addBOS: false)
        // Reject literal/spoofed media control tokens instead of guessing which
        // rows should receive image embeddings.
        guard tokenIDs.filter({ $0 == beginningOfImageID }).count == expectedCounts.count,
              tokenIDs.filter({ $0 == endOfImageID }).count == expectedCounts.count,
              tokenIDs.filter({ $0 == imageID }).count == expectedCounts.reduce(0, +) else {
            throw GFTokenizerError.invalidChatTemplate(
                "image marker sequence is ambiguous")
        }
        guard !expectedCounts.isEmpty else {
            return PreparedChatPrompt(tokenIDs: tokenIDs, imageSpans: [])
        }

        var spans: [Range<Int>] = []
        spans.reserveCapacity(expectedCounts.count)
        var cursor = tokenIDs.startIndex
        for count in expectedCounts {
            guard let boundary = tokenIDs[cursor...].firstIndex(of: beginningOfImageID) else {
                throw GFTokenizerError.invalidChatTemplate("image start marker is missing")
            }
            let lower = boundary + 1
            let upper = lower + count
            guard upper < tokenIDs.endIndex,
                  tokenIDs[lower..<upper].allSatisfy({ $0 == imageID }),
                  tokenIDs[upper] == endOfImageID else {
                throw GFTokenizerError.invalidChatTemplate(
                    "image placeholder count does not match projected features")
            }
            spans.append(lower..<upper)
            cursor = upper + 1
        }
        return PreparedChatPrompt(tokenIDs: tokenIDs, imageSpans: spans)
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition]) throws -> [Int32] {
        guard tokenizer.hasChatTemplate else {
            throw GFTokenizerError.missingToolTemplate
        }
        guard messages.allSatisfy({ $0.imageTokenCounts.isEmpty }) else {
            throw GFTokenizerError.invalidChatTemplate(
                "image inputs are not supported in tool-calling chats")
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
        return [endOfTurnID] + encode(
            "\n\(Self.turnOpen)user\n\(content)\(Self.turnClose)\n"
                + "\(Self.turnOpen)model\n<|channel>thought\n<channel|>",
            addBOS: false)
    }

    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition]
    ) throws -> [Int32] {
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
