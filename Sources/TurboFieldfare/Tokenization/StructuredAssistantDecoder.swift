import Foundation

public enum StructuredAssistantEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

public final class StructuredAssistantDecoder: @unchecked Sendable {
    private enum Channel {
        case thought
        case visible
        case label
    }

    private let tokenizer: GFTokenizer
    private let allowedTools: Set<String>
    private let idGenerator: @Sendable () -> String
    private var channel: Channel = .visible
    private var label = ""
    private var toolTokens: [Int32]?
    private var emittedCalls = 0
    private var failed = false

    public init(tokenizer: GFTokenizer,
                allowedTools: Set<String>,
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.tokenizer = tokenizer
        self.allowedTools = allowedTools
        self.idGenerator = idGenerator
    }

    public func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw GemmaToolCallParserError.malformed }
        if tokenID == tokenizer.channelStartID {
            label = ""
            channel = .label
            return []
        }
        if tokenID == tokenizer.channelEndID {
            channel = .visible
            return []
        }
        if tokenID == tokenizer.toolCallStartID {
            guard toolTokens == nil else {
                failed = true
                throw GemmaToolCallParserError.malformed
            }
            toolTokens = []
            return []
        }
        if tokenID == tokenizer.toolCallEndID {
            guard let tokens = toolTokens else {
                failed = true
                throw GemmaToolCallParserError.malformed
            }
            toolTokens = nil
            let text = tokenizer.decode(tokens, skipSpecialTokens: false)
            do {
                let call = try GemmaToolCallParser().parse(
                    text, allowedTools: allowedTools, id: idGenerator())
                emittedCalls += 1
                return [.toolCall(call)]
            } catch {
                failed = true
                throw error
            }
        }
        if tokenID == tokenizer.toolResponseID || tokenID == tokenizer.toolResponseEndID {
            guard emittedCalls > 0, toolTokens == nil else {
                failed = true
                throw GemmaToolCallParserError.malformed
            }
            return []
        }
        if var tokens = toolTokens {
            tokens.append(tokenID)
            guard tokens.count * MemoryLayout<Int32>.size <= GemmaToolCallParser.maximumBytes else {
                failed = true
                throw GemmaToolCallParserError.oversized
            }
            toolTokens = tokens
            return []
        }
        switch channel {
        case .thought:
            return []
        case .visible:
            return delta.isEmpty ? [] : [.content(delta)]
        case .label:
            label += delta
            guard let newline = label.firstIndex(of: "\n") else { return [] }
            let name = label[..<newline].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let contentStart = label.index(after: newline)
            let content = String(label[contentStart...])
            channel = name == "final" || name == "answer" ? .visible : .thought
            label = ""
            if channel == .visible, !content.isEmpty {
                return [.content(content)]
            }
            return []
        }
    }

    public func finish() throws {
        guard !failed, toolTokens == nil else {
            throw GemmaToolCallParserError.malformed
        }
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }
}
