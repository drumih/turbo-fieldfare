import Foundation
import Tokenizers

/// Splits Inkling 2's generated tokens (the Inkling tokenizer dialect) into answer text, thinking and tool
/// calls, by the control tokens the chat format uses rather than by their
/// text (those tokens decode to nothing).
///
/// A model turn is a run of messages, each `<|message_model|>[name]
/// <|content_<kind>|>…<|end_message|>`, closed by
/// `<|content_model_end_sampling|>`. `content_text` is the answer,
/// `content_thinking` the reasoning, and `content_invoke_tool_json` a call
/// whose body is `{"name":…,"args":{…}}`. A call is reported only for a tool
/// the request offered; anything else the model invokes stays visible as text
/// rather than vanishing.
package struct InklingAssistantParser {
    /// The control-token ids the parser needs, read from the tokenizer.
    package struct Vocabulary: Sendable {
        package let messageModel: Int
        package let contentText: Int
        package let contentThinking: Int
        package let invokeToolJSON: Int
        package let invokeToolText: Int
        package let endMessage: Int
        /// Tokens that open a turn the model must not write (user, system,
        /// tool): the turn is over when one appears.
        package let foreignTurn: Set<Int>
        /// Every other control token; never part of decoded text.
        package let control: Set<Int>

        /// The ids from a tokenizer of the Inkling dialect. Every added token
        /// is a control token in this vocabulary: none of them is ever text.
        package init(tokenizer: GFTokenizer) throws {
            func id(_ name: String) throws -> Int {
                guard let id = tokenizer.tokenizer.convertTokenToId(name),
                    tokenizer.tokenizer.convertIdToToken(id) == name
                else {
                    throw GFTokenizerError.missingSpecialToken(name)
                }
                return id
            }
            messageModel = try id("<|message_model|>")
            contentText = try id("<|content_text|>")
            contentThinking = try id("<|content_thinking|>")
            invokeToolJSON = try id("<|content_invoke_tool_json|>")
            invokeToolText = try id("<|content_invoke_tool_text|>")
            endMessage = try id("<|end_message|>")
            foreignTurn = Set(
                ["<|message_user|>", "<|message_system|>", "<|message_tool|>"]
                    .compactMap { try? id($0) })
            control = Set(tokenizer.byteLevelDecoderConfiguration.addedTokens.keys.map(Int.init))
        }

        package init(
            messageModel: Int, contentText: Int, contentThinking: Int, invokeToolJSON: Int,
            invokeToolText: Int, endMessage: Int, foreignTurn: Set<Int>, control: Set<Int>
        ) {
            self.messageModel = messageModel
            self.contentText = contentText
            self.contentThinking = contentThinking
            self.invokeToolJSON = invokeToolJSON
            self.invokeToolText = invokeToolText
            self.endMessage = endMessage
            self.foreignTurn = foreignTurn
            self.control = control
        }
    }

    private enum Kind { case text, thinking, tool(name: [Int]), ignored }

    private enum State {
        /// Between messages: waiting for `<|message_model|>`.
        case idle
        /// After `<|message_model|>`: the optional name, then a content token.
        case header(name: [Int])
        case segment(Kind, tokens: [Int], emitted: String)
    }

    private let vocabulary: Vocabulary
    private let decode: ([Int]) -> String
    private let offeredTools: Set<String>
    private var state: State
    /// True once the model has started a turn it must not write.
    package private(set) var ended = false

    /// `startInText` is for a prompt that already opened a text message
    /// (`openTextTurn`) and for a raw completion.
    package init(
        vocabulary: Vocabulary, decode: @escaping ([Int]) -> String,
        offeredTools: Set<String>, startInText: Bool
    ) {
        self.vocabulary = vocabulary
        self.decode = decode
        self.offeredTools = offeredTools
        state = startInText ? .segment(.text, tokens: [], emitted: "") : .idle
    }

    /// One generated token as zero or more events.
    package mutating func consume(_ token: Int) -> [StructuredAssistantEvent] {
        if vocabulary.foreignTurn.contains(token) {
            ended = true
            return close()
        }
        if token == vocabulary.messageModel {
            let events = close()
            state = .header(name: [])
            return events
        }
        if token == vocabulary.endMessage {
            let events = close()
            state = .idle
            return events
        }
        switch state {
        case .idle:
            // Text outside any message: keep it rather than drop it.
            guard !vocabulary.control.contains(token) else { return [] }
            state = .segment(.text, tokens: [token], emitted: "")
            return delta()
        case .header(let name):
            switch token {
            case vocabulary.contentText: state = .segment(.text, tokens: [], emitted: "")
            case vocabulary.contentThinking: state = .segment(.thinking, tokens: [], emitted: "")
            case vocabulary.invokeToolJSON, vocabulary.invokeToolText:
                state = .segment(.tool(name: name), tokens: [], emitted: "")
            default:
                if vocabulary.control.contains(token) {
                    state = .segment(.ignored, tokens: [], emitted: "")
                } else {
                    state = .header(name: name + [token])
                }
            }
            return []
        case .segment(let kind, let tokens, let emitted):
            guard !vocabulary.control.contains(token) else { return [] }
            state = .segment(kind, tokens: tokens + [token], emitted: emitted)
            return delta()
        }
    }

    /// Whatever the last, unterminated message still holds.
    package mutating func finish() -> [StructuredAssistantEvent] {
        let events = close()
        state = .idle
        return events
    }

    /// The new text of a streaming segment. A trailing U+FFFD is a character
    /// still being spelled across tokens, so it waits for the next token.
    private mutating func delta(final: Bool = false) -> [StructuredAssistantEvent] {
        guard case .segment(let kind, let tokens, let emitted) = state else { return [] }
        let isText: Bool
        switch kind {
        case .text: isText = true
        case .thinking: isText = false
        case .tool, .ignored: return []
        }
        let decoded = decode(tokens)
        guard decoded.hasPrefix(emitted), decoded.count > emitted.count,
            final || !decoded.hasSuffix("\u{FFFD}")
        else { return [] }
        let piece = String(decoded.dropFirst(emitted.count))
        state = .segment(kind, tokens: tokens, emitted: decoded)
        return [isText ? .content(piece) : .reasoning(piece)]
    }

    private mutating func close() -> [StructuredAssistantEvent] {
        guard case .segment(let kind, let tokens, _) = state else {
            state = .idle
            return []
        }
        var events = delta(final: true)
        if case .tool(let nameTokens) = kind {
            events += toolEvents(header: decode(nameTokens), body: decode(tokens))
        }
        state = .idle
        return events
    }

    private func toolEvents(header: String, body: String) -> [StructuredAssistantEvent] {
        let trimmedHeader = header.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = body.data(using: .utf8),
            let parsed = try? JSONDecoder().decode(JSONValue.self, from: data),
            case .object(let fields) = parsed
        else {
            return body.isEmpty ? [] : [.content(body)]
        }
        var name = trimmedHeader
        if case .string(let inner)? = fields["name"], !inner.isEmpty { name = inner }
        let arguments: JSONValue
        if let args = fields["args"] ?? fields["arguments"] {
            arguments = InklingChatTemplate.argumentsObject(args)
        } else {
            arguments = .object([:])
        }
        guard !name.isEmpty, offeredTools.contains(name) else { return [.content(body)] }
        let call = ParsedToolCall(
            id: Self.callID(), name: name, arguments: arguments,
            argumentsJSON: InklingChatTemplate.canonicalJSON(arguments))
        return [.toolCall(call)]
    }

    static func callID() -> String {
        "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
    }
}
