import Foundation
import Tokenizers

/// Where the messages of a rendered chat prompt start, for the prompt cache's
/// interior checkpoints (system end, last-turn start).
///
/// ChatML opens every message and the generation prompt with `<|im_start|>`,
/// so the final start is always the generation header. Inkling opens each
/// message with its role token (`<|message_user|>` …) and has no generation
/// prompt: only a prompt that opened the answer's text message itself
/// (`<|message_model|><|content_text|>`) ends in a header.
public struct PromptMessageMarkers: Sendable, Equatable {
    public let startIDs: Set<Int32>
    /// nil: the last start is always the generation header (ChatML). Else the
    /// header is these ids when the prompt ends with them.
    public let headerSuffix: [Int32]?

    public init(startIDs: Set<Int32>, headerSuffix: [Int32]?) {
        self.startIDs = startIDs
        self.headerSuffix = headerSuffix
    }

    /// The ChatML markers: one start token, the last start the header.
    public static func chatML(imStartID: Int32) -> Self {
        Self(startIDs: [imStartID], headerSuffix: nil)
    }

    /// Message starts on `promptIDs`, the generation header excluded. Empty
    /// when the ids are not a chat render these markers describe.
    public func messageStarts(_ promptIDs: [Int32]) -> [Int] {
        var starts = promptIDs.indices.filter { startIDs.contains(promptIDs[$0]) }
        guard let headerSuffix else {
            guard starts.count >= 2 else { return [] }
            return Array(starts.dropLast())
        }
        if !headerSuffix.isEmpty, promptIDs.count >= headerSuffix.count,
            Array(promptIDs.suffix(headerSuffix.count)) == headerSuffix,
            starts.last == promptIDs.count - headerSuffix.count
        {
            starts.removeLast()
        }
        return starts
    }
}

extension GFTokenizer {
    /// This tokenizer's message markers.
    public var promptMessageMarkers: PromptMessageMarkers {
        guard dialect == .inkling else { return .chatML(imStartID: imStartID) }
        let ids = [
            "<|message_user|>", "<|message_system|>", "<|message_model|>", "<|message_tool|>",
        ]
        .compactMap { tokenizer.convertTokenToId($0).map(Int32.init) }
        let header = ["<|message_model|>", "<|content_text|>"]
            .compactMap { tokenizer.convertTokenToId($0).map(Int32.init) }
        return PromptMessageMarkers(
            startIDs: Set(ids), headerSuffix: header.count == 2 ? header : [])
    }
}
