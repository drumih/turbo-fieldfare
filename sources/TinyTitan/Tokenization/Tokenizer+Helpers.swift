import Foundation
import Tokenizers
import TinyTitanCore

// Tokenizer introspection helpers: the Jinja context, the resolved-special-token
// type, the streaming-decoder check and the special-token resolution.
//
// Split out of `Tokenizer.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion. Four members widened from
// `private` to internal because the initializer that calls them stays in
// Tokenizer.swift: `templateContext`, `ResolvedSpecialTokens`,
// `validateStreamingDecoder` and `resolveChatMLTokens`.
extension GFTokenizer {

    /// The Jinja context shared by every bundled-template render: the binary
    /// switch always, plus the effort override only when one is active (an
    /// absent key selects the template's own default level).
    static func templateContext(
        thinkingEnabled: Bool,
        reasoningEffort: ModelReasoningEffort?
    ) -> [String: any Sendable] {
        var context: [String: any Sendable] = ["enable_thinking": thinkingEnabled]
        if let reasoningEffort {
            context["reasoning_effort"] = reasoningEffort.rawValue
        }
        return context
    }

    struct ResolvedSpecialTokens {
        let dialect: TokenizerDialect
        let imStartID: Int32
        let bosID: Int32
        let eosID: Int32
        let padID: Int32
        let endOfTurnID: Int32
        let toolCallStartID: Int32
        let toolCallEndID: Int32
        /// The outer `<ifm|tool_calls>` wrapper K2 writes around one or more
        /// inner calls. Optional: the ChatML families wrap nothing.
        let toolCallsStartID: Int32?
        let toolCallsEndID: Int32?
        let toolResponseID: Int32
        let toolResponseEndID: Int32
        let channelStartID: Int32
        let channelEndID: Int32
        let thinkStartID: Int32?
        let thinkEndID: Int32?
        let thinkStartMark: String
        let thinkEndMark: String
        let toolCallStartMark: String
        let toolCallEndMark: String
        let toolCallsStartMark: String
        let toolCallsEndMark: String
        let stopTokenIDs: Set<Int32>
        let vocabSize: Int
    }

    static func validateStreamingDecoder(
        _ decoder: GFByteLevelDecoderConfiguration,
        tokenizer: any Tokenizer,
        resolved: ResolvedSpecialTokens
    ) throws {
        let literalMarkers: [(Int32, String)] = [
            (resolved.toolCallStartID, resolved.toolCallStartMark),
            (resolved.toolCallEndID, resolved.toolCallEndMark),
            (resolved.toolResponseID, "<tool_response>"),
            (resolved.toolResponseEndID, "</tool_response>"),
            (resolved.channelStartID, resolved.thinkStartMark),
            (resolved.channelEndID, resolved.thinkEndMark),
        ]
        for (id, content) in literalMarkers {
            guard let added = decoder.addedTokens[id],
                added.content == content, !added.special
            else {
                throw GFTokenizerError.unsupportedForDialect(
                    "ChatML control token \(content) must be a literal ByteLevel barrier")
            }
        }

        let filteredMarkers = [resolved.eosID, resolved.endOfTurnID]
        for id in filteredMarkers {
            guard decoder.addedTokens[id]?.special == true else {
                let token = tokenizer.convertIdToToken(Int(id)) ?? "id \(id)"
                throw GFTokenizerError.unsupportedForDialect(
                    "ChatML stop token \(token) must be marked special")
            }
        }
    }

    /// Resolves a token string to its ID, rejecting the unk-token fallback
    /// some tokenizers substitute for out-of-vocabulary strings.
    private static func specialTokenID(_ tokenizer: any Tokenizer, _ token: String) -> Int? {
        guard let id = tokenizer.convertTokenToId(token),
            tokenizer.convertIdToToken(id) == token
        else { return nil }
        return id
    }

    /// The model's padded embedding/lm_head row count. The tokenizer's own
    /// vocab (248 077 for Qwen) is smaller; logits buffers and the
    /// embedding/lm_head are sized to the padded rows, and `vocabSize`
    /// reports at least this many.
    private static let paddedLogitsVocabSize = 248_320

    /// Derive the tokenizer's actual vocab by probing `convertIdToToken` for
    /// the first invalid id. Standard vocab files keep ids dense from 0, so
    /// the first nil is the vocab count. Bounded so a pathological tokenizer
    /// cannot make init scan forever; nil means "no reliable derivation".
    private static func derivedVocabSize(_ tokenizer: any Tokenizer) -> Int? {
        // 2,097,152 — far above any shipping vocab.
        let upperBound = 1 << 21
        for id in 0..<upperBound where tokenizer.convertIdToToken(id) == nil {
            return id
        }
        return nil
    }

    /// The tokenizer's control dialect. ChatML is the Qwen/Ornith family;
    /// K2-Horizon writes `<|ifm|…>` markers instead.
    static func resolveSpecialTokens(
        _ tokenizer: any Tokenizer,
        reasoningEffort: ModelReasoningEffort?
    ) throws -> ResolvedSpecialTokens {
        if let chatML = try? resolveChatMLTokens(tokenizer) { return chatML }
        return try resolveK2Tokens(tokenizer, effort: reasoningEffort ?? .high)
    }

    static func resolveChatMLTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw GFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        // Template rendering recognizes `<|im_start|>` as text; the id is
        // also what interior-checkpoint boundaries are located by.
        let imStart = try id(Self.imStartMark)
        let imEnd = try id(Self.imEndMark)
        let endOfText = try id("<|endoftext|>")
        let toolCallStart = try id("<tool_call>")
        let toolCallEnd = try id("</tool_call>")
        let toolResponse = try id("<tool_response>")
        let toolResponseEnd = try id("</tool_response>")
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        return ResolvedSpecialTokens(
            dialect: .chatML,
            imStartID: imStart,
            bosID: endOfText,
            eosID: endOfText,
            padID: endOfText,
            endOfTurnID: imEnd,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolCallsStartID: nil,
            toolCallsEndID: nil,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            thinkStartMark: "<think>",
            thinkEndMark: "</think>",
            toolCallStartMark: "<tool_call>",
            toolCallEndMark: "</tool_call>",
            toolCallsStartMark: "",
            toolCallsEndMark: "",
            stopTokenIDs: [imEnd, endOfText],
            // At least the model's padded embedding/lm_head rows; larger when
            // the tokenizer's own vocab (derived from `convertIdToToken`)
            // exceeds them.
            vocabSize: max(
                Self.derivedVocabSize(tokenizer) ?? 0,
                Self.paddedLogitsVocabSize))
    }

    /// K2-Horizon's dialect: `<|ifm|im_start|>` turns, `<ifm|tool_call>` tool
    /// blocks and `<ifm|think>` reasoning. The tool-call wrapper
    /// (`<ifm|tool_calls>`) is left to `K2ToolCallParser`; the streaming
    /// decoder buffers the inner call.
    static func resolveK2Tokens(
        _ tokenizer: any Tokenizer,
        effort: ModelReasoningEffort
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw GFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        let imStart = try id("<|ifm|im_start|>")
        let imEnd = try id("<|ifm|im_end|>")
        let endOfText = try id("<|ifm|endoftext|>")
        let toolCallStart = try id("<ifm|tool_call>")
        let toolCallEnd = try id("</ifm|tool_call>")
        let toolResponse = try id("<tool_response>")
        let toolResponseEnd = try id("</tool_response>")
        // The generation prompt's tag is chosen by effort: high opens
        // `<ifm|think>`, medium `<ifm|think_fast>`, low `<ifm|think_faster>`.
        // The decoder has to follow the tag the prompt actually used.
        let tag: String
        switch effort {
        case .medium: tag = "think_fast"
        case .low: tag = "think_faster"
        case .high, .xhigh: tag = "think"
        }
        let thinkStart = try id("<ifm|\(tag)>")
        let thinkEnd = try id("</ifm|\(tag)>")
        return ResolvedSpecialTokens(
            dialect: .k2,
            imStartID: imStart,
            bosID: endOfText,
            eosID: endOfText,
            padID: endOfText,
            endOfTurnID: imEnd,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolCallsStartID: try? id("<ifm|tool_calls>"),
            toolCallsEndID: try? id("</ifm|tool_calls>"),
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            thinkStartMark: "<ifm|\(tag)>",
            thinkEndMark: "</ifm|\(tag)>",
            toolCallStartMark: "<ifm|tool_call>",
            toolCallEndMark: "</ifm|tool_call>",
            toolCallsStartMark: "<ifm|tool_calls>",
            toolCallsEndMark: "</ifm|tool_calls>",
            stopTokenIDs: [imEnd, endOfText],
            vocabSize: max(Self.derivedVocabSize(tokenizer) ?? 0, 250_624))
    }
}
