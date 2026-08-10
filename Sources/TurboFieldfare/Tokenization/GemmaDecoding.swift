import Foundation
import Tokenizers

/// The pinned Gemma 4 detokenization pipeline, without HF's cleanup pass.
///
/// `tokenizer.json` declares `decoder: Sequence[Replace("▁" -> " "),
/// ByteFallback, Fuse]`. swift-transformers runs that sequence and then applies
/// `clean_up_tokenization_spaces`, which defaults to **true** when the key is
/// absent — and Gemma's `tokenizer_config.json` omits it.
///
/// That cleanup pass is a legacy heuristic for whitespace tokenizers and is
/// wrong for a metaspace + byte-fallback BPE, which is lossless by
/// construction. It rewrites text the model deliberately produced:
///
/// ```text
/// "he said ' ok ' now"  ->  "he said'ok'now"
/// "step 1 . done"       ->  "step 1. done"
/// "    . indented"      ->  "   . indented"
/// ```
///
/// So `decode(encode(x)) != x`. It also breaks streaming: because the pass runs
/// over the whole string on every call, appending a token can rewrite text that
/// was already emitted, which no append-only stream can represent.
///
/// This type reproduces the declared decoder sequence and stops there. Both the
/// batch (`GFTokenizer.decode`) and streaming (`GFDetokenizer`) paths go through
/// it, so they agree by construction.
enum GemmaDecoding {
    /// `<0xXX>` byte-fallback token, e.g. `<0xE2>`.
    static func isByteFallback(_ token: String) -> Bool {
        token.count == 6
            && token.hasPrefix("<0x")
            && token.hasSuffix(">")
            && token.dropFirst(3).dropLast().allSatisfy { $0.isHexDigit }
    }

    static func byteValue(_ token: String) -> UInt8? {
        guard isByteFallback(token) else { return nil }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16)
    }

    /// One token's contribution to the output: the `Replace` decoder's
    /// `"▁" -> " "` substitution. Byte-fallback tokens are handled by
    /// `assembleBytes` instead — they only decode as a complete run.
    static func fragment(_ token: String) -> String {
        token.replacingOccurrences(of: sentencePieceUnderline, with: " ")
    }

    /// Assemble a run of byte-fallback tokens into text.
    ///
    /// Returns `nil` while the run is not yet valid UTF-8, which is how a
    /// codepoint split across several tokens stays whole instead of streaming
    /// out as replacement characters (issue #58).
    static func assembleBytes(_ bytes: [UInt8]) -> String? {
        guard !bytes.isEmpty else { return "" }
        return String(bytes: bytes, encoding: .utf8)
    }

    private static let sentencePieceUnderline = "▁"
}

/// Caches which token IDs `skipSpecialTokens` drops.
///
/// The library owns the added/special token set and does not expose it, so the
/// answer is probed once per distinct ID and memoized: a special token decodes
/// to nothing when skipped, and to its literal text when not. A generation
/// touches a few thousand distinct IDs, so the probe cost is bounded.
struct GemmaSpecialTokenFilter {
    private let tokenizer: any Tokenizer
    private var cache: [Int: Bool] = [:]

    init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    mutating func isSpecial(_ id: Int) -> Bool {
        if let cached = cache[id] { return cached }
        let skipped = tokenizer.decode(tokens: [id], skipSpecialTokens: true)
        let kept = tokenizer.decode(tokens: [id], skipSpecialTokens: false)
        let value = skipped.isEmpty && !kept.isEmpty
        cache[id] = value
        return value
    }
}
