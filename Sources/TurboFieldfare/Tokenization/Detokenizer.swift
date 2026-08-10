import Foundation
import Tokenizers

/// Streaming detokenizer for generation loops.
///
/// Emits each token's own contribution to the output as it arrives. Two
/// properties make that safe:
///
/// 1. The Gemma decoder sequence (`Replace`, `ByteFallback`, `Fuse`) is
///    position-independent, so the decode of a token stream is the
///    concatenation of its per-token fragments. `GemmaDecoding` reproduces that
///    sequence without HF's `clean_up_tokenization_spaces` pass, which is the
///    one stage that rewrites already-decoded text and is wrong for this
///    tokenizer anyway (see `GemmaDecoding`).
/// 2. BPE byte fallback splits a multi-byte codepoint across several tokens, so
///    a run of `<0xXX>` tokens is held until its bytes form valid UTF-8. That
///    keeps a split codepoint whole instead of streaming replacement characters
///    (issue #58).
///
/// Cost is O(1) per token and independent of how much has already been
/// generated. The previous implementation re-decoded the entire accumulated
/// token list on every push, which made a generation O(n²) — roughly 3.6·10⁹
/// dictionary lookups over the app's 64K-token budget — and compared the result
/// against the full emitted prefix to recover a delta.
struct GFDetokenizer {
    @usableFromInline let tokenizer: any Tokenizer
    /// Bytes of an in-flight byte-fallback run that is not yet valid UTF-8.
    @usableFromInline var pendingBytes: [UInt8] = []
    @usableFromInline var specials: GemmaSpecialTokenFilter

    init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer.tokenizer
        self.specials = GemmaSpecialTokenFilter(tokenizer: tokenizer.tokenizer)
    }

    /// Text contributed by `id`, ready to append to the stream.
    ///
    /// Returns `""` while a byte-fallback run is still incomplete; those bytes
    /// come out with the token that completes them, or at `flush()`.
    mutating func push(_ id: Int32) -> String {
        guard let token = tokenizer.convertIdToToken(Int(id)) else { return "" }

        if let byte = GemmaDecoding.byteValue(token) {
            pendingBytes.append(byte)
            guard let assembled = GemmaDecoding.assembleBytes(pendingBytes) else {
                return ""
            }
            pendingBytes.removeAll(keepingCapacity: true)
            return assembled
        }

        var text = drainPendingBytes()
        if !specials.isSpecial(Int(id)) {
            text += GemmaDecoding.fragment(token)
        }
        return text
    }

    /// Remainder held back at a stop boundary.
    mutating func flush() -> String {
        drainPendingBytes()
    }

    /// Close an in-flight byte-fallback run.
    ///
    /// A run that never became valid UTF-8 is decoded leniently, matching what a
    /// batch `decode` of the same IDs produces, rather than being dropped.
    @usableFromInline
    mutating func drainPendingBytes() -> String {
        guard !pendingBytes.isEmpty else { return "" }
        let text = String(decoding: pendingBytes, as: UTF8.self)
        pendingBytes.removeAll(keepingCapacity: true)
        return text
    }
}
