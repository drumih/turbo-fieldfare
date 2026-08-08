import Foundation

/// Streaming detokenizer for K3 generation loops, the K3 counterpart of the
/// house `GFDetokenizer`.
///
/// `K3Tokenizer.decode` works on whole id lists, so `push` re-decodes the
/// accumulated ids and emits the suffix delta against what was already
/// emitted (byte-wise prefix compare, resyncing on the rare decoder rewrite,
/// same as the house detokenizer). Byte-level BPE can split one codepoint
/// across tokens; a trailing U+FFFD run in the decoded text means "tail bytes
/// incomplete", so those characters are held back until more tokens arrive or
/// `flush` is called. Special ids decode to their literal names, which is
/// what `K3ToolCallParser` consumes.
public struct K3Detokenizer: Sendable {
    private let tokenizer: K3Tokenizer
    private var ids: [Int] = []
    private var emitted = ""

    public init(tokenizer: K3Tokenizer) {
        self.tokenizer = tokenizer
    }

    public mutating func push(_ id: Int32) -> String {
        ids.append(Int(id))
        var current = tokenizer.decode(ids)
        // Hold back a trailing replacement-char run: it marks tail bytes that
        // a later token may still complete into a codepoint.
        while current.hasSuffix("\u{FFFD}") {
            current.removeLast()
        }
        return commitDelta(current)
    }

    public mutating func flush() -> String {
        commitDelta(tokenizer.decode(ids))
    }

    private mutating func commitDelta(_ current: String) -> String {
        let currentUTF8 = current.utf8
        var boundary = currentUTF8.startIndex
        for byte in emitted.utf8 {
            guard boundary != currentUTF8.endIndex,
                  currentUTF8[boundary] == byte else {
                // Decoder altered the prefix — extremely rare in append-only
                // streams. Resync rather than emit garbage.
                emitted = current
                return ""
            }
            currentUTF8.formIndex(after: &boundary)
        }
        let delta = String(current[boundary...])
        emitted = current
        return delta
    }
}
