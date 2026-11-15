import Foundation

/// Row addressing for Qwen3.8-Flash-Next's n-gram (PLE) embedding table.
///
/// Each token gathers `headsPerNgram * (ngramSize - 1)` rows — 16 in
/// production — from a ~320M-row table. The row ids are pure integer math over
/// token ids, which is what makes this a good fit for SSD streaming: a row
/// depends only on the token and its predecessors, never on router output or
/// any activation, so the next token's rows can be resolved and prefetched the
/// moment it is sampled.
///
/// Matches the reference implementation (`mlx_qwen4exp/ple.py`), including two
/// details that are easy to get wrong:
///
/// - The `cut` latches. Walking back from the current token, as soon as a slot
///   is missing or is EOS, that slot *and every older one* read as EOS.
/// - A token that is itself EOS does **not** cut its own context: `ctx[0]` is
///   always the raw token.
///
/// One mixed value serves all heads of an n-gram order; the heads differ only
/// by modulus and offset. Computing a separate mix per head would be both
/// slower and wrong.
public struct PLEHash: Sendable {
    public let multipliers: [UInt64]
    public let offsets: [UInt64]
    public let vocabSizes: [UInt64]
    public let ngramSize: Int
    public let headsPerNgram: Int
    public let eosTokenID: Int32

    /// Total rows gathered per token: one head set per n-gram order from 2 up
    /// to `ngramSize`.
    public var headCount: Int { headsPerNgram * (ngramSize - 1) }

    public init(
        multipliers: [UInt64], offsets: [UInt64], vocabSizes: [UInt64],
        ngramSize: Int, headsPerNgram: Int, eosTokenID: Int32
    ) {
        precondition(
            multipliers.count >= ngramSize,
            "need one multiplier per n-gram slot")
        precondition(
            offsets.count == vocabSizes.count,
            "offsets and vocab sizes must pair up")
        precondition(
            offsets.count == headsPerNgram * (ngramSize - 1),
            "expected one head set per n-gram order")
        precondition(vocabSizes.allSatisfy { $0 > 0 }, "vocab sizes must be > 0")
        self.multipliers = multipliers
        self.offsets = offsets
        self.vocabSizes = vocabSizes
        self.ngramSize = ngramSize
        self.headsPerNgram = headsPerNgram
        self.eosTokenID = eosTokenID
    }

    /// Row ids for one token.
    ///
    /// - Parameters:
    ///   - context: the token followed by its predecessors, nearest first
    ///     (`context[0]` is the token itself, `context[s]` is `s` positions
    ///     back). Shorter than `ngramSize` is fine — missing slots behave as
    ///     absent, exactly like a `-1` in the reference.
    /// - Returns: `headCount` row ids, ordered by n-gram order then head.
    public func rows(context: [Int32]) -> [UInt32] {
        // Resolve the context window with the latching cut.
        var resolved = [Int64](repeating: Int64(eosTokenID), count: ngramSize)
        resolved[0] = context.isEmpty ? Int64(eosTokenID) : Int64(context[0])
        var cut = false
        for s in 1..<ngramSize {
            let raw: Int64 = s < context.count ? Int64(context[s]) : -1
            let t = cut ? -1 : raw
            cut = cut || t < 0 || t == Int64(eosTokenID)
            resolved[s] = cut ? Int64(eosTokenID) : t
        }

        // Wrapping uint64 multiply, one term per slot.
        var terms = [UInt64](repeating: 0, count: ngramSize)
        for s in 0..<ngramSize {
            terms[s] = UInt64(bitPattern: resolved[s]) &* multipliers[s]
        }

        var out = [UInt32](repeating: 0, count: headCount)
        for order in 2...ngramSize {
            var mixed = terms[0]
            for j in 1..<order { mixed ^= terms[j] }
            let base = (order - 2) * headsPerNgram
            for h in 0..<headsPerNgram {
                // The row is truncated to 32 bits, matching the reference's
                // `(int32_t) value` reinterpretation.
                let row = (mixed % vocabSizes[base + h]) &+ offsets[base + h]
                out[base + h] = UInt32(truncatingIfNeeded: row)
            }
        }
        return out
    }

    /// Row ids for a whole sequence, resolving each token's context from the
    /// sequence itself plus any carried predecessors.
    public func rows(tokens: [Int32], previous: [Int32] = []) -> [[UInt32]] {
        tokens.indices.map { i in
            var context: [Int32] = [tokens[i]]
            for s in 1..<ngramSize {
                let back = i - s
                if back >= 0 {
                    context.append(tokens[back])
                } else {
                    // Reach into the carried predecessors, nearest first.
                    let p = previous.count + back
                    context.append(p >= 0 ? previous[p] : -1)
                }
            }
            return rows(context: context)
        }
    }
}
