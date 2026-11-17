/// Drafts by copying from the context: the longest recent n-gram that matches
/// the end of the token history proposes the tokens that followed it.
///
/// This is the draft source of the wide-verify plan (docs/plan-wide-verify.md,
/// task 5). It touches no weights and runs on the CPU between passes, so its
/// cost is a dictionary probe per n per proposal and a dictionary write per n
/// per token. Semantics are exactly those of `NgramIndex` in
/// `benchmark/draft_coverage_replay.py`, so the replay's coverage figures
/// describe what this type proposes:
///
/// - n runs from `maxN` down to `minN`; the first n with a prior occurrence of
///   the history's last n tokens wins.
/// - "Prior occurrence" is the most recent one whose continuation exists. The
///   history's own suffix is never registered as an occurrence of itself.
/// - The proposal is the tokens after that occurrence, up to `limit`, and may
///   run into the suffix itself (a repeating pattern drafts its next period).
public struct PromptLookupDrafter: Sendable {
    /// An n-gram of at most eight tokens, padded with -1 (never a token id),
    /// so different lengths never collide and nothing allocates per key.
    struct Key: Hashable, Sendable {
        let lanes: SIMD8<Int32>
    }

    public static let maximumN = 8
    public let minN: Int
    public let maxN: Int
    public private(set) var history: [Int32] = []
    /// n-gram -> index in `history` of the token that followed its most recent
    /// occurrence.
    private var continuation: [Key: Int] = [:]

    public init(minN: Int = 2, maxN: Int = 8) {
        precondition(
            1 <= minN && minN <= maxN && maxN <= Self.maximumN,
            "PromptLookupDrafter needs 1 <= minN <= maxN <= \(Self.maximumN)")
        self.minN = minN
        self.maxN = maxN
    }

    public mutating func reset() {
        history.removeAll(keepingCapacity: true)
        continuation.removeAll(keepingCapacity: true)
    }

    public mutating func append(_ token: Int32) {
        precondition(token >= 0, "token ids are non-negative")
        history.append(token)
        // Register the n-grams ending just before the new token: their
        // continuation (this token) now exists.
        let end = history.count - 1
        for n in minN...maxN where end - n >= 0 {
            continuation[key(endingBefore: end, length: n)] = end
        }
    }

    public mutating func append<S: Sequence>(contentsOf tokens: S) where S.Element == Int32 {
        for token in tokens { append(token) }
    }

    /// The draft for the next positions and the n it matched on (0 when there
    /// is no match). Empty when `limit` is not positive.
    public func propose(limit: Int) -> (tokens: ArraySlice<Int32>, matchedN: Int) {
        guard limit > 0 else { return ([], 0) }
        for n in stride(from: maxN, through: minN, by: -1) where history.count >= n {
            if let start = continuation[key(endingBefore: history.count, length: n)],
                start < history.count
            {
                let stop = min(history.count, start + limit)
                return (history[start..<stop], n)
            }
        }
        return ([], 0)
    }

    private func key(endingBefore end: Int, length: Int) -> Key {
        var lanes = SIMD8<Int32>(repeating: -1)
        for lane in 0..<length {
            lanes[lane] = history[end - length + lane]
        }
        return Key(lanes: lanes)
    }
}
