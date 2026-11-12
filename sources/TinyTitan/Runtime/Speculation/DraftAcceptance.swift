/// Which drafted tokens a W-row verify pass accepts, for any sampler.
///
/// A pass runs rows `0..<W`: row 0 takes the boundary token (already emitted),
/// row `i > 0` takes draft `i - 1`, and row `i`'s logits predict the token after
/// it. Scalar decode would have emitted `t0 = sample(row 0)`, then -- only if
/// `t0` equals draft 0 -- `t1 = sample(row 1)`, and so on. This routine replays
/// exactly that: it samples row `i` with the history scalar decode would have
/// had at that point (the committed history plus drafts `0..<i`) and stops at the
/// first sample that differs from its draft. That sample is the correction, and
/// every row after it is discarded unread.
///
/// Because each emitted token is the target's own sample given an
/// already-accepted prefix, the output distribution is the target's, whatever
/// the sampler. With a fixed `seed` the sampler's seed is a function of the
/// position alone (`Sampler.seedFor`), so the emitted text is byte-identical to
/// scalar decode -- provided the rows' logits are, which is task 3's job. This
/// is what lets speculation run under the default presence penalty instead of
/// only under `isPureGreedy` (docs/plan-wide-verify.md, task 4).
public enum DraftAcceptance {
    public struct Outcome: Sendable, Equatable {
        /// Tokens to emit, in order: the accepted drafts and then the
        /// target's own token for the first row that was not accepted (or
        /// for the last row, when every draft was accepted).
        public let emitted: [Int32]
        public let acceptedDrafts: Int

        public init(emitted: [Int32], acceptedDrafts: Int) {
            self.emitted = emitted
            self.acceptedDrafts = acceptedDrafts
        }
    }

    /// - Parameters:
    ///   - drafts: the `W - 1` drafted tokens the pass verified.
    ///   - sampleRow: samples row `row` given the drafts accepted before it
    ///     (`drafts[0..<row]`), which the caller appends to its committed
    ///     history for the penalties, at position `boundary + row`.
    public static func resolve(
        drafts: ArraySlice<Int32>,
        sampleRow: (_ row: Int, _ acceptedBefore: ArraySlice<Int32>) throws -> Int32
    ) rethrows -> Outcome {
        var emitted: [Int32] = []
        emitted.reserveCapacity(drafts.count + 1)
        let base = drafts.startIndex
        for row in 0...drafts.count {
            let token = try sampleRow(row, drafts[base..<(base + row)])
            emitted.append(token)
            guard row < drafts.count, token == drafts[base + row] else {
                return Outcome(emitted: emitted, acceptedDrafts: row)
            }
        }
        // Unreachable: the last row always returns above.
        return Outcome(emitted: emitted, acceptedDrafts: drafts.count)
    }
}
