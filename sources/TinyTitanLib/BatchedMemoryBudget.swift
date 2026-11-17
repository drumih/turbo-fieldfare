import Foundation
import TinyTitanCore

/// The headroom rule for batched serving.
///
/// Batching bounds peak KV to one sequence per slot; four slots is four
/// sequences' worth, and the wired routed-expert cache cannot be paged out to
/// rescue an over-commit. Width is therefore capped by what the worst-case
/// per-slot stores can actually hold beside the expert cache, not by the
/// requested number alone.
enum BatchedMemoryBudget {
    /// Bytes the slots may occupy: half of physical memory, less the wired
    /// expert cache. The other half is left for the weights, the allocator and
    /// the rest of the machine. Never negative.
    static func slotBudgetBytes(
        physicalMemory: UInt64,
        expertCacheBudgetBytes: Int
    ) -> Int {
        guard physicalMemory > 0 else { return Int.max }
        return max(0, Int(physicalMemory / 2) - max(0, expertCacheBudgetBytes))
    }

    /// The expert cache to hold back from the KV headroom.
    ///
    /// A dense family has no routed experts, so the profile's expert-cache
    /// budget is memory it will never allocate; subtracting it took 8 GiB off
    /// the headroom of a dense install and clamped a width that fits.
    static func expertCacheHeldBack(numExperts: Int, configured: Int) -> Int {
        numExperts > 0 ? max(0, configured) : 0
    }

    /// The width to budget before the memory clamp.
    ///
    /// A family with per-sequence runtime state that is not slot-keyed -- the
    /// QSA indexer and the PLE block on Qwen3.8-Flash-Next -- cannot run two
    /// sequences without cross-talk: the indexer's pooled keys are per layer,
    /// not per slot, and `resetSequence` only clears it on slot 0. Serving one
    /// at a time is the correct fallback until that state is slot-keyed.
    /// Returning one here is what makes the queueing honest rather than
    /// silently emitting another sequence's tokens.
    static func effectiveRequestedSlots(
        requested: Int,
        hasPerSequenceState: Bool
    ) -> Int {
        hasPerSequenceState ? 1 : max(1, requested)
    }

    /// The largest width whose worst-case stores fit the budget.
    ///
    /// Always at least one: the single-sequence path must stay available even
    /// when the model is too large for the budget to hold a second slot.
    static func effectiveSlots(
        requested: Int,
        perSlotBytes: Int,
        budgetBytes: Int
    ) -> Int {
        guard requested > 1 else { return max(1, requested) }
        guard perSlotBytes > 0 else {
            // Unknown size cannot be budgeted; trust the request rather than
            // silently pin it to one.
            return requested
        }
        // A zero budget is a real constraint (the expert cache already claims
        // the headroom), not an unknown: serve one sequence at a time.
        guard budgetBytes > 0 else { return 1 }
        return max(1, min(requested, budgetBytes / perSlotBytes))
    }
}
