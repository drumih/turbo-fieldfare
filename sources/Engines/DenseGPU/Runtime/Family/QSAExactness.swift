import Foundation
import TinyTitanCore

/// When dense attention is bit-exact for Qwen Sparse Attention.
///
/// QSA scores mean-pooled key blocks with a small indexer head set and keeps
/// the `budget / compressRatio` highest-scoring complete blocks, plus the
/// incomplete tail block always. Below a certain context that selection is
/// vacuous -- every visible block is kept -- and dense attention computes
/// exactly the same thing the sparse path would.
///
/// That window is worth pinning as a type rather than a scattered comparison,
/// because it is the boundary between "correct" and "silently attending to a
/// subset of the context". A query beyond it needs the real indexer; running
/// dense there is not slow, it is wrong in a way nothing downstream reports.
///
/// Derived from the geometry, and independently equal to the 2,051 the
/// upstream model card cites for its mask becoming quadratic: at budget 2048
/// and compress 4 the indexer keeps 512 complete blocks, so a query stays
/// exact while `visibleKeys / 4 <= 512`, i.e. through 2,051 visible keys.
public struct QSAExactness: Sendable, Equatable {
    public let budget: Int
    public let compressRatio: Int

    public init(budget: Int, compressRatio: Int) {
        precondition(budget > 0 && compressRatio > 0)
        self.budget = budget
        self.compressRatio = compressRatio
    }

    public init(_ config: SparseIndexerConfig) {
        self.init(budget: config.budget, compressRatio: config.compressRatio)
    }

    /// Complete blocks the indexer retains.
    public var keptBlocks: Int { budget / compressRatio }

    /// The largest number of visible keys for which dense attention is exact.
    ///
    /// Not simply `budget`: the tail block is always kept, so a query can see
    /// up to `compressRatio - 1` keys beyond the last complete block and
    /// still have everything selected.
    public var maximumExactVisibleKeys: Int {
        keptBlocks * compressRatio + (compressRatio - 1)
    }

    /// Whether a query seeing `visibleKeys` keys is exactly served by dense
    /// attention.
    public func isDenseExact(visibleKeys: Int) -> Bool {
        precondition(visibleKeys >= 0)
        return visibleKeys / compressRatio <= keptBlocks
    }

    /// Whether every query in a causal window of `tokens` is exactly served.
    /// The last query sees the most keys, so it decides the whole window.
    public func isDenseExact(forContext tokens: Int) -> Bool {
        tokens <= 0 || isDenseExact(visibleKeys: tokens)
    }

    /// The prompt-plus-generation length at which the indexer becomes
    /// mandatory. Callers gate on this rather than on `budget`, which would be
    /// off by the tail block.
    public var indexerRequiredFromContext: Int { maximumExactVisibleKeys + 1 }
}

/// Raised when a context outgrows the window where dense attention is exact
/// and the model's sparse indexer was not built for it.
///
/// This is deliberately an error and not a warning. Past the window, dense
/// attention does not degrade -- it attends to keys the trained model's
/// selection would have dropped, and produces fluent output that nothing
/// downstream flags. Refusing is the only honest behaviour when there is no
/// indexer to select with.
public struct QSAIndexerRequired: Error, CustomStringConvertible, Equatable {
    public let visibleKeys: Int
    public let exactWindow: Int

    public init(visibleKeys: Int, exactWindow: Int) {
        self.visibleKeys = visibleKeys
        self.exactWindow = exactWindow
    }

    public var description: String {
        "context reaches \(visibleKeys) tokens, past the \(exactWindow) where "
            + "dense attention matches this model's sparse selection exactly. "
            + "This model's QSA indexer was not built, and running dense here "
            + "would attend to keys the model would have dropped -- silently, "
            + "and with plausible output. Check the load diagnostics for the "
            + "indexer, or shorten the prompt or lower --max-new so prompt + "
            + "generation stays within \(exactWindow)."
    }
}
