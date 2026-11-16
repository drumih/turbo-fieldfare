import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// The dense-exactness window for Qwen Sparse Attention. This boundary decides
/// whether dense attention is correct or is silently attending to a subset of
/// the context, so the off-by-one at its edge matters more than most.
@Suite("QSA exactness")
struct QSAExactnessTests {
    private static let production = QSAExactness(
        ArchConfig.qwen38FlashNext.sparseIndexer)

    @Test("Production geometry keeps 512 complete blocks")
    func keptBlocks() {
        #expect(Self.production.budget == 2048)
        #expect(Self.production.compressRatio == 4)
        #expect(Self.production.keptBlocks == 512)
    }

    @Test("The window extends past the budget by the tail block")
    func windowIncludesTail() {
        // 2051, not 2048: the incomplete tail block is always kept, so a query
        // may see up to compressRatio-1 keys beyond the last complete block.
        // Gating on `budget` would wrongly reject three exact positions.
        #expect(Self.production.maximumExactVisibleKeys == 2051)
        #expect(Self.production.indexerRequiredFromContext == 2052)
    }

    @Test("The boundary matches the value the upstream card cites")
    func matchesUpstream() {
        // The model card states the mask becomes quadratic above 2,051 tokens.
        // Derived here from block accounting alone, so agreement is a genuine
        // cross-check of the geometry rather than a copied constant.
        #expect(Self.production.isDenseExact(visibleKeys: 2051))
        #expect(!Self.production.isDenseExact(visibleKeys: 2052))
    }

    @Test("Exactness holds across the window and fails past it")
    func acrossTheWindow() {
        let q = Self.production
        for v in [0, 1, 4, 1000, 2048, 2049, 2050, 2051] {
            #expect(q.isDenseExact(visibleKeys: v), "visible \(v)")
        }
        for v in [2052, 2053, 4096, 262_144] {
            #expect(!q.isDenseExact(visibleKeys: v), "visible \(v)")
        }
    }

    @Test("A context is exact when its longest query is")
    func contextGate() {
        let q = Self.production
        #expect(q.isDenseExact(forContext: 2051))
        #expect(!q.isDenseExact(forContext: 2052))
        // Degenerate contexts are exact, not a precondition failure.
        #expect(q.isDenseExact(forContext: 0))
    }

    @Test("Past the window the runtime refuses rather than attending densely")
    func refusalCarriesTheNumbers() {
        // The gate exists because dense attention past the window is not
        // slow, it is wrong with nothing downstream reporting it. A caller
        // that hits this needs both numbers to know what to shorten.
        let error = QSAIndexerRequired(visibleKeys: 2052, exactWindow: 2051)
        #expect(error.description.contains("2052"))
        #expect(error.description.contains("2051"))
        #expect(error.description.contains("--max-new"))
    }

    @Test("The refusal names the missing indexer, not an unimplemented feature")
    func refusalNamesTheRealCause() {
        // The indexer is implemented; this error can only come from a family
        // that declares one which was never built. The old text said the
        // indexer did not exist and sent the operator to shorten a prompt for
        // what would be a load defect.
        let error = QSAIndexerRequired(visibleKeys: 2052, exactWindow: 2051)
        #expect(error.description.contains("was not built"))
        #expect(!error.description.contains("not implemented"))
    }

    @Test("The window tracks the geometry rather than being hardcoded")
    func otherGeometries() {
        // compress 1: every block is complete, so the window is the budget.
        #expect(
            QSAExactness(budget: 64, compressRatio: 1)
                .maximumExactVisibleKeys == 64)
        // compress 8: seven tail keys ride along past 512 complete blocks.
        let wide = QSAExactness(budget: 512, compressRatio: 8)
        #expect(wide.keptBlocks == 64)
        #expect(wide.maximumExactVisibleKeys == 64 * 8 + 7)
    }
}
