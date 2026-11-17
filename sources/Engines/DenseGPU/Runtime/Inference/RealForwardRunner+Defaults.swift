import Foundation

// Runner-wide tuning constants. Moved verbatim out of the class body (2026-10-20)
// under the 500-line-per-file rule; the values, comments and their order are
// unchanged, so this is pure code motion plus the `Self.` prefix at the call
// sites.

extension RealForwardRunner {
    enum Defaults {
        static let rdadviseBoundedMissCap = 12
        static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
        static let rdadviseAdaptiveMissCap = 12
        static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
        static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
        /// Prefill routed-tile shape. The shipped default is an eight-expert tile
        /// with a one-tile lookahead; both are overridable for measurement so a
        /// wider tile or a deeper pipeline can be A/B'd without a rebuild. The
        /// config's own initializer clamps experts to 1...16 and depth to >= 1.
        static let prefillRoutedTileSchedulerConfig: PrefillRoutedTileSchedulerConfig = {
            let environment = ProcessInfo.processInfo.environment
            // Depth 2 measured 49.8 s against 51.8 s over three order-rotated
            // rounds on a 2.2K prompt (all three faster); tile width 16 was noise.
            var config = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 2, tileExperts: 8)
            if let raw = environment["TINYTITAN_PREFILL_TILE_EXPERTS"], let value = Int(raw) {
                config = PrefillRoutedTileSchedulerConfig(
                    maxPendingDepth: config.maxPendingDepth, tileExperts: value)
            }
            if let raw = environment["TINYTITAN_PREFILL_TILE_DEPTH"], let value = Int(raw) {
                config = PrefillRoutedTileSchedulerConfig(
                    maxPendingDepth: value, tileExperts: config.tileExperts)
            }
            return config
        }()
    }
}
