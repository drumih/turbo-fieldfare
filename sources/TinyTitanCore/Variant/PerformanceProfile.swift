import Foundation

/// A variant's performance constants (7.0 spec §7), held in the variant target.
/// A tuning change is a source change in one variant, reviewed against that
/// variant's benchmark and golden output. There is no string-keyed
/// `(modelID, bits)` table that can silently fall back to a family default; a
/// missing constant is a compile error in the variant. Environment overrides
/// are debug-only and compile out of release builds.
public struct PerformanceProfile: Sendable, Equatable {
    public let expertCacheBytes: Int
    public let prefetchDepth: Int
    public let prefillChunk: Int
    public let routerTopKSimd: Bool
    public let attentionSimdPartial: Bool
    public let keepCacheWired: Bool

    public init(
        expertCacheBytes: Int, prefetchDepth: Int, prefillChunk: Int,
        routerTopKSimd: Bool, attentionSimdPartial: Bool, keepCacheWired: Bool
    ) {
        self.expertCacheBytes = expertCacheBytes
        self.prefetchDepth = prefetchDepth
        self.prefillChunk = prefillChunk
        self.routerTopKSimd = routerTopKSimd
        self.attentionSimdPartial = attentionSimdPartial
        self.keepCacheWired = keepCacheWired
    }
}
