import Darwin
import Foundation

/// The expert-streaming statistics a run accumulates, promoted from the 6.x
/// runtime's `ExpertCacheTypes.swift` (whose shipped variant carried a
/// byte-identical copy) because `EngineRunner.expertStreamingStatistics()`
/// returns it. The latency bucket bounds moved with it: the struct's
/// percentile query was the only caller of the streamer's static, so the
/// definition now lives beside its only consumer and the engine copies are
/// gone. `adding` is `package` — the engine modules merge per-streamer stats
/// with it; nothing else about the type changed.
public struct ExpertStreamingStatistics: Sendable, Equatable {
    public let plans: UInt64
    public let requestedExperts: UInt64
    public let hits: UInt64
    public let misses: UInt64
    public let bytesRead: UInt64
    public let readOperations: UInt64
    public let evictions: UInt64
    public let reloads: UInt64
    public let loadBatches: UInt64
    public let totalLoadNanos: UInt64
    public let maximumLoadNanos: UInt64
    public let latencyHistogram: [UInt64]
    public let residentSlots: Int
    public let loadingSlots: Int
    public let pinnedSlots: Int
    public let peakLoadingSlots: Int

    public init(
        plans: UInt64,
        requestedExperts: UInt64,
        hits: UInt64,
        misses: UInt64,
        bytesRead: UInt64,
        readOperations: UInt64,
        evictions: UInt64,
        reloads: UInt64,
        loadBatches: UInt64,
        totalLoadNanos: UInt64,
        maximumLoadNanos: UInt64,
        latencyHistogram: [UInt64],
        residentSlots: Int,
        loadingSlots: Int,
        pinnedSlots: Int,
        peakLoadingSlots: Int
    ) {
        self.plans = plans
        self.requestedExperts = requestedExperts
        self.hits = hits
        self.misses = misses
        self.bytesRead = bytesRead
        self.readOperations = readOperations
        self.evictions = evictions
        self.reloads = reloads
        self.loadBatches = loadBatches
        self.totalLoadNanos = totalLoadNanos
        self.maximumLoadNanos = maximumLoadNanos
        self.latencyHistogram = latencyHistogram
        self.residentSlots = residentSlots
        self.loadingSlots = loadingSlots
        self.pinnedSlots = pinnedSlots
        self.peakLoadingSlots = peakLoadingSlots
    }

    public var hitRate: Double {
        requestedExperts == 0 ? 0 : Double(hits) / Double(requestedExperts)
    }

    /// An upper-bound estimate from a fixed, bounded power-of-two histogram.
    public func loadLatencyPercentile(_ percentile: Double) -> UInt64 {
        guard loadBatches > 0 else { return 0 }
        let clamped = min(1, max(0, percentile))
        let rank = max(UInt64(1), UInt64(ceil(clamped * Double(loadBatches))))
        var cumulative: UInt64 = 0
        for (index, count) in latencyHistogram.enumerated() {
            cumulative &+= count
            if cumulative >= rank {
                return Self.latencyBucketUpperBound(index: index)
            }
        }
        return UInt64.max
    }

    /// The upper bound a histogram bucket stands for: 125 µs doubling per
    /// bucket, 16 buckets, and `UInt64.max` for the overflow bucket. Moved
    /// here from `PreadExpertStreamer` with the struct.
    static func latencyBucketUpperBound(index: Int) -> UInt64 {
        guard index < 16 else { return UInt64.max }
        return 125_000 << UInt64(index)
    }

    public static let zero = ExpertStreamingStatistics(
        plans: 0, requestedExperts: 0, hits: 0, misses: 0,
        bytesRead: 0, readOperations: 0, evictions: 0, reloads: 0,
        loadBatches: 0, totalLoadNanos: 0, maximumLoadNanos: 0,
        latencyHistogram: [UInt64](repeating: 0, count: 17),
        residentSlots: 0, loadingSlots: 0, pinnedSlots: 0, peakLoadingSlots: 0)

    package func adding(_ other: ExpertStreamingStatistics) -> ExpertStreamingStatistics {
        ExpertStreamingStatistics(
            plans: plans &+ other.plans,
            requestedExperts: requestedExperts &+ other.requestedExperts,
            hits: hits &+ other.hits,
            misses: misses &+ other.misses,
            bytesRead: bytesRead &+ other.bytesRead,
            readOperations: readOperations &+ other.readOperations,
            evictions: evictions &+ other.evictions,
            reloads: reloads &+ other.reloads,
            loadBatches: loadBatches &+ other.loadBatches,
            totalLoadNanos: totalLoadNanos &+ other.totalLoadNanos,
            maximumLoadNanos: max(maximumLoadNanos, other.maximumLoadNanos),
            latencyHistogram: zip(latencyHistogram, other.latencyHistogram)
                .map { $0 &+ $1 },
            residentSlots: residentSlots + other.residentSlots,
            loadingSlots: loadingSlots + other.loadingSlots,
            pinnedSlots: pinnedSlots + other.pinnedSlots,
            peakLoadingSlots: max(peakLoadingSlots, other.peakLoadingSlots))
    }

    public func subtracting(_ baseline: ExpertStreamingStatistics) -> ExpertStreamingStatistics {
        func delta(_ value: UInt64, _ base: UInt64) -> UInt64 {
            value >= base ? value - base : 0
        }
        return ExpertStreamingStatistics(
            plans: delta(plans, baseline.plans),
            requestedExperts: delta(requestedExperts, baseline.requestedExperts),
            hits: delta(hits, baseline.hits),
            misses: delta(misses, baseline.misses),
            bytesRead: delta(bytesRead, baseline.bytesRead),
            readOperations: delta(readOperations, baseline.readOperations),
            evictions: delta(evictions, baseline.evictions),
            reloads: delta(reloads, baseline.reloads),
            loadBatches: delta(loadBatches, baseline.loadBatches),
            totalLoadNanos: delta(totalLoadNanos, baseline.totalLoadNanos),
            maximumLoadNanos: maximumLoadNanos,
            latencyHistogram: zip(latencyHistogram, baseline.latencyHistogram)
                .map { delta($0, $1) },
            residentSlots: residentSlots,
            loadingSlots: loadingSlots,
            pinnedSlots: pinnedSlots,
            peakLoadingSlots: peakLoadingSlots)
    }
}
