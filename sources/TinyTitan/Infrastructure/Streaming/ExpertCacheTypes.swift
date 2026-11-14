//
//  ExpertCacheTypes.swift
//  TinyTitan
//
//  The value types the expert streaming stack speaks in: a cache plan, its
//  policy, layout and backend, the advice a read returns, and the statistics a
//  run accumulates. Split out of `PreadExpertStreamer.swift`, which owns the
//  reader that produces them.
//

import Darwin
import Foundation
import Metal
import Synchronization
import TinyTitanCore

public struct ExpertIOAdviceResult: Sendable, Equatable {
    public let requested: Int
    public let failed: Int
    public let calls: Int
    public let bytes: UInt64
    public let skipped: Int
    public let maxCallNanos: UInt64

    public init(
        requested: Int,
        failed: Int,
        calls: Int? = nil,
        bytes: UInt64 = 0,
        skipped: Int = 0,
        maxCallNanos: UInt64 = 0
    ) {
        self.requested = requested
        self.failed = failed
        self.calls = calls ?? requested
        self.bytes = bytes
        self.skipped = skipped
        self.maxCallNanos = maxCallNanos
    }

    public static func skipped(requested: Int, bytes: UInt64 = 0) -> ExpertIOAdviceResult {
        ExpertIOAdviceResult(
            requested: requested,
            failed: 0,
            calls: 0,
            bytes: bytes,
            skipped: requested)
    }

}

public struct ExpertCachePlan: Sendable, Equatable {
    /// K11: the layer the plan's pread offsets are computed against.
    ///
    /// NOTE: the streamer is bound to ONE layer file at construction, and
    /// `StreamLayout.expertOffset(layer: 0, ...)` is the branch that consults
    /// that file's per-layer `expertOffsets` table — so 0 is the correct value
    /// for the current per-layer design (passing the real layer would select
    /// the dense cross-layer formula and mis-offset). Callers must only pass a
    /// nonzero layer if the streamer ever serves a multi-layer file.
    public let layer: Int
    public let experts: [Int]
    public let assignedSlots: [Int]
    /// Slot incarnation captured when this plan reserved or hit each slot.
    /// A command may use the slot only while this generation still matches.
    public let assignedGenerations: [UInt64]
    public let misses: [Int]
    public let hits: Int

    public init(
        experts: [Int], assignedSlots: [Int],
        assignedGenerations: [UInt64], misses: [Int], hits: Int,
        layer: Int = 0
    ) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.assignedGenerations = assignedGenerations
        self.misses = misses
        self.hits = hits
        self.layer = layer
    }
}

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
                return PreadExpertStreamer.latencyBucketUpperBound(index: index)
            }
        }
        return UInt64.max
    }

    public static let zero = ExpertStreamingStatistics(
        plans: 0, requestedExperts: 0, hits: 0, misses: 0,
        bytesRead: 0, readOperations: 0, evictions: 0, reloads: 0,
        loadBatches: 0, totalLoadNanos: 0, maximumLoadNanos: 0,
        latencyHistogram: [UInt64](repeating: 0, count: 17),
        residentSlots: 0, loadingSlots: 0, pinnedSlots: 0, peakLoadingSlots: 0)

    func adding(_ other: ExpertStreamingStatistics) -> ExpertStreamingStatistics {
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

/// The two eviction orders a run may select. The experiment variants beside
/// them (aging-lfu, a decayed use count with a tunable half-life, and the two
/// env switches that reached them) all measured washes or losses and have been
/// removed; LFU is the shipped default, LRU the configured alternative.
public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
}

public enum ExpertIOBackend: String, Sendable {
    case pread
    case metal

    static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ExpertIOBackend {
        guard let raw = environment["TINYTITAN_EXPERT_IO_BACKEND"] else { return .pread }
        guard let backend = ExpertIOBackend(rawValue: raw) else {
            throw ModelError.internalInconsistency(
                detail: "unsupported TINYTITAN_EXPERT_IO_BACKEND '\(raw)'; allowed: pread, metal")
        }
        return backend
    }
}
