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
