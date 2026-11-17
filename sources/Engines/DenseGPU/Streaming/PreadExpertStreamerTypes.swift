import Darwin
import Foundation
import Synchronization
import Metal

// The streamer's destination holder and cache lease.
//
// Split out of `PreadExpertStreamer.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// Members whose callers stay behind widened from `private` to internal.
/// Raw staging pointers are allocated by Metal and remain valid until the
/// owning prefetch ring releases them. This wrapper makes that lifetime
/// invariant explicit at the scheduler boundary.
/// unchecked-invariant: the ring retains every backing MTLBuffer until this
/// request has reached a terminal state.
final class PrefetchDestinations: @unchecked Sendable {
    let values: [UnsafeMutableRawPointer]

    init(_ values: [UnsafeMutableRawPointer]) {
        self.values = values
    }
}

/// Pins exact slot generations until every GPU command using them completes.
/// Release is idempotent so error cleanup and normal command completion can
/// safely converge on the same lifetime operation.
/// unchecked-invariant: immutable slot metadata is published at init and the
/// only mutable release flag is guarded by `lock`; streamer state has its own lock.
final class ExpertCacheLease: @unchecked Sendable {
    private weak var streamer: PreadExpertStreamer?
    private let slots: [Int]
    private let generations: [UInt64]
    private let lock = NSLock()
    private var released = false

    init(
        streamer: PreadExpertStreamer,
        slots: [Int],
        generations: [UInt64]
    ) {
        self.streamer = streamer
        self.slots = slots
        self.generations = generations
    }

    func release() {
        lock.lock()
        guard !released else {
            lock.unlock()
            return
        }
        released = true
        lock.unlock()
        streamer?.unpin(slots: slots, generations: generations)
    }

    deinit { release() }
}
