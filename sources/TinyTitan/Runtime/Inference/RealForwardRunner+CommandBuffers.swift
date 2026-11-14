import Foundation
import Metal
import TinyTitanCore

// Command-buffer completion plumbing: the wait helper, the routed command whose
// completion is deferred to the next layer, and the diagnostic completion clock.
// Split out of RealForwardRunner.swift (2026-09-28) under the 500-line-per-file
// rule as pure code motion.

extension RealForwardRunner {
    var nextLayerPredictionEnabled: Bool {
        prefetchTraceFD >= 0 || predictivePrefetch != nil
    }

    // MARK: - Routing trace (TINYTITAN_ROUTE_TRACE)

    nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw ModelError.commandBufferFailed(detail: String(describing: err))
        }
    }

    /// A routed-expert command whose completion is deferred to the next layer.
    struct PendingRoutedCommand {
        let cb: MTLCommandBuffer
        let sharedCB: MTLCommandBuffer?
        let phase1HitCB: MTLCommandBuffer?
        /// The miss phase-1 fixup committed early, when the load split its
        /// reads and the front range landed before the tail.
        let phase1MissCB: MTLCommandBuffer?
        let expertLease: RoutedExpertLease?
        let storageOperation: RoutedExpertLoadOperation?
        let overlapCompletionClock: CommandCompletionClock?
        let expectedOverlapCompletions: Int
        let kernelRole: String
        let encodeAndCommitNanos: UInt64
    }

    /// Diagnostic-only completion clock used to measure the I/O tail left
    /// after already-runnable GPU work. It is allocated only with
    /// TINYTITAN_RUNNER_STATS, never in the production hot path.
    /// unchecked-invariant: completion timestamps are mutated and read only
    /// while holding `lock`.
    final class CommandCompletionClock: @unchecked Sendable {
        let lock = NSLock()
        var completionCount = 0
        var latestCompletion: UInt64 = 0

        func track(_ commandBuffer: MTLCommandBuffer) {
            commandBuffer.addCompletedHandler { [self] _ in
                lock.lock()
                completionCount += 1
                latestCompletion = max(
                    latestCompletion,
                    clock_gettime_nsec_np(CLOCK_UPTIME_RAW))
                lock.unlock()
            }
        }

        func latest(expected: Int) -> UInt64? {
            lock.lock()
            defer { lock.unlock() }
            guard completionCount == expected else { return nil }
            return latestCompletion
        }
    }
}
