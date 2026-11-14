// The admission queue and the per-runner counters the server infers from.
//
// Split out of ServerInference.swift (2026-09-27) under the 500-line-per-file
// rule; declarations and their doc comments are unchanged, except that
// `private` became internal where a declaration now lives beside an extension
// in another file.
import CryptoKit
import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

public actor ServerCoordinator {

    let queueLimit: Int
    /// How many generations may run at once. One is the historical
    /// single-generation server; more lets the batched slots through while the
    /// excess still queues. The engine's `ForwardStepGate` keeps their forward
    /// passes from interleaving.
    let width: Int
    var admittedCount = 0
    var activeCount = 0
    var waiters: [SuspensionSlot] = []
    var shuttingDown = false
    /// Raised for the duration of every client generation. The side-engine
    /// reads it to choose its width: one thread while a person is waiting,
    /// four in the gaps.
    package nonisolated let generating = GenerationSignal()

    public init(queueLimit: Int, width: Int = 1) {
        self.queueLimit = queueLimit
        self.width = max(1, width)
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await runPreparing(
            onQueued: onQueued,
            prepare: { () },
            operation: { _ in try await operation() })
    }

    func runPreparing<Prepared: Sendable, T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        prepare: @escaping @Sendable () async throws -> Prepared,
        operation: @escaping @Sendable (Prepared) async throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        // S6: at most `width` running and `queueLimit` queued behind them, so
        // `width + queueLimit` admitted. Width 1 reproduces the original
        // single-generation bound exactly.
        guard admittedCount < width + queueLimit else {
            // Shed load rather than queue without bound.
            throw ServerRequestError.queueFull
        }
        admittedCount += 1
        defer { admittedCount -= 1 }

        let prepared = try await prepare()
        try Task.checkCancellation()
        try await acquire(onQueued: onQueued)
        defer { release() }
        generating.enter()
        defer { generating.leave() }
        return try await operation(prepared)
    }

    func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        guard let slot = try queueOrAdmit(onQueued: onQueued) else { return }
        await withTaskCancellationHandler {
            await withCheckedContinuation { slot.handOver($0) }
        } onCancel: {
            Task { await self.cancel(slot) }
        }
        if slot.received == .cancel { throw CancellationError() }
        if Task.isCancelled {
            // Admitted and cancelled in the same breath: the width is held now, so
            // hand it back before throwing rather than leaking a slot.
            release()
            throw CancellationError()
        }
    }

    /// Admit the caller, or queue it on a fresh slot, in one step with no
    /// suspension. Nil means admitted. Registering in the same step as the check is
    /// what keeps a `release` from landing between them, and the slot is the only
    /// thing the continuation closure may touch: see `SuspensionSlot`.
    private func queueOrAdmit(onQueued: @Sendable () -> Void) throws -> SuspensionSlot? {
        if activeCount < width {
            activeCount += 1
            return nil
        }
        guard waiters.count < queueLimit else { throw ServerRequestError.queueFull }
        onQueued()
        let slot = SuspensionSlot()
        waiters.append(slot)
        return slot
    }

    func cancel(_ slot: SuspensionSlot) {
        if let index = waiters.firstIndex(where: { $0 === slot }) {
            waiters.remove(at: index)
        }
        slot.signal(.cancel)
    }

    func release() {
        if waiters.isEmpty {
            activeCount = max(0, activeCount - 1)
        } else {
            waiters.removeFirst().signal(.wake)
        }
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.signal(.cancel)
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { activeCount > 0 }
    /// Running generations, for tests and the readiness view.
    public var runningCount: Int { activeCount }
    /// The admission width this coordinator was built with.
    public var concurrencyWidth: Int { width }
}
