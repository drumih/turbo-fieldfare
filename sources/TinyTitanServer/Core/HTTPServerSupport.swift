//
//  HTTPServerSupport.swift
//  TinyTitanServer
//
//  Types the server's handler and actor share: the SSE outbox, the child-channel
//  registry, and the small per-request/per-stream state holders.
//

import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TinyTitanCore

/// Bounded FIFO of pre-encoded SSE frames for one stream, with a cap that
/// fails the stream when a slow reader outruns the drainer (S4).
/// unchecked-invariant: every field is guarded by `lock`. The queue is written
/// from the generation task and drained from the event loop, which is exactly
/// why the lock is here rather than relying on loop confinement.
/// Internal rather than private so the cancellation ordering below can be
/// exercised directly: it is a scheduling property, not a routing one.
final class SSEOutbox: @unchecked Sendable {
    private let lock = NSLock()
    private var frames: [Data] = []
    private var pendingDrain: CheckedContinuation<Data?, Never>?
    private var closed = false
    private var overflowed = false
    private var abandoned = false
    private var closeAfterDrain = false
    /// Set by the cancellation handler. The handler runs *before* the
    /// continuation is installed when the task is already cancelled, so a flag
    /// is what the installing side can see; `pendingDrain` alone is nil at that
    /// moment and the continuation would park with nothing left to resume it.
    private var drainCancelled = false
    let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
    }

    var closeWhenDrained: Bool {
        lock.withLock { closeAfterDrain }
    }

    /// Enqueue a regular content frame. Returns false when the outbox is
    /// closed, already failed, or the cap is exceeded (slow reader).
    func enqueue(_ frame: Data) -> Bool {
        lock.withLock {
            guard !closed, !overflowed else { return false }
            if frames.count >= capacity {
                overflowed = true
                return false
            }
            push(frame)
            return true
        }
    }

    /// Enqueue terminal frames ([DONE] / error) and close the outbox. Later
    /// frames are rejected; the drainer writes everything already queued in
    /// order and then (when `closeWhenDrained`) closes the connection.
    func enqueueTerminal(_ frames: [Data], closeWhenDrained: Bool) {
        lock.withLock {
            guard !closed else { return }
            for frame in frames { push(frame) }
            closed = true
            if closeWhenDrained { closeAfterDrain = true }
            // A terminal with no frames still ends the stream, so a drainer parked
            // on `next()` must be released here: `push` resumes it only when it
            // has a frame to hand over. The responses and messages surfaces send
            // an empty terminal on cancellation, and without this the drainer
            // waited forever — leaving the request task, and the in-flight count
            // it decrements, outstanding.
            if frames.isEmpty, let continuation = pendingDrain {
                pendingDrain = nil
                continuation.resume(returning: nil)
            }
        }
    }

    /// Retire the outbox without emitting anything and without ending the HTTP
    /// response.
    ///
    /// For a streaming request refused *before* its SSE head exists: the error is
    /// written whole by the ordinary response paths, so a frame here would be a
    /// body with no head and the drainer's `end` would be a second one on the same
    /// request. Releasing a parked drainer is the same requirement as above.
    func abandon() {
        lock.withLock {
            guard !closed else { return }
            closed = true
            abandoned = true
            if let continuation = pendingDrain {
                pendingDrain = nil
                continuation.resume(returning: nil)
            }
        }
    }

    var isAbandoned: Bool { lock.withLock { abandoned } }

    /// Await the next frame; nil once the outbox is closed and drained.
    func next() async -> Data? {
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                lock.withLock {
                    if !frames.isEmpty {
                        continuation.resume(returning: frames.removeFirst())
                    } else if closed || drainCancelled {
                        // A frame or a terminal close that arrived after the
                        // cancellation is still delivered by the branches above;
                        // only an empty, open outbox ends the drain here.
                        continuation.resume(returning: nil)
                    } else {
                        pendingDrain = continuation
                    }
                }
            }
        } onCancel: {
            self.cancelPendingDrain()
        }
    }

    private func push(_ frame: Data) {
        if let continuation = pendingDrain {
            // Hand the frame directly to the awaiting drainer; do NOT also
            // append it to frames, or the drainer's next() would remove and
            // deliver the same frame a second time.
            pendingDrain = nil
            continuation.resume(returning: frame)
        } else {
            frames.append(frame)
        }
    }

    private func cancelPendingDrain() {
        lock.withLock {
            drainCancelled = true
            if let continuation = pendingDrain {
                pendingDrain = nil
                continuation.resume(returning: nil)
            }
        }
    }
}

final class ChildChannelRegistry: Sendable {
    private struct State {
        var channels: [ObjectIdentifier: Channel] = [:]
        var tasks: [UUID: Task<Void, Never>] = [:]
        var shuttingDown = false
    }

    private let state = Mutex(State())
    private let maximumChannels: Int

    init(maximumChannels: Int) {
        self.maximumChannels = maximumChannels
    }

    func insert(_ channel: Channel) {
        let shouldClose = state.withLock {
            guard !$0.shuttingDown else { return true }
            // S1: connection cap — reject beyond maximumChannels.
            if $0.channels.count >= maximumChannels {
                return true
            }
            $0.channels[ObjectIdentifier(channel)] = channel
            return false
        }
        if shouldClose {
            channel.close(promise: nil)
        }
    }

    func remove(_ channel: Channel) {
        _ = state.withLock {
            $0.channels.removeValue(forKey: ObjectIdentifier(channel))
        }
    }

    func startTask(
        _ operation: @escaping @Sendable () async -> Void
    ) -> Task<Void, Never> {
        state.withLock { state in
            let id = UUID()
            let task = Task { [self] in
                defer {
                    _ = self.state.withLock {
                        $0.tasks.removeValue(forKey: id)
                    }
                }
                await operation()
            }
            state.tasks[id] = task
            if state.shuttingDown {
                task.cancel()
            }
            return task
        }
    }

    func closeAll() async {
        let channels = state.withLock {
            $0.shuttingDown = true
            return Array($0.channels.values)
        }
        for channel in channels {
            try? await channel.close().get()
        }
        let tasks = state.withLock { Array($0.tasks.values) }
        for task in tasks {
            task.cancel()
        }
        for task in tasks {
            await task.value
        }
    }

    var count: Int {
        state.withLock { $0.channels.count }
    }
}

/// unchecked-invariant: a transport for one `ChannelHandlerContext` across a
/// `@Sendable` boundary. The context itself is NOT thread-safe -- every use of
/// `.value` below hops back to the channel's event loop first (writeJSON,
/// writeHeartbeat, and `.eventLoop` which is itself immutable). The box makes
/// the capture legal; the event-loop hop is what makes it correct.
final class SendableContext: @unchecked Sendable {
    let value: ChannelHandlerContext

    init(_ value: ChannelHandlerContext) {
        self.value = value
    }
}

final class RequestPhaseState: Sendable {
    private let state = Mutex("accepted")

    var value: String { state.withLock { $0 } }

    func set(_ value: String) {
        state.withLock { $0 = value }
    }
}

/// unchecked-invariant: every field is guarded by `lock`. Start/stop are
/// driven from the generation task while the heartbeat fires on the event loop,
/// so both sides contend and neither can rely on loop confinement.
final class StreamState: @unchecked Sendable {
    private let lock = NSLock()
    private var started = false
    private var stopped = false
    private var heartbeat: RepeatedTask?
    private var startFuture: EventLoopFuture<Void>?
    private var toolIndex = 0

    var isStarted: Bool {
        lock.withLock { started }
    }

    func start(
        eventLoop: EventLoop,
        interval: TimeAmount,
        ping: @escaping @Sendable () -> Void
    ) -> Bool {
        lock.withLock {
            guard !started else { return false }
            started = true
            stopped = false
            startFuture = nil
            heartbeat = eventLoop.scheduleRepeatedTask(
                initialDelay: interval,
                delay: interval
            ) { [weak self] _ in
                guard self?.shouldPing == true else { return }
                ping()
            }
            return true
        }
    }

    func setStartFuture(_ future: EventLoopFuture<Void>) {
        lock.withLock { startFuture = future }
    }

    func waitUntilStarted() async throws {
        let future = lock.withLock { startFuture }
        if let future {
            try await future.get()
        }
    }

    private var shouldPing: Bool {
        lock.withLock { started && !stopped }
    }

    func stop() {
        lock.withLock {
            stopped = true
            heartbeat?.cancel()
            heartbeat = nil
        }
    }

    func nextToolIndex() -> Int {
        lock.withLock {
            defer { toolIndex += 1 }
            return toolIndex
        }
    }
}
