import Darwin
import Foundation
import TurboFieldfareDecodeProtocol

/// Why a wait for a decode-service event ended without one.
public enum DecodeServiceTransportError: Error, CustomStringConvertible {
    /// The service stopped answering while still connected: no frame of any
    /// kind arrived within the deadline. A closed socket surfaces as the
    /// reader's own error instead.
    case timedOut(seconds: Double)
    /// The connection is already known to be dead; a new one must be built.
    case connectionLost(underlying: String)

    public var description: String {
        switch self {
        case .timedOut(let seconds):
            "the decode service stopped responding after \(Int(seconds))s"
        case .connectionLost(let underlying):
            "the decode service connection is gone: \(underlying)"
        }
    }
}

/// Demultiplexes the service's event stream onto the callers waiting for it.
///
/// Every wait is bounded and cancellable. A blocking wait meant that a service
/// which stayed alive but stopped answering — its writer thread died, or a
/// command buffer wedged — left the app in `.running` forever with Cancel
/// inert, because nothing downstream could observe that no event was coming.
final class DecodeServiceResponseRouter: @unchecked Sendable {
    private struct Waiter {
        let requestID: UUID
        let continuation: CheckedContinuation<DecodeServiceEvent, Error>
    }

    private let lock = NSLock()
    private var pending: [UUID: [DecodeServiceEvent]] = [:]
    private var waiters: [UUID: Waiter] = [:]
    private var terminalError: Error?
    /// Requests whose waiter gave up. Their later events are dropped rather
    /// than buffered: a request that was cancelled or timed out is never waited
    /// on again, so buffering its remaining events grew the map for the life of
    /// the connection.
    private var abandoned: [UUID] = []
    private static let abandonedMemory = 32

    /// Called once, off the reader thread, when the stream ends for any reason.
    /// The client uses it to tear the connection down rather than keep handing
    /// out handles that can only fail.
    private let onTerminate: @Sendable (DecodeServiceResponseRouter, Error) -> Void

    private let output: FileHandle

    init(
        output: FileHandle,
        onTerminate: @escaping @Sendable (DecodeServiceResponseRouter, Error) -> Void
            = { _, _ in }
    ) {
        self.onTerminate = onTerminate
        self.output = output
        let reader = Thread { [weak self] in
            self?.readFrames(from: output)
        }
        reader.name = "TurboFieldfare.DecodeService.ResponseRouter"
        reader.qualityOfService = .userInitiated
        reader.start()
    }

    /// Closes the stream this router reads.
    ///
    /// The reader owns a `dup` of the socket, so closing the client's write
    /// handle does not wake it: against a service that is alive but no longer
    /// answering commands — the case the deadlines exist for — the thread would
    /// otherwise stay blocked in `read` for the life of the process, holding a
    /// descriptor nothing will ever close.
    func closeStream() {
        // `shutdown` first. Closing a descriptor another thread is already
        // blocked reading is unspecified by POSIX — measured here it does wake
        // the reader, but this is called precisely when a thread is parked in
        // `read`, and shutting the socket down removes the reliance. On a pipe
        // it fails with ENOTSOCK and costs nothing.
        shutdown(output.fileDescriptor, SHUT_RDWR)
        try? output.close()
    }

    /// True once the stream has ended. The connection cannot recover from this;
    /// only a new one can.
    var isTerminated: Bool {
        lock.lock()
        defer { lock.unlock() }
        return terminalError != nil
    }

    func next(matching requestID: UUID, timeout: Duration = .seconds(300)) async throws
        -> DecodeServiceEvent {
        let waiterID = UUID()
        let timeoutSeconds = Double(timeout.components.seconds)
            + Double(timeout.components.attoseconds) / 1e18
        // Scheduled by libdispatch rather than the cooperative pool: this
        // deadline exists precisely for the moments when the machine is
        // saturated, which is when a cooperative timer is starved worst.
        let deadline = DispatchSource.makeTimerSource(
            queue: DispatchQueue.global(qos: .utility))
        deadline.schedule(deadline: .now() + timeoutSeconds)
        deadline.setEventHandler { [self] in
            fail(waiterID, with: DecodeServiceTransportError.timedOut(
                seconds: timeoutSeconds))
        }
        deadline.resume()
        defer { deadline.cancel() }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                register(Waiter(requestID: requestID, continuation: continuation),
                         id: waiterID)
            }
        } onCancel: {
            self.fail(waiterID, with: CancellationError())
        }
    }

    private func register(_ waiter: Waiter, id: UUID) {
        lock.lock()
        // Cancellation before the wait even starts. `withTaskCancellationHandler`
        // runs its handler immediately when the task is already cancelled, which
        // is before this call, so `fail` had nothing to find and the waiter
        // registered here would have sat until its deadline — 900 s on a load,
        // which is the hang this router exists to end. The same applies to a
        // cancel landing between the handler being installed and this lock.
        if Task.isCancelled {
            pending.removeValue(forKey: waiter.requestID)
            abandoned.append(waiter.requestID)
            if abandoned.count > Self.abandonedMemory { abandoned.removeFirst() }
            lock.unlock()
            waiter.continuation.resume(throwing: CancellationError())
            return
        }
        // A frame that arrived before this call, or a stream that has already
        // ended, resolves the wait without it ever being stored.
        if var events = pending[waiter.requestID], !events.isEmpty {
            let event = events.removeFirst()
            if events.isEmpty {
                pending.removeValue(forKey: waiter.requestID)
            } else {
                pending[waiter.requestID] = events
            }
            lock.unlock()
            waiter.continuation.resume(returning: event)
            return
        }
        if let terminalError {
            lock.unlock()
            waiter.continuation.resume(throwing: terminalError)
            return
        }
        waiters[id] = waiter
        lock.unlock()
    }

    /// Resolves a waiter that timed out or was cancelled. A waiter already
    /// resumed by an arriving frame is simply absent.
    private func fail(_ id: UUID, with error: Error) {
        lock.lock()
        let waiter = waiters.removeValue(forKey: id)
        if let waiter {
            pending.removeValue(forKey: waiter.requestID)
            abandoned.append(waiter.requestID)
            if abandoned.count > Self.abandonedMemory { abandoned.removeFirst() }
        }
        lock.unlock()
        waiter?.continuation.resume(throwing: error)
    }

    private func deliver(_ event: DecodeServiceEvent) {
        lock.lock()
        if let entry = waiters.first(where: { $0.value.requestID == event.generationID }) {
            waiters.removeValue(forKey: entry.key)
            lock.unlock()
            entry.value.continuation.resume(returning: event)
            return
        }
        guard !abandoned.contains(event.generationID) else {
            lock.unlock()
            return
        }
        pending[event.generationID, default: []].append(event)
        lock.unlock()
    }

    private func terminate(with error: Error) {
        lock.lock()
        guard terminalError == nil else {
            lock.unlock()
            return
        }
        terminalError = error
        // Events that arrived before the stream ended are still answers; a
        // service that replies and then exits is ordinary. Only waiters with
        // nothing buffered learn the connection is gone.
        var delivered: [(Waiter, DecodeServiceEvent)] = []
        var stranded: [Waiter] = []
        for waiter in waiters.values {
            if var events = pending[waiter.requestID], !events.isEmpty {
                let event = events.removeFirst()
                if events.isEmpty {
                    pending.removeValue(forKey: waiter.requestID)
                } else {
                    pending[waiter.requestID] = events
                }
                delivered.append((waiter, event))
            } else {
                stranded.append(waiter)
            }
        }
        waiters.removeAll()
        lock.unlock()
        for (waiter, event) in delivered {
            waiter.continuation.resume(returning: event)
        }
        for waiter in stranded {
            waiter.continuation.resume(throwing: error)
        }
        onTerminate(self, error)
    }

    private func readFrames(from output: FileHandle) {
        do {
            while true {
                deliver(try DecodeFrameCodec.read(
                    DecodeServiceEvent.self, from: output))
            }
        } catch {
            terminate(with: error)
        }
    }
}
