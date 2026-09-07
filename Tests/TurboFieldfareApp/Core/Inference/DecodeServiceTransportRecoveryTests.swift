import Darwin
import Foundation
import Synchronization
import Testing
@testable import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

/// The failure this covers is a decode service that is *alive but mute*: its
/// writer thread died, or a command buffer wedged. Nothing closes the socket,
/// so a blocking wait never returns, Cancel is inert, and the app stays in
/// `.running` until it is quit.
@Suite struct DecodeServiceTransportRecoveryTests {
    @Test func aSilentServiceEndsTheWaitInsteadOfHangingForever() async throws {
        // Open at both ends and never written to: alive, connected, mute.
        let pipe = Pipe()
        defer {
            try? pipe.fileHandleForWriting.close()
            try? pipe.fileHandleForReading.close()
        }
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)

        let started = ContinuousClock.now
        var description = ""
        do {
            _ = try await router.next(matching: UUID(), timeout: .milliseconds(300))
            Issue.record("a mute service produced an event")
        } catch let error as DecodeServiceTransportError {
            description = "\(error)"
        }
        let elapsed = ContinuousClock.now - started
        #expect(elapsed >= .milliseconds(250), "gave up before its deadline")
        // Generous upper bound: this runs alongside the GPU suites, and the
        // point is that the wait ends at all, not that it ends punctually.
        #expect(elapsed < .seconds(120), "the wait never ended")
        #expect(description.contains("stopped responding"),
                "the error must name the cause: \(description)")
    }

    /// The wait used to run in a detached task, so cancelling the caller left
    /// the thread parked and the caller waiting on it.
    @Test func cancellingTheCallerEndsTheWait() async throws {
        let pipe = Pipe()
        defer {
            try? pipe.fileHandleForWriting.close()
            try? pipe.fileHandleForReading.close()
        }
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)

        let waiting = Task {
            try await router.next(matching: UUID(), timeout: .seconds(60))
        }
        try await Task.sleep(for: .milliseconds(100))
        waiting.cancel()

        let started = ContinuousClock.now
        await #expect(throws: CancellationError.self) { try await waiting.value }
        #expect(ContinuousClock.now - started < .seconds(120),
                "cancellation never reached the wait")
    }

    /// A wait entered by a task that is *already* cancelled must end at once.
    ///
    /// `withTaskCancellationHandler` runs its handler immediately in that case
    /// — before the wait registers — so the handler found nothing to fail and
    /// the waiter that registered afterwards sat until its deadline. Fifteen
    /// minutes, for a load, which is indistinguishable from the hang the
    /// deadline was added to end. The app reaches this whenever Cancel is
    /// pressed while the client is still inside its connect retries.
    @Test func awaitEnteredAlreadyCancelledEndsImmediately() async throws {
        let pipe = Pipe()
        defer {
            try? pipe.fileHandleForWriting.close()
            try? pipe.fileHandleForReading.close()
        }
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)

        let waited = Mutex<Duration?>(nil)
        let waiting = Task {
            // Absorbed, so the wait below is entered by a task that is already
            // cancelled — the real ordering, since the client's connect retries
            // are not cancellable either.
            try? await Task.sleep(for: .milliseconds(50))
            let entered = ContinuousClock.now
            defer { waited.withLock { $0 = ContinuousClock.now - entered } }
            // Timed from inside the wait: this suite runs beside the GPU tests,
            // and scheduling delay outside it says nothing about the deadline.
            return try await router.next(matching: UUID(), timeout: .seconds(120))
        }
        waiting.cancel()

        await #expect(throws: CancellationError.self) { try await waiting.value }
        let elapsed = try #require(waited.withLock { $0 }, "the wait was never entered")
        #expect(elapsed < .seconds(30),
                "the wait ran on toward its deadline instead of ending: \(elapsed)")
    }

    /// Closing the stream from the owning side has to end the reader too.
    ///
    /// It holds its own `dup` of the socket, so closing the write handle does
    /// not wake it: against a service too wedged to act on the shutdown command
    /// the client sends first, the thread would stay in `read` for the life of
    /// the process, holding a descriptor nothing else can close.
    @Test func closingTheStreamEndsTheReader() async throws {
        let pipe = Pipe()
        defer { try? pipe.fileHandleForWriting.close() }
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)
        #expect(!router.isTerminated)

        router.closeStream()
        let deadline = ContinuousClock.now + .seconds(5)
        while !router.isTerminated, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(router.isTerminated, "the reader outlived the connection it read")
    }

    /// The same, on the transport this actually runs over rather than a pipe:
    /// the reader is parked in `read` on a socket when the stream is closed,
    /// which is the shape of the real thing.
    @Test func closingTheStreamEndsTheReaderOnASocket() async throws {
        var pair: [Int32] = [0, 0]
        #expect(socketpair(AF_UNIX, SOCK_STREAM, 0, &pair) == 0)
        let peer = FileHandle(fileDescriptor: pair[0], closeOnDealloc: true)
        defer { try? peer.close() }
        let router = DecodeServiceResponseRouter(
            output: FileHandle(fileDescriptor: pair[1], closeOnDealloc: true))
        // Let the reader reach its blocking read before the stream is closed.
        try await Task.sleep(for: .milliseconds(100))
        #expect(!router.isTerminated)

        router.closeStream()
        let deadline = ContinuousClock.now + .seconds(5)
        while !router.isTerminated, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(router.isTerminated,
                "the reader stayed blocked on a socket nothing else can close")
    }

    /// A closed stream must be reported once, so the owner can tear the
    /// connection down rather than keep handing out handles that only throw.
    @Test func theStreamEndingIsReportedOnceAndLatches() async throws {
        let pipe = Pipe()
        let notified = Mutex(0)
        let router = DecodeServiceResponseRouter(
            output: pipe.fileHandleForReading,
            onTerminate: { _, _ in notified.withLock { $0 += 1 } })
        #expect(!router.isTerminated)

        try pipe.fileHandleForWriting.close()
        let deadline = ContinuousClock.now + .seconds(5)
        while !router.isTerminated, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(router.isTerminated)
        #expect(notified.withLock { $0 } == 1)

        // Every later wait fails at once rather than waiting out its timeout.
        let started = ContinuousClock.now
        await #expect(throws: (any Error).self) {
            _ = try await router.next(matching: UUID(), timeout: .seconds(60))
        }
        #expect(ContinuousClock.now - started < .seconds(30),
                "a wait on a dead router blocked for its full timeout")
        try? pipe.fileHandleForReading.close()
    }

    @Test func aTerminatedRouterCanDeallocateAfterItsCallback() async throws {
        let pipe = Pipe()
        let reportedIdentity = Mutex<ObjectIdentifier?>(nil)
        weak var releasedRouter: DecodeServiceResponseRouter?

        do {
            let router = DecodeServiceResponseRouter(
                output: pipe.fileHandleForReading,
                onTerminate: { router, _ in
                    reportedIdentity.withLock { $0 = ObjectIdentifier(router) }
                })
            releasedRouter = router
            try pipe.fileHandleForWriting.close()
            let deadline = ContinuousClock.now + .seconds(5)
            while !router.isTerminated, ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(20))
            }
            #expect(reportedIdentity.withLock { $0 } == ObjectIdentifier(router))
        }

        let deadline = ContinuousClock.now + .seconds(5)
        while releasedRouter != nil, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(releasedRouter == nil, "the termination callback retained its router")
    }

    /// A service that answers and then exits is ordinary — launchd reaps it as
    /// soon as it is idle — so events that arrived before the stream ended are
    /// still answers and must be drainable after it.
    @Test func endOfStreamDoesNotDiscardEventsThatAlreadyArrived() async throws {
        let pipe = Pipe()
        let id = UUID()
        for index in 0..<4 {
            try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
                DecodeServiceEvent(kind: .snapshot, generationID: id,
                                   tokenCount: index)))
        }
        try pipe.fileHandleForWriting.close()
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)

        for index in 0..<4 {
            let event = try await router.next(matching: id, timeout: .seconds(5))
            #expect(event.tokenCount == index)
        }
        // Only once they are drained does the closed stream surface.
        await #expect(throws: (any Error).self) {
            _ = try await router.next(matching: id, timeout: .seconds(5))
        }
        try? pipe.fileHandleForReading.close()
    }

    /// Once a wait gives up, that request's later events must be dropped, not
    /// buffered: a stream abandoned early — a cancelled load, a timed-out
    /// generation — otherwise grew the map for the life of the connection.
    @Test func eventsForAnAbandonedRequestAreDropped() async throws {
        let pipe = Pipe()
        defer {
            try? pipe.fileHandleForWriting.close()
            try? pipe.fileHandleForReading.close()
        }
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)
        let abandoned = UUID()

        // Nobody answers, so this wait gives up.
        await #expect(throws: DecodeServiceTransportError.self) {
            _ = try await router.next(matching: abandoned, timeout: .milliseconds(150))
        }

        // The service keeps talking about that request anyway.
        for index in 0..<64 {
            try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
                DecodeServiceEvent(kind: .snapshot, generationID: abandoned,
                                   tokenCount: index)))
        }
        // A live request on the same connection still works, which is how we
        // know the reader is running and the drop is selective.
        let live = UUID()
        try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(kind: .finished, generationID: live)))
        let event = try await router.next(matching: live, timeout: .seconds(5))
        #expect(event.kind == .finished)

        // Nothing was kept for the abandoned one.
        await #expect(throws: DecodeServiceTransportError.self) {
            _ = try await router.next(matching: abandoned, timeout: .milliseconds(150))
        }
    }

    /// A frame that arrives before anyone waits for it must still be delivered.
    @Test func eventsBufferedBeforeAWaitAreDelivered() async throws {
        let id = UUID()
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(kind: .snapshot, generationID: id, tokenCount: 1)))
        try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(kind: .finished, generationID: id, tokenCount: 2)))
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)

        let first = try await router.next(matching: id, timeout: .seconds(5))
        let second = try await router.next(matching: id, timeout: .seconds(5))
        #expect(first.kind == .snapshot)
        #expect(second.kind == .finished)
        try? pipe.fileHandleForWriting.close()
        try? pipe.fileHandleForReading.close()
    }

    /// Writing to a socket whose peer has exited raises SIGPIPE, and the
    /// default disposition kills the process. This is the app's own guarantee,
    /// so assert it on a real closed socket pair rather than trusting the call.
    @Test func writingToADeadPeerFailsInsteadOfKillingTheProcess() throws {
        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
        var descriptors: [Int32] = [0, 0]
        #expect(socketpair(AF_UNIX, SOCK_STREAM, 0, &descriptors) == 0)
        DecodeUnixSocket.disableSIGPIPE(on: descriptors[0])
        Darwin.close(descriptors[1])

        let payload = [UInt8](repeating: 0x41, count: 1_024)
        let written = payload.withUnsafeBytes {
            Darwin.send(descriptors[0], $0.baseAddress!, $0.count, 0)
        }
        // Reaching this line at all is the assertion: an unhandled SIGPIPE
        // would have killed the test process on the send above.
        #expect(written == -1)
        #expect(errno == EPIPE)
        Darwin.close(descriptors[0])
    }
}
