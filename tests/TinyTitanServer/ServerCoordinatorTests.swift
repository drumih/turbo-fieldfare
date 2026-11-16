import Testing

@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

private actor TestGate {
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        await withCheckedContinuation { continuation = $0 }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

@Suite("Server coordinator")
struct ServerCoordinatorTests {
    @Test func boundsFIFOAndRecoversAfterCancellation() async throws {
        let coordinator = ServerCoordinator(queueLimit: 1)
        let gate = TestGate()
        let active = Task {
            try await coordinator.run {
                await gate.wait()
                return 1
            }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.isActive }
        let queued = Task {
            try await coordinator.run { 2 }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.queuedCount == 1 }
        await #expect(throws: ServerRequestError.queueFull) {
            try await coordinator.run { 3 }
        }
        queued.cancel()
        _ = try? await queued.value
        await gate.open()
        #expect(try await active.value == 1)
        #expect(await coordinator.queuedCount == 0)
    }

    /// Width 2 runs two generations at once, queues the third behind the
    /// `queueLimit`, and sheds the one after that — the batched admission rule.
    @Test func widthRunsSeveralAtOnceAndQueuesTheRest() async throws {
        let coordinator = ServerCoordinator(queueLimit: 1, width: 2)
        let release = AsyncSemaphore()

        let a = Task {
            try await coordinator.run {
                await release.wait()
                return 1
            }
        }
        let b = Task {
            try await coordinator.run {
                await release.wait()
                return 2
            }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.runningCount == 2 }

        let queued = Task { try await coordinator.run { 3 } }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.queuedCount == 1 }
        await #expect(throws: ServerRequestError.queueFull) {
            try await coordinator.run { 4 }
        }

        await release.signal()
        #expect(try await a.value == 1)
        #expect(try await b.value == 2)
        // The queued one takes the freed slot without waiting again.
        #expect(try await queued.value == 3)
        #expect(await coordinator.runningCount == 0)
        #expect(await coordinator.queuedCount == 0)
    }

    /// Bounded poll so a state that never reaches `condition` fails fast
    /// instead of spinning forever.
    private func waitUntil(
        timeout: Duration,
        _ condition: () async -> Bool
    ) async throws {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while await !condition() {
            guard ContinuousClock.now < deadline else {
                throw CoordinatorTimeout()
            }
            await Task.yield()
        }
    }

    private struct CoordinatorTimeout: Error {}

    /// The side-engine reads this before every token to decide how wide to
    /// run, from whatever thread it is on, so it has to be readable without
    /// awaiting the actor — and it has to be accurate, because the whole
    /// point is not to slow down the answer someone is waiting for.
    @Test func generationSignalIsRaisedForTheDurationOfAGeneration() async throws {
        let coordinator = ServerCoordinator(queueLimit: 2)
        #expect(coordinator.generating.isBusy == false)

        let started = AsyncSemaphore()
        let release = AsyncSemaphore()
        let work = Task {
            try await coordinator.run {
                await started.signal()
                await release.wait()
                return 1
            }
        }
        await started.wait()
        #expect(coordinator.generating.isBusy, "raised while the generation runs")
        await release.signal()
        _ = try await work.value
        #expect(coordinator.generating.isBusy == false, "and lowered after it")
    }

    /// It must survive a failing generation too: a signal that stuck high
    /// after one error would pin the side-engine to one thread forever.
    @Test func generationSignalIsLoweredWhenTheGenerationThrows() async throws {
        let coordinator = ServerCoordinator(queueLimit: 2)
        struct Boom: Error {}
        _ = try? await coordinator.run { throw Boom() }
        #expect(coordinator.generating.isBusy == false)
    }
}

/// A one-shot signal between two tasks, so the test can hold a generation
/// open while it observes the flag.
private actor AsyncSemaphore {
    private var signalled = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func signal() {
        signalled = true
        for waiter in waiters { waiter.resume() }
        waiters.removeAll()
    }

    func wait() async {
        if signalled { return }
        await withCheckedContinuation { waiters.append($0) }
    }

}
