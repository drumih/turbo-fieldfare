import Testing
@testable import TurboFieldfareServerCore

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
        while await !coordinator.isActive {
            await Task.yield()
        }
        let queued = Task {
            try await coordinator.run { 2 }
        }
        while await coordinator.queuedCount != 1 {
            await Task.yield()
        }
        await #expect(throws: ServerRequestError.queueFull) {
            try await coordinator.run { 3 }
        }
        queued.cancel()
        _ = try? await queued.value
        await gate.open()
        #expect(try await active.value == 1)
        #expect(await coordinator.queuedCount == 0)
    }
}
