import Darwin
import Synchronization
import Testing

@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

@Suite("Server termination signals", .serialized)
struct ServerTerminationSignalTests {
    /// Polls rather than awaiting, because the hook runs on the dispatch
    /// source's queue: a test that assumed it had already run would be racy, and
    /// one that let the real `exit(1)` run killed the whole test process.
    private func waitForExitCount(
        _ exits: borrowing Mutex<Int>, toReach target: Int,
        timeout: Duration = .seconds(2)
    ) async -> Int {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline, exits.withLock({ $0 }) < target {
            try? await Task.sleep(for: .milliseconds(5))
        }
        return exits.withLock { $0 }
    }

    @Test func dispatchSignalCrossesIntoAsyncCodeWithoutExecutorTrap() async {
        let signals = ServerTerminationSignals([SIGUSR1], forceExit: {})
        let waiter = Task {
            await signals.wait()
        }

        kill(getpid(), SIGUSR1)

        #expect(await waiter.value == SIGUSR1)
        await signals.cancel()
    }

    /// Cancelling before any signal resolves the wait with nil instead of
    /// trapping. `wait()` used to `preconditionFailure` when the stream ended
    /// without a signal -- reachable simply by ordering the two calls, since the
    /// API is public. Production waits first, so it never hit it.
    @Test func cancellingBeforeWaitingResolvesWithNil() async {
        let signals = ServerTerminationSignals([SIGUSR1], forceExit: {})
        await signals.cancel()

        #expect(await signals.wait() == nil)
    }

    /// The first signal reaches the waiter, and a second one during shutdown
    /// forces exit instead of being dropped (S33). The forced exit is injected:
    /// with the real `exit(1)` this test ended the test process -- asynchronously,
    /// so it aborted whichever suite ran next and left no summary.
    @Test func aSecondSignalForcesExitInsteadOfBeingDropped() async {
        let exits = Mutex(0)
        let signals = ServerTerminationSignals(
            [SIGUSR1],
            forceExit: {
                exits.withLock { $0 += 1 }
            })
        let waiter = Task {
            await signals.wait()
        }

        kill(getpid(), SIGUSR1)
        #expect(await waiter.value == SIGUSR1)

        // Dispatch coalesces pending signals, so one `kill` is not one handler
        // invocation: keep signalling until the hook fires. Any delivery after
        // the first is a "second signal during shutdown" by definition.
        let deadline = ContinuousClock.now + .seconds(2)
        while ContinuousClock.now < deadline, exits.withLock({ $0 }) == 0 {
            kill(getpid(), SIGUSR1)
            try? await Task.sleep(for: .milliseconds(10))
        }

        #expect(exits.withLock { $0 } >= 1, "a second signal did not force an exit")
        await signals.cancel()
    }

    /// Repeated delivery keeps the first signal for the waiter and calls the
    /// forced exit for each later one.
    @Test func repeatedDeliveryKeepsTheFirstSignalAndForcesExitOnce() async {
        let exits = Mutex(0)
        let signals = ServerTerminationSignals(
            [SIGUSR1],
            forceExit: {
                exits.withLock { $0 += 1 }
            })
        let waiter = Task {
            await signals.wait()
        }

        kill(getpid(), SIGUSR1)
        #expect(await waiter.value == SIGUSR1)
        kill(getpid(), SIGUSR1)
        let forced = await waitForExitCount(exits, toReach: 1)

        #expect(forced >= 1)
        await signals.cancel()
    }
}
