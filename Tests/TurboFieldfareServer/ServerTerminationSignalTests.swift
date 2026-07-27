import Darwin
import Testing
@testable import TurboFieldfareServerCore

@Suite("Server termination signals", .serialized)
struct ServerTerminationSignalTests {
    @Test func dispatchSignalCrossesIntoAsyncCodeWithoutExecutorTrap() async {
        let signals = ServerTerminationSignals([SIGUSR1])
        let waiter = Task {
            await signals.wait()
        }

        kill(getpid(), SIGUSR1)

        #expect(await waiter.value == SIGUSR1)
        await signals.cancel()
    }

    @Test func repeatedSignalDeliveryKeepsTheFirstSignal() async {
        let signals = ServerTerminationSignals([SIGUSR1])
        let waiter = Task {
            await signals.wait()
        }

        kill(getpid(), SIGUSR1)
        kill(getpid(), SIGUSR1)

        #expect(await waiter.value == SIGUSR1)
        await signals.cancel()
    }
}
