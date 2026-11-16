import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

@Suite("Routing gate")
struct RoutingGateTests {
    /// Every waiter on one gate has to be released by the open that took the gate
    /// from it. The gate used to store a single continuation, so a second arrival
    /// overwrote the first and that caller parked forever — and a test that awaits
    /// its own parked task cannot time out, because a task group joins every child.
    /// So the count is polled, and the deadline is the failure.
    @Test func everyWaiterOnOneGateIsReleased() async throws {
        let gate = RoutingGate()
        let log = RoutingEventLog()
        for _ in 0..<4 {
            Task {
                await gate.wait()
                log.append("released")
            }
        }
        await RoutingFixture.eventually("a waiter to park") { await gate.isWaiting }
        try? await Task.sleep(for: .milliseconds(50))
        await gate.open()
        await RoutingFixture.eventually("all four waiters to be released", timeout: .seconds(5)) {
            log.releases == 4
        }
        #expect(log.releases == 4, "only \(log.releases) of 4 waiters were released")
        // A caller that arrives after the gate opened goes straight through.
        _ = Task {
            await gate.wait()
            log.append("released")
        }
        await RoutingFixture.eventually("the late caller to return", timeout: .seconds(5)) {
            log.releases == 5
        }
        // No join: the count above is only reached by a body that ran to the end,
        // and joining a parked task is exactly the hang this test is written to avoid.
        #expect(log.releases == 5)
    }
}
