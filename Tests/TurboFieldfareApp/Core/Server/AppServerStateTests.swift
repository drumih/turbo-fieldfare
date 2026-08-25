import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppServerStateTests {
    @Test func runningStatesWithSamePortAreEqual() {
        #expect(AppServerState.running(port: 8080) == .running(port: 8080))
    }

    @Test func runningStatesWithDifferentPortsAreNotEqual() {
        #expect(AppServerState.running(port: 8080) != .running(port: 9090))
    }

    @Test func failedStatesWithDifferentMessagesAreNotEqual() {
        #expect(AppServerState.failed(message: "a") != .failed(message: "b"))
    }

    @Test func stoppedIsNotStarting() {
        #expect(AppServerState.stopped != .starting)
    }
}
