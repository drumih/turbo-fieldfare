import Dispatch
import Testing

@testable import TinyTitan
import TinyTitanCore

@Suite(.serialized)
struct GPUKeepWarmRunnerTests {
    @Test func repeatedImmediateStartStopJoinsEachOneShotRunner() throws {
        let context = try MetalContext()
        for iteration in 0..<128 {
            let runner = try GPUKeepWarmRunner(context: context, intervalUs: 100)
            try #require(
                completesWithinDeadline {
                    runner.start()
                    runner.stop()
                },
                "Immediate stop failed to join runner \(iteration)")
        }
    }

    @Test func absentSettingDoesNotStartARunner() throws {
        let context = try MetalContext()
        let runner = try GPUKeepWarmRunner.startIfEnabled(context: context, environment: [:])
        // Clean up even if the disabled path incorrectly starts a stream.
        if let runner {
            try #require(completesWithinDeadline { runner.stop() })
        }
        #expect(runner == nil)
    }

    @Test func enabledSettingStartsARunnerThatCanBeStoppedImmediately() throws {
        let context = try MetalContext()
        let runner = try #require(
            try GPUKeepWarmRunner.startIfEnabled(
                context: context, environment: ["TINYTITAN_GPU_KEEPWARM_US": "100"]))
        try #require(
            completesWithinDeadline { runner.stop() },
            "The enabled runner did not join after stop")
    }

    // stop() blocks, so run the lifecycle off the test thread and assert that
    // it returns. The deadline bounds failure; it does not synchronize a sleep.
    private func completesWithinDeadline(_ operation: @escaping @Sendable () -> Void) -> Bool {
        let completed = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            operation()
            completed.signal()
        }
        return completed.wait(timeout: .now() + 5) == .success
    }
}
