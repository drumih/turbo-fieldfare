import Testing

@testable import TurboFieldfare

@Suite struct RealForwardRunnerTests {
    private enum SyntheticCommandBufferError: Error, Equatable {
        case failed
    }

    @Test func commandBufferErrorsArePropagated() throws {
        try RealForwardRunner.checkCommandBufferError(nil)

        do {
            try RealForwardRunner.checkCommandBufferError(SyntheticCommandBufferError.failed)
            Issue.record("expected command-buffer error")
        } catch let error as SyntheticCommandBufferError {
            #expect(error == .failed)
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }
}
