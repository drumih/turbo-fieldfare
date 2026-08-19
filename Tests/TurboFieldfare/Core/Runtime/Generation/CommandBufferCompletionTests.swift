import Testing

@testable import TurboFieldfare

@Suite struct CommandBufferCompletionTests {
    private enum SyntheticCommandBufferError: Error, Equatable {
        case failed
    }

    @Test func commandBufferErrorsArePropagated() throws {
        try checkCommandBufferError(nil)

        do {
            try checkCommandBufferError(SyntheticCommandBufferError.failed)
            Issue.record("expected command-buffer error")
        } catch let error as SyntheticCommandBufferError {
            #expect(error == .failed)
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }
}
