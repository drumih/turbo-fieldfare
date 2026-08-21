import Foundation
import Synchronization
import Testing
@testable import TurboFieldfareAppCore

/// Cancelling work that runs off the calling task.
///
/// The vision pack's activation hash ran inside a bare `Task.detached`, which
/// inherits nothing — so its cooperative `Task.checkCancellation()` tested a
/// flag no one ever set. Cancel did nothing: the multi-minute hash ran to the
/// end and the pack activated anyway, while the UI sat on "Cancelling" with
/// every model action disabled. The suite's own mock hid it by running the
/// check inline in the caller's task, where cancellation does propagate.
@Suite struct CancellableDetachedWorkTests {
    private let client = RepackVisionPackInstallerClient()

    /// The regression: without the forwarding this hangs until the body
    /// finishes on its own and returns normally, instead of throwing.
    @Test func cancellingTheCallerCancelsTheDetachedBody() async throws {
        let started = Mutex(false)
        let observedCancellation = Mutex(false)

        let task = Task {
            try await client.runCancellableDetached {
                started.withLock { $0 = true }
                // Stands in for the verification hash: cooperative, and long
                // enough that it cannot finish before the cancel lands. Bounded
                // so that losing the forwarding fails this test instead of
                // hanging it — the body then simply returns.
                let deadline = Date().addingTimeInterval(3)
                while !Task.isCancelled, Date() < deadline {
                    usleep(1_000)
                }
                guard Task.isCancelled else { return }
                observedCancellation.withLock { $0 = true }
                try Task.checkCancellation()
            }
        }

        let deadline = Date().addingTimeInterval(5)
        while !started.withLock({ $0 }), Date() < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(started.withLock { $0 }, "the body never ran")

        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(observedCancellation.withLock { $0 },
                "the detached body never saw the cancellation")
    }

    /// Work that is not cancelled still runs and returns its value: forwarding
    /// cancellation must not turn into cancelling everything.
    @Test func uncancelledWorkReturnsItsValue() async throws {
        let ran = Mutex(false)

        let value = try await client.runCancellableDetached { () -> Int in
            ran.withLock { $0 = true }
            return 41 + 1
        }

        #expect(value == 42)
        #expect(ran.withLock { $0 })
    }

    /// A failure inside the body reaches the caller unchanged, rather than
    /// being flattened into a cancellation.
    @Test func aFailureInTheBodyPropagates() async throws {
        struct Failure: Error, Equatable {}

        await #expect(throws: Failure.self) {
            try await client.runCancellableDetached { throw Failure() }
        }
    }
}
