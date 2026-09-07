import Foundation
import Testing
@testable import TurboFieldfareDecodeService

/// `.cancel` means two things on the wire — stop this generation, and abandon
/// this load — and the load half used to latch with no scope. Stopping an
/// answer left a cancel armed, and the next model load the user asked for died
/// on arrival with "decode service returned cancelled for a load request".
@Suite struct ServiceLoadCancellationTests {
    private func makeTask() -> Task<Void, Error> {
        Task { try await Task.sleep(for: .seconds(30)) }
    }

    @Test func stoppingAGenerationDoesNotPoisonTheNextLoad() async throws {
        let work = ServiceLoadCancellation()
        // No load queued: this cancel belongs to a generation.
        work.cancel()
        #expect(!work.isHoldingCancel,
                "a generation's cancel was held against a future load")

        // The user changes a setting and reloads, well after the stop.
        work.enqueueLoad()
        let load = makeTask()
        work.begin(load)
        defer { load.cancel() }
        #expect(!load.isCancelled, "the reload was cancelled by an unrelated stop")
        work.finish()
    }

    /// The race the latch exists for: the app writes `.load`, gives up, and
    /// writes `.cancel` before the command loop has started the load.
    @Test func acancelThatOvertakesItsLoadStillLands() async throws {
        let work = ServiceLoadCancellation()
        work.enqueueLoad()
        work.cancel()
        #expect(work.isHoldingCancel)

        let load = makeTask()
        work.begin(load)
        #expect(load.isCancelled, "the cancel never reached the load it was for")
        #expect(!work.isHoldingCancel, "the cancel was held past the load it was for")
        work.finish()
    }

    /// A cancel arriving while the load is running cancels that load and
    /// nothing else.
    @Test func acancelDuringALoadCancelsOnlyThatLoad() async throws {
        let work = ServiceLoadCancellation()
        work.enqueueLoad()
        let first = makeTask()
        work.begin(first)
        work.cancel()
        #expect(first.isCancelled)
        #expect(!work.isHoldingCancel)
        work.finish()

        work.enqueueLoad()
        let second = makeTask()
        work.begin(second)
        defer { second.cancel() }
        #expect(!second.isCancelled, "the next load inherited the last one's cancel")
    }

    /// A load can fail before it ever starts — its runtime options are parsed
    /// after the input thread has already counted it — and the count has to
    /// clear anyway. Otherwise the next `.cancel` from a generation finds a
    /// load still queued, holds itself, and kills the load after that.
    @Test func aloadThatNeverStartsStillClearsItsQueueSlot() async throws {
        let work = ServiceLoadCancellation()
        work.enqueueLoad()
        // The service's `.load` handler threw before `begin`; its defer runs.
        work.finish()

        // Now an ordinary generation stop.
        work.cancel()
        #expect(!work.isHoldingCancel,
                "an abandoned load left its slot counted, so a generation's cancel latched")

        work.enqueueLoad()
        let load = makeTask()
        work.begin(load)
        defer { load.cancel() }
        #expect(!load.isCancelled, "the next load paid for a load that never ran")
        work.finish()
    }

    /// Two cancels with no load between them must not accumulate a debt that a
    /// later load pays.
    @Test func repeatedGenerationCancelsAccumulateNothing() async throws {
        let work = ServiceLoadCancellation()
        for _ in 0..<5 { work.cancel() }
        work.enqueueLoad()
        let load = makeTask()
        work.begin(load)
        defer { load.cancel() }
        #expect(!load.isCancelled)
    }
}
