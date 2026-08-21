import Synchronization
import Testing
@testable import TurboFieldfare

/// The cancellation guard for chunked text prefill. It lives here rather than
/// inline in `RealForwardRunner` because the runner's protocol seam is
/// `prefillChunked` itself, so every test double replaces the loop under test and
/// nothing in the suite could reach it. The defect this pins was found by timing
/// an interrupt by hand.
@Suite struct PrefillSpanIterationTests {
    private func spans(_ count: Int) -> [PrefillChunkSpan] {
        (0..<count).map {
            PrefillChunkSpan(tokenOffset: $0 * 128,
                             tokenCount: 128,
                             startPosition: $0 * 128,
                             completedCount: $0 * 128)
        }
    }

    @Test func everySpanRunsWhenNothingCancels() async throws {
        var seen: [Int] = []
        try await PrefillSpanIteration.forEachSpan(spans(5)) { index, _ in
            seen.append(index)
        }
        #expect(seen == [0, 1, 2, 3, 4])
    }

    /// The poll happens *before* a span, never during one. A chunk's GPU work
    /// does not observe cancellation, so the loop must not be expected to abandon
    /// it midway - which is why the measured latency after the fix is one chunk
    /// rather than zero. Modelled with a body that ignores cancellation, because
    /// `Task.sleep` does not: an earlier version of this test used `sleep` and so
    /// asserted a property of the body rather than of the loop.
    @Test func cancellationStopsTheLoopAndLetsTheSpanInFlightFinish() async throws {
        let finished = Mutex(0)
        let task = Task {
            try await PrefillSpanIteration.forEachSpan(spans(50)) { _, _ in
                var sink = 0
                for i in 0..<2_000_000 { sink &+= i }
                finished.withLock { $0 += 1 &+ (sink & 0) }
            }
        }
        try await Task.sleep(nanoseconds: 5_000_000)
        task.cancel()
        _ = try? await task.value

        let completed = finished.withLock { $0 }
        #expect(completed > 0, "no span completed, so the loop bailed before doing work")
        #expect(completed < 50, "cancellation did not stop the loop")
    }
}
