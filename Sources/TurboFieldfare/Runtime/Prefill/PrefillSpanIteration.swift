import Foundation

/// Iterates prefill spans, polling cancellation before each one.
///
/// This exists to be testable. The cancellation check lives inside
/// `RealForwardRunner.prefillChunked`, and every test double for that runner
/// implements `prefillChunked` itself - the protocol seam *is* the loop - so no
/// test in the suite could reach it. The defect it guards against was found by
/// timing a `SIGINT` by hand: a cancelled 7,000-token prompt kept prefilling for
/// 50 seconds, against 0.58 seconds when the same interrupt arrived during
/// decode, because this loop never polled while the multimodal loop beside it
/// always had.
enum PrefillSpanIteration {
    /// Runs `body` for each span, throwing `CancellationError` before the next
    /// one if the task was cancelled. A chunk already in flight still finishes:
    /// its GPU work is not interruptible, which is why the observed latency
    /// after the fix is one chunk rather than zero.
    static func forEachSpan(
        _ spans: [PrefillChunkSpan],
        _ body: (Int, PrefillChunkSpan) async throws -> Void
    ) async throws {
        for (index, span) in spans.enumerated() {
            try Task.checkCancellation()
            try await body(index, span)
        }
    }
}
