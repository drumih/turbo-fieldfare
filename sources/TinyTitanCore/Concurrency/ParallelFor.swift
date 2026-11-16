import Dispatch

/// Runs `body` for every index in `0..<iterations` concurrently, then waits
/// for all of them.
///
/// The safe replacement for `DispatchQueue.concurrentPerform` on a fixed-width
/// worker pool. `concurrentPerform` blocks the calling thread and counts it as
/// one of the workers; called from the expert-I/O scheduler's four worker
/// threads, concurrent full-chunk prefill fetches starve the GCD pool and
/// deadlock mid-layer at 0% CPU (a 5,818-token prompt stalled at layer 18–25
/// and never finished). Here the caller only waits on a group while the
/// iterations run as independent members of the global pool, so the work
/// always gets threads regardless of how many callers are waiting.
package func parallelFor(
    iterations: Int,
    _ body: @escaping @Sendable (Int) -> Void
) {
    guard iterations > 0 else { return }
    let group = DispatchGroup()
    let queue = DispatchQueue.global(qos: .userInitiated)
    for index in 0..<iterations {
        queue.async(group: group) { body(index) }
    }
    group.wait()
}
