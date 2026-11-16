import Testing
import Foundation
import Dispatch
import Synchronization
@testable import TinyTitanCore

/// The dispatch-group parallel-for that replaces `concurrentPerform` on the
/// expert-I/O worker pool (the full-chunk prefill deadlock fix). These pin its
/// correctness and the property the fix rests on: it completes when called
/// concurrently from a fixed-width worker pool, where `concurrentPerform`
/// counts the calling thread as a worker, starves the GCD pool and hangs.
struct ParallelForTests {

    @Test func runsEveryIndexExactlyOnce() {
        let count = 256
        let hits = Mutex<[Int]>([Int](repeating: 0, count: count))
        parallelFor(iterations: count) { index in
            hits.withLock { $0[index] += 1 }
        }
        #expect(hits.withLock { $0 } == [Int](repeating: 1, count: count))
    }

    @Test func zeroIterationsReturnsImmediately() {
        let hits = Mutex<Int>(0)
        parallelFor(iterations: 0) { _ in hits.withLock { $0 += 1 } }
        #expect(hits.withLock { $0 } == 0)
    }

    @Test func completesFromAFixedWorkerPool() {
        // The expert-I/O scheduler shape: a few dedicated worker threads each
        // running a parallel-for at once. concurrentPerform counts the calling
        // thread as a worker and deadlocks here under load; parallelFor must
        // finish well inside the timeout.
        let workers = 4
        let perWorker = 16
        let done = Mutex<Int>(0)
        let group = DispatchGroup()
        for worker in 0..<workers {
            DispatchQueue(label: "parallelfor-test.\(worker)").async(group: group) {
                parallelFor(iterations: perWorker) { _ in
                    done.withLock { $0 += 1 }
                }
            }
        }
        #expect(group.wait(timeout: .now() + 15) == .success)
        #expect(done.withLock { $0 } == workers * perWorker)
    }
}
