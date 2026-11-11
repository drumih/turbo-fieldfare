import Foundation
import Metal

/// The read mechanics behind `GGUFExpertBufferCache`: one `pread` loop per
/// matrix, and a batch of them that runs concurrently and remembers the first
/// failure. The cache decides what to read and into which slot; this file only
/// moves bytes.
extension GGUFExpertBufferCache {
    /// Reads one matrix into its buffer with `pread`, which carries its own
    /// offset, so concurrent reads can share one descriptor per shard.
    static func read(
        _ range: GGUFTensorRange, into buffer: MTLBuffer, name: String, descriptor: Int32
    ) throws -> Int {
        guard Int(range.byteCount) <= buffer.length else {
            throw GGUFExpertBufferCacheError.shortRead(
                name: name, expected: Int(range.byteCount), got: buffer.length)
        }
        let destination = buffer.contents()
        var total = 0
        while total < Int(range.byteCount) {
            let count = min(Int(range.byteCount) - total, 1 << 20)
            let read = pread(
                descriptor, destination.advanced(by: total), count,
                off_t(range.fileOffset) + off_t(total))
            if read <= 0 {
                throw GGUFExpertBufferCacheError.shortRead(
                    name: name, expected: Int(range.byteCount), got: total)
            }
            total += read
        }
        return total
    }

    /// unchecked-invariant: each job index is run by exactly one task and the
    /// inputs are immutable; only the failure slot is shared, behind a lock.
    final class ReadJobs: @unchecked Sendable {
        struct Job {
            let range: GGUFTensorRange
            let buffer: MTLBuffer
            let descriptor: Int32
            let name: String
        }

        let jobs: [Job]
        private let lock = NSLock()
        private var failure: Error?

        init(_ jobs: [Job]) { self.jobs = jobs }

        func run(_ index: Int) {
            let job = jobs[index]
            do {
                _ = try GGUFExpertBufferCache.read(
                    job.range, into: job.buffer, name: job.name, descriptor: job.descriptor)
            } catch {
                lock.lock()
                if failure == nil { failure = error }
                lock.unlock()
            }
        }

        /// Runs every job on the calling thread's concurrent pool and returns
        /// when all are done.
        func runAll() throws {
            DispatchQueue.concurrentPerform(iterations: jobs.count) { run($0) }
            if let error = firstFailure { throw error }
        }

        var firstFailure: Error? {
            lock.lock()
            defer { lock.unlock() }
            return failure
        }
    }
}
