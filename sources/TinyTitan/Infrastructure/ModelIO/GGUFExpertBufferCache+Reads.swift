import Foundation
import Metal

/// The read mechanics behind `GGUFExpertBufferCache`: `pread` loops over
/// pieces of a matrix (`pread` carries its own file offset, so concurrent
/// reads share one descriptor per shard), and a batch of them that runs
/// concurrently and remembers the first failure. The cache decides what to
/// read and into which slot; this file only moves bytes.
extension GGUFExpertBufferCache {
    /// Reads bytes `piece` of one matrix (relative to its start) into its
    /// place at `offset + piece.lowerBound` of `buffer`.
    static func readPiece(
        _ range: GGUFTensorRange, piece: Range<Int>, into buffer: MTLBuffer, offset: Int,
        capacity: Int, name: String, descriptor: Int32
    ) throws {
        guard Int(range.byteCount) <= capacity, offset + capacity <= buffer.length,
            piece.upperBound <= Int(range.byteCount)
        else {
            throw GGUFExpertBufferCacheError.shortRead(
                name: name, expected: Int(range.byteCount), got: capacity)
        }
        let destination = buffer.contents().advanced(by: offset)
        var total = piece.lowerBound
        while total < piece.upperBound {
            let read = pread(
                descriptor, destination.advanced(by: total), piece.upperBound - total,
                off_t(range.fileOffset) + off_t(total))
            if read <= 0 {
                throw GGUFExpertBufferCacheError.shortRead(
                    name: name, expected: piece.count, got: total - piece.lowerBound)
            }
            total += read
        }
    }

    /// unchecked-invariant: each (job, piece) is run by exactly one task and the
    /// inputs are immutable; only the failure slot is shared, behind a lock.
    final class ReadJobs: @unchecked Sendable {
        struct Job {
            let range: GGUFTensorRange
            let buffer: MTLBuffer
            let offset: Int
            let capacity: Int
            let descriptor: Int32
            let name: String
        }

        let jobs: [Job]
        private let lock = NSLock()
        private var failure: Error?

        /// Bytes per `pread`. A layer's misses are a few matrices of 0.3-0.9
        /// MB, so at 1 MiB a matrix is one read. Measured at `--ram-budget
        /// 10G`, two interleaved rounds, mean fetch per token: 64 KiB 163/205
        /// ms, 128 KiB 198/161, 256 KiB 157/165, 1 MiB 124/133 - smaller
        /// pieces only added syscalls and thread hand-offs.
        static let pieceBytes = 1 << 20

        /// The pool the pieces run on. `concurrentPerform` runs one item per
        /// CPU core, so a blocking read held the queue depth at the core count;
        /// work submitted to a concurrent queue gets more threads when the
        /// ones it has block in the kernel.
        private static let queue = DispatchQueue(
            label: "tinytitan.gguf.expert-reads", qos: .userInitiated, attributes: .concurrent)

        init(_ jobs: [Job]) { self.jobs = jobs }

        private func run(_ index: Int, piece: Range<Int>) {
            let job = jobs[index]
            do {
                try GGUFExpertBufferCache.readPiece(
                    job.range, piece: piece, into: job.buffer, offset: job.offset,
                    capacity: job.capacity, name: job.name, descriptor: job.descriptor)
            } catch {
                lock.lock()
                if failure == nil { failure = error }
                lock.unlock()
            }
        }

        private let group = DispatchGroup()

        /// Starts every job, cut into `pieceBytes` reads, on the concurrent
        /// queue and returns at once; `wait()` returns when all are done.
        func start() {
            for (index, job) in jobs.enumerated() {
                let length = Int(job.range.byteCount)
                for begin in stride(from: 0, to: length, by: Self.pieceBytes) {
                    let piece = begin..<min(length, begin + Self.pieceBytes)
                    Self.queue.async(group: group) { self.run(index, piece: piece) }
                }
            }
        }

        /// Waits for every started read and throws the first failure.
        func wait() throws {
            group.wait()
            if let error = firstFailure { throw error }
        }

        /// Runs every job and returns when all are done.
        func runAll() throws {
            start()
            try wait()
        }

        var firstFailure: Error? {
            lock.lock()
            defer { lock.unlock() }
            return failure
        }
    }
}
