import Foundation

/// A process-wide pool of staging buffers for compressed expert frame reads.
///
/// One buffer per concurrent decode, and they are shared across every open
/// layer. The alternative — a scratch buffer per streamer — costs 40 open
/// layers times a frame each, ~50 MB before the pool has to hold more than one,
/// and the decode path only ever has a handful of reads in flight: one per
/// demand worker, plus the prefetch ring's.
///
/// **Why a pool at all.** The first version of the compressed path reused one
/// buffer under a lock. On Ornith 4-bit that put every expert decode in the
/// token behind one mutex and one core, and decode fell from 10.0 to 3.2 tok/s —
/// exactly the single-core LZFSE bound for the ~204 MB a token misses. A
/// per-decode buffer is what lets the misses decode in parallel.
/// unchecked-invariant: `free` and the borrow/give-back pairing are guarded by
/// `lock`, and a buffer is removed from `free` before it is handed out, so no
/// two decoders can hold the same pointer.
final class CompressedScratchPool: @unchecked Sendable {
    static let shared = CompressedScratchPool()

    /// Buffers kept for reuse. More than this is discarded rather than held:
    /// the steady state is one per concurrent decode, and a burst should not
    /// turn into retained memory.
    private static let limit = 8

    private let lock = NSLock()
    private var free: [(pointer: UnsafeMutableRawPointer, bytes: Int)] = []

    /// A buffer of at least `bytes`, from the pool or freshly allocated.
    ///
    /// `giveBack` must receive the same `bytes`, and the pointer must not be
    /// used after it.
    func borrow(bytes: Int) -> UnsafeMutableRawPointer {
        precondition(bytes > 0)
        lock.lock()
        if let index = free.lastIndex(where: { $0.bytes >= bytes }) {
            let entry = free.remove(at: index)
            lock.unlock()
            return entry.pointer
        }
        lock.unlock()
        return .allocate(byteCount: bytes, alignment: 4096)
    }

    func giveBack(_ pointer: UnsafeMutableRawPointer, bytes: Int) {
        lock.lock()
        guard free.count < Self.limit else {
            lock.unlock()
            pointer.deallocate()
            return
        }
        free.append((pointer, bytes))
        lock.unlock()
    }
}
