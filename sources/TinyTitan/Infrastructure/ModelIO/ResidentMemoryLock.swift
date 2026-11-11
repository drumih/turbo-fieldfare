import Darwin
import Foundation
import Metal

/// Pins a shared Metal buffer's pages in RAM with `mlock`.
///
/// A shared-storage buffer is ordinary process memory between command
/// buffers: under pressure the system compresses or swaps it, and the next
/// dispatch that reads it waits for the pages to come back. A process given a
/// RAM budget has already been told it may hold these bytes, so it pins them.
/// Measured at `--ram-budget 10G` on the 24 GB M3 with 12 GB of system swap in
/// use: unpinned, a token alternated between ~0.3 s and 1.1-1.7 s of GPU waits
/// as the pages were evicted and faulted back.
package enum ResidentMemoryLock {
    /// Locks `buffer`'s pages. Returns nil on success, or the `errno` of the
    /// refusal; a refusal is not fatal (the buffer still works, unpinned), so
    /// the caller reports it and carries on.
    @discardableResult
    package static func lock(_ buffer: MTLBuffer) -> Int32? {
        guard buffer.storageMode == .shared, buffer.length > 0 else { return nil }
        return mlock(buffer.contents(), buffer.length) == 0 ? nil : errno
    }

    /// Locks every buffer and writes one stderr line naming how many bytes
    /// could not be pinned, if any.
    package static func lock(_ buffers: [MTLBuffer], label: String) {
        var refused = 0
        var lastError: Int32 = 0
        for buffer in buffers {
            if let error = lock(buffer) {
                refused += buffer.length
                lastError = error
            }
        }
        guard refused > 0 else { return }
        FileHandle.standardError.write(
            Data(
                ("TinyTitan memory: could not pin \(refused / 1_048_576) MiB of \(label) "
                    + "(errno \(lastError)); it stays in RAM only while the system allows\n").utf8))
    }
}
