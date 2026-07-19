import Foundation
import Darwin

/// Shared building blocks for the resident LM and routed-expert layer writers.
enum WriterCore {

    /// Tile size for pwrite (and the subsequent SHA-256 hashing pass). Chosen
    /// so per-worker scratch and per-syscall payload both stay well under 1 MB.
    static let tileBytes: Int = 512 * 1024

    /// Compute SHA-256 of an entire (presumed-written) file by streaming it
    /// through `tileBytes` pread chunks. Drops pages with `F_NOCACHE` style
    /// behaviour via fcntl. Allocates one bounded scratch buffer.
    static func hashEntireFile(path: String, size: UInt64,
                                      audit: RepackAudit,
                                      cancellationCheck: () throws -> Void = {}) throws -> String {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        // Hint the kernel that we will read this file sequentially and then
        // drop it from cache — keeps the post-write working set from blowing
        // up the dev box.
        _ = fcntl(fd, F_NOCACHE, 1)

        let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: WriterCore.tileBytes,
                                                         alignment: 16_384)
        defer { buf.deallocate() }
        if buf.count > audit.largestScratchBytes {
            audit.largestScratchBytes = buf.count
        }

        var hasher = Sha256Stream()
        var off: UInt64 = 0
        let total = Int(size)
        var remaining = total
        while remaining > 0 {
            try cancellationCheck()
            let want = min(remaining, WriterCore.tileBytes)
            let got = pread(fd, buf.baseAddress, want, off_t(off))
            if got <= 0 {
                throw RepackError.preadShort(path: path, expected: want, got: 0, errno: errno)
            }
            hasher.update(UnsafeRawBufferPointer(start: buf.baseAddress, count: got))
            audit.byteCopyTiles &+= 1
            off += UInt64(got)
            remaining -= got
        }
        return hasher.finalizeHexString()
    }
}
