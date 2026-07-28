import Foundation
import Darwin

/// Shared building blocks for the resident LM and routed-expert layer writers.
public enum WriterCore {

    /// Tile size for pwrite (and the subsequent SHA-256 hashing pass). Chosen
    /// so per-worker scratch and per-syscall payload both stay well under
    /// the 1 MB BoundedScratch budget.
    public static let tileBytes: Int = 512 * 1024

    /// Copy `size` bytes from `srcShard.base + srcOffset` to file `dstFd` at
    /// `dstOffset`, in pwrite-sized tiles. Pages consumed from the source map
    /// are evicted via madvise after each tile, capping the source-side RSS.
    public static func pwriteTensorRegion(srcShard: MmapHandle,
                                          srcAbsoluteOffset: UInt64,
                                          size: UInt64,
                                          dstFd: Int32, dstPath: String,
                                          dstOffset: UInt64,
                                          audit: RepackAudit) throws {
        var remaining = Int(size)
        var srcOff = srcAbsoluteOffset
        var dstOff = dstOffset
        let tile = WriterCore.tileBytes
        while remaining > 0 {
            let n = min(remaining, tile)
            let p = srcShard.base.advanced(by: Int(srcOff))
            try Posix.pwriteAll(fd: dstFd, path: dstPath, buf: p, count: n, offset: dstOff)
            audit.recordTile(bytes: n)
            audit.recordWrite(bytes: n)
            audit.recordRead(bytes: n)
            srcShard.adviseDontNeed(offset: srcOff, count: n)
            srcOff += UInt64(n)
            dstOff += UInt64(n)
            remaining -= n
        }
    }

    /// Compute SHA-256 of an entire (presumed-written) file by streaming it
    /// through `tileBytes` pread chunks. Drops pages with `F_NOCACHE` style
    /// behaviour via fcntl. Allocates one bounded scratch buffer.
    public static func hashEntireFile(path: String, size: UInt64,
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
