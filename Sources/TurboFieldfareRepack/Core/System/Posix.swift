import Foundation
import Darwin

/// Thin POSIX helpers used by the writer hot path. The shapes here are picked
/// so callers can stay inside tile-bounded scratch budgets without a
/// Foundation `FileHandle` allocation per write.
enum Posix {
    static func openRead(_ path: String) throws -> Int32 {
        let fd = open(path, O_RDONLY)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        return fd
    }

    static func openCreateRW(_ path: String) throws -> Int32 {
        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        return fd
    }

    static func ftruncate(_ fd: Int32, path: String, size: UInt64) throws {
        if Darwin.ftruncate(fd, off_t(size)) != 0 {
            throw RepackError.ftruncateFailed(path: path, errno: errno)
        }
    }

    static func pwriteAll(fd: Int32, path: String,
                                 buf: UnsafeRawPointer, count: Int,
                                 offset: UInt64) throws {
        var remaining = count
        var off = off_t(offset)
        var ptr = buf
        while remaining > 0 {
            let n = pwrite(fd, ptr, remaining, off)
            if n <= 0 {
                throw RepackError.pwriteShort(path: path, expected: count,
                                              wrote: count - remaining, errno: errno)
            }
            remaining -= n
            off += off_t(n)
            ptr = ptr.advanced(by: n)
        }
    }

    static func preadAll(fd: Int32, path: String,
                                buf: UnsafeMutableRawPointer, count: Int,
                                offset: UInt64) throws {
        var remaining = count
        var off = off_t(offset)
        var ptr = buf
        while remaining > 0 {
            let n = pread(fd, ptr, remaining, off)
            if n <= 0 {
                throw RepackError.preadShort(path: path, expected: count,
                                             got: count - remaining, errno: errno)
            }
            remaining -= n
            off += off_t(n)
            ptr = ptr.advanced(by: n)
        }
    }

    static func fsync(_ fd: Int32, path: String) throws {
        if Darwin.fsync(fd) != 0 {
            throw RepackError.fsyncFailed(path: path, errno: errno)
        }
    }

    static func rename(from src: String, to dst: String) throws {
        if Darwin.rename(src, dst) != 0 {
            throw RepackError.renameFailed(from: src, to: dst, errno: errno)
        }
    }

    static func mkdirP(_ path: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    static func fileSize(fd: Int32, path: String) throws -> UInt64 {
        var st = stat()
        if fstat(fd, &st) != 0 {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        return UInt64(st.st_size)
    }

}
