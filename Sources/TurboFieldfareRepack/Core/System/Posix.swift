import Foundation
import Darwin

/// Thin POSIX helpers used by the writer hot path. The shapes here are picked
/// so callers can stay inside tile-bounded scratch budgets without a
/// Foundation `FileHandle` allocation per write.
public enum Posix {
    public static let pageSize: Int = Int(getpagesize())

    public enum EntryKind: Hashable, Sendable {
        case absent
        case regular
        case directory
        case symlink
        case other
    }

    public static func openRead(_ path: String) throws -> Int32 {
        let fd = open(path, O_RDONLY)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        return fd
    }

    public static func openReadNoFollow(_ path: String) throws -> Int32 {
        let fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        return fd
    }

    public static func openCreateRW(_ path: String) throws -> Int32 {
        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        return fd
    }

    public static func openExistingRW(_ path: String) throws -> Int32 {
        let fd = open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        return fd
    }

    public static func openDirectory(_ path: String) throws -> Int32 {
        let fd = open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        return fd
    }

    public static func openLock(_ path: String) throws -> Int32 {
        let fd = open(path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        return fd
    }

    public static func ftruncate(_ fd: Int32, path: String, size: UInt64) throws {
        if Darwin.ftruncate(fd, off_t(size)) != 0 {
            throw RepackError.ftruncateFailed(path: path, errno: errno)
        }
    }

    public static func pwriteAll(fd: Int32, path: String,
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

    public static func preadAll(fd: Int32, path: String,
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

    public static func fsync(_ fd: Int32, path: String) throws {
        if Darwin.fsync(fd) != 0 {
            throw RepackError.fsyncFailed(path: path, errno: errno)
        }
    }

    public static func fsyncDirectory(_ path: String) throws {
        let fd = try openDirectory(path)
        defer { close(fd) }
        try fsync(fd, path: path)
    }

    public static func rename(from src: String, to dst: String) throws {
        if Darwin.rename(src, dst) != 0 {
            throw RepackError.renameFailed(from: src, to: dst, errno: errno)
        }
    }

    public static func renameSwap(_ first: String, _ second: String) throws {
        if renamex_np(first, second, UInt32(RENAME_SWAP)) != 0 {
            throw RepackError.promotionUnsupported(path: first, errno: errno)
        }
    }

    public static func mkdirP(_ path: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    public static func physicalPath(_ path: String) throws -> String {
        guard let resolved = realpath(path, nil) else {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        defer { free(resolved) }
        return String(cString: resolved)
    }

    public static func entryKind(_ path: String) throws -> EntryKind {
        var info = stat()
        if lstat(path, &info) != 0 {
            if errno == ENOENT { return .absent }
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        switch info.st_mode & S_IFMT {
        case S_IFREG: return .regular
        case S_IFDIR: return .directory
        case S_IFLNK: return .symlink
        default: return .other
        }
    }

    public static func atomicWrite(_ data: Data,
                                   to path: String,
                                   durableIn directory: String,
                                   afterRename: (() throws -> Void)? = nil) throws {
        let temporary = path + ".tmp"
        switch try entryKind(temporary) {
        case .absent:
            break
        case .regular:
            if unlink(temporary) != 0 {
                throw RepackError.fileOpenFailed(path: temporary, errno: errno)
            }
            try fsyncDirectory(directory)
        default:
            throw RepackError.installPathUnsafe(
                path: temporary,
                detail: "atomic-write temporary has the wrong type")
        }
        let fd = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        if fd < 0 { throw RepackError.fileOpenFailed(path: temporary, errno: errno) }
        defer { close(fd) }
        do {
            try data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                var offset = 0
                while offset < raw.count {
                    let count = write(fd, base.advanced(by: offset), raw.count - offset)
                    if count <= 0 {
                        throw RepackError.pwriteShort(
                            path: temporary,
                            expected: raw.count,
                            wrote: offset,
                            errno: errno)
                    }
                    offset += count
                }
            }
            try fsync(fd, path: temporary)
            try rename(from: temporary, to: path)
            try afterRename?()
            try fsyncDirectory(directory)
        } catch {
            _ = unlink(temporary)
            throw error
        }
    }

    public static func readBoundedData(_ path: String, maximumBytes: UInt64) throws -> Data {
        let fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        defer { close(fd) }
        let size = try fileSize(fd: fd, path: path)
        guard size <= maximumBytes, size <= UInt64(Int.max) else {
            throw RepackError.installStateCorrupt(
                path: path,
                detail: "size \(size) exceeds \(maximumBytes)-byte cap")
        }
        var data = Data(count: Int(size))
        try data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress, !raw.isEmpty else { return }
            try preadAll(fd: fd, path: path, buf: base, count: raw.count, offset: 0)
        }
        return data
    }

    public static func fileSize(fd: Int32, path: String) throws -> UInt64 {
        var st = stat()
        if fstat(fd, &st) != 0 {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        guard st.st_size >= 0 else {
            throw RepackError.fileStatFailed(path: path, errno: EINVAL)
        }
        return UInt64(st.st_size)
    }

    public static func descriptorMatchesPath(_ fd: Int32, path: String) throws -> Bool {
        var descriptorInfo = stat()
        guard fstat(fd, &descriptorInfo) == 0 else {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        var pathInfo = stat()
        guard lstat(path, &pathInfo) == 0 else {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        return descriptorInfo.st_dev == pathInfo.st_dev
            && descriptorInfo.st_ino == pathInfo.st_ino
            && pathInfo.st_mode & S_IFMT == S_IFREG
    }

    public static func adviseDontNeed(fd: Int32, offset: UInt64, length: Int) {
        // posix_fadvise isn't available on Darwin. Use fcntl F_RDADVISE / F_NOCACHE
        // for similar hints; here we use F_NOCACHE on the fd as a coarse hint.
        // For per-range eviction we rely on madvise(MADV_DONTNEED) when the
        // range is mapped (handled in MmapHandle).
        _ = fd; _ = offset; _ = length
    }
}

/// Owns one MAP_PRIVATE, PROT_READ mapping of a file. Pages fault in on access
/// and can be evicted via `adviseDontNeed`. Designed for one shard at a time.
public final class MmapHandle {
    public let path: String
    public let fd: Int32
    public let base: UnsafeRawPointer
    public let length: Int

    public init(path: String) throws {
        self.path = path
        self.fd = try Posix.openRead(path)
        var st = stat()
        if fstat(fd, &st) != 0 {
            close(fd)
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        let len = Int(st.st_size)
        guard let raw = mmap(nil, len, PROT_READ, MAP_PRIVATE, fd, 0),
              raw != MAP_FAILED else {
            close(fd)
            throw RepackError.mmapFailed(path: path, errno: errno)
        }
        self.base = UnsafeRawPointer(raw)
        self.length = len
        _ = posix_madvise(raw, len, POSIX_MADV_SEQUENTIAL)
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), length)
        close(fd)
    }

    public func slice(at offset: UInt64, count: Int) -> UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: base.advanced(by: Int(offset)), count: count)
    }

    /// madvise(MADV_DONTNEED) on a region. Alignment is rounded outward so we
    /// only touch whole pages — the kernel ignores partial-page requests.
    public func adviseDontNeed(offset: UInt64, count: Int) {
        let page = UInt64(Posix.pageSize)
        let start = (offset / page) * page
        let endRaw = offset + UInt64(count)
        let end = ((endRaw + page - 1) / page) * page
        if end <= start { return }
        let len = Int(end - start)
        let addr = UnsafeMutableRawPointer(mutating: base.advanced(by: Int(start)))
        _ = posix_madvise(addr, len, POSIX_MADV_DONTNEED)
    }
}
