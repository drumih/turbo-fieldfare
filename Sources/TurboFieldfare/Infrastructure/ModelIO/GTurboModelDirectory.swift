import Darwin
import Foundation
import TurboFieldfareFormat

package final class GTurboModelDirectory {
    package let rootURL: URL
    private let rootFD: Int32

    package init(rootURL: URL) throws {
        let standardized = rootURL.standardizedFileURL
        let fd = open(standardized.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(standardized.path))", errno: errno)
        }
        self.rootURL = standardized
        self.rootFD = fd
    }

    deinit { close(rootFD) }

    package func openFile(_ relativePath: String) throws -> Int32 {
        do {
            try GTurboPathValidator.validateRelativePath(relativePath,
                                                         field: "path.\(relativePath)")
        } catch {
            throw ModelError.indexCorrupt(detail: "unsafe path \(relativePath): \(error)")
        }
        let components = relativePath.components(separatedBy: "/")
        var directoryFD = fcntl(rootFD, F_DUPFD_CLOEXEC, 0)
        guard directoryFD >= 0 else {
            throw ModelError.posixFailed(call: "fcntl(F_DUPFD_CLOEXEC, model root)", errno: errno)
        }
        for component in components.dropLast() {
            let next = openat(directoryFD, component,
                              O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            let savedErrno = errno
            close(directoryFD)
            guard next >= 0 else {
                throw openError(relativePath: relativePath, errno: savedErrno)
            }
            directoryFD = next
        }
        let fd = openat(directoryFD, components.last!,
                        O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        let savedErrno = errno
        close(directoryFD)
        guard fd >= 0 else { throw openError(relativePath: relativePath, errno: savedErrno) }
        var st = stat()
        guard fstat(fd, &st) == 0, (st.st_mode & S_IFMT) == S_IFREG else {
            let statErrno = errno
            close(fd)
            throw ModelError.posixFailed(call: "fstat(\(relativePath))", errno: statErrno)
        }
        return fd
    }

    package func readMetadata(_ relativePath: String,
                              maxBytes: UInt64) throws -> Data {
        let fd = try openFile(relativePath)
        defer { close(fd) }
        return try readMetadata(fileDescriptor: fd, relativePath: relativePath,
                                maxBytes: maxBytes)
    }

    package func readMetadata(fileDescriptor fd: Int32,
                              relativePath: String,
                              maxBytes: UInt64) throws -> Data {
        let size = try fileSize(fileDescriptor: fd, relativePath: relativePath)
        guard size <= maxBytes, size <= UInt64(Int.max) else {
            throw ModelError.indexCorrupt(
                detail: "\(relativePath) size \(size) exceeds metadata cap \(maxBytes)")
        }
        var data = Data(count: Int(size))
        var total = 0
        while total < data.count {
            let remaining = data.count - total
            let got = data.withUnsafeMutableBytes { raw in
                pread(fd, raw.baseAddress!.advanced(by: total), remaining, off_t(total))
            }
            if got < 0, errno == EINTR { continue }
            guard got > 0 else {
                throw ModelError.posixFailed(call: "pread(\(relativePath))",
                                             errno: got < 0 ? errno : EIO)
            }
            total += got
        }
        return data
    }

    package func fileSize(_ relativePath: String) throws -> UInt64 {
        let fd = try openFile(relativePath)
        defer { close(fd) }
        return try fileSize(fileDescriptor: fd, relativePath: relativePath)
    }

    package func fileSize(fileDescriptor fd: Int32,
                          relativePath: String) throws -> UInt64 {
        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size >= 0 else {
            throw ModelError.posixFailed(call: "fstat(\(relativePath))", errno: errno)
        }
        return UInt64(st.st_size)
    }

    private func openError(relativePath: String, errno: Int32) -> ModelError {
        if errno == ENOENT {
            return ModelError.missingFile(name: relativePath)
        }
        return ModelError.posixFailed(call: "openat(\(relativePath))", errno: errno)
    }
}
