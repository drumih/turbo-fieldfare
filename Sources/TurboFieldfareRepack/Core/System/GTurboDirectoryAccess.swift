import Darwin
import Foundation
import TurboFieldfareFormat

package final class GTurboDirectoryAccess {
    package let rootPath: String
    private let rootFD: Int32

    package init(rootPath: String) throws {
        let standardized = URL(fileURLWithPath: rootPath).standardizedFileURL.path
        let fd = open(standardized, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard fd >= 0 else {
            throw RepackError.fileOpenFailed(path: standardized, errno: errno)
        }
        self.rootPath = standardized
        self.rootFD = fd
    }

    deinit { close(rootFD) }

    package var directoryFileDescriptor: Int32 { rootFD }

    package func openFile(_ relativePath: String) throws -> Int32 {
        do {
            try GTurboPathValidator.validateRelativePath(relativePath,
                                                         field: "path.\(relativePath)")
        } catch {
            throw RepackError.configurationInvalid(detail: "unsafe path \(relativePath): \(error)")
        }
        let components = relativePath.components(separatedBy: "/")
        var directoryFD = fcntl(rootFD, F_DUPFD_CLOEXEC, 0)
        guard directoryFD >= 0 else {
            throw RepackError.fileOpenFailed(path: rootPath, errno: errno)
        }
        for component in components.dropLast() {
            let next = openat(directoryFD, component,
                              O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            let savedErrno = errno
            close(directoryFD)
            guard next >= 0 else {
                if savedErrno == ENOENT {
                    throw RepackError.fileStatFailed(
                        path: "\(rootPath)/\(relativePath)", errno: savedErrno)
                }
                throw RepackError.fileOpenFailed(
                    path: "\(rootPath)/\(relativePath)", errno: savedErrno)
            }
            directoryFD = next
        }
        let fd = openat(directoryFD, components.last!,
                        O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        let savedErrno = errno
        close(directoryFD)
        guard fd >= 0 else {
            if savedErrno == ENOENT {
                throw RepackError.fileStatFailed(path: "\(rootPath)/\(relativePath)",
                                                 errno: savedErrno)
            }
            throw RepackError.fileOpenFailed(path: "\(rootPath)/\(relativePath)",
                                             errno: savedErrno)
        }
        var st = stat()
        guard fstat(fd, &st) == 0, (st.st_mode & S_IFMT) == S_IFREG else {
            let statErrno = errno
            close(fd)
            throw RepackError.fileStatFailed(path: "\(rootPath)/\(relativePath)",
                                             errno: statErrno)
        }
        return fd
    }

    package func readMetadata(_ relativePath: String,
                              maxBytes: UInt64) throws -> Data {
        let fd = try openFile(relativePath)
        defer { close(fd) }
        return try readMetadata(fileDescriptor: fd,
                                relativePath: relativePath,
                                maxBytes: maxBytes)
    }

    package func readMetadata(fileDescriptor fd: Int32,
                              relativePath: String,
                              maxBytes: UInt64) throws -> Data {
        let size = try fileSize(fileDescriptor: fd, relativePath: relativePath)
        guard size <= maxBytes, size <= UInt64(Int.max) else {
            throw RepackError.configurationInvalid(
                detail: "\(relativePath) size \(size) exceeds metadata cap \(maxBytes)")
        }
        var data = Data(count: Int(size))
        if !data.isEmpty {
            try data.withUnsafeMutableBytes {
                try Posix.preadAll(
                    fd: fd, path: "\(rootPath)/\(relativePath)",
                    buf: $0.baseAddress!, count: $0.count, offset: 0)
            }
        }
        return data
    }

    package func fileSize(_ relativePath: String) throws -> UInt64 {
        let fd = try openFile(relativePath)
        defer { close(fd) }
        return try fileSize(fileDescriptor: fd, relativePath: relativePath)
    }

    package func relativeEntries(maxDepth: Int = 16,
                                 maxEntries: Int = 100_000) throws -> [String] {
        guard maxDepth >= 0, maxEntries > 0 else {
            throw RepackError.configurationInvalid(detail: "invalid directory scan bounds")
        }
        var result: [String] = []
        try collectEntries(directoryFD: rootFD,
                           prefix: "",
                           depth: 0,
                           maxDepth: maxDepth,
                           maxEntries: maxEntries,
                           result: &result)
        return result.sorted()
    }

    private func collectEntries(directoryFD: Int32,
                                prefix: String,
                                depth: Int,
                                maxDepth: Int,
                                maxEntries: Int,
                                result: inout [String]) throws {
        let scanFD = openat(directoryFD, ".",
                            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard scanFD >= 0, let directory = fdopendir(scanFD) else {
            if scanFD >= 0 { close(scanFD) }
            throw RepackError.fileOpenFailed(
                path: prefix.isEmpty ? rootPath : "\(rootPath)/\(prefix)", errno: errno)
        }
        defer { closedir(directory) }
        while true {
            errno = 0
            guard let entry = readdir(directory) else {
                if errno != 0 {
                    throw RepackError.fileOpenFailed(
                        path: prefix.isEmpty ? rootPath : "\(rootPath)/\(prefix)", errno: errno)
                }
                break
            }
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) {
                    String(cString: $0)
                }
            }
            if name == "." || name == ".." { continue }
            let relativePath = prefix.isEmpty ? name : "\(prefix)/\(name)"
            var info = stat()
            guard fstatat(directoryFD, name, &info, AT_SYMLINK_NOFOLLOW) == 0 else {
                // readdir returns a buffered snapshot, so an entry can be
                // renamed or unlinked before this stat runs. A vanished entry
                // is ordinary concurrent-directory behavior, not a damaged
                // install, and reporting it also has to be skipped.
                if errno == ENOENT { continue }
                throw RepackError.fileStatFailed(
                    path: "\(rootPath)/\(relativePath)", errno: errno)
            }
            result.append(relativePath)
            guard result.count <= maxEntries else {
                throw RepackError.configurationInvalid(detail: "artifact directory has too many entries")
            }
            if (info.st_mode & S_IFMT) == S_IFDIR {
                if prefix.isEmpty, name == "tokenizer" { continue }
                // Directory-shaped OS sidecars (.Spotlight-V100, .fseventsd,
                // .TemporaryItems) land beside an artifact after any Finder or
                // SMB round-trip. Descending into a Spotlight index is never
                // right, and refusing the whole artifact over one is worse:
                // list the entry so callers can filter it, and stop there.
                if GTurboVisionFormatV1.isSidecarEntry(name) { continue }
                guard depth < maxDepth else {
                    throw RepackError.configurationInvalid(
                        detail: "artifact directory nesting exceeds \(maxDepth)")
                }
                let childFD = openat(directoryFD, name,
                                     O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                guard childFD >= 0 else {
                    // Same snapshot race as the stat above, one level up: the
                    // directory can be removed between fstatat and openat. Drop
                    // the entry recorded for it rather than reporting a name
                    // that no longer exists.
                    if errno == ENOENT {
                        result.removeLast()
                        continue
                    }
                    throw RepackError.fileOpenFailed(
                        path: "\(rootPath)/\(relativePath)", errno: errno)
                }
                do {
                    try collectEntries(directoryFD: childFD,
                                       prefix: relativePath,
                                       depth: depth + 1,
                                       maxDepth: maxDepth,
                                       maxEntries: maxEntries,
                                       result: &result)
                    close(childFD)
                } catch {
                    close(childFD)
                    throw error
                }
            }
        }
    }

    package func hash(_ relativePath: String, noCache: Bool) throws -> String {
        let fd = try openFile(relativePath)
        defer { close(fd) }
        return try hash(fileDescriptor: fd,
                        relativePath: relativePath,
                        noCache: noCache)
    }

    package func hash(fileDescriptor fd: Int32,
                      relativePath: String,
                      noCache: Bool) throws -> String {
        return try Sha256Stream.hashFileDescriptor(
            fd, displayPath: "\(rootPath)/\(relativePath)", noCache: noCache)
    }

    package func atomicWrite(_ data: Data, to relativePath: String) throws {
        do {
            try GTurboPathValidator.validateBasename(
                relativePath, field: "output.\(relativePath)")
        } catch {
            throw RepackError.configurationInvalid(
                detail: "unsafe output path \(relativePath): \(error)")
        }
        let temporary = ".gturbo-\(UUID().uuidString).tmp"
        var fd = openat(rootFD, temporary,
                        O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                        0o600)
        guard fd >= 0 else {
            throw RepackError.fileOpenFailed(path: "\(rootPath)/\(temporary)", errno: errno)
        }
        do {
            try data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                try Posix.pwriteAll(
                    fd: fd, path: "\(rootPath)/\(temporary)",
                    buf: base, count: raw.count, offset: 0)
            }
            try Posix.fsync(fd, path: "\(rootPath)/\(temporary)")
            close(fd)
            fd = -1
            guard renameat(rootFD, temporary, rootFD, relativePath) == 0 else {
                throw RepackError.renameFailed(
                    from: "\(rootPath)/\(temporary)",
                    to: "\(rootPath)/\(relativePath)", errno: errno)
            }
            try Posix.fsync(rootFD, path: rootPath)
        } catch {
            if fd >= 0 { close(fd) }
            _ = unlinkat(rootFD, temporary, 0)
            throw error
        }
    }

    package func fileSize(fileDescriptor fd: Int32,
                          relativePath: String) throws -> UInt64 {
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw RepackError.fileStatFailed(path: "\(rootPath)/\(relativePath)", errno: errno)
        }
        guard st.st_size >= 0 else {
            throw RepackError.configurationInvalid(detail: "\(relativePath) has negative file size")
        }
        return UInt64(st.st_size)
    }
}
