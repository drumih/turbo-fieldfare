import Darwin
import Foundation

public enum DecodeUnixSocket {
    public static func connect(path: String) throws -> (input: FileHandle, output: FileHandle) {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        do {
            var address = try makeAddress(path: path)
            let result = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
                }
            }
            guard result == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            return try handles(for: fd)
        } catch {
            Darwin.close(fd)
            throw error
        }
    }

    public static func listenAndAccept(path: String) throws
        -> (input: FileHandle, output: FileHandle) {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        do {
            var address = try makeAddress(path: path)
            unlink(path)
            let bindResult = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
                }
            }
            guard bindResult == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            guard listen(fd, 1) == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            let accepted = Darwin.accept(fd, nil, nil)
            guard accepted >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            Darwin.close(fd)
            return try handles(for: accepted)
        } catch {
            Darwin.close(fd)
            unlink(path)
            throw error
        }
    }

    private static func handles(for fd: Int32) throws
        -> (input: FileHandle, output: FileHandle) {
        let outputFD = dup(fd)
        guard outputFD >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        return (FileHandle(fileDescriptor: fd, closeOnDealloc: true),
                FileHandle(fileDescriptor: outputFD, closeOnDealloc: true))
    }

    private static func makeAddress(path: String) throws -> sockaddr_un {
        let bytes = Array(path.utf8)
        var address = sockaddr_un()
        guard bytes.count < MemoryLayout.size(ofValue: address.sun_path) else {
            throw POSIXError(.ENAMETOOLONG)
        }
        address.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutableBytes(of: &address.sun_path) { destination in
            destination.copyBytes(from: bytes)
            destination[bytes.count] = 0
        }
        return address
    }
}
