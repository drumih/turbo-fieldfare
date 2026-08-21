import Darwin
import Foundation

public enum DecodeUnixSocket {
    /// Both ends write to a peer that can die at any time, so the signal has to
    /// be off before the first write - otherwise a dead decode service takes
    /// the app with it, and a dead app takes the service.
    public static func ignoreSIGPIPEProcessWide() {
        signal(SIGPIPE, SIG_IGN)
    }

    /// Per-descriptor belt to the process-wide braces above: it works on pipes
    /// as well as sockets, and it survives anything that resets the handler.
    public static func disableSIGPIPE(on descriptor: Int32) {
        var enabled: Int32 = 1
        _ = fcntl(descriptor, F_SETNOSIGPIPE, 1)
        _ = setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE,
                       &enabled, socklen_t(MemoryLayout<Int32>.size))
    }

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
        var listeningClosed = false
        do {
            var address = try makeAddress(path: path)
            unlink(path)
            let bindResult = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
                }
            }
            guard bindResult == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            // The socket lives in a world-writable directory, so without this
            // any local user can connect in the window before the app does and
            // drive the service.
            guard chmod(path, 0o600) == 0 else {
                throw POSIXError(.init(rawValue: errno) ?? .EIO)
            }
            guard listen(fd, 1) == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            let accepted = try acceptFromOwner(fd)
            Darwin.close(fd)
            listeningClosed = true
            do {
                return try handles(for: accepted)
            } catch {
                Darwin.close(accepted)
                throw error
            }
        } catch {
            // The listening descriptor is closed once, at the handover. Closing
            // it again here could close a number another thread now owns.
            if !listeningClosed { Darwin.close(fd) }
            unlink(path)
            throw error
        }
    }

    /// Accepts only a connection from this user, dropping anything else and
    /// continuing to listen, so a probe cannot deny the real client its
    /// connection.
    private static func acceptFromOwner(_ listening: Int32) throws -> Int32 {
        let owner = getuid()
        while true {
            let accepted = Darwin.accept(listening, nil, nil)
            guard accepted >= 0 else {
                if errno == EINTR { continue }
                throw POSIXError(.init(rawValue: errno) ?? .EIO)
            }
            var peerUID = uid_t(0)
            var peerGID = gid_t(0)
            if getpeereid(accepted, &peerUID, &peerGID) == 0, peerUID == owner {
                return accepted
            }
            Darwin.close(accepted)
        }
    }

    private static func handles(for fd: Int32) throws
        -> (input: FileHandle, output: FileHandle) {
        let outputFD = dup(fd)
        guard outputFD >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        disableSIGPIPE(on: fd)
        disableSIGPIPE(on: outputFD)
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
