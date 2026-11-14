import Darwin
import Testing

@testable import TinyTitanServerCore
import TinyTitanCore

/// The server binds its listener only after the model is resident, so the cheap
/// startup check has to answer the same question the bind will: is anything
/// already listening on 127.0.0.1:port?
@Suite struct ServerPortCheckTests {
    /// A real listening socket on an ephemeral loopback port.
    private static func listeningSocket() throws -> (fd: Int32, port: Int) {
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        try #require(fd >= 0)
        var reuse: Int32 = 1
        _ = Darwin.setsockopt(
            fd, SOL_SOCKET, SO_REUSEADDR, &reuse,
            socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        #expect(inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1)
        let bound: Int32 = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        try #require(bound == 0)
        try #require(Darwin.listen(fd, 8) == 0)
        var local = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named: Int32 = withUnsafeMutablePointer(to: &local) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &length)
            }
        }
        try #require(named == 0)
        return (fd, Int(UInt16(bigEndian: local.sin_port)))
    }

    @Test func aPortWithAListenerIsUnavailable() throws {
        let listener = try Self.listeningSocket()
        defer { Darwin.close(listener.fd) }

        #expect(!ServerPortCheck.isAvailable(port: listener.port))
    }

    /// The same port once its socket is closed: the check must not refuse a
    /// port the real bind would accept, which is what an unconditional
    /// SO_REUSEADDR-less probe would do after a restart.
    @Test func aClosedPortIsAvailableAgain() throws {
        let listener = try Self.listeningSocket()
        Darwin.close(listener.fd)

        #expect(ServerPortCheck.isAvailable(port: listener.port))
    }
}
