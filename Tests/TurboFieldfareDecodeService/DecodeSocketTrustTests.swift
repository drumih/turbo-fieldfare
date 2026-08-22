import Darwin
import Foundation
import Testing
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

/// The decode socket lives in a world-writable directory and carries model
/// paths, attachment paths and generated text. Two rules keep it honest: only
/// this user may connect, and only files this app staged may be opened.
@Suite struct DecodeSocketTrustTests {
    @Test func aConnectionFromThisUserIsAccepted() async throws {
        let path = "/private/tmp/turbofieldfare-trust-\(UUID().uuidString).sock"
        defer { unlink(path) }
        let accepted = Mutex2<(input: FileHandle, output: FileHandle)?>(nil)
        let listener = Thread {
            accepted.set(try? DecodeUnixSocket.listenAndAccept(path: path))
        }
        listener.start()
        let ready = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: path), Date() < ready {
            usleep(5_000)
        }

        // The socket file exists from bind(), which is before listen(), so a
        // connect in that window is refused. The production client retries for
        // the same reason.
        let client = try connectWithRetry(path: path)
        let deadline = Date().addingTimeInterval(5)
        while accepted.get() == nil, Date() < deadline { usleep(5_000) }
        #expect(accepted.get() != nil, "the owner's own connection was refused")
        try? client.input.close()
        try? client.output.close()
    }

    /// The socket is created before the app connects, so anyone can reach it in
    /// that window. It must not be readable by other users at all.
    @Test func theSocketIsNotReachableByOtherUsers() async throws {
        let path = "/private/tmp/turbofieldfare-mode-\(UUID().uuidString).sock"
        defer { unlink(path) }
        let listener = Thread {
            _ = try? DecodeUnixSocket.listenAndAccept(path: path)
        }
        listener.start()
        let ready = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: path), Date() < ready {
            usleep(5_000)
        }

        var info = stat()
        #expect(lstat(path, &info) == 0)
        let mode = info.st_mode & 0o777
        #expect(mode & 0o077 == 0,
                "the socket is reachable by other users: \(String(mode, radix: 8))")

        // Unblock the listener.
        let client = try? connectWithRetry(path: path)
        try? client?.input.close()
        try? client?.output.close()
    }

    /// The service opens and hashes whatever path it is handed, so a path
    /// outside the attachment store would make it an oracle for any file the
    /// user can read.
    @Test func onlyStagedAttachmentPathsAreAccepted() throws {
        let root = AppImageAttachmentStore.root
        let staged = root
            .appendingPathComponent("pid-\(getpid())", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathComponent("image.png")
        #expect(AppImageAttachmentStore.contains(staged))

        #expect(!AppImageAttachmentStore.contains(URL(fileURLWithPath: "/etc/passwd")))
        #expect(!AppImageAttachmentStore.contains(
            URL(fileURLWithPath: FileManager.default.temporaryDirectory.path)
                .appendingPathComponent("elsewhere.png")))
        #expect(!AppImageAttachmentStore.contains(root),
                "the root itself is a directory, not an attachment")
        // Traversal back out of the root must not pass.
        #expect(!AppImageAttachmentStore.contains(
            root.appendingPathComponent("../../etc/passwd")))
    }

    /// A symlink planted inside the root must not smuggle a target outside it.
    @Test func aSymlinkOutOfTheStoreIsRejected() throws {
        let manager = FileManager.default
        let root = AppImageAttachmentStore.root
            .appendingPathComponent("pid-\(getpid())", isDirectory: true)
            .appendingPathComponent("trust-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: root) }
        let outside = manager.temporaryDirectory
            .appendingPathComponent("outside-\(UUID().uuidString).png")
        try Data("secret".utf8).write(to: outside)
        defer { try? manager.removeItem(at: outside) }

        let link = root.appendingPathComponent("link.png")
        try manager.createSymbolicLink(at: link, withDestinationURL: outside)
        #expect(!AppImageAttachmentStore.contains(link),
                "a symlink pointed the store at a file outside it")
    }
}

private func connectWithRetry(
    path: String,
    timeout: TimeInterval = 10
) throws -> (input: FileHandle, output: FileHandle) {
    let deadline = Date().addingTimeInterval(timeout)
    var lastError: Error?
    while Date() < deadline {
        do { return try DecodeUnixSocket.connect(path: path) }
        catch { lastError = error; usleep(5_000) }
    }
    throw lastError ?? POSIXError(.ECONNREFUSED)
}

/// A tiny lock, because `Synchronization.Mutex` cannot hold a non-Sendable
/// tuple of file handles.
private final class Mutex2<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value

    init(_ value: Value) { self.value = value }

    func get() -> Value {
        lock.lock(); defer { lock.unlock() }
        return value
    }

    func set(_ newValue: Value) {
        lock.lock(); defer { lock.unlock() }
        value = newValue
    }
}
