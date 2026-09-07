import Darwin
import Foundation
import TurboFieldfareRepackCore

/// The single-writer lock on the conversation store.
///
/// Two processes appending to one JSONL file is the documented corruption in
/// every client that has hit it: out-of-order records, a torn line in the
/// middle rather than at the end, and a file that then fails to open at all.
/// A second instance of the app takes no lock, reads the store, and says so.
///
/// Held for the instance's lifetime, released when this object is deallocated.
/// `flock` is per open file description, so a crash releases it with the
/// process and no stale lock survives a kill.
public final class ConversationStoreLock: @unchecked Sendable {
    public static let fileName = ".writer.lock"

    private let descriptor: Int32

    private init(descriptor: Int32) {
        self.descriptor = descriptor
    }

    deinit {
        _ = flock(descriptor, LOCK_UN)
        close(descriptor)
    }

    /// Takes the exclusive lock, or returns nil when another instance holds it.
    ///
    /// Nil rather than a throw for the busy case: a second window is a normal
    /// thing for a user to open, and it is not an error — it is read-only, and
    /// the sidebar tells them so. Anything else that goes wrong opening the lock
    /// file still throws, because a store that cannot be locked for an unknown
    /// reason must not be written to.
    public static func acquire(storeRoot: URL) throws -> ConversationStoreLock? {
        try FileManager.default.createDirectory(
            at: storeRoot, withIntermediateDirectories: true)
        let path = storeRoot.appendingPathComponent(fileName, isDirectory: false).path
        let descriptor = open(path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else {
            throw ConversationStoreError.lockUnavailable(
                path: path, errno: errno)
        }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let saved = errno
            close(descriptor)
            if saved == EWOULDBLOCK { return nil }
            throw ConversationStoreError.lockUnavailable(path: path, errno: saved)
        }
        // The file could have been replaced between open and flock — the
        // store root removed and recreated under a running instance — in which
        // case this holds the lock on an unlinked inode while the next
        // instance takes the new file's, and both write the same transcripts.
        // The same check the install and companion-pack locks make.
        let matches: Bool
        do {
            matches = try Posix.descriptorMatchesPath(descriptor, path: path)
        } catch RepackError.fileStatFailed(_, let code) {
            close(descriptor)
            throw ConversationStoreError.lockUnavailable(path: path, errno: code)
        } catch {
            close(descriptor)
            throw ConversationStoreError.lockUnavailable(path: path, errno: EIO)
        }
        guard matches else {
            close(descriptor)
            throw ConversationStoreError.lockUnavailable(path: path, errno: ESTALE)
        }
        return ConversationStoreLock(descriptor: descriptor)
    }
}
