import Darwin
import Foundation
import TurboFieldfareFormat

public final class VisionPackUseLease: @unchecked Sendable {
    /// How long `acquireShared` waits for a mutation to finish before giving
    /// up. A blocking `flock` would wait forever, so a stuck installer would
    /// hang model load with nothing to report.
    public static let acquisitionTimeout: Duration = .seconds(5)

    public let lockFileURL: URL
    private let descriptor: Int32

    private init(lockFileURL: URL, descriptor: Int32) {
        self.lockFileURL = lockFileURL
        self.descriptor = descriptor
    }

    deinit {
        _ = flock(descriptor, LOCK_UN)
        close(descriptor)
    }

    public static func acquireShared(
        companionURL: URL,
        timeout: Duration = acquisitionTimeout
    ) throws -> VisionPackUseLease {
        let lockURL = lockURL(for: companionURL)
        let created = open(
            lockURL.path,
            O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
            0o600)
        if created >= 0 {
            return try lease(descriptor: created, lockURL: lockURL, timeout: timeout)
        }
        let createErrno = errno
        guard createErrno == EROFS || createErrno == EACCES || createErrno == EPERM else {
            throw VisionPackError.invalidMetadata(
                "could not open use lock at \(lockURL.path): errno \(createErrno)")
        }
        // A verified pack can live where the process cannot write: a read-only
        // DMG or share, or another user's directory. `flock(LOCK_SH)` works on
        // an O_RDONLY descriptor, so an existing lock file still coordinates
        // with a writer even when this process cannot create one. O_NONBLOCK
        // guards the open itself: a FIFO planted at the lock path would block
        // a plain O_RDONLY open forever, before any flock timeout applies;
        // opened nonblocking it proceeds and the S_IFREG check rejects it.
        let readOnly = open(lockURL.path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        if readOnly >= 0 {
            return try lease(descriptor: readOnly, lockURL: lockURL, timeout: timeout)
        }
        let openErrno = errno
        guard openErrno == ENOENT, createErrno == EROFS else {
            throw VisionPackError.invalidMetadata(
                "could not open use lock at \(lockURL.path): errno \(openErrno)")
        }
        // No lock file and a read-only volume: no mutator can exist there, so
        // using the pack without a lock is sound. Hold the companion itself so
        // the lease still owns a descriptor whose lifetime brackets the use.
        // An unwritable directory on a writable volume deliberately stays an
        // error above: a privileged writer could still mutate the pack there.
        let companion = open(companionURL.standardizedFileURL.path, O_RDONLY | O_CLOEXEC)
        guard companion >= 0 else {
            let companionErrno = errno
            throw VisionPackError.invalidMetadata(
                "could not open vision companion at \(companionURL.path): "
                    + "errno \(companionErrno)")
        }
        return VisionPackUseLease(lockFileURL: lockURL, descriptor: companion)
    }

    private static func lease(
        descriptor: Int32, lockURL: URL, timeout: Duration
    ) throws -> VisionPackUseLease {
        var info = stat()
        guard fstat(descriptor, &info) == 0,
              (info.st_mode & S_IFMT) == S_IFREG else {
            let savedErrno = errno
            close(descriptor)
            throw VisionPackError.invalidMetadata(
                "invalid use lock at \(lockURL.path): errno \(savedErrno)")
        }
        do {
            try lock(descriptor, path: lockURL.path, timeout: timeout)
            // The lock file could have been replaced between `open` and the
            // successful `flock` — by a reinstall, say — in which case this
            // holds a shared lock on an unlinked inode and the writer's
            // exclusive lock on the new one excludes nothing.
            guard try descriptorMatchesPath(descriptor, path: lockURL.path) else {
                throw VisionPackError.invalidMetadata(
                    "use lock at \(lockURL.path) was replaced during acquisition")
            }
        } catch {
            _ = flock(descriptor, LOCK_UN)
            close(descriptor)
            throw error
        }
        return VisionPackUseLease(lockFileURL: lockURL, descriptor: descriptor)
    }

    /// The derivation is shared with any writer-side mutation lock through
    /// `GTurboVisionLockPathsV1`, next to the format it protects; deriving it
    /// independently per module does not fail on mismatch — it silently leaves
    /// the two sides locking different files.
    public static func lockURL(for companionURL: URL) -> URL {
        GTurboVisionLockPathsV1.useLockURL(forCompanion: companionURL)
    }

    /// Polls instead of blocking, so a mutation that never finishes surfaces as
    /// an error rather than a hang.
    private static func lock(
        _ descriptor: Int32, path: String, timeout: Duration
    ) throws {
        let deadline = ContinuousClock.now + timeout
        while true {
            if flock(descriptor, LOCK_SH | LOCK_NB) == 0 { return }
            let savedErrno = errno
            guard savedErrno == EWOULDBLOCK else {
                throw VisionPackError.invalidMetadata(
                    "could not acquire use lock at \(path): errno \(savedErrno)")
            }
            guard ContinuousClock.now < deadline else {
                throw VisionPackError.invalidMetadata(
                    "vision companion at \(path) is being modified; "
                        + "timed out after \(timeout)")
            }
            var pause = timespec(tv_sec: 0, tv_nsec: 20_000_000)
            nanosleep(&pause, nil)
        }
    }

    private static func descriptorMatchesPath(
        _ descriptor: Int32, path: String
    ) throws -> Bool {
        var descriptorInfo = stat()
        guard fstat(descriptor, &descriptorInfo) == 0 else {
            throw VisionPackError.invalidMetadata(
                "could not stat use lock descriptor for \(path): errno \(errno)")
        }
        var pathInfo = stat()
        guard lstat(path, &pathInfo) == 0 else {
            throw VisionPackError.invalidMetadata(
                "could not stat use lock at \(path): errno \(errno)")
        }
        return descriptorInfo.st_dev == pathInfo.st_dev
            && descriptorInfo.st_ino == pathInfo.st_ino
            && (pathInfo.st_mode & S_IFMT) == S_IFREG
    }
}
