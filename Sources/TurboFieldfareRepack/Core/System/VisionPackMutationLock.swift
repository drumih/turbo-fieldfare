import Darwin
import Foundation

final class VisionPackMutationLock {
    private let descriptor: Int32

    private init(descriptor: Int32) {
        self.descriptor = descriptor
    }

    deinit {
        _ = flock(descriptor, LOCK_UN)
        close(descriptor)
    }

    /// The path both sides of this lock must agree on. Readers derive it in
    /// another module, from a companion URL; if the two derivations ever differ
    /// the lock silently stops excluding anything, so the rule is written once:
    /// resolve the parent physically, then name the file after the companion's
    /// own last component.
    static func lockPath(outputDirectory: String) throws -> String {
        let output = URL(fileURLWithPath: outputDirectory).standardizedFileURL
        let parent = try Posix.physicalPath(output.deletingLastPathComponent().path)
        return (parent as NSString).appendingPathComponent(
            ".\(output.lastPathComponent).use.lock")
    }

    static func acquire(outputDirectory: String) throws -> VisionPackMutationLock {
        let path = try lockPath(outputDirectory: outputDirectory)
        let descriptor = try Posix.openLock(path)
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let savedErrno = errno
            close(descriptor)
            if savedErrno == EWOULDBLOCK {
                throw RepackError.installBusy(
                    path: path, holder: .installerOrLoadedModel)
            }
            throw RepackError.fileOpenFailed(path: path, errno: savedErrno)
        }
        do {
            // The file could have been replaced between open and flock, which
            // would leave this holding an exclusive lock on an unlinked inode
            // while a reader locks the new one. `InstallLock` already checks
            // this; both mutation paths have to.
            guard try Posix.descriptorMatchesPath(descriptor, path: path) else {
                throw RepackError.installPathUnsafe(
                    path: path, detail: "lock path was replaced during acquisition")
            }
        } catch {
            _ = flock(descriptor, LOCK_UN)
            close(descriptor)
            throw error
        }
        return VisionPackMutationLock(descriptor: descriptor)
    }
}
