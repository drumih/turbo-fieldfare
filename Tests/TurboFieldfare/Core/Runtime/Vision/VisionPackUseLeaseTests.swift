import Darwin
import Foundation
import Testing
@testable import TurboFieldfare

/// The shared use lease against real filesystem permissions: the writable
/// happy path, the read-only fallbacks, and the fail-closed cases. The
/// EROFS lock-free path needs an actual read-only mount and stays covered by
/// code review only.
@Suite struct VisionPackUseLeaseTests {

    private func makePack() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("use-lease-\(UUID().uuidString)", isDirectory: true)
        let pack = root.appendingPathComponent("model.vision.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
        return pack
    }

    @Test func sharedLeaseCreatesAndNamesTheDottedLockFile() throws {
        let pack = try makePack()
        defer { try? FileManager.default.removeItem(at: pack.deletingLastPathComponent()) }
        let lease = try VisionPackUseLease.acquireShared(companionURL: pack)
        #expect(lease.lockFileURL.lastPathComponent == ".model.vision.gturbo.use.lock")
        #expect(FileManager.default.fileExists(atPath: lease.lockFileURL.path))
    }

    @Test func readOnlyLockFileStillGrantsASharedLease() throws {
        let pack = try makePack()
        defer { try? FileManager.default.removeItem(at: pack.deletingLastPathComponent()) }
        let lockURL = VisionPackUseLease.lockURL(for: pack)
        #expect(FileManager.default.createFile(atPath: lockURL.path, contents: nil))
        #expect(chmod(lockURL.path, 0o444) == 0)
        // The O_RDWR create-open gets EACCES; the O_RDONLY retry must lock.
        let lease = try VisionPackUseLease.acquireShared(companionURL: pack)
        #expect(lease.lockFileURL.path == lockURL.path)
    }

    @Test func unwritableParentWithoutALockFileFailsClosed() throws {
        let pack = try makePack()
        let parent = pack.deletingLastPathComponent()
        defer {
            _ = chmod(parent.path, 0o755)
            try? FileManager.default.removeItem(at: parent)
        }
        #expect(chmod(parent.path, 0o555) == 0)
        // A writable volume whose directory merely denies us could still host
        // a privileged writer, so no lock file means no lease.
        #expect(throws: VisionPackError.self) {
            _ = try VisionPackUseLease.acquireShared(companionURL: pack)
        }
    }

    @Test func fifoAtTheLockPathIsRejectedNotHung() throws {
        let pack = try makePack()
        defer { try? FileManager.default.removeItem(at: pack.deletingLastPathComponent()) }
        let lockURL = VisionPackUseLease.lockURL(for: pack)
        // Read-only FIFO: the create-open gets EACCES, and the O_RDONLY retry
        // would block forever on a FIFO without O_NONBLOCK; the S_IFREG check
        // must reject it instead.
        #expect(mkfifo(lockURL.path, 0o444) == 0)
        #expect(throws: VisionPackError.self) {
            _ = try VisionPackUseLease.acquireShared(companionURL: pack)
        }
    }

    @Test func anExclusiveWriterBlocksTheSharedLeaseUntilTimeout() throws {
        let pack = try makePack()
        defer { try? FileManager.default.removeItem(at: pack.deletingLastPathComponent()) }
        let lockURL = VisionPackUseLease.lockURL(for: pack)
        let writer = open(lockURL.path, O_RDWR | O_CREAT | O_CLOEXEC, 0o600)
        #expect(writer >= 0)
        defer { close(writer) }
        #expect(flock(writer, LOCK_EX) == 0)
        #expect(throws: VisionPackError.self) {
            _ = try VisionPackUseLease.acquireShared(
                companionURL: pack, timeout: .milliseconds(120))
        }
    }
}
