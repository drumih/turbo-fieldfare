import Darwin
import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareRepackCore

/// The runtime's shared lease and the installer's exclusive lock live in
/// different modules and derive their lock path independently. Nothing fails
/// loudly when those derivations disagree — the two sides simply lock different
/// files and stop excluding each other, which is how a companion pack gets
/// replaced underneath a runtime that is mmap'ing it.
@Suite struct VisionPackLockAgreementTests {
    private static func makeParent(_ label: String) throws -> URL {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(label)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: false)
        return parent
    }

    @Test func bothSidesDeriveTheSameLockPath() throws {
        let parent = try Self.makeParent("lock-agreement")
        defer { try? FileManager.default.removeItem(at: parent) }
        let companion = parent.appendingPathComponent("model.vision.gturbo")

        let mutation = try VisionPackMutationLock.lockPath(
            outputDirectory: companion.path)
        let lease = VisionPackUseLease.lockURL(for: companion).path
        #expect(mutation == lease)
    }

    /// An explicit companion path routed through a symlinked parent — what a
    /// `--vision-pack` override or a relocated model directory produces. The
    /// two sides used to resolve it differently, so both locks succeeded and
    /// neither excluded the other.
    @Test func symlinkedParentsStillAgree() throws {
        let real = try Self.makeParent("lock-agreement-real")
        let link = FileManager.default.temporaryDirectory
            .appendingPathComponent("lock-agreement-link-\(UUID().uuidString)")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
        defer {
            try? FileManager.default.removeItem(at: link)
            try? FileManager.default.removeItem(at: real)
        }
        let throughLink = link.appendingPathComponent("model.vision.gturbo")

        let mutation = try VisionPackMutationLock.lockPath(
            outputDirectory: throughLink.path)
        let lease = VisionPackUseLease.lockURL(for: throughLink).path
        #expect(mutation == lease)

        // And the exclusion actually holds through the symlink.
        let held = try VisionPackMutationLock.acquire(outputDirectory: throughLink.path)
        #expect(throws: VisionPackError.self) {
            _ = try VisionPackUseLease.acquireShared(
                companionURL: throughLink, timeout: .milliseconds(200))
        }
        withExtendedLifetime(held) {}
    }

    /// A shared lease used to block indefinitely, so an installer that hung —
    /// or crashed while another process held the lock — hung model load with
    /// nothing to report. It has to give up and say why.
    @Test func sharedLeaseTimesOutInsteadOfBlockingForever() throws {
        let parent = try Self.makeParent("lock-timeout")
        defer { try? FileManager.default.removeItem(at: parent) }
        let companion = parent.appendingPathComponent("model.vision.gturbo")
        let held = try VisionPackMutationLock.acquire(outputDirectory: companion.path)

        let started = ContinuousClock.now
        var message = ""
        do {
            _ = try VisionPackUseLease.acquireShared(
                companionURL: companion, timeout: .milliseconds(300))
            Issue.record("shared lease ignored the exclusive mutation lock")
        } catch let error as VisionPackError {
            message = "\(error)"
        }
        let elapsed = ContinuousClock.now - started
        #expect(elapsed >= .milliseconds(250), "gave up before waiting")
        #expect(elapsed < .seconds(3), "waited far past its timeout")
        #expect(message.contains("being modified"),
                "the error must say what is holding the pack: \(message)")
        withExtendedLifetime(held) {}
    }

    /// Once the mutation finishes, readers get in.
    @Test func sharedLeaseSucceedsAfterTheMutationReleases() throws {
        let parent = try Self.makeParent("lock-release")
        defer { try? FileManager.default.removeItem(at: parent) }
        let companion = parent.appendingPathComponent("model.vision.gturbo")
        var held: VisionPackMutationLock? =
            try VisionPackMutationLock.acquire(outputDirectory: companion.path)
        held = nil
        _ = held

        let lease = try VisionPackUseLease.acquireShared(
            companionURL: companion, timeout: .milliseconds(300))
        #expect(lease.lockFileURL.lastPathComponent
                    == ".model.vision.gturbo.use.lock")
        // Two readers may hold it at once; only mutation is exclusive.
        let second = try VisionPackUseLease.acquireShared(
            companionURL: companion, timeout: .milliseconds(300))
        withExtendedLifetime((lease, second)) {}
    }

    /// The lock file is replaced between `open` and `flock` — a reinstall
    /// racing a load. Locking the unlinked inode excludes nothing, so it has to
    /// be caught rather than reported as success.
    @Test func aReplacedLockFileIsRejected() throws {
        let parent = try Self.makeParent("lock-replaced")
        defer { try? FileManager.default.removeItem(at: parent) }
        let companion = parent.appendingPathComponent("model.vision.gturbo")
        let path = VisionPackUseLease.lockURL(for: companion).path

        // Stand in for the race: a descriptor on the old inode, then the path
        // pointing at a new one.
        let stale = open(path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        #expect(stale >= 0)
        defer { close(stale) }
        try FileManager.default.removeItem(atPath: path)
        let fresh = open(path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        #expect(fresh >= 0)
        defer { close(fresh) }

        var staleInfo = stat()
        var freshInfo = stat()
        #expect(fstat(stale, &staleInfo) == 0)
        #expect(fstat(fresh, &freshInfo) == 0)
        #expect(staleInfo.st_ino != freshInfo.st_ino,
                "the fixture did not actually replace the inode")

        // A lease taken now must bind to the live inode, not the stale one.
        let lease = try VisionPackUseLease.acquireShared(
            companionURL: companion, timeout: .milliseconds(300))
        var leaseInfo = stat()
        #expect(lstat(lease.lockFileURL.path, &leaseInfo) == 0)
        #expect(leaseInfo.st_ino == freshInfo.st_ino)
        withExtendedLifetime(lease) {}
    }
}
