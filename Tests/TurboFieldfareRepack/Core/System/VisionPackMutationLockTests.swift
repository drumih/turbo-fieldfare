import Darwin
import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite struct VisionPackMutationLockTests {
    @Test func mutationFailsWhileRuntimeHoldsSharedLease() throws {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-use-lock-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: parent,
            withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: parent) }
        let output = parent.appendingPathComponent("model.vision.gturbo")
        let lockPath = parent.appendingPathComponent(
            ".model.vision.gturbo.use.lock").path
        let descriptor = open(
            lockPath,
            O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
            0o600)
        #expect(descriptor >= 0)
        defer {
            _ = flock(descriptor, LOCK_UN)
            close(descriptor)
        }
        #expect(flock(descriptor, LOCK_SH | LOCK_NB) == 0)

        do {
            _ = try VisionPackMutationLock.acquire(
                outputDirectory: output.path)
            Issue.record("exclusive mutation lock ignored the shared runtime lease")
        } catch let error as RepackError {
            guard case .installBusy = error else {
                Issue.record("expected installBusy, got \(error)")
                return
            }
        }
    }

    /// The message a user actually sees when a loaded model blocks a remove.
    /// It used to say "another installer holds ...", so the one case people hit
    /// sent them looking for a second install that does not exist.
    @Test func aBlockedMutationNamesTheLoadedModelNotJustAnInstaller() throws {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-busy-text-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: parent) }
        let output = parent.appendingPathComponent("model.vision.gturbo")
        let lockPath = parent.appendingPathComponent(
            ".model.vision.gturbo.use.lock").path
        let descriptor = open(
            lockPath, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        #expect(descriptor >= 0)
        defer {
            _ = flock(descriptor, LOCK_UN)
            close(descriptor)
        }
        #expect(flock(descriptor, LOCK_SH | LOCK_NB) == 0)

        do {
            _ = try VisionPackMutationLock.acquire(outputDirectory: output.path)
            Issue.record("expected the shared lease to block the mutation")
        } catch let error as RepackError {
            let message = error.description
            #expect(message.contains("loaded model"),
                    "the message must name the holder people actually hit: \(message)")
            #expect(message.contains("unload the model"),
                    "the message must say what to do: \(message)")
        }
    }

    /// The installer's own lock has no reader side, so it keeps the plain
    /// wording; widening every busy message would have been a lie in that case.
    @Test func theInstallerLockKeepsItsOwnWording() {
        #expect(RepackError.installBusy(path: "/tmp/x.install.lock").description
                == "another installer holds /tmp/x.install.lock")
    }

    @Test func transactionLockCanCoexistWithRuntimeLeaseWithoutDeadlock() throws {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-lock-order-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: parent,
            withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: parent) }
        let output = parent.appendingPathComponent("model.vision.gturbo")
        let lockPath = parent.appendingPathComponent(
            ".model.vision.gturbo.use.lock").path
        let descriptor = open(
            lockPath,
            O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
            0o600)
        #expect(descriptor >= 0)
        defer {
            _ = flock(descriptor, LOCK_UN)
            close(descriptor)
        }
        #expect(flock(descriptor, LOCK_SH | LOCK_NB) == 0)

        let transaction = try InstallLock.acquire(outputDirectory: output.path)
        #expect(throws: RepackError.self) {
            _ = try VisionPackMutationLock.acquire(outputDirectory: output.path)
        }
        withExtendedLifetime(transaction) {}
    }
}
