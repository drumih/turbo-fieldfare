import Darwin
import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// The store is a trust boundary before it is anything else: whatever it
/// vouches for gets opened and hashed by another process, so a path it accepts
/// from outside its own root would make that process an oracle for any file the
/// user can read.
@Suite struct AppImageAttachmentStoreTests {
    @Test func onlyStagedPathsAreAccepted() throws {
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
    @Test func asymlinkOutOfTheStoreIsRejected() throws {
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

    /// Staging copies rather than referencing: the source may be a file the
    /// user is about to move, rename or delete, and a run that reads it later
    /// would then read nothing.
    @Test func stagingCopiesTheSourceAndRemovingItLeavesNothingBehind() throws {
        let manager = FileManager.default
        let source = manager.temporaryDirectory
            .appendingPathComponent("source-\(UUID().uuidString).png")
        try Data("pretend png".utf8).write(to: source)
        defer { try? manager.removeItem(at: source) }

        let store = AppImageAttachmentStore()
        let attachment = try store.stage(source)
        #expect(AppImageAttachmentStore.contains(attachment.fileURL))
        #expect(manager.fileExists(atPath: attachment.fileURL.path))

        try manager.removeItem(at: source)
        #expect(manager.fileExists(atPath: attachment.fileURL.path),
                "the staged copy died with its source")

        store.remove(attachment)
        #expect(!manager.fileExists(atPath: attachment.fileURL.path),
                "removing an attachment left its bytes on disk")
    }
}
