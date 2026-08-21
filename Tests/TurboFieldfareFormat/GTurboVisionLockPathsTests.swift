import Foundation
import Testing
import TurboFieldfareFormat

/// Every reader and writer of a vision companion must derive the same lock
/// file; these pin the shared derivation itself.
@Suite struct GTurboVisionLockPathsTests {

    @Test func lockFileIsADottedSiblingOfTheCompanion() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appendingPathComponent("lockpaths-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let url = GTurboVisionLockPathsV1.useLockURL(
            forCompanion: root.appendingPathComponent("pack.vision.gturbo"))
        #expect(url.lastPathComponent == ".pack.vision.gturbo.use.lock")
        #expect(url.deletingLastPathComponent().lastPathComponent
            == root.lastPathComponent)
    }

    @Test func symlinkedParentDerivesTheSameLockFileAsThePhysicalPath() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appendingPathComponent("lockpaths-\(UUID().uuidString)", isDirectory: true)
        let real = root.appendingPathComponent("real", isDirectory: true)
        try fm.createDirectory(at: real, withIntermediateDirectories: true)
        let link = root.appendingPathComponent("link")
        try fm.createSymbolicLink(at: link, withDestinationURL: real)
        defer { try? fm.removeItem(at: root) }
        let viaReal = GTurboVisionLockPathsV1.useLockURL(
            forCompanion: real.appendingPathComponent("p.gturbo"))
        let viaLink = GTurboVisionLockPathsV1.useLockURL(
            forCompanion: link.appendingPathComponent("p.gturbo"))
        #expect(viaReal.path == viaLink.path)
    }

    @Test func nonexistentParentStillDerivesDeterministically() {
        let a = GTurboVisionLockPathsV1.useLockURL(
            forCompanion: URL(fileURLWithPath: "/nonexistent-lockpath-test/p.gturbo"))
        let b = GTurboVisionLockPathsV1.useLockURL(
            forCompanion: URL(fileURLWithPath: "/nonexistent-lockpath-test/x/../p.gturbo"))
        #expect(a.path == b.path)
        #expect(a.lastPathComponent == ".p.gturbo.use.lock")
    }
}
