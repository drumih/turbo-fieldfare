import Darwin
import Foundation
import Testing

@testable import TinyTitanRepackCore
@testable import TinyTitanCore

/// The `--share-ngram-table` link path.
///
/// The table is 102 GB and byte-identical in every quantization, so a second
/// install hardlinks it instead of copying it. What that has to get right is
/// small and easy to get wrong in a way nothing notices: one inode with two
/// links, an empty origin refused rather than shared, an existing destination
/// replaced rather than linked over, and an optional file that is simply
/// absent when the snapshot does not carry one.
///
/// The whole-repack path needs a `qwen38flash` snapshot to reach this code, and
/// this does not: the decision being pinned is the link itself.
@Suite struct SharedNgramTableTests {
    private func temporaryRoot(_ tag: String) -> String {
        let base = (FileManager.default.currentDirectoryPath as NSString)
            .appendingPathComponent(".build/test-artifacts")
        try? FileManager.default.createDirectory(
            atPath: base,
            withIntermediateDirectories: true)
        let path = (base as NSString)
            .appendingPathComponent("\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }

    private func makeDirectories(_ root: String) throws -> (snapshot: String, partial: String) {
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let partial = (root as NSString).appendingPathComponent("partial")
        try FileManager.default.createDirectory(
            atPath: snapshot,
            withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            atPath: partial,
            withIntermediateDirectories: true)
        return (snapshot, partial)
    }

    private func write(_ count: Int, to path: String) throws {
        try Data(repeating: 0x5A, count: count).write(to: URL(fileURLWithPath: path))
    }

    private func statOf(_ path: String) throws -> stat {
        var info = stat()
        try #require(stat(path, &info) == 0, "stat failed for \(path)")
        return info
    }

    @Test func linksTheTableIntoTheInstallInsteadOfCopyingIt() throws {
        let root = temporaryRoot("share-ngram")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let (snapshot, partial) = try makeDirectories(root)
        let origin = (snapshot as NSString).appendingPathComponent("ngram_table.bin")
        try write(4096, to: origin)

        let linked = try RemoteStreamingRepacker.linkPassthroughFile(
            named: "ngram_table.bin", from: snapshot, into: partial)

        let destination = try #require(linked)
        #expect(destination == (partial as NSString).appendingPathComponent("ngram_table.bin"))
        let source = try statOf(origin)
        let shared = try statOf(destination)
        #expect(source.st_ino == shared.st_ino, "one inode, not two copies")
        #expect(source.st_nlink == 2, "the snapshot's copy now has two links")
        #expect(shared.st_nlink == 2)
        #expect(shared.st_size == 4096)

        // Deleting the snapshot leaves the install whole, which a symlink
        // would not: that is the reason this links rather than points.
        try FileManager.default.removeItem(atPath: origin)
        #expect(FileManager.default.fileExists(atPath: destination))
        #expect(try statOf(destination).st_nlink == 1)
    }

    @Test func aTableTheSnapshotDoesNotCarryIsNotAnError() throws {
        let root = temporaryRoot("share-ngram-absent")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let (snapshot, partial) = try makeDirectories(root)

        // The table is an optional requirement, so a backbone built without
        // one must not fail here.
        let linked = try RemoteStreamingRepacker.linkPassthroughFile(
            named: "ngram_table.bin", from: snapshot, into: partial)
        #expect(linked == nil)
    }

    @Test func anEmptyTableIsRefusedRatherThanShared() throws {
        let root = temporaryRoot("share-ngram-empty")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let (snapshot, partial) = try makeDirectories(root)
        let origin = (snapshot as NSString).appendingPathComponent("ngram_table.bin")
        try write(0, to: origin)

        // A zero-length table links happily and then reads as a table with no
        // rows, so the guard exists to fail here rather than at load.
        #expect(throws: RepackError.self) {
            _ = try RemoteStreamingRepacker.linkPassthroughFile(
                named: "ngram_table.bin", from: snapshot, into: partial)
        }
    }

    @Test func anExistingDestinationIsReplacedByTheLink() throws {
        let root = temporaryRoot("share-ngram-existing")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let (snapshot, partial) = try makeDirectories(root)
        let origin = (snapshot as NSString).appendingPathComponent("ngram_table.bin")
        let destination = (partial as NSString).appendingPathComponent("ngram_table.bin")
        try write(4096, to: origin)
        try write(16, to: destination)

        let linked = try RemoteStreamingRepacker.linkPassthroughFile(
            named: "ngram_table.bin", from: snapshot, into: partial)

        #expect(linked == destination)
        #expect(try statOf(destination).st_size == 4096)
        #expect(try statOf(destination).st_ino == statOf(origin).st_ino)
    }
}
