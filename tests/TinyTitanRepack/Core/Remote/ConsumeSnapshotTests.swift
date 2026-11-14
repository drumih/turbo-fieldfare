import Foundation
import Testing

@testable import TinyTitanRepackCore
import TinyTitanCore

/// `--consume-snapshot`: a source shard is deleted only after the last
/// coalesced copy that reads it has committed, never before, and a shard no
/// copy names is untouched. The option is what lets a 125 GB install grow into
/// the space its 125 GB snapshot is releasing on a host that cannot hold both.
@Suite("Consume snapshot")
struct ConsumeSnapshotTests {
    private static func writeFile(_ url: URL, bytes: Int, fill: UInt8) throws {
        try Data(repeating: fill, count: bytes).write(to: url)
    }

    private static func copy(
        id: String, shard: String, offset: UInt64, size: UInt64,
        destination: String, destinationOffset: UInt64
    ) -> CoalescedRangeCopy {
        CoalescedRangeCopy(
            id: id, shardID: shard, sourceOffset: offset, size: size,
            destinations: [
                RangeCopy(
                    shardID: shard, sourceOffset: offset, size: size,
                    destinationPath: destination,
                    destinationOffset: destinationOffset)
            ])
    }

    private static func run(
        directory: URL, copies: [CoalescedRangeCopy], consume: Bool
    ) async throws {
        let provider = LocalSourceByteProvider(
            snapshotDirectory: directory.path, writeTileBytes: 64 * 1024)
        var committed: [String] = []
        try await provider.copyBatch(
            copies,
            completedRangeIDs: [],
            partialDirectory: directory.path,
            temporaryPath: directory.appendingPathComponent("range.tmp").path,
            audit: RepackAudit(),
            progress: { _ in },
            commit: { completed in committed.append(completed.id) },
            consumeSource: consume)
        #expect(committed.count == copies.count)
    }

    @Test func deletesEachShardAfterItsLastCopy() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("consume-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = directory.appendingPathComponent("a.safetensors")
        let second = directory.appendingPathComponent("b.safetensors")
        let unused = directory.appendingPathComponent("c.safetensors")
        try Self.writeFile(first, bytes: 128, fill: 0x11)
        try Self.writeFile(second, bytes: 128, fill: 0x22)
        try Self.writeFile(unused, bytes: 128, fill: 0x33)
        let outA = directory.appendingPathComponent("out-a.bin")
        let outB = directory.appendingPathComponent("out-b.bin")
        try Self.writeFile(outA, bytes: 128, fill: 0)
        try Self.writeFile(outB, bytes: 128, fill: 0)

        // Two copies read shard `a`, one reads `b`; `c` is named by no copy.
        let copies = [
            Self.copy(
                id: "a-0", shard: "a.safetensors", offset: 0, size: 64,
                destination: outA.path, destinationOffset: 0),
            Self.copy(
                id: "a-1", shard: "a.safetensors", offset: 64, size: 64,
                destination: outA.path, destinationOffset: 64),
            Self.copy(
                id: "b-0", shard: "b.safetensors", offset: 0, size: 128,
                destination: outB.path, destinationOffset: 0),
        ]
        try await Self.run(directory: directory, copies: copies, consume: true)
        #expect(!FileManager.default.fileExists(atPath: first.path))
        #expect(!FileManager.default.fileExists(atPath: second.path))
        #expect(FileManager.default.fileExists(atPath: unused.path))
        let a = try Data(contentsOf: outA)
        let b = try Data(contentsOf: outB)
        #expect(a == Data(repeating: 0x11, count: 128))
        #expect(b == Data(repeating: 0x22, count: 128))
    }

    @Test func leavesTheSnapshotAloneByDefault() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("consume-off-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("a.safetensors")
        try Self.writeFile(source, bytes: 64, fill: 0x44)
        let destination = directory.appendingPathComponent("out.bin")
        try Self.writeFile(destination, bytes: 64, fill: 0)
        let copies = [
            Self.copy(
                id: "a-0", shard: "a.safetensors", offset: 0, size: 64,
                destination: destination.path, destinationOffset: 0)
        ]
        try await Self.run(directory: directory, copies: copies, consume: false)
        #expect(FileManager.default.fileExists(atPath: source.path))
    }
}
