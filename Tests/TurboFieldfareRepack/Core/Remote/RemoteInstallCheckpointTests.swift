import Foundation
import Darwin
import Testing

@testable import TurboFieldfareRepackCore

@Suite
struct RemoteInstallCheckpointTests {
    @Test func roundTripsCompactCheckpoint() throws {
        let root = tmpDirForRemote("checkpoint")
        let path = (root as NSString).appendingPathComponent("resume.json")
        defer { cleanUpRemote([root]) }
        try Posix.mkdirP(root)
        let checkpoint = sampleCheckpoint()

        try checkpoint.write(to: path, parentDirectory: root)

        #expect(try RemoteInstallCheckpoint.load(from: path) == checkpoint)
        #expect(try Data(contentsOf: URL(fileURLWithPath: path)).count < 1024)
        #expect(!checkpoint.matches(
            repoID: checkpoint.repoID,
            requestedRevision: checkpoint.requestedRevision,
            sourceIndexSHA256: String(repeating: "e", count: 64),
            planFingerprint: checkpoint.planFingerprint))
    }

    @Test func oversizedCheckpointIsRejectedBeforeDecode() throws {
        let root = tmpDirForRemote("checkpoint-large")
        let path = (root as NSString).appendingPathComponent("resume.json")
        defer { cleanUpRemote([root]) }
        try Posix.mkdirP(root)
        let descriptor = try Posix.openCreateRW(path)
        try Posix.ftruncate(
            descriptor,
            path: path,
            size: RemoteInstallCheckpoint.maximumBytes + 1)
        close(descriptor)

        #expect(throws: RepackError.self) {
            _ = try RemoteInstallCheckpoint.load(from: path)
        }
    }

    @Test func damagedDestinationInvalidatesItsRange() throws {
        let root = tmpDirForRemote("checkpoint-digest")
        let path = (root as NSString).appendingPathComponent("weights.bin")
        defer { cleanUpRemote([root]) }
        try Posix.mkdirP(root)
        let descriptor = try Posix.openCreateRW(path)
        var bytes = [UInt8](repeating: 7, count: 16)
        try bytes.withUnsafeBytes {
            try Posix.pwriteAll(
                fd: descriptor,
                path: path,
                buf: $0.baseAddress!,
                count: $0.count,
                offset: 0)
        }
        close(descriptor)

        let copy = CoalescedRangeCopy(
            id: "range-00000000",
            shardID: "source.bin",
            sourceOffset: 0,
            size: 16,
            destinations: [
                RangeCopy(
                    shardID: "source.bin",
                    sourceOffset: 0,
                    size: 16,
                    destinationPath: path,
                    destinationOffset: 0),
            ])
        let digest = try HTTPRangeSourceByteProvider.destinationDigest(
            copy,
            partialDirectory: root)
        let completed = RemoteCompletedRange(
            id: copy.id,
            destinationDigest: digest,
            sourceBytes: 16,
            destinationBytes: 16)
        #expect(try RemoteStreamingRepacker.validatedCompletedRanges(
            [completed],
            copies: [copy],
            partialDirectory: root) == [completed])

        let writeDescriptor = try Posix.openExistingRW(path)
        bytes[0] = 8
        try bytes.withUnsafeBytes {
            try Posix.pwriteAll(
                fd: writeDescriptor,
                path: path,
                buf: $0.baseAddress!,
                count: 1,
                offset: 0)
        }
        close(writeDescriptor)
        #expect(try RemoteStreamingRepacker.validatedCompletedRanges(
            [completed],
            copies: [copy],
            partialDirectory: root).isEmpty)
    }
}

private func sampleCheckpoint() -> RemoteInstallCheckpoint {
    RemoteInstallCheckpoint(
        repoID: "owner/model",
        requestedRevision: "main",
        resolvedCommit: String(repeating: "a", count: 40),
        sourceIndexSHA256: String(repeating: "b", count: 64),
        planFingerprint: String(repeating: "c", count: 64),
        totalSourceBytes: 10,
        completedRanges: [
            RemoteCompletedRange(
                id: "range-00000000",
                destinationDigest: String(repeating: "d", count: 64),
                sourceBytes: 10,
                destinationBytes: 8),
        ])
}
