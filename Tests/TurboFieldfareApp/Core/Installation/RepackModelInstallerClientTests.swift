import Foundation
@testable import TurboFieldfareRepackCore
import Synchronization
import Testing
@testable import TurboFieldfareAppCore

@Suite struct RepackModelInstallerClientTests {
    @Test func mapsCoreProgressAndCompletion() async throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("scripted.gturbo")
        let client = RepackModelInstallerClient { _, outputDirectory, progress in
            progress(.downloadingMetadata)
            progress(.planning(downloadBytes: 10, outputBytes: 20))
            progress(.checkingDisk(DiskSpaceRequirement(path: "/",
                                                        requiredBytes: 30,
                                                        availableBytes: 40)))
            progress(.reservingOutput(bytes: 20))
            progress(.copyingPayload(
                reusedBytes: 1,
                downloadedThisRunBytes: 3,
                totalBytes: 10))
            progress(.hashingOutput("model_weights.bin"))
            progress(.finalizing)
            return outputDirectory
        }

        var events: [AppModelInstallEvent] = []
        for try await event in client.installDefaultModel(outputDirectory: output) {
            events.append(event)
        }

        #expect(events == [
            .checking,
            .downloadingMetadata,
            .planning,
            .checking,
            .reservingOutput,
            .copyingPayload(
                reusedBytes: 1,
                downloadedThisRunBytes: 3,
                totalBytes: 10),
            .hashingOutput("model_weights.bin"),
            .finalizing,
            .installed(output.standardizedFileURL),
        ])
    }

    @Test func cancelPropagatesCancellationToConsumer() async throws {
        let started = Flag()
        let client = RepackModelInstallerClient { _, outputDirectory, progress in
            progress(.downloadingMetadata)
            started.set()
            try await Task.sleep(for: .seconds(60))
            return outputDirectory
        }
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("cancelled.gturbo")
        let consume = Task {
            for try await _ in client.installDefaultModel(outputDirectory: output) {}
        }

        for _ in 0..<200 where !started.value {
            await Task.yield()
        }
        #expect(started.value)
        client.cancel()

        await #expect(throws: CancellationError.self) {
            try await consume.value
        }
    }

    @Test func rejectsUInt64MaxCheckpointDestinationBytes() throws {
        try expectCorruptCheckpoint(destinationBytes: [UInt64.max])
    }

    @Test func rejectsCheckpointDestinationByteOverflow() throws {
        try expectCorruptCheckpoint(destinationBytes: [UInt64.max, 1])
    }

    @Test func rejectsCheckpointDestinationBytesAboveInstalledModel() throws {
        try expectCorruptCheckpoint(destinationBytes: [101])
    }

    private func expectCorruptCheckpoint(destinationBytes: [UInt64]) throws {
        let descriptor = AppModelInstallDescriptor(
            displayName: "Test",
            repoID: "owner/model",
            revision: "main",
            sourceIndexSHA256: String(repeating: "b", count: 64),
            approximateDownloadBytes: 100,
            installedBytes: 100,
            rangeStagingBytes: 0,
            reserveBytes: 0)
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("checkpoint-bytes-\(UUID().uuidString).gturbo")
        let paths = try RemoteInstallPaths(outputDirectory: output.path)
        defer {
            for path in [
                paths.partialDirectory,
                paths.checkpointFile,
                paths.lockFile,
            ] {
                try? FileManager.default.removeItem(atPath: path)
            }
        }
        try FileManager.default.createDirectory(
            atPath: paths.partialDirectory,
            withIntermediateDirectories: true)
        let checkpoint = RemoteInstallCheckpoint(
            repoID: descriptor.repoID,
            requestedRevision: descriptor.revision,
            resolvedCommit: String(repeating: "a", count: 40),
            sourceIndexSHA256: descriptor.sourceIndexSHA256,
            planFingerprint: String(repeating: "c", count: 64),
            totalSourceBytes: UInt64(destinationBytes.count),
            completedRanges: destinationBytes.enumerated().map { index, bytes in
                RemoteCompletedRange(
                    id: "range-\(index)",
                    destinationDigest: String(repeating: "d", count: 64),
                    sourceBytes: 1,
                    destinationBytes: bytes)
            })
        try JSONEncoder().encode(checkpoint).write(
            to: URL(fileURLWithPath: paths.checkpointFile))
        let client = RepackModelInstallerClient(descriptor: descriptor)

        do {
            _ = try client.checkInstallRequirement(outputDirectory: output)
            Issue.record("invalid checkpoint byte total was accepted")
        } catch let error as RepackError {
            guard case .installStateCorrupt = error else {
                Issue.record("expected installStateCorrupt, got \(error)")
                return
            }
        }
    }
}

private final class Flag: Sendable {
    private let storage = Mutex(false)
    var value: Bool { storage.withLock { $0 } }
    func set() { storage.withLock { $0 = true } }
}
