import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLib
@testable import TinyTitanCore

@Suite struct ServerModelSessionStorageBudgetTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "ornith",
        sourceSnapshotHash: "source",
        runtimeProfileHash: "profile",
        maximumContext: 8192,
        kvStorage: "int8",
        fp16RingEnabled: false,
        templateSHA256: "template")

    private func metadata() -> PromptCheckpointMetadata {
        let descriptor = InferenceStateSnapshotDescriptor(
            position: 3, kvSegmentLengths: [16], gdnSegmentLengths: [],
            payloadBytes: 16)
        return PromptCheckpointMetadata(
            key: PromptCheckpointKey.make(
                domain: domain, prefixTokenIDs: [1, 2, 3]),
            domain: domain,
            kind: .systemEnd,
            conversationKey: "conversation",
            prefixTokenIDs: [1, 2, 3],
            descriptor: descriptor,
            createdAt: Date())
    }

    private func recordIn(_ root: URL) async throws -> PromptCheckpointMetadata {
        let meta = metadata()
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        let writer = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root,
                diskLimitBytes: 1 << 20))
        try await writer.save(
            meta,
            snapshot: InferenceStateSnapshot(
                descriptor: meta.descriptor,
                payload: Data(repeating: 7, count: 16)))
        return meta
    }

    @Test func theDiskBudgetIsSplitOnlyWhenBothStoresExist() {
        let eightGiB = 8_192 * 1_048_576
        #expect(
            ServerModelSession.promptCachePerStoreDiskLimit(
                total: eightGiB, checkpointsEnabled: true)
                == 4_096 * 1_048_576)
        #expect(
            ServerModelSession.promptCachePerStoreDiskLimit(
                total: eightGiB, checkpointsEnabled: false)
                == eightGiB)
    }

    @Test func checkpointsAreOnUnlessExplicitlyDisabled() {
        #expect(
            ServerModelSession.checkpointsEnabled(environment: [:]))
        #expect(
            ServerModelSession.checkpointsEnabled(
                environment: ["TINYTITAN_PROMPT_CHECKPOINTS": "1"]))
        #expect(
            !ServerModelSession.checkpointsEnabled(
                environment: ["TINYTITAN_PROMPT_CHECKPOINTS": "0"]))
    }

    /// With checkpoints off, the records a previous enabled run left behind
    /// have no consumer: they are deleted, and the turn store gets the whole
    /// budget back.
    @Test func staleCheckpointRecordsAreReclaimedWhenCheckpointsAreOff() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("budget-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = try await recordIn(root)
        let staleDirectory = root.appendingPathComponent(meta.key.contentHash)
        #expect(FileManager.default.fileExists(atPath: staleDirectory.path))

        let survivors = try await ServerModelSession.reclaimStaleCheckpoints(
            directory: root, totalDiskLimitBytes: 1 << 20)
        #expect(survivors == 0)
        #expect(!FileManager.default.fileExists(atPath: staleDirectory.path))
    }

    /// A record that refuses deletion still occupies disk, so its bytes come
    /// back as survivors and are charged against the turn store's share
    /// instead of disappearing from the accounting.
    @Test func undelatableCheckpointBytesAreChargedToTheTurnStore() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("budget-\(UUID().uuidString)")
        let meta = try await recordIn(root)
        let staleDirectory = root.appendingPathComponent(meta.key.contentHash)
        defer {
            try? FileManager.default.setAttributes(
                [.immutable: false], ofItemAtPath: staleDirectory.path)
            try? FileManager.default.removeItem(at: root)
        }
        try FileManager.default.setAttributes(
            [.immutable: true], ofItemAtPath: staleDirectory.path)

        let survivors = try await ServerModelSession.reclaimStaleCheckpoints(
            directory: root, totalDiskLimitBytes: 1 << 20)
        #expect(survivors == 16)
        #expect(
            ServerModelSession.promptCachePerStoreDiskLimit(
                total: (1 << 20) - survivors, checkpointsEnabled: false)
                == (1 << 20) - 16)
        #expect(FileManager.default.fileExists(atPath: staleDirectory.path))
    }
}
