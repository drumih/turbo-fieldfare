import Foundation
import Synchronization
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

@Suite("Server prompt state store")
struct ServerPromptStateStoreTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    @Test func diskSnapshotSurvivesStoreRecreation() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0,
            diskDirectory: root,
            diskLimitBytes: 1_024)

        let writer = try ServerPromptStateStore(configuration: configuration)
        let saved = await writer.save(entry: entry, snapshot: snapshot)
        #expect(saved.diskError == nil)
        #expect(saved.diskBytes == snapshot.payload.count)
        let entryDirectory = root.appendingPathComponent(
            entry.id.uuidString.lowercased())
        #expect(permissions(of: root) & 0o077 == 0)
        #expect(permissions(of: entryDirectory) & 0o077 == 0)
        #expect(
            permissions(of: entryDirectory.appendingPathComponent("metadata.json"))
                & 0o077 == 0)
        #expect(
            permissions(of: entryDirectory.appendingPathComponent("state.bin"))
                & 0o077 == 0)

        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain) == [entry])
        #expect(reader.loadEntries(domain: domain) == [entry])
        let restored = try reader.loadSnapshot(entryID: entry.id)
        #expect(restored.tier == "ssd")
        #expect(restored.snapshot == snapshot)
    }

    @Test func memoryLRUEvictsOldestUnbackedSnapshot() async throws {
        let store = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 10,
                diskDirectory: nil,
                diskLimitBytes: 0))
        let first = makeEntry(tokens: [1])
        let second = makeEntry(tokens: [2])
        let firstSnapshot = makeSnapshot(position: 1, payload: Data(repeating: 1, count: 6))
        let secondSnapshot = makeSnapshot(position: 1, payload: Data(repeating: 2, count: 6))

        _ = await store.save(entry: first, snapshot: firstSnapshot)
        let saved = await store.save(entry: second, snapshot: secondSnapshot)

        #expect(saved.unbackedEntryIDs.contains(first.id))
        #expect(!store.contains(first.id))
        #expect(store.contains(second.id))
        #expect(try store.loadSnapshot(entryID: second.id).snapshot == secondSnapshot)
    }

    @Test func diskLRUEvictsOldestUnbackedSnapshot() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0,
                diskDirectory: root,
                diskLimitBytes: 10))
        let first = makeEntry(tokens: [1])
        let second = makeEntry(tokens: [2])

        _ = await store.save(
            entry: first,
            snapshot: makeSnapshot(
                position: 1,
                payload: Data(repeating: 1, count: 6)))
        let saved = await store.save(
            entry: second,
            snapshot: makeSnapshot(
                position: 1,
                payload: Data(repeating: 2, count: 6)))

        #expect(saved.unbackedEntryIDs.contains(first.id))
        #expect(!store.contains(first.id))
        #expect(store.contains(second.id))
        #expect(saved.diskBytes == 6)
    }

    @Test func diskEvictionTheVolumeRefusesKeepsItsBytesCounted() async throws {
        let root = temporaryDirectory()
        let first = makeEntry(tokens: [1])
        let second = makeEntry(tokens: [2])
        let firstPayload =
            root
            .appendingPathComponent(first.id.uuidString.lowercased())
            .appendingPathComponent("state.bin")
        defer {
            // Clearing the flag is what lets the cleanup below finish; left set, the
            // temp directory survives for the same reason the eviction failed.
            chflags(firstPayload.path, 0)
            try? FileManager.default.removeItem(at: root)
        }
        let store = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0,
                diskDirectory: root,
                diskLimitBytes: 10))

        _ = await store.save(
            entry: first,
            snapshot: makeSnapshot(
                position: 1,
                payload: Data(repeating: 1, count: 6)))
        // UF_IMMUTABLE on one file inside the entry directory is the volume this
        // store could not otherwise be made to meet: writing into the directory
        // still succeeds, and unlinking the directory answers EPERM. Measured with
        // the same two calls outside the store before this test was written.
        #expect(chflags(firstPayload.path, UInt32(UF_IMMUTABLE)) == 0)

        let saved = await store.save(
            entry: second,
            snapshot: makeSnapshot(
                position: 1,
                payload: Data(repeating: 2, count: 6)))

        // Twelve bytes are on the volume, so twelve must be what the store reports.
        // Subtracting `payloadBytes` before the `try? removeItem` that has to free
        // them is the defect: the cap then believes it has room.
        #expect(saved.diskBytes == 12)
        #expect(saved.diskError != nil)
        #expect(store.contains(first.id))
        #expect(!saved.unbackedEntryIDs.contains(first.id))
        let restored = try store.loadSnapshot(entryID: first.id)
        #expect(restored.snapshot.payload == Data(repeating: 1, count: 6))
    }

    @Test func corruptDiskPayloadFailsClosedAndIsRemoved() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0,
            diskDirectory: root,
            diskLimitBytes: 1_024)
        let writer = try ServerPromptStateStore(configuration: configuration)
        _ = await writer.save(entry: entry, snapshot: snapshot)
        let payload =
            root
            .appendingPathComponent(entry.id.uuidString.lowercased())
            .appendingPathComponent("state.bin")
        try Data([9, 2, 3, 4, 5, 6, 7]).write(to: payload)

        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain) == [entry])
        #expect(throws: ServerPromptStateStoreError.self) {
            try reader.loadSnapshot(entryID: entry.id)
        }
        #expect(
            !FileManager.default.fileExists(
                atPath: payload.deletingLastPathComponent().path))
    }

    /// A payload swapped to a larger file *after* the record was indexed is
    /// refused on the declared size, and the reader stops there: it asks the
    /// handle for `expected + 1` bytes and never walks to the end of the
    /// planted tail. The indexing pass checks the file size, so this is the
    /// window between indexing and reading — the only place the unbounded read
    /// could still put a planted file's bytes in memory.
    @Test func swappedOversizedDiskPayloadIsRefusedAtTheDeclaredSize() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0,
            diskDirectory: root,
            diskLimitBytes: 1_024)
        let writer = try ServerPromptStateStore(configuration: configuration)
        _ = await writer.save(entry: entry, snapshot: snapshot)

        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain) == [entry])

        let payload =
            root
            .appendingPathComponent(entry.id.uuidString.lowercased())
            .appendingPathComponent("state.bin")
        let handle = try FileHandle(forWritingTo: payload)
        try handle.truncate(atOffset: 16 * 1_024 * 1_024)
        try handle.close()

        #expect(throws: ServerPromptStateStoreError.self) {
            try reader.loadSnapshot(entryID: entry.id)
        }
        #expect(
            !FileManager.default.fileExists(
                atPath: payload.deletingLastPathComponent().path))
    }

    /// AUD-142: a record whose `metadata.json` crosses the store's bound is
    /// skipped, and skipped *without being read*. The shape this replaced stat'd
    /// the path, checked the size, then re-read the path — so the bytes checked
    /// were not the bytes used (K17), and the whole document was allocated before
    /// the refusal. The fixture is a sparse tail for that reason: it is 16 MiB and
    /// one byte of *size* with almost nothing behind it, which is exactly what a
    /// planted cache entry looks like and what an unbounded read would put in
    /// memory.
    @Test func oversizedDiskMetadataIsSkippedWithoutBeingRead() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0,
            diskDirectory: root,
            diskLimitBytes: 1_024)
        let writer = try ServerPromptStateStore(configuration: configuration)
        _ = await writer.save(entry: entry, snapshot: snapshot)

        let metadata =
            root
            .appendingPathComponent(entry.id.uuidString.lowercased())
            .appendingPathComponent("metadata.json")
        let handle = try FileHandle(forWritingTo: metadata)
        try handle.truncate(atOffset: 16 * 1_024 * 1_024 + 1)
        try handle.close()

        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain).isEmpty)
    }

    /// A link standing in for a record's metadata is not a record. The check that
    /// covers this used to be a `resourceValues` look at the path before the read;
    /// it now lives in the read, on the descriptor that is actually used.
    @Test func symlinkedDiskMetadataIsSkipped() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0,
            diskDirectory: root,
            diskLimitBytes: 1_024)
        let writer = try ServerPromptStateStore(configuration: configuration)
        _ = await writer.save(entry: entry, snapshot: snapshot)

        let record = root.appendingPathComponent(entry.id.uuidString.lowercased())
        let metadata = record.appendingPathComponent("metadata.json")
        let real = root.appendingPathComponent("elsewhere.json")
        try FileManager.default.moveItem(at: metadata, to: real)
        try FileManager.default.createSymbolicLink(
            atPath: metadata.path, withDestinationPath: real.path)

        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain).isEmpty)
    }

    private func makeEntry(tokens: [Int32]) -> ServerPromptCacheEntry {
        ServerPromptCacheEntry(
            id: UUID(),
            domain: domain,
            inputMessages: [GFTokenizer.Message(role: .user, content: "prompt")],
            tools: [],
            assistantTurn: CachedAssistantTurn(
                message: GFTokenizer.Message(role: .assistant, content: "answer"),
                rawStopReason: .endOfTurn),
            kvBackedTokenIDs: tokens,
            uncommittedBoundaryTokenIDs: [99],
            kvPosition: tokens.count)
    }

    private func makeSnapshot(position: Int, payload: Data) -> InferenceStateSnapshot {
        InferenceStateSnapshot(
            descriptor: InferenceStateSnapshotDescriptor(
                position: position,
                kvSegmentLengths: [payload.count - 3],
                gdnSegmentLengths: [3],
                payloadBytes: payload.count),
            payload: payload)
    }

    /// AUD2-048: the persisted LRU touch is not fatal, but a failure that changes
    /// restart order must not be silent. The checkpoint store reports the same
    /// write through `ServerLog.diagnostic` (AUD2-007); the turn store answered
    /// with `try?`, so a read-only volume or a replaced directory left the hot
    /// record looking untouched with nothing in the log to say why.
    @Test func aFailedDiskTouchIsReportedRatherThanSwallowed() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1])
        let store = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1_024))
        _ = await store.save(
            entry: entry,
            snapshot: makeSnapshot(position: 1, payload: Data(repeating: 1, count: 6)))

        // UF_IMMUTABLE makes the directory's own attributes refuse the write while
        // leaving the entry readable, which is the failure shape a read-only volume
        // or a directory replaced under the store presents.
        let directory = root.appendingPathComponent(entry.id.uuidString.lowercased())
        #expect(chflags(directory.path, UInt32(UF_IMMUTABLE)) == 0)
        defer { chflags(directory.path, 0) }

        let lines = Mutex<[String]>([])
        ServerLog.useSink { line in lines.withLock { $0.append(line) } }
        defer { ServerLog.useSink(nil) }

        store.touchDisk(entry.id)
        var message: String?
        for _ in 0..<100 {
            message = lines.withLock { captured in
                captured.first { $0.contains("could not record access on \(entry.id)") }
            }
            if message != nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(message != nil, "the dropped LRU touch must be logged, not swallowed")
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-prompt-cache-tests-\(UUID().uuidString)")
    }

    private func permissions(of url: URL) -> Int {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return attributes?[.posixPermissions] as? Int ?? 0o777
    }

    @Test func auxiliaryStateSurvivesStoreRecreation() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(
            position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let auxiliary = InferenceAuxiliaryState(
            position: 2, payload: Data(repeating: 8, count: 16))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1_024)
        let writer = try ServerPromptStateStore(configuration: configuration)
        let saved = await writer.save(
            entry: entry, snapshot: snapshot, auxiliary: auxiliary)
        #expect(saved.diskError == nil)
        #expect(saved.diskBytes == snapshot.payload.count + 16)
        #expect(
            permissions(
                of: root.appendingPathComponent(entry.id.uuidString.lowercased())
                    .appendingPathComponent("auxiliary.bin")) & 0o077 == 0)

        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain) == [entry])
        let loaded = try reader.loadSnapshot(entryID: entry.id)
        #expect(loaded.snapshot == snapshot)
        #expect(loaded.auxiliary == auxiliary)
    }

    @Test func corruptAuxiliaryStateIsRemovedRatherThanRestored() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let entry = makeEntry(tokens: [1, 2])
        let snapshot = makeSnapshot(
            position: 2, payload: Data([1, 2, 3, 4, 5, 6, 7]))
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1_024)
        let writer = try ServerPromptStateStore(configuration: configuration)
        _ = await writer.save(
            entry: entry, snapshot: snapshot,
            auxiliary: InferenceAuxiliaryState(
                position: 2, payload: Data(repeating: 3, count: 8)))
        let auxiliary =
            root.appendingPathComponent(entry.id.uuidString.lowercased())
            .appendingPathComponent("auxiliary.bin")
        try Data(repeating: 4, count: 8).write(to: auxiliary)
        let reader = try ServerPromptStateStore(configuration: configuration)
        #expect(reader.loadEntries(domain: domain) == [entry])
        #expect(throws: ServerPromptStateStoreError.self) {
            try reader.loadSnapshot(entryID: entry.id)
        }
        #expect(
            !FileManager.default.fileExists(
                atPath: auxiliary.deletingLastPathComponent().path))
    }

    @Test func auxiliaryStateCountsAgainstTheMemoryBudget() async throws {
        let store = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 10, diskDirectory: nil, diskLimitBytes: 0))
        let entry = makeEntry(tokens: [1])
        let snapshot = makeSnapshot(
            position: 1, payload: Data(repeating: 1, count: 6))
        let saved = await store.save(
            entry: entry, snapshot: snapshot,
            auxiliary: InferenceAuxiliaryState(
                position: 1, payload: Data(repeating: 2, count: 6)))
        #expect(saved.memoryBytes <= 10)
        #expect(!store.contains(entry.id))
    }
}
