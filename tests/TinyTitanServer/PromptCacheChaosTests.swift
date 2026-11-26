import Darwin
import Foundation
import Testing
import TinyTitanCore
@testable import TinyTitanLib

/// Pre-release chaos, concurrency, and fault-injection stress suite for the
/// prompt-state disk store (regression shield for GitHub #35 and TSan coverage).
@Suite("Prompt cache chaos and fault injection")
struct PromptCacheChaosTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "chaos-model",
        sourceSnapshotHash: "chaos-snapshot",
        runtimeProfileHash: "chaos-profile",
        maximumContext: 16_384,
        kvStorage: "int8",
        fp16RingEnabled: false,
        templateSHA256: "chaos-template")

    private func makeEntry(tokens: [Int32]) -> ServerPromptCacheEntry {
        ServerPromptCacheEntry(
            id: UUID(),
            domain: domain,
            inputMessages: [GFTokenizer.Message(role: .user, content: "hi")],
            tools: [],
            assistantTurn: CachedAssistantTurn(
                message: GFTokenizer.Message(role: .assistant, content: "ok"),
                rawStopReason: .eos),
            kvBackedTokenIDs: tokens,
            uncommittedBoundaryTokenIDs: [],
            kvPosition: tokens.count)
    }

    private func makeSnapshot(position: Int, payload: Data) -> InferenceStateSnapshot {
        InferenceStateSnapshot(
            descriptor: InferenceStateSnapshotDescriptor(
                position: position,
                kvSegmentLengths: [payload.count],
                gdnSegmentLengths: [],
                payloadBytes: payload.count),
            payload: payload)
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-prompt-chaos-\(UUID().uuidString)")
    }

    /// Concurrent overlapping saves, evictions, reads, touches, and removals
    /// under a tight disk budget — exercises the Task.detached race from #35.
    @Test func concurrentSavesAndEvictionsStayBoundedWithoutErrors() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 128, diskDirectory: root, diskLimitBytes: 384))
        try await withThrowingTaskGroup(of: Void.self) { group in
            for worker in 0..<8 {
                group.addTask {
                    for round in 0..<6 {
                        let token = Int32(worker * 100 + round)
                        let entry = makeEntry(tokens: [token, token + 1])
                        let payload = Data(repeating: UInt8((worker + round) & 0xFF), count: 96)
                        let saved = await store.save(
                            entry: entry,
                            snapshot: makeSnapshot(position: 2, payload: payload))
                        #expect(saved.diskError == nil)
                        store.touchDisk(entry.id)
                        _ = try? store.loadSnapshot(entryID: entry.id)
                        if round.isMultiple(of: 3) {
                            store.remove(entryIDs: [entry.id])
                        }
                    }
                }
            }
            try await group.waitForAll()
        }
        let reloaded = try ServerPromptStateStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 384))
        let entries = reloaded.loadEntries(domain: domain)
        #expect(entries.count * 96 <= 384)
    }

    /// Mid-session filesystem chaos: external ENOENT deletions, truncated
    /// payloads, immutable refusal + recovery, and dirty-restart accounting.
    @Test func externalDeletionsCorruptionAndImmutableRecoverySelfHeal() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let config = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 256)
        let store = try ServerPromptStateStore(configuration: config)

        let e1 = makeEntry(tokens: [1])
        let e2 = makeEntry(tokens: [2])
        let e3 = makeEntry(tokens: [3])
        let e4 = makeEntry(tokens: [4])
        let e5 = makeEntry(tokens: [5])
        let payload = Data(repeating: 0xAB, count: 100)

        _ = await store.save(entry: e1, snapshot: makeSnapshot(position: 1, payload: payload))
        _ = await store.save(entry: e2, snapshot: makeSnapshot(position: 1, payload: payload))

        // 1) External deletion of e1 (ENOENT): next save evicts e1 cleanly.
        try FileManager.default.removeItem(
            at: root.appendingPathComponent(e1.id.uuidString.lowercased()))
        let s3 = await store.save(entry: e3, snapshot: makeSnapshot(position: 1, payload: payload))
        #expect(s3.diskError == nil)
        #expect(!store.contains(e1.id))
        #expect(s3.diskBytes == 200)

        // 2) Corrupt e2 payload on disk: load fails closed and removes e2.
        let e2Payload = root.appendingPathComponent(e2.id.uuidString.lowercased())
            .appendingPathComponent("state.bin")
        try Data([0x00, 0x01]).write(to: e2Payload)
        #expect(throws: ServerPromptStateStoreError.self) {
            try store.loadSnapshot(entryID: e2.id)
        }
        #expect(!store.contains(e2.id))

        // 3) Immutable refusal on e3 is reported honestly, then clears once unflagged.
        let e3Dir = root.appendingPathComponent(e3.id.uuidString.lowercased())
        #expect(chflags(e3Dir.path, UInt32(UF_IMMUTABLE)) == 0)
        defer { chflags(e3Dir.path, 0) }
        _ = await store.save(entry: e4, snapshot: makeSnapshot(position: 1, payload: payload))
        let blocked = await store.save(
            entry: e5, snapshot: makeSnapshot(position: 1, payload: payload))
        #expect(blocked.diskError != nil)
        #expect(store.contains(e3.id))

        #expect(chflags(e3Dir.path, 0) == 0)
        let e6 = makeEntry(tokens: [6])
        let recovered = await store.save(
            entry: e6, snapshot: makeSnapshot(position: 1, payload: payload))
        #expect(recovered.diskError == nil)
        #expect(!store.contains(e3.id))
        #expect(recovered.diskBytes <= 256)

        // 4) Dirty restart with stray .tmp directory reconciles exact bytes.
        let stray = root.appendingPathComponent(".tmp-stray-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: stray, withIntermediateDirectories: true)
        let restarted = try ServerPromptStateStore(configuration: config)
        let surviving = restarted.loadEntries(domain: domain)
        #expect(surviving.count == 2)
        #expect(surviving.allSatisfy { (try? restarted.loadSnapshot(entryID: $0.id)) != nil })
    }
}
