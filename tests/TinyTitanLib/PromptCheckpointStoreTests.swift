import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLib
@testable import TinyTitanCore

@Suite struct PromptCheckpointStoreTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "ornith",
        sourceSnapshotHash: "source",
        runtimeProfileHash: "profile",
        maximumContext: 8192,
        kvStorage: "int8",
        fp16RingEnabled: false,
        templateSHA256: "template")

    private func descriptor(
        position: Int, payloadBytes: Int
    ) -> InferenceStateSnapshotDescriptor {
        InferenceStateSnapshotDescriptor(
            position: position,
            kvSegmentLengths: [payloadBytes],
            gdnSegmentLengths: [],
            payloadBytes: payloadBytes)
    }

    private func metadata(
        tokens: [Int32],
        kind: PromptCheckpointKind = .systemEnd,
        conversation: String = "conversation",
        position: Int? = nil,
        key: PromptCheckpointKey? = nil,
        createdAt: Date = Date(),
        domain: ServerPromptCacheDomain? = nil
    ) -> PromptCheckpointMetadata {
        let resolvedDomain = domain ?? self.domain
        return PromptCheckpointMetadata(
            key: key
                ?? PromptCheckpointKey.make(
                    domain: resolvedDomain, prefixTokenIDs: tokens),
            domain: resolvedDomain,
            kind: kind,
            conversationKey: conversation,
            prefixTokenIDs: tokens,
            descriptor: descriptor(
                position: position ?? tokens.count,
                payloadBytes: 16),
            createdAt: createdAt)
    }

    private func snapshot(
        for metadata: PromptCheckpointMetadata, byte: UInt8 = 7
    ) -> InferenceStateSnapshot {
        InferenceStateSnapshot(
            descriptor: metadata.descriptor,
            payload: Data(repeating: byte, count: metadata.descriptor.payloadBytes))
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("checkpoints-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    private func memoryStore(limit: Int = 1 << 20) throws -> PromptCheckpointStore {
        try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: limit, diskDirectory: nil,
                diskLimitBytes: 0))
    }

    @Test func memoryRoundTrip() async throws {
        let store = try memoryStore()
        let meta = metadata(tokens: [1, 2, 3])
        try await store.save(meta, snapshot: snapshot(for: meta))
        let present = await store.contains(meta.key)
        #expect(present)
        let loaded = try await store.load(meta.key, domain: domain)
        #expect(loaded.metadata == meta)
        #expect(loaded.snapshot.payload == snapshot(for: meta).payload)
    }

    @Test func contentHashIsPrefixAndDomainSensitive() {
        let base = PromptCheckpointKey.make(
            domain: domain, prefixTokenIDs: [1, 2, 3])
        #expect(
            base
                == PromptCheckpointKey.make(
                    domain: domain, prefixTokenIDs: [1, 2, 3]))
        #expect(
            base
                != PromptCheckpointKey.make(
                    domain: domain, prefixTokenIDs: [1, 2, 4]))
        #expect(
            base
                != PromptCheckpointKey.make(
                    domain: domain, prefixTokenIDs: [1, 2]))
        var other = domain
        other = ServerPromptCacheDomain(
            modelID: "other", sourceSnapshotHash: other.sourceSnapshotHash,
            runtimeProfileHash: other.runtimeProfileHash,
            maximumContext: other.maximumContext, kvStorage: other.kvStorage,
            fp16RingEnabled: other.fp16RingEnabled,
            templateSHA256: other.templateSHA256)
        #expect(
            base
                != PromptCheckpointKey.make(
                    domain: other, prefixTokenIDs: [1, 2, 3]))
    }

    @Test func positionAndKeyMismatchesAreRefused() async throws {
        let store = try memoryStore()
        let wrongPosition = metadata(tokens: [1, 2], position: 3)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(
                wrongPosition,
                snapshot: self.snapshot(
                    for: wrongPosition))
        }
        let foreignKey = PromptCheckpointKey.make(
            domain: domain, prefixTokenIDs: [9, 9])
        let wrongKey = metadata(tokens: [1, 2], key: foreignKey)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(wrongKey, snapshot: self.snapshot(for: wrongKey))
        }
    }

    @Test func payloadSizeMismatchIsRefused() async throws {
        let store = try memoryStore()
        let meta = metadata(tokens: [1, 2, 3])
        let bad = InferenceStateSnapshot(
            descriptor: meta.descriptor,
            payload: Data(repeating: 1, count: 8))
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(meta, snapshot: bad)
        }
    }

    @Test func domainMismatchOnLoadIsRefused() async throws {
        let store = try memoryStore()
        let meta = metadata(tokens: [1, 2, 3])
        try await store.save(meta, snapshot: snapshot(for: meta))
        let other = ServerPromptCacheDomain(
            modelID: "other", sourceSnapshotHash: domain.sourceSnapshotHash,
            runtimeProfileHash: domain.runtimeProfileHash,
            maximumContext: domain.maximumContext, kvStorage: domain.kvStorage,
            fp16RingEnabled: domain.fp16RingEnabled,
            templateSHA256: domain.templateSHA256)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.load(meta.key, domain: other)
        }
        let stillThere = try await store.load(meta.key, domain: domain)
        #expect(stillThere.metadata == meta)
    }

    @Test func memoryLimitEvictsTheOldest() async throws {
        let store = try memoryStore(limit: 16)
        let first = metadata(
            tokens: [1, 2, 3], createdAt: Date(timeIntervalSince1970: 1))
        let second = metadata(
            tokens: [4, 5, 6], createdAt: Date(timeIntervalSince1970: 2))
        try await store.save(first, snapshot: snapshot(for: first))
        try await store.save(second, snapshot: snapshot(for: second))
        let hasSecond = await store.contains(second.key)
        let hasFirst = await store.contains(first.key)
        #expect(hasSecond)
        #expect(!hasFirst)
    }

    @Test func diskRoundTripsAcrossStores() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let first = try PromptCheckpointStore(configuration: configuration)
            try await first.save(meta, snapshot: snapshot(for: meta))
        }
        let second = try PromptCheckpointStore(configuration: configuration)
        let loaded = try await second.load(meta.key, domain: domain)
        #expect(loaded.metadata == meta)
        #expect(loaded.snapshot.payload.count == meta.descriptor.payloadBytes)
        let listed = await second.metadata(for: domain)
        #expect(listed == [meta])
    }

    @Test func corruptDiskPayloadIsDeletedAndReported() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let first = try PromptCheckpointStore(configuration: configuration)
            try await first.save(meta, snapshot: snapshot(for: meta))
        }
        let payload = root.appendingPathComponent(meta.key.contentHash)
            .appendingPathComponent("state.bin")
        var bytes = try Data(contentsOf: payload)
        bytes[0] ^= 0xFF
        try bytes.write(to: payload)
        let second = try PromptCheckpointStore(configuration: configuration)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await second.load(meta.key, domain: domain)
        }
        #expect(!FileManager.default.fileExists(atPath: payload.path))
    }

    @Test(
        arguments: [-1, PromptCheckpointStore.maximumAuxiliaryBytes + 1, Int.max],
        [false, true])
    func invalidAuxiliaryByteCountsAreRefused(bytes: Int, indexFirst: Bool) async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: Int.max)
        let meta = metadata(tokens: [1, 2, 3])
        let writer = try PromptCheckpointStore(configuration: configuration)
        try await writer.save(meta, snapshot: snapshot(for: meta))
        let path = root.appendingPathComponent(meta.key.contentHash)
            .appendingPathComponent("metadata.json")
        var json = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        json["auxiliaryBytes"] = bytes
        try JSONSerialization.data(withJSONObject: json).write(to: path)
        let reader = try PromptCheckpointStore(configuration: configuration)
        if indexFirst {
            #expect(!(await reader.contains(meta.key)))
            // `contains` can also answer from the directory itself; this
            // asserts the index skipped the record rather than charging it.
            #expect(await reader.metadata(for: domain).isEmpty)
        } else {
            await #expect(throws: PromptCheckpointStoreError.self) {
                try await reader.load(meta.key, domain: domain)
            }
        }
        #expect(await reader.diskBytes == 0)
    }

    @Test(
        arguments: [-1, PromptCheckpointStore.maximumCaptureBytes + 1, Int.max],
        [false, true])
    func invalidPayloadByteCountsAreRefused(bytes: Int, indexFirst: Bool) async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: Int.max)
        let meta = metadata(tokens: [1, 2, 3])
        let writer = try PromptCheckpointStore(configuration: configuration)
        try await writer.save(meta, snapshot: snapshot(for: meta))
        let path = root.appendingPathComponent(meta.key.contentHash)
            .appendingPathComponent("metadata.json")
        var json = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
        var checkpoint = try #require(json["metadata"] as? [String: Any])
        var descriptor = try #require(checkpoint["descriptor"] as? [String: Any])
        descriptor["payloadBytes"] = bytes
        descriptor["kvSegmentLengths"] = [bytes]
        checkpoint["descriptor"] = descriptor
        json["metadata"] = checkpoint
        try JSONSerialization.data(withJSONObject: json).write(to: path)
        let reader = try PromptCheckpointStore(configuration: configuration)
        if indexFirst {
            #expect(!(await reader.contains(meta.key)))
            // `contains` can also answer from the directory itself; this
            // asserts the index skipped the record rather than charging it.
            #expect(await reader.metadata(for: domain).isEmpty)
        } else {
            await #expect(throws: PromptCheckpointStoreError.self) {
                try await reader.load(meta.key, domain: domain)
            }
        }
        #expect(await reader.diskBytes == 0)
    }

    @Test func diskLimitEvictsTheOldestDirectory() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 20)
        let store = try PromptCheckpointStore(configuration: configuration)
        let first = metadata(
            tokens: [1, 2, 3], createdAt: Date(timeIntervalSince1970: 1))
        let second = metadata(
            tokens: [4, 5, 6], createdAt: Date(timeIntervalSince1970: 2))
        try await store.save(first, snapshot: snapshot(for: first))
        try await store.save(second, snapshot: snapshot(for: second))
        let hasSecond = await store.contains(second.key)
        let hasFirst = await store.contains(first.key)
        #expect(hasSecond)
        #expect(!hasFirst)
        let firstDirectory = root.appendingPathComponent(
            first.key.contentHash)
        #expect(!FileManager.default.fileExists(atPath: firstDirectory.path))
    }

    @Test func oversizedPayloadIsRefused() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 8)
        let store = try PromptCheckpointStore(configuration: configuration)
        let meta = metadata(tokens: [1, 2, 3])
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(meta, snapshot: self.snapshot(for: meta))
        }
    }

    @Test func aDiskOnlyRecordIsFoundWithoutLoadingIt() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let first = try PromptCheckpointStore(configuration: configuration)
            try await first.save(meta, snapshot: snapshot(for: meta))
        }
        let second = try PromptCheckpointStore(configuration: configuration)
        // `contains` must see the record the first process wrote, or the
        // capture path rewrites the same multi-gigabyte payload every turn.
        #expect(await second.contains(meta.key))
        let absent = PromptCheckpointKey.make(
            domain: domain, prefixTokenIDs: [7])
        #expect(!(await second.contains(absent)))
    }

    @Test func rewritingTheSameDiskRecordChargesItsBytesOnce() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let store = try PromptCheckpointStore(configuration: configuration)
        let meta = metadata(tokens: [1, 2, 3])
        try await store.save(meta, snapshot: snapshot(for: meta))
        let afterFirst = await store.diskBytes
        try await store.save(meta, snapshot: snapshot(for: meta, byte: 8))
        let afterSecond = await store.diskBytes
        #expect(afterFirst == meta.descriptor.payloadBytes)
        #expect(afterSecond == meta.descriptor.payloadBytes)
    }

    @Test func metadataListsOneConversationOldestFirst() async throws {
        let store = try memoryStore()
        let first = metadata(
            tokens: [1, 2], conversation: "a",
            createdAt: Date(timeIntervalSince1970: 1))
        let second = metadata(
            tokens: [3, 4], conversation: "a",
            createdAt: Date(timeIntervalSince1970: 2))
        let other = metadata(
            tokens: [5, 6], conversation: "b",
            createdAt: Date(timeIntervalSince1970: 3))
        try await store.save(first, snapshot: snapshot(for: first))
        try await store.save(second, snapshot: snapshot(for: second))
        try await store.save(other, snapshot: snapshot(for: other))
        let listed = await store.metadata(for: domain)
        #expect(listed.map(\.conversationKey) == ["a", "a", "b"])
        #expect(listed.map(\.position) == [2, 2, 2])
    }

    @Test func concurrentSavesAndLoadsSeeWholeRecords() async throws {
        let store = try memoryStore(limit: 1 << 22)
        let domain = self.domain
        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<16 {
                group.addTask {
                    let tokens = [Int32(index), 40, 41]
                    let key = PromptCheckpointKey.make(
                        domain: domain, prefixTokenIDs: tokens)
                    let descriptor = InferenceStateSnapshotDescriptor(
                        position: tokens.count,
                        kvSegmentLengths: [16],
                        gdnSegmentLengths: [],
                        payloadBytes: 16)
                    let metadata = PromptCheckpointMetadata(
                        key: key,
                        domain: domain,
                        kind: .systemEnd,
                        conversationKey: "conversation",
                        prefixTokenIDs: tokens,
                        descriptor: descriptor)
                    let snapshot = InferenceStateSnapshot(
                        descriptor: descriptor,
                        payload: Data(repeating: UInt8(index), count: 16))
                    try await store.save(metadata, snapshot: snapshot)
                    let loaded = try await store.load(key, domain: domain)
                    #expect(loaded.metadata == metadata)
                    #expect(
                        loaded.snapshot.payload
                            == Data(repeating: UInt8(index), count: 16))
                }
            }
            try await group.waitForAll()
        }
        #expect(await store.memoryBytes == 16 * 16)
    }

    @Test func aLoadedDiskRecordIsListedOnceAndOldestFirst() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 1 << 20, diskDirectory: root,
            diskLimitBytes: 1 << 20)
        let first = metadata(
            tokens: [1, 2], createdAt: Date(timeIntervalSince1970: 1))
        let second = metadata(
            tokens: [3, 4], createdAt: Date(timeIntervalSince1970: 2))
        do {
            // Force both onto disk by saving through a memory-less store, so
            // the reading store has to index them from the filesystem.
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root,
                    diskLimitBytes: 1 << 20))
            try await writer.save(first, snapshot: snapshot(for: first))
            try await writer.save(second, snapshot: snapshot(for: second))
        }
        let reader = try PromptCheckpointStore(configuration: configuration)
        _ = try await reader.load(second.key, domain: domain)
        let listed = await reader.metadata(for: domain)
        #expect(listed.count == 2)
        #expect(listed.map(\.key) == [first.key, second.key])
    }

    @Test func auxiliaryQSAStateTravelsWithTheCheckpoint() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let meta = metadata(tokens: [1, 2, 3])
        let auxiliary = InferenceAuxiliaryState(
            position: 3, payload: Data(repeating: 9, count: 64))
        do {
            let writer = try PromptCheckpointStore(configuration: configuration)
            try await writer.save(
                meta, snapshot: snapshot(for: meta), auxiliary: auxiliary)
        }
        let reader = try PromptCheckpointStore(configuration: configuration)
        let loaded = try await reader.load(meta.key, domain: domain)
        #expect(loaded.auxiliary == auxiliary)
        #expect(loaded.snapshot.payload == snapshot(for: meta).payload)
    }

    @Test func corruptAuxiliaryStateIsDeletedRatherThanRestored() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let meta = metadata(tokens: [1, 2, 3])
        let auxiliary = InferenceAuxiliaryState(
            position: 3, payload: Data(repeating: 4, count: 32))
        do {
            let writer = try PromptCheckpointStore(configuration: configuration)
            try await writer.save(
                meta, snapshot: snapshot(for: meta), auxiliary: auxiliary)
        }
        let path = root.appendingPathComponent(meta.key.contentHash)
            .appendingPathComponent("auxiliary.bin")
        var bytes = try Data(contentsOf: path)
        bytes[0] ^= 0xFF
        try bytes.write(to: path)
        let reader = try PromptCheckpointStore(configuration: configuration)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await reader.load(meta.key, domain: domain)
        }
        #expect(!FileManager.default.fileExists(atPath: path.path))
    }

    @Test func rewritingWithoutAuxiliaryDropsTheOldPayload() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let store = try PromptCheckpointStore(configuration: configuration)
        let meta = metadata(tokens: [1, 2, 3])
        try await store.save(
            meta, snapshot: snapshot(for: meta),
            auxiliary: InferenceAuxiliaryState(
                position: 3, payload: Data(repeating: 1, count: 32)))
        let firstBytes = await store.diskBytes
        try await store.save(meta, snapshot: snapshot(for: meta))
        let secondBytes = await store.diskBytes
        #expect(firstBytes == meta.descriptor.payloadBytes + 32)
        #expect(secondBytes == meta.descriptor.payloadBytes)
        let loaded = try await store.load(meta.key, domain: domain)
        #expect(loaded.auxiliary == nil)
    }

    @Test func auxiliaryStateCountsAgainstTheMemoryLimit() async throws {
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 20, diskDirectory: nil, diskLimitBytes: 0))
        let meta = metadata(tokens: [1, 2, 3])
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(
                meta, snapshot: self.snapshot(for: meta),
                auxiliary: InferenceAuxiliaryState(
                    position: 3, payload: Data(repeating: 1, count: 8)))
        }
    }

    /// The write goes through a staging directory and a rename swap, so a
    /// successful save — first write or rewrite — must leave exactly the
    /// record directory under the root: no `.tmp-` staging and no `.bak-`
    /// backup from the swap the rewrite performs.
    @Test func aRewriteLeavesOnlyTheRecordDirectory() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let store = try PromptCheckpointStore(configuration: configuration)
        let meta = metadata(tokens: [1, 2, 3])
        try await store.save(meta, snapshot: snapshot(for: meta))
        try await store.save(meta, snapshot: snapshot(for: meta, byte: 8))
        let loaded = try await store.load(meta.key, domain: domain)
        #expect(loaded.snapshot.payload == Data(repeating: 8, count: 16))
        let entries = try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil)
        #expect(entries.map(\.lastPathComponent) == [meta.key.contentHash])
    }

    /// A `.tmp-` staging directory is what a crash mid-write leaves behind;
    /// it holds no metadata, so it must stay invisible to both indexing and
    /// lookup rather than being served or charged.
    @Test func aStaleStagingDirectoryIsNotIndexedOrServed() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        let staging = root.appendingPathComponent(
            ".tmp-\(meta.key.contentHash)-\(UUID().uuidString)",
            isDirectory: true)
        try FileManager.default.createDirectory(
            at: staging, withIntermediateDirectories: true)
        try Data(repeating: 5, count: 16).write(
            to: staging.appendingPathComponent("state.bin"))
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root,
                diskLimitBytes: 1 << 20))
        #expect(!(await store.contains(meta.key)))
        let listed = await store.metadata(for: domain)
        #expect(listed.isEmpty)
        #expect(await store.diskBytes == 0)
        #expect(
            !FileManager.default.fileExists(atPath: staging.path),
            "an unpublished staging directory is reclaimed, not left on disk")
    }

    /// A record directory a crashed in-place write left incomplete (payload,
    /// no metadata) is replaced whole by the next save of that key.
    @Test func anIncompleteRecordDirectoryIsReplacedWhole() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        let directory = root.appendingPathComponent(
            meta.key.contentHash, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data(repeating: 5, count: 16).write(
            to: directory.appendingPathComponent("state.bin"))
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root,
                diskLimitBytes: 1 << 20))
        try await store.save(meta, snapshot: snapshot(for: meta, byte: 9))
        let loaded = try await store.load(meta.key, domain: domain)
        #expect(loaded.snapshot.payload == Data(repeating: 9, count: 16))
        let names = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .map(\.lastPathComponent).sorted()
        #expect(names == ["metadata.json", "state.bin"])
    }

    /// A directory whose metadata.json cannot be decoded is corrupt, not
    /// absent: `load` reports it as corrupt, and the directory is deleted so
    /// its bytes are reclaimed instead of staying uncounted forever.
    @Test func corruptDiskMetadataIsQuarantinedNotMissing() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        let directory = root.appendingPathComponent(
            meta.key.contentHash, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data("not json".utf8).write(
            to: directory.appendingPathComponent("metadata.json"))
        try Data(repeating: 1, count: 16).write(
            to: directory.appendingPathComponent("state.bin"))
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root,
                diskLimitBytes: 1 << 20))
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.load(meta.key, domain: domain)
        }
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    /// `contains` cannot throw, but a corrupt record it trips over must not
    /// survive as uncounted disk either.
    @Test func containsQuarantinesACorruptRecordItFinds() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        let directory = root.appendingPathComponent(
            meta.key.contentHash, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data("not json".utf8).write(
            to: directory.appendingPathComponent("metadata.json"))
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root,
                diskLimitBytes: 1 << 20))
        #expect(!(await store.contains(meta.key)))
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    /// After a restart the eviction order is the directories' modification
    /// dates, not the order the volume happened to list them in.
    @Test func restartEvictionOrderFollowsModificationDates() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let oldest = metadata(tokens: [1, 2, 3])
        let newest = metadata(tokens: [4, 5, 6])
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root,
                    diskLimitBytes: 1 << 20))
            try await writer.save(oldest, snapshot: snapshot(for: oldest))
            try await writer.save(newest, snapshot: snapshot(for: newest))
        }
        // Backdate the first record so the listing order cannot decide the
        // outcome: whichever order the volume returns, the older mtime goes.
        let oldestDirectory = root.appendingPathComponent(
            oldest.key.contentHash)
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 0)],
            ofItemAtPath: oldestDirectory.path)
        // A fresh store with room for two records (each payload is 16 bytes):
        // indexing both, then saving a third, must evict the record with the
        // oldest modification date.
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 32))
        let third = metadata(tokens: [7, 8, 9])
        try await store.save(third, snapshot: snapshot(for: third))
        #expect(!(await store.contains(oldest.key)))
        #expect(await store.contains(newest.key))
        #expect(await store.contains(third.key))
        #expect(!FileManager.default.fileExists(atPath: oldestDirectory.path))
    }

    /// A real access has to survive the restart it exists to protect: save A,
    /// save B, load A, restart and fill the budget must evict B — the record
    /// nobody touched. Updating only the in-memory date on touch makes the
    /// loaded record the first eviction candidate again.
    @Test func aLoadedRecordKeepsItsPlaceAcrossARestart() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let loaded = metadata(tokens: [1, 2, 3])
        let untouched = metadata(tokens: [4, 5, 6])
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root,
                    diskLimitBytes: 1 << 20))
            try await writer.save(loaded, snapshot: snapshot(for: loaded))
            try await writer.save(untouched, snapshot: snapshot(for: untouched))
        }
        // Both directories are pinned in the past, A behind B, so the outcome
        // rests on the access alone and not on timestamp resolution: only a
        // persisted touch can move A past B.
        let fm = FileManager.default
        try fm.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 0)],
            ofItemAtPath: root.appendingPathComponent(
                loaded.key.contentHash
            ).path)
        try fm.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 1)],
            ofItemAtPath: root.appendingPathComponent(
                untouched.key.contentHash
            ).path)
        let reader = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root,
                diskLimitBytes: 1 << 20))
        _ = try await reader.load(loaded.key, domain: domain)
        // A restarted store with room for two of the three 16-byte records.
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 32))
        let third = metadata(tokens: [7, 8, 9])
        try await store.save(third, snapshot: snapshot(for: third))
        #expect(await store.contains(loaded.key))
        #expect(!(await store.contains(untouched.key)))
        #expect(await store.contains(third.key))
    }

    /// `removeAll` on a store that has not indexed the disk yet still has to
    /// delete what is there.
    @Test func removeAllOnAFreshStoreDeletesTheDiskRecords() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root,
                    diskLimitBytes: 1 << 20))
            try await writer.save(meta, snapshot: snapshot(for: meta))
        }
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root,
                diskLimitBytes: 1 << 20))
        await store.removeAll()
        let directory = root.appendingPathComponent(meta.key.contentHash)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(await store.diskBytes == 0)
    }

    /// A record loaded before anything indexed the root is the most recently
    /// used, not the oldest: the backdated directory still goes first.
    @Test func aRecordLoadedBeforeIndexingKeepsItsPlaceInTheLRU() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let oldest = metadata(tokens: [1, 2, 3])
        let newest = metadata(tokens: [4, 5, 6])
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root,
                    diskLimitBytes: 1 << 20))
            try await writer.save(oldest, snapshot: snapshot(for: oldest))
            try await writer.save(newest, snapshot: snapshot(for: newest))
        }
        let oldestDirectory = root.appendingPathComponent(oldest.key.contentHash)
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 0)],
            ofItemAtPath: oldestDirectory.path)
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 32))
        _ = try await store.load(newest.key, domain: domain)
        let third = metadata(tokens: [7, 8, 9])
        try await store.save(third, snapshot: snapshot(for: third))
        #expect(!(await store.contains(oldest.key)))
        #expect(await store.contains(newest.key))
        #expect(await store.contains(third.key))
        #expect(!FileManager.default.fileExists(atPath: oldestDirectory.path))
    }

    /// A failed listing must not latch indexing off for the store's lifetime.
    /// Root replaced by a regular file makes the listing fail with ENOTDIR.
    @Test func aFailedListingDoesNotLatchIndexingOff() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root,
                    diskLimitBytes: 1 << 20))
            try await writer.save(meta, snapshot: snapshot(for: meta))
        }
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root,
                diskLimitBytes: 1 << 20))
        let displaced = root.appendingPathExtension("displaced")
        try FileManager.default.moveItem(at: root, to: displaced)
        try Data().write(to: root)
        #expect(!(await store.contains(meta.key)))
        try FileManager.default.removeItem(at: root)
        try FileManager.default.moveItem(at: displaced, to: root)
        await store.removeAll()
        let directory = root.appendingPathComponent(meta.key.contentHash)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(await store.diskBytes == 0)
    }

    /// A directory that refuses removal still occupies disk, so `remove` must
    /// keep it indexed and charged until a later unlink succeeds.
    @Test func aFailedRemovalKeepsTheRecordCharged() async throws {
        let root = try temporaryDirectory()
        let meta = metadata(tokens: [1, 2, 3])
        let directory = root.appendingPathComponent(meta.key.contentHash)
        defer {
            try? FileManager.default.setAttributes(
                [.immutable: false], ofItemAtPath: directory.path)
            try? FileManager.default.removeItem(at: root)
        }
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root,
                    diskLimitBytes: 1 << 20))
            try await writer.save(meta, snapshot: snapshot(for: meta))
        }
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root,
                diskLimitBytes: 1 << 20))
        _ = try await store.load(meta.key, domain: domain)
        try FileManager.default.setAttributes(
            [.immutable: true], ofItemAtPath: directory.path)
        await store.remove(meta.key)
        // Charged before `contains`, which re-indexes a record it finds on
        // disk and would silently restore the accounting under test.
        #expect(await store.diskBytes == 16)
        #expect(FileManager.default.fileExists(atPath: directory.path))
        #expect(await store.contains(meta.key))
        try FileManager.default.setAttributes(
            [.immutable: false], ofItemAtPath: directory.path)
        await store.remove(meta.key)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(await store.diskBytes == 0)
    }

    /// Eviction that cannot unlink the oldest record must not subtract its
    /// bytes: the budget stays truthful and the record stays listed.
    @Test func aFailedEvictionKeepsTheRecordCharged() async throws {
        let root = try temporaryDirectory()
        let oldest = metadata(tokens: [1, 2, 3])
        let newest = metadata(tokens: [4, 5, 6])
        let oldestDirectory = root.appendingPathComponent(oldest.key.contentHash)
        defer {
            try? FileManager.default.setAttributes(
                [.immutable: false], ofItemAtPath: oldestDirectory.path)
            try? FileManager.default.removeItem(at: root)
        }
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root,
                    diskLimitBytes: 1 << 20))
            try await writer.save(oldest, snapshot: snapshot(for: oldest))
            try await writer.save(newest, snapshot: snapshot(for: newest))
        }
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 32))
        try FileManager.default.setAttributes(
            [.immutable: true], ofItemAtPath: oldestDirectory.path)
        let third = metadata(tokens: [7, 8, 9])
        try await store.save(third, snapshot: snapshot(for: third))
        #expect(await store.diskBytes == 48)
        let listed = await store.metadata(for: domain)
        #expect(listed.count == 3)
        #expect(FileManager.default.fileExists(atPath: oldestDirectory.path))
        try FileManager.default.setAttributes(
            [.immutable: false], ofItemAtPath: oldestDirectory.path)
        await store.evictDiskIfNeeded()
        #expect(!FileManager.default.fileExists(atPath: oldestDirectory.path))
        #expect(await store.diskBytes == 32)
    }

    /// `removeAll` cannot free a directory it fails to unlink; it has to leave
    /// that record indexed rather than reset the accounting to zero.
    @Test func aFailedRemoveAllKeepsTheRecordItCouldNotDelete() async throws {
        let root = try temporaryDirectory()
        let meta = metadata(tokens: [1, 2, 3])
        let other = metadata(tokens: [4, 5, 6])
        let directory = root.appendingPathComponent(meta.key.contentHash)
        defer {
            try? FileManager.default.setAttributes(
                [.immutable: false], ofItemAtPath: directory.path)
            try? FileManager.default.removeItem(at: root)
        }
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root,
                    diskLimitBytes: 1 << 20))
            try await writer.save(meta, snapshot: snapshot(for: meta))
            try await writer.save(other, snapshot: snapshot(for: other))
        }
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root,
                diskLimitBytes: 1 << 20))
        try FileManager.default.setAttributes(
            [.immutable: true], ofItemAtPath: directory.path)
        await store.removeAll()
        #expect(await store.contains(meta.key))
        #expect(!(await store.contains(other.key)))
        #expect(await store.diskBytes == 16)
        try FileManager.default.setAttributes(
            [.immutable: false], ofItemAtPath: directory.path)
        await store.removeAll()
        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(await store.diskBytes == 0)
    }

    /// AUD2-001/002: a canonical directory with only a payload and no metadata is
    /// crash debris, not a record. Startup reclaims it instead of leaving bytes
    /// uncounted forever.
    @Test func payloadOnlyDirectoryIsReclaimedAtStartup() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        let directory = root.appendingPathComponent(meta.key.contentHash, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 16).write(
            to: directory.appendingPathComponent("state.bin"))
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20))
        await store.indexDiskIfNeeded()
        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(await store.diskBytes == 0)
    }

    /// AUD2-002: corrupt metadata used to be skipped by indexing and only found
    /// by an exact-key lookup, so it stayed uncounted until someone asked for it.
    @Test func corruptMetadataIsReclaimedAtStartup() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        let directory = root.appendingPathComponent(meta.key.contentHash, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data("not json".utf8).write(
            to: directory.appendingPathComponent("metadata.json"))
        try Data(repeating: 1, count: 16).write(
            to: directory.appendingPathComponent("state.bin"))
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20))
        await store.indexDiskIfNeeded()
        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(await store.diskBytes == 0)
    }

    /// AUD2-001: a directory is a record only under the hash its metadata names.
    /// A renamed or planted directory must not be indexed as a different key.
    @Test func aDirectoryNamedForTheWrongHashIsReclaimedAtStartup() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20))
            try await writer.save(meta, snapshot: snapshot(for: meta))
        }
        let wrong = String(repeating: "a", count: 64)
        #expect(wrong != meta.key.contentHash)
        try FileManager.default.moveItem(
            at: root.appendingPathComponent(meta.key.contentHash),
            to: root.appendingPathComponent(wrong))
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20))
        await store.indexDiskIfNeeded()
        #expect(
            !FileManager.default.fileExists(atPath: root.appendingPathComponent(wrong).path))
        #expect(await store.diskBytes == 0)
    }

    /// AUD2-001/002: a payload shorter than its descriptor declares is an
    /// incomplete record. Indexing used to accept the metadata alone, so the
    /// directory was counted until a later load failed.
    @Test func truncatedPayloadIsReclaimedAtStartup() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20))
            try await writer.save(meta, snapshot: snapshot(for: meta))
        }
        let directory = root.appendingPathComponent(meta.key.contentHash)
        try Data(repeating: 1, count: 8).write(
            to: directory.appendingPathComponent("state.bin"))
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20))
        await store.indexDiskIfNeeded()
        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(await store.diskBytes == 0)
    }

    /// AUD2-001: the publish swap renames the canonical directory to `.bak-` and
    /// then the staging directory into place. A crash between those renames
    /// leaves only the backup; startup restores it instead of deleting the last
    /// good record.
    @Test func aValidBackupIsRestoredWhenTheCanonicalRecordIsGone() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20))
            try await writer.save(meta, snapshot: snapshot(for: meta))
        }
        let canonical = root.appendingPathComponent(meta.key.contentHash)
        let backup = root.appendingPathComponent(
            ".bak-\(meta.key.contentHash)-\(UUID().uuidString)")
        try FileManager.default.moveItem(at: canonical, to: backup)
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20))
        await store.indexDiskIfNeeded()
        #expect(FileManager.default.fileExists(atPath: canonical.path))
        #expect(!FileManager.default.fileExists(atPath: backup.path))
        let loaded = try await store.load(meta.key, domain: domain)
        #expect(loaded.snapshot.payload == snapshot(for: meta).payload)
    }

    /// The other half: a backup left behind after both renames completed is
    /// stale, and must not be indexed as a second copy of the record.
    @Test func aStaleBackupIsReclaimedWhenTheCanonicalRecordExists() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meta = metadata(tokens: [1, 2, 3])
        do {
            let writer = try PromptCheckpointStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20))
            try await writer.save(meta, snapshot: snapshot(for: meta))
        }
        let canonical = root.appendingPathComponent(meta.key.contentHash)
        let backup = root.appendingPathComponent(
            ".bak-\(meta.key.contentHash)-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: canonical, to: backup)
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20))
        await store.indexDiskIfNeeded()
        #expect(!FileManager.default.fileExists(atPath: backup.path))
        #expect(FileManager.default.fileExists(atPath: canonical.path))
        #expect(await store.diskBytes == 16)
    }

    /// Regression for GitHub #35: an externally deleted checkpoint directory
    /// must not wedge LRU eviction or inflate `diskBytes`.
    @Test func missingDiskCheckpointDoesNotWedgeEviction() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 24))
        let first = metadata(tokens: [1, 2], createdAt: Date(timeIntervalSince1970: 1))
        let second = metadata(tokens: [3, 4], createdAt: Date(timeIntervalSince1970: 2))
        try await store.save(first, snapshot: snapshot(for: first))
        try FileManager.default.removeItem(
            at: root.appendingPathComponent(first.key.contentHash))
        try await store.save(second, snapshot: snapshot(for: second))
        #expect(await store.diskBytes == 16)
    }
}
