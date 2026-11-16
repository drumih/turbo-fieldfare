import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLib
@testable import TinyTitanCore

/// K1: the system-end checkpoint is content-addressed by the rendered system
/// prefix and the runtime domain, validated before a restore, and accounted
/// for by the same budget/LRU/disk rules as every other interior checkpoint.
@Suite struct SharedSystemPrefixTests {
    private let imStart: Int32 = 9

    private let domain = ServerPromptCacheDomain(
        modelID: "qwen-toy",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 4096,
        kvStorage: "int8",
        fp16RingEnabled: false,
        templateSHA256: "template")

    private func message(_ role: GFTokenizer.Role, _ content: String)
        -> GFTokenizer.Message
    {
        GFTokenizer.Message(role: role, content: content)
    }

    /// A rendered-looking ChatML prompt: one `<|im_start|>` per message plus
    /// the generation prompt's trailing assistant header.
    private func rendered(_ messages: [[Int32]]) -> [Int32] {
        messages.reduce(into: []) { $0 += [imStart] + $1 } + [imStart]
    }

    private func prefix(
        _ promptIDs: [Int32],
        _ messages: [GFTokenizer.Message],
        domain: ServerPromptCacheDomain? = nil
    ) -> PromptCheckpointSharedSystem.Prefix? {
        PromptCheckpointSharedSystem.prefix(
            promptIDs: promptIDs,
            messages: messages,
            imStartID: imStart,
            domain: domain ?? self.domain)
    }

    private func descriptor(
        position: Int, payloadBytes: Int = 16
    ) -> InferenceStateSnapshotDescriptor {
        InferenceStateSnapshotDescriptor(
            position: position,
            kvSegmentLengths: [payloadBytes],
            gdnSegmentLengths: [],
            payloadBytes: payloadBytes)
    }

    private func snapshot(
        position: Int, byte: UInt8 = 3, payloadBytes: Int = 16
    ) -> InferenceStateSnapshot {
        let descriptor = descriptor(
            position: position, payloadBytes: payloadBytes)
        return InferenceStateSnapshot(
            descriptor: descriptor,
            payload: Data(repeating: byte, count: descriptor.payloadBytes))
    }

    private func metadata(
        for prefix: PromptCheckpointSharedSystem.Prefix,
        kind: PromptCheckpointKind = .sharedSystem,
        domain: ServerPromptCacheDomain? = nil,
        position: Int? = nil,
        prefixTokenIDs: [Int32]? = nil
    ) -> PromptCheckpointMetadata {
        let resolvedDomain = domain ?? self.domain
        let resolvedPosition = position ?? prefix.boundary.position
        return PromptCheckpointMetadata(
            key: prefix.key,
            domain: resolvedDomain,
            kind: kind,
            conversationKey: "conversation",
            prefixTokenIDs: prefixTokenIDs ?? prefix.prefixTokenIDs,
            descriptor: descriptor(position: resolvedPosition))
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("shared-system-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    // MARK: - Locating the system end

    @Test func systemEndIsThePlannerBoundary() {
        let promptIDs = rendered([[1, 2], [3, 4], [5], [6]])
        let messages = [
            message(.system, "system"),
            message(.user, "first"),
            message(.assistant, "answer"),
            message(.user, "second"),
        ]
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: promptIDs, messages: messages, imStartID: imStart)
        #expect(
            boundaries.contains(
                PromptCheckpointBoundary(kind: .systemEnd, position: 3)))
        #expect(
            PromptCheckpointPlanner.systemEndPosition(
                promptIDs: promptIDs, messages: messages, imStartID: imStart) == 3)
    }

    @Test func aSingleUserTurnSharesTheSystemEndPosition() {
        // system | user | generation prompt. The planner labels the shared
        // position `.lastTurnStart`; the position itself is still the system
        // end, which is what the shared cache keys on.
        let promptIDs = rendered([[1, 2], [3, 4]])
        let messages = [
            message(.system, "system"),
            message(.user, "only"),
        ]
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: promptIDs, messages: messages, imStartID: imStart)
        #expect(
            boundaries
                == [PromptCheckpointBoundary(kind: .lastTurnStart, position: 3)])
        #expect(
            PromptCheckpointPlanner.systemEndPosition(
                promptIDs: promptIDs, messages: messages, imStartID: imStart) == 3)
    }

    @Test func noGuidanceOrNoSecondMessageHasNoSystemEnd() {
        let noGuidance = rendered([[1, 2], [3, 4]])
        #expect(
            PromptCheckpointPlanner.systemEndPosition(
                promptIDs: noGuidance,
                messages: [message(.user, "a"), message(.user, "b")],
                imStartID: imStart) == nil)

        // A system message with no following turn: only one message start.
        let single = rendered([[1, 2]])
        #expect(
            PromptCheckpointPlanner.systemEndPosition(
                promptIDs: single,
                messages: [message(.system, "system")],
                imStartID: imStart) == nil)
    }

    // MARK: - The key

    @Test func identicalSystemPromptsShareAKeyAcrossUserTurns() throws {
        let first = try #require(
            prefix(
                rendered([[1, 2], [10, 11]]),
                [message(.system, "system"), message(.user, "first")]))
        let second = try #require(
            prefix(
                rendered([[1, 2], [20, 21]]),
                [message(.system, "system"), message(.user, "second")]))
        #expect(first.key == second.key)
        #expect(first.boundary.position == second.boundary.position)
        #expect(first.prefixTokenIDs == [imStart, 1, 2])
        // The user turn is outside the shared prefix.
        #expect(!first.prefixTokenIDs.contains(10))
    }

    @Test func aDifferentSystemPromptOrDomainMisses() throws {
        let base = try #require(
            prefix(
                rendered([[1, 2], [10, 11]]),
                [message(.system, "system"), message(.user, "first")]))
        let otherSystem = try #require(
            prefix(
                rendered([[7, 8], [10, 11]]),
                [message(.system, "other"), message(.user, "first")]))
        #expect(base.key != otherSystem.key)

        let otherDomain = ServerPromptCacheDomain(
            modelID: domain.modelID,
            sourceSnapshotHash: domain.sourceSnapshotHash,
            runtimeProfileHash: domain.runtimeProfileHash,
            maximumContext: domain.maximumContext + 1,
            kvStorage: domain.kvStorage,
            fp16RingEnabled: domain.fp16RingEnabled,
            templateSHA256: domain.templateSHA256)
        let samePromptOtherDomain = try #require(
            prefix(
                rendered([[1, 2], [10, 11]]),
                [message(.system, "system"), message(.user, "first")],
                domain: otherDomain))
        #expect(base.key != samePromptOtherDomain.key)

        let otherModel = ServerPromptCacheDomain(
            modelID: "other-model",
            sourceSnapshotHash: domain.sourceSnapshotHash,
            runtimeProfileHash: domain.runtimeProfileHash,
            maximumContext: domain.maximumContext,
            kvStorage: domain.kvStorage,
            fp16RingEnabled: domain.fp16RingEnabled,
            templateSHA256: domain.templateSHA256)
        let sameSystemOtherModel = try #require(
            prefix(
                rendered([[1, 2], [10, 11]]),
                [message(.system, "system"), message(.user, "first")],
                domain: otherModel))
        #expect(base.key != sameSystemOtherModel.key)
    }

    // MARK: - Validation

    @Test func aValidRecordPassesAndEveryMismatchIsRefused() throws {
        let prefix = try #require(
            prefix(
                rendered([[1, 2], [10, 11]]),
                [message(.system, "system"), message(.user, "first")]))
        let captured = snapshot(position: prefix.boundary.position)
        let valid = metadata(for: prefix)
        try PromptCheckpointSharedSystem.validate(
            metadata: valid,
            snapshot: captured,
            auxiliary: nil,
            against: prefix,
            domain: domain)

        // Different domain.
        let otherDomain = ServerPromptCacheDomain(
            modelID: "other",
            sourceSnapshotHash: "snapshot",
            runtimeProfileHash: "profile",
            maximumContext: 4096,
            kvStorage: "int8",
            fp16RingEnabled: false,
            templateSHA256: "template")
        #expect(throws: PromptCheckpointStoreError.self) {
            try PromptCheckpointSharedSystem.validate(
                metadata: metadata(for: prefix, domain: otherDomain),
                snapshot: captured,
                auxiliary: nil,
                against: prefix,
                domain: self.domain)
        }

        // Non-shareable kind, e.g. a grid point.
        #expect(throws: PromptCheckpointStoreError.self) {
            try PromptCheckpointSharedSystem.validate(
                metadata: metadata(for: prefix, kind: .grid),
                snapshot: captured,
                auxiliary: nil,
                against: prefix,
                domain: self.domain)
        }

        // Position does not match the request's boundary.
        #expect(throws: PromptCheckpointStoreError.self) {
            try PromptCheckpointSharedSystem.validate(
                metadata: metadata(for: prefix, position: 1),
                snapshot: captured,
                auxiliary: nil,
                against: prefix,
                domain: self.domain)
        }

        // Prefix does not match the rendered system tokens.
        #expect(throws: PromptCheckpointStoreError.self) {
            try PromptCheckpointSharedSystem.validate(
                metadata: metadata(
                    for: prefix, prefixTokenIDs: [imStart, 9, 9]),
                snapshot: captured,
                auxiliary: nil,
                against: prefix,
                domain: self.domain)
        }

        // The snapshot is not the boundary the metadata claims.
        #expect(throws: PromptCheckpointStoreError.self) {
            try PromptCheckpointSharedSystem.validate(
                metadata: valid,
                snapshot: snapshot(position: 1),
                auxiliary: nil,
                against: prefix,
                domain: self.domain)
        }

        // Auxiliary state for a different position.
        #expect(throws: PromptCheckpointStoreError.self) {
            try PromptCheckpointSharedSystem.validate(
                metadata: valid,
                snapshot: captured,
                auxiliary: InferenceAuxiliaryState(
                    position: 1, payload: Data([1, 2, 3])),
                against: prefix,
                domain: self.domain)
        }
    }

    @Test func legacyKindsAreAcceptedOnlyAtTheSystemBoundary() throws {
        let promptIDs = rendered([[1, 2], [3, 4]])
        let messages = [
            message(.system, "system"), message(.user, "only"),
        ]
        let prefix = try #require(prefix(promptIDs, messages))
        let captured = snapshot(position: prefix.boundary.position)
        // Pre-K1 records spell this position `.systemEnd` or, with a single
        // user turn, `.lastTurnStart`. Both describe the same system prefix.
        for kind in [PromptCheckpointKind.systemEnd, .lastTurnStart] {
            try PromptCheckpointSharedSystem.validate(
                metadata: metadata(for: prefix, kind: kind),
                snapshot: captured,
                auxiliary: nil,
                against: prefix,
                domain: domain)
        }
    }

    // MARK: - Budget, LRU and disk

    @Test func aSharedEntryIsChargedAndEvictedByTheMemoryLru() async throws {
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 16, diskDirectory: nil, diskLimitBytes: 0))
        let firstPrefix = try #require(
            prefix(
                rendered([[1, 2], [10]]),
                [message(.system, "first"), message(.user, "u")]))
        let secondPrefix = try #require(
            prefix(
                rendered([[4, 5], [11]]),
                [message(.system, "second"), message(.user, "u")]))
        for prefix in [firstPrefix, secondPrefix] {
            let captured = snapshot(position: prefix.boundary.position)
            let metadata = try PromptCheckpointSharedSystem.claimedMetadata(
                prefix: prefix,
                domain: domain,
                conversationKey: "conversation",
                snapshot: captured)
            try await store.save(metadata, snapshot: captured)
        }
        let memoryBytes = await store.memoryBytes
        let firstPresent = await store.contains(firstPrefix.key)
        let secondPresent = await store.contains(secondPrefix.key)
        #expect(memoryBytes == 16)
        #expect(!firstPresent)
        #expect(secondPresent)
    }

    @Test func aSharedEntryIsEvictedByTheDiskLru() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 16))
        let prefixes = try [1, 4].map { token in
            try #require(
                prefix(
                    rendered([[Int32(token), 2], [10]]),
                    [message(.system, "s\(token)"), message(.user, "u")]))
        }
        for prefix in prefixes {
            let captured = snapshot(position: prefix.boundary.position)
            let metadata = try PromptCheckpointSharedSystem.claimedMetadata(
                prefix: prefix,
                domain: domain,
                conversationKey: "conversation",
                snapshot: captured)
            try await store.save(metadata, snapshot: captured)
        }
        let diskBytes = await store.diskBytes
        let firstPresent = await store.contains(prefixes[0].key)
        let secondPresent = await store.contains(prefixes[1].key)
        #expect(diskBytes == 16)
        #expect(!firstPresent)
        #expect(secondPresent)
    }

    @Test func theDiskTierSurvivesANewStoreWithAuxiliaryState() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let prefix = try #require(
            prefix(
                rendered([[1, 2], [10]]),
                [message(.system, "system"), message(.user, "u")]))
        let captured = snapshot(position: prefix.boundary.position)
        let auxiliary = InferenceAuxiliaryState(
            position: prefix.boundary.position,
            payload: Data([9, 8, 7, 6, 5]))
        let metadata = try PromptCheckpointSharedSystem.claimedMetadata(
            prefix: prefix,
            domain: domain,
            conversationKey: "conversation",
            snapshot: captured)
        do {
            // One process writes...
            let writer = try PromptCheckpointStore(
                configuration: configuration)
            try await writer.save(
                metadata, snapshot: captured, auxiliary: auxiliary)
        }
        // ...a new one reads the directory with no in-memory state.
        let reader = try PromptCheckpointStore(configuration: configuration)
        let loaded = try await reader.load(prefix.key, domain: domain)
        #expect(loaded.tier == "ssd")
        #expect(loaded.metadata == metadata)
        #expect(loaded.metadata.kind == .sharedSystem)
        #expect(loaded.metadata.position == prefix.boundary.position)
        #expect(loaded.auxiliary == auxiliary)
    }

    @Test func aDurableSharedEntryIsOnDiskEvenWhenItFitsRAM() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 1 << 20, diskDirectory: root,
            diskLimitBytes: 1 << 20)
        let prefix = try #require(
            prefix(
                rendered([[1, 2], [10]]),
                [message(.system, "system"), message(.user, "u")]))
        let captured = snapshot(position: prefix.boundary.position)
        let metadata = try PromptCheckpointSharedSystem.claimedMetadata(
            prefix: prefix,
            domain: domain,
            conversationKey: "conversation",
            snapshot: captured)
        try await PromptCheckpointStore(
            configuration: configuration
        ).save(metadata, snapshot: captured, durable: true)

        // A new process sees the disk record even though the payload fit RAM
        // in the writer, which is the cross-restart guarantee K1 needs.
        let restarted = try PromptCheckpointStore(
            configuration: configuration)
        let loaded = try await restarted.load(prefix.key, domain: domain)
        #expect(loaded.tier == "ssd")
        #expect(loaded.metadata == metadata)
    }

    @Test func aNonDurableEntryThatFitsRAMStaysOffDisk() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 1 << 20, diskDirectory: root,
            diskLimitBytes: 1 << 20)
        let prefix = try #require(
            prefix(
                rendered([[1, 2], [10]]),
                [message(.system, "system"), message(.user, "u")]))
        let captured = snapshot(position: prefix.boundary.position)
        let metadata = try PromptCheckpointSharedSystem.claimedMetadata(
            prefix: prefix,
            domain: domain,
            conversationKey: "conversation",
            snapshot: captured)
        try await PromptCheckpointStore(configuration: configuration)
            .save(metadata, snapshot: captured)

        // Existing behaviour for turn-local checkpoints is preserved: the RAM
        // copy is the only copy, so a new process does not find it.
        let restarted = try PromptCheckpointStore(
            configuration: configuration)
        let present = await restarted.contains(prefix.key)
        #expect(!present)
    }

    @Test func aMissingAuxiliaryPayloadIsNotInvented() async throws {
        let store = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 1 << 20, diskDirectory: nil,
                diskLimitBytes: 0))
        let prefix = try #require(
            prefix(
                rendered([[1, 2], [10]]),
                [message(.system, "system"), message(.user, "u")]))
        let captured = snapshot(position: prefix.boundary.position)
        let metadata = try PromptCheckpointSharedSystem.claimedMetadata(
            prefix: prefix,
            domain: domain,
            conversationKey: "conversation",
            snapshot: captured)
        try await store.save(metadata, snapshot: captured)
        let loaded = try await store.load(prefix.key, domain: domain)
        #expect(loaded.auxiliary == nil)
    }

    @Test func aCorruptStoredPositionIsRefusedAndRemoved() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 0, diskDirectory: root, diskLimitBytes: 1 << 20)
        let prefix = try #require(
            prefix(
                rendered([[1, 2], [10]]),
                [message(.system, "system"), message(.user, "u")]))
        let captured = snapshot(position: prefix.boundary.position)
        let metadata = try PromptCheckpointSharedSystem.claimedMetadata(
            prefix: prefix,
            domain: domain,
            conversationKey: "conversation",
            snapshot: captured)
        let writer = try PromptCheckpointStore(configuration: configuration)
        try await writer.save(metadata, snapshot: captured)

        // Move the descriptor's position off the prefix length. `load` must
        // refuse the record, remove it, and never hand it to a restore.
        let metadataURL =
            root
            .appendingPathComponent(prefix.key.contentHash, isDirectory: true)
            .appendingPathComponent(PromptCheckpointStore.metadataName)
        var json = try #require(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: metadataURL))
                as? [String: Any])
        var record = try #require(json["metadata"] as? [String: Any])
        var descriptor = try #require(
            record["descriptor"] as? [String: Any])
        descriptor["position"] = 1
        record["descriptor"] = descriptor
        json["metadata"] = record
        try JSONSerialization.data(withJSONObject: json)
            .write(to: metadataURL)

        let reader = try PromptCheckpointStore(configuration: configuration)
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await reader.load(prefix.key, domain: self.domain)
        }
        #expect(
            !FileManager.default.fileExists(
                atPath: root.appendingPathComponent(
                    prefix.key.contentHash
                ).path))
        let present = await reader.contains(prefix.key)
        #expect(!present)
    }

    // MARK: - The process-wide store and the log line

    @Test func sharedInstancesAreOneStorePerConfiguration() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let configuration = ServerPromptCacheStorageConfiguration(
            memoryLimitBytes: 1 << 20, diskDirectory: root,
            diskLimitBytes: 1 << 20)
        let first = try PromptCheckpointStore.sharedInstance(
            configuration: configuration)
        let second = try PromptCheckpointStore.sharedInstance(
            configuration: configuration)
        #expect(first === second)

        let other = try PromptCheckpointStore.sharedInstance(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 1 << 20, diskDirectory: root,
                diskLimitBytes: 2 << 20))
        #expect(first !== other)
    }

    @Test func theHitLineNamesTheTierAndTheReusedAndSavedTokens() {
        let line = PromptCheckpointSharedSystem.hitLine(
            tier: "ssd", prefixTokens: 64, promptTokens: 100)
        #expect(
            line
                == "TinyTitan prompt_checkpoint shared_system_hit "
                + "tier=ssd prefix_tokens=64 reused_tokens=64 suffix_tokens=36")
        // Logged through `ServerLog`, whose sink takes whole lines; no stdout
        // write happens here.
        #expect(line.hasPrefix("TinyTitan prompt_checkpoint shared_system_hit "))
        #expect(!line.contains("suffix_tokens=0"))
    }
}
