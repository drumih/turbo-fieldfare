import CryptoKit
import Foundation
import TinyTitanCore

/// Content-addressed store for interior checkpoints.
///
/// The turn-end prompt cache keeps one snapshot per completed assistant turn.
/// This store holds the points *inside* a prompt (system end, last-turn start)
/// that survive an edit to the last turn. Payloads are the same
/// `InferenceStateSnapshot` the turn cache already captures; the metadata adds
/// the exact prefix token IDs and a content hash, so a restore can never use
/// a snapshot for a different prefix.
///
/// The store is an actor: capture, restore and eviction are serialized, and
/// disk writes happen inside it so two saves cannot interleave. The same
/// memory/disk accounting as `ServerPromptStateStore` applies: a payload goes
/// to memory when it fits the memory budget, otherwise to disk, otherwise it
/// is refused rather than silently dropped.
actor PromptCheckpointStore {
    /// A single captured snapshot is capped, like the turn store, so a huge
    /// context cannot allocate an unbounded payload regardless of the budget.
    static let maximumCaptureBytes = 4 * 1_024 * 1_048_576
    /// QSA pooled blocks plus the PLE window. At 8K this is tens of MiB; the
    /// cap keeps a pathological context from turning an auxiliary file into a
    /// second unbounded snapshot.
    static let maximumAuxiliaryBytes = 1_024 * 1_048_576

    struct DiskMetadata: Codable {
        static let currentVersion = 1

        let version: Int
        let metadata: PromptCheckpointMetadata
        let payloadSHA256: String
        // Additive in v1: an old record has no auxiliary state, a new Qwen3.8
        // record carries the QSA/PLE payload and its hash.
        let auxiliaryBytes: Int?
        let auxiliarySHA256: String?

        func validate() throws {
            try metadata.validate()
            let payloadBytes = metadata.descriptor.payloadBytes
            guard payloadBytes > 0, payloadBytes <= PromptCheckpointStore.maximumCaptureBytes else {
                throw PromptCheckpointStoreError.invalidMetadata(
                    "payload size \(payloadBytes) is out of range")
            }
            let auxiliaryBytes = auxiliaryBytes ?? 0
            guard auxiliaryBytes >= 0,
                auxiliaryBytes <= PromptCheckpointStore.maximumAuxiliaryBytes
            else {
                throw PromptCheckpointStoreError.invalidMetadata(
                    "auxiliary size \(auxiliaryBytes) is out of range")
            }
        }
    }

    struct DiskRecord {
        let metadata: PromptCheckpointMetadata
        let directory: URL
        let payload: URL
        let auxiliaryBytes: Int
        var modificationDate: Date
    }

    struct MemoryRecord {
        let metadata: PromptCheckpointMetadata
        let snapshot: InferenceStateSnapshot
        let auxiliary: InferenceAuxiliaryState?
    }

    struct State {
        var memory: [PromptCheckpointKey: MemoryRecord] = [:]
        var memoryLRU: [PromptCheckpointKey] = []
        var memoryBytes = 0
        var disk: [PromptCheckpointKey: DiskRecord] = [:]
        var diskLRU: [PromptCheckpointKey] = []
        var diskBytes = 0
    }

    static let metadataName = "metadata.json"
    static let payloadName = "state.bin"
    static let auxiliaryName = "auxiliary.bin"

    let configuration: ServerPromptCacheStorageConfiguration
    let fileManager: FileManager
    var state = State()
    var diskIndexed = false

    init(
        configuration: ServerPromptCacheStorageConfiguration,
        fileManager: FileManager = .default
    ) throws {
        self.configuration = configuration
        self.fileManager = fileManager
        if let root = configuration.diskDirectory {
            try fileManager.createDirectory(
                at: root, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            try fileManager.setAttributes(
                [.posixPermissions: 0o700], ofItemAtPath: root.path)
        }
    }

    /// Saves one checkpoint. The metadata is validated against the payload
    /// before anything is written.
    ///
    /// `durable` asks for a disk copy as well as a RAM copy. Without it the
    /// store keeps a payload in whichever tier holds it -- RAM when it fits --
    /// which is right for a turn-local checkpoint and wrong for the shared
    /// system prefix: a small prefix would never reach disk, and a restart
    /// would lose exactly the entry K1 exists to keep. The turn store has
    /// always written both tiers; this is the checkpoint store's equivalent.
    func save(
        _ metadata: PromptCheckpointMetadata,
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState? = nil,
        durable: Bool = false
    ) throws {
        try metadata.validate()
        guard snapshot.descriptor == metadata.descriptor else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "snapshot descriptor does not match the metadata")
        }
        guard snapshot.payload.count == metadata.descriptor.payloadBytes else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "payload is \(snapshot.payload.count) bytes, descriptor says "
                    + "\(metadata.descriptor.payloadBytes)")
        }
        let bytes = snapshot.payload.count
        guard bytes <= Self.maximumCaptureBytes else {
            throw PromptCheckpointStoreError.exceedsLimit(
                bytes: bytes, limit: Self.maximumCaptureBytes)
        }
        let auxiliaryBytes = auxiliary?.payload.count ?? 0
        guard auxiliaryBytes <= Self.maximumAuxiliaryBytes else {
            throw PromptCheckpointStoreError.exceedsLimit(
                bytes: auxiliaryBytes, limit: Self.maximumAuxiliaryBytes)
        }
        let total = bytes + auxiliaryBytes
        let fitsMemory = total <= configuration.memoryLimitBytes
        let fitsDisk =
            configuration.diskDirectory != nil
            && total <= configuration.diskLimitBytes
        if fitsDisk, durable || !fitsMemory {
            try writeDisk(
                metadata: metadata, snapshot: snapshot, auxiliary: auxiliary)
        } else if !fitsMemory {
            // No tier can hold it: RAM is too small and there is no disk (or
            // the payload is larger than the whole disk budget). Refuse rather
            // than accept an entry nothing backs.
            throw PromptCheckpointStoreError.exceedsLimit(
                bytes: total,
                limit: configuration.diskDirectory == nil
                    ? configuration.memoryLimitBytes
                    : configuration.diskLimitBytes)
        }
        if fitsMemory {
            insertMemory(
                metadata: metadata, snapshot: snapshot, auxiliary: auxiliary)
        }
    }

    /// Loads a checkpoint and verifies it against `domain` and its content
    /// hash. A corrupt record is deleted and reported, never returned.
    ///
    /// `tier` is the tier this call actually read from, captured before a
    /// disk record is promoted into memory: the shared-system log line has to
    /// name where the bytes came from, not where a previous lookup left them.
    func load(
        _ key: PromptCheckpointKey,
        domain: ServerPromptCacheDomain
    ) throws -> (
        metadata: PromptCheckpointMetadata,
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?,
        tier: String
    ) {
        if let record = state.memory[key] {
            guard record.metadata.domain == domain else {
                throw PromptCheckpointStoreError.domainMismatch
            }
            try Self.verifyKey(record.metadata)
            touchMemory(key)
            return (record.metadata, record.snapshot, record.auxiliary, "ram")
        }
        if state.disk[key] == nil {
            // A corrupt record is deleted and reported (see load's catch),
            // never surfaced as `missing`: `missing` means the space is free,
            // and an uncounted directory left behind is neither evictable
            // nor reclaimed.
            if let restored = try diskRecordFromFileSystem(for: key) {
                state.disk[key] = restored
                state.diskLRU.append(key)
                state.diskBytes +=
                    restored.metadata.descriptor.payloadBytes
                    + restored.auxiliaryBytes
            }
        }
        guard let diskRecord = state.disk[key] else {
            throw PromptCheckpointStoreError.missing(key)
        }
        guard diskRecord.metadata.domain == domain else {
            throw PromptCheckpointStoreError.domainMismatch
        }
        do {
            let metadataData = try BoundedMetadataRead.read(
                fileAt: diskRecord.directory
                    .appendingPathComponent(Self.metadataName),
                maxBytes: 16 * 1_048_576)
            let diskMetadata = try JSONDecoder().decode(
                DiskMetadata.self, from: metadataData)
            guard diskMetadata.version == DiskMetadata.currentVersion else {
                throw PromptCheckpointStoreError.corrupt(
                    key, "unsupported metadata version \(diskMetadata.version)")
            }
            guard diskMetadata.metadata.key == key else {
                throw PromptCheckpointStoreError.keyMismatch
            }
            try diskMetadata.validate()
            // Verify the file's hash while streaming it, before any payload-sized
            // allocation. `readPayload` used to read the declared size first and
            // hash the Data afterwards, so a corrupt or swapped multi-GiB record
            // was materialized before it was refused (AUD2-004).
            guard
                try Sha256Verifier.hashFile(at: diskRecord.payload)
                    == diskMetadata.payloadSHA256
            else {
                throw PromptCheckpointStoreError.corrupt(
                    key, "payload hash does not match")
            }
            let payload = try readPayload(
                at: diskRecord.payload,
                expectedBytes: diskMetadata.metadata.descriptor.payloadBytes)
            let auxiliary: InferenceAuxiliaryState?
            if let auxiliaryBytes = diskMetadata.auxiliaryBytes,
                let auxiliarySHA256 = diskMetadata.auxiliarySHA256
            {
                let auxiliaryURL = diskRecord.directory.appendingPathComponent(
                    Self.auxiliaryName)
                guard try Sha256Verifier.hashFile(at: auxiliaryURL) == auxiliarySHA256
                else {
                    throw PromptCheckpointStoreError.corrupt(
                        key, "auxiliary payload hash does not match")
                }
                let auxiliaryData = try readPayload(
                    at: auxiliaryURL, expectedBytes: auxiliaryBytes)
                auxiliary = InferenceAuxiliaryState(
                    position: diskMetadata.metadata.position,
                    payload: auxiliaryData)
            } else {
                auxiliary = nil
            }
            touchDisk(key)
            let snapshot = InferenceStateSnapshot(
                descriptor: diskMetadata.metadata.descriptor, payload: payload)
            if payload.count + (auxiliary?.payload.count ?? 0)
                <= configuration.memoryLimitBytes
            {
                insertMemory(
                    metadata: diskMetadata.metadata, snapshot: snapshot,
                    auxiliary: auxiliary)
            }
            return (diskMetadata.metadata, snapshot, auxiliary, "ssd")
        } catch let error as PromptCheckpointStoreError {
            remove(key)
            throw error
        } catch {
            remove(key)
            throw PromptCheckpointStoreError.corrupt(
                key, String(describing: error))
        }
    }

    func contains(_ key: PromptCheckpointKey) -> Bool {
        indexDiskIfNeeded()
        if state.memory[key] != nil || state.disk[key] != nil { return true }
        // A checkpoint written by an earlier process is on disk but not in
        // either in-memory index until someone loads it. The capture path asks
        // first so a repeated request does not rewrite a multi-gigabyte
        // payload that is already there.
        guard configuration.diskDirectory != nil else { return false }
        return (try? diskRecordFromFileSystem(for: key)) != nil
    }

    /// The metadata for `key` without materializing its payload.
    ///
    /// The interior-checkpoint lookup asks this first, so a record captured
    /// by another conversation is skipped without reading (and promoting) a
    /// potentially large payload. Nil for a missing record, a different
    /// domain, or one that cannot be read; `load` makes the same decision
    /// again on the records this returns.
    func metadataIfPresent(
        _ key: PromptCheckpointKey,
        domain: ServerPromptCacheDomain
    ) -> PromptCheckpointMetadata? {
        if let record = state.memory[key] {
            return record.metadata.domain == domain ? record.metadata : nil
        }
        indexDiskIfNeeded()
        if let record = state.disk[key] {
            return record.metadata.domain == domain ? record.metadata : nil
        }
        guard configuration.diskDirectory != nil else { return nil }
        guard let record = try? diskRecordFromFileSystem(for: key) else {
            return nil
        }
        return record.metadata.domain == domain ? record.metadata : nil
    }

    /// Every checkpoint recorded for a domain, oldest first.
    ///
    /// A record loaded from disk also lands in memory (when it fits), so the
    /// two indexes can name the same key; the key is the identity and the
    /// memory copy is the newer one.
    func metadata(for domain: ServerPromptCacheDomain) -> [PromptCheckpointMetadata] {
        indexDiskIfNeeded()
        var byKey: [PromptCheckpointKey: PromptCheckpointMetadata] = [:]
        for record in state.disk.values {
            byKey[record.metadata.key] = record.metadata
        }
        for record in state.memory.values {
            byKey[record.metadata.key] = record.metadata
        }
        return byKey.values.filter { $0.domain == domain }
            .sorted { $0.createdAt < $1.createdAt }
    }

    var memoryBytes: Int { state.memoryBytes }
    var diskBytes: Int { state.diskBytes }

    private func diskRecordFromFileSystem(for key: PromptCheckpointKey)
        throws -> DiskRecord?
    {
        guard let root = configuration.diskDirectory else { return nil }
        let directory = root.appendingPathComponent(
            key.contentHash, isDirectory: true)
        let metadataURL = directory.appendingPathComponent(Self.metadataName)
        guard fileManager.fileExists(atPath: metadataURL.path) else {
            return nil
        }
        // A directory named for this key whose metadata cannot be read,
        // decoded or validated is corrupt, not absent: quarantine it (delete,
        // so its bytes are reclaimed) and say so. Returning nil here would
        // surface as `missing` and leave the directory uncounted forever.
        do {
            let data = try BoundedMetadataRead.read(
                fileAt: metadataURL, maxBytes: 16 * 1_048_576)
            let diskMetadata = try JSONDecoder().decode(
                DiskMetadata.self, from: data)
            guard diskMetadata.version == DiskMetadata.currentVersion,
                diskMetadata.metadata.key == key
            else {
                throw PromptCheckpointStoreError.corrupt(
                    key, "metadata does not match the record directory")
            }
            try diskMetadata.validate()
            return DiskRecord(
                metadata: diskMetadata.metadata, directory: directory,
                payload: directory.appendingPathComponent(Self.payloadName),
                auxiliaryBytes: diskMetadata.auxiliaryBytes ?? 0,
                modificationDate: Date())
        } catch {
            try? fileManager.removeItem(at: directory)
            ServerLog.diagnostic(
                "prompt checkpoints: quarantined corrupt record "
                    + "\(key.contentHash): \(error)")
            throw PromptCheckpointStoreError.corrupt(
                key, String(describing: error))
        }
    }

    private func insertMemory(
        metadata: PromptCheckpointMetadata,
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?
    ) {
        func bytes(_ record: MemoryRecord) -> Int {
            record.snapshot.payload.count + (record.auxiliary?.payload.count ?? 0)
        }
        if let existing = state.memory[metadata.key] {
            state.memoryBytes -= bytes(existing)
        }
        state.memory[metadata.key] = MemoryRecord(
            metadata: metadata, snapshot: snapshot, auxiliary: auxiliary)
        state.memoryLRU.removeAll { $0 == metadata.key }
        state.memoryLRU.append(metadata.key)
        state.memoryBytes += snapshot.payload.count + (auxiliary?.payload.count ?? 0)
        while state.memoryBytes > configuration.memoryLimitBytes,
            let oldest = state.memoryLRU.first
        {
            if let record = state.memory.removeValue(forKey: oldest) {
                state.memoryBytes -= bytes(record)
            }
            state.memoryLRU.removeFirst()
        }
    }

    private func touchMemory(_ key: PromptCheckpointKey) {
        state.memoryLRU.removeAll { $0 == key }
        state.memoryLRU.append(key)
    }

    private func touchDisk(_ key: PromptCheckpointKey) {
        state.diskLRU.removeAll { $0 == key }
        state.diskLRU.append(key)
        guard var record = state.disk[key] else { return }
        let date = Date()
        record.modificationDate = date
        state.disk[key] = record
        // The in-process LRU above only lives as long as this actor, so the
        // directory's own modification date is what carries an access across a
        // restart; `indexDiskIfNeeded` orders by it. Without this write a
        // record read every session stays the oldest on disk and is the first
        // one evicted. The turn store persists its touch the same way.
        do {
            try fileManager.setAttributes(
                [.modificationDate: date],
                ofItemAtPath: record.directory.path)
        } catch {
            // Not fatal: the record keeps its older date, so it is simply an
            // earlier eviction candidate next restart. Say so rather than
            // swallow it.
            ServerLog.diagnostic(
                "prompt checkpoints: could not record access on "
                    + "\(key.contentHash): \(error)")
        }
    }

    private static func verifyKey(_ metadata: PromptCheckpointMetadata) throws {
        let recomputed = PromptCheckpointKey.make(
            domain: metadata.domain, prefixTokenIDs: metadata.prefixTokenIDs)
        guard recomputed == metadata.key else {
            throw PromptCheckpointStoreError.keyMismatch
        }
    }
}
