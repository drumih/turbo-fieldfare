import CryptoKit
import Foundation
import Synchronization
import TinyTitan
import TinyTitanCore

/// Persistent + in-memory backing for published prompt-cache entries.
///
/// Concurrency model: the store is shared between the session actor (restore,
/// remove, contains) and the detached save tasks. All in-memory bookkeeping is
/// guarded by `state`; all disk writes are serialized on `diskQueue` so
/// concurrent saves (a later generation's snapshot while an earlier write is
/// still in flight) never interleave file operations or clobber each other.
/// unchecked-invariant: see the concurrency note above -- in-memory
/// bookkeeping is guarded by `state`, and every disk write is serialized on
/// `diskQueue` so concurrent saves cannot interleave file operations.
final class ServerPromptStateStore: @unchecked Sendable {
    struct DiskMetadata: Codable {
        static let currentVersion = 1

        let version: Int
        let entry: ServerPromptCacheEntry
        let descriptor: InferenceStateSnapshotDescriptor
        let payloadSHA256: String
        // Additive in v1: Qwen3.8 records carry the QSA/PLE payload and hash.
        let auxiliaryBytes: Int?
        let auxiliarySHA256: String?
    }

    struct DiskRecord {
        let metadata: DiskMetadata
        let directory: URL
        let payload: URL
        let auxiliary: URL
        let auxiliaryBytes: Int
        var modificationDate: Date
    }

    static let metadataName = "metadata.json"
    static let payloadName = "state.bin"
    static let auxiliaryName = "auxiliary.bin"
    /// Applied by `BoundedMetadataRead` before the buffer is allocated, not to
    /// the bytes after they have been read — a stat-then-re-read check bounded
    /// the decision but not the memory (K17).
    private static let maximumMetadataBytes = 16 * 1_048_576

    /// Hard ceiling for one snapshot capture (S2). A snapshot above this is
    /// never allocated, bounding the multi-GiB capture + write cost regardless
    /// of a large configured disk budget (default 8 GiB).
    static let maximumCaptureBytes = 4 * 1_024 * 1_048_576
    static let maximumAuxiliaryBytes = 1_024 * 1_048_576

    struct MemoryRecord {
        let snapshot: InferenceStateSnapshot
        let auxiliary: InferenceAuxiliaryState?

        var bytes: Int {
            snapshot.payload.count + (auxiliary?.payload.count ?? 0)
        }
    }

    struct State {
        var memory: [UUID: MemoryRecord] = [:]
        var memoryLRU: [UUID] = []
        var memoryBytes = 0
        var disk: [UUID: DiskRecord] = [:]
        var diskLRU: [UUID] = []
        var diskBytes = 0
    }

    let configuration: ServerPromptCacheStorageConfiguration
    let fileManager: FileManager
    let state = Mutex(State())
    private let diskQueue = DispatchQueue(
        label: "tinytitan.prompt-state-store.disk",
        qos: .utility)

    var maximumSnapshotBytes: Int {
        min(
            max(
                configuration.memoryLimitBytes,
                configuration.diskDirectory == nil ? 0 : configuration.diskLimitBytes),
            Self.maximumCaptureBytes)
    }

    init(
        configuration: ServerPromptCacheStorageConfiguration,
        fileManager: FileManager = .default
    ) throws {
        self.configuration = configuration
        self.fileManager = fileManager
        if let root = configuration.diskDirectory {
            try fileManager.createDirectory(
                at: root,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            try fileManager.setAttributes(
                [.posixPermissions: 0o700],
                ofItemAtPath: root.path)
        }
    }

    func loadEntries(domain: ServerPromptCacheDomain) -> [ServerPromptCacheEntry] {
        state.withLock { state in
            state.disk.removeAll(keepingCapacity: true)
            state.diskLRU.removeAll(keepingCapacity: true)
            state.diskBytes = 0
        }
        guard let root = configuration.diskDirectory,
            configuration.diskLimitBytes > 0
        else { return [] }
        let keys: Set<URLResourceKey> = [
            .isDirectoryKey,
            .isSymbolicLinkKey,
            .contentModificationDateKey,
            .fileSizeKey,
        ]
        guard
            let children = try? fileManager.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: Array(keys),
                options: [.skipsHiddenFiles])
        else { return [] }

        var loaded: [DiskRecord] = []
        for directory in children {
            guard let directoryID = UUID(uuidString: directory.lastPathComponent),
                let values = try? directory.resourceValues(forKeys: keys),
                values.isDirectory == true,
                values.isSymbolicLink != true
            else { continue }
            let metadataURL = directory.appendingPathComponent(Self.metadataName)
            let payloadURL = directory.appendingPathComponent(Self.payloadName)
            let auxiliaryURL = directory.appendingPathComponent(Self.auxiliaryName)
            guard
                let metadataData = try? BoundedMetadataRead.read(
                    fileAt: metadataURL,
                    maxBytes: UInt64(Self.maximumMetadataBytes)),
                let metadata = try? JSONDecoder().decode(
                    DiskMetadata.self,
                    from: metadataData),
                metadata.version == DiskMetadata.currentVersion,
                metadata.entry.id == directoryID,
                metadata.descriptor.version
                    == InferenceStateSnapshotDescriptor.currentVersion,
                metadata.entry.kvPosition == metadata.descriptor.position,
                (try? metadata.descriptor.validatedPayloadBytes()) != nil,
                let payloadValues = try? payloadURL.resourceValues(
                    forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey]),
                payloadValues.isRegularFile == true,
                payloadValues.isSymbolicLink != true,
                payloadValues.fileSize == metadata.descriptor.payloadBytes
            else {
                continue
            }
            let auxiliaryBytes = metadata.auxiliaryBytes ?? 0
            if auxiliaryBytes > 0 {
                guard
                    let auxiliaryValues = try? auxiliaryURL.resourceValues(
                        forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey]),
                    auxiliaryValues.isRegularFile == true,
                    auxiliaryValues.isSymbolicLink != true,
                    auxiliaryValues.fileSize == auxiliaryBytes
                else { continue }
            }
            loaded.append(
                DiskRecord(
                    metadata: metadata,
                    directory: directory,
                    payload: payloadURL,
                    auxiliary: auxiliaryURL,
                    auxiliaryBytes: auxiliaryBytes,
                    modificationDate: values.contentModificationDate ?? .distantPast))
        }

        loaded.sort { $0.modificationDate < $1.modificationDate }
        state.withLock { state in
            for record in loaded {
                let id = record.metadata.entry.id
                state.disk[id] = record
                state.diskLRU.append(id)
                state.diskBytes +=
                    record.metadata.descriptor.payloadBytes + record.auxiliaryBytes
            }
        }
        // The evicted ids mean nothing to a listing; a refusal is the cap not being
        // honoured, and this path has no result to carry it in.
        if let refusal = evictDiskIfNeeded().refusal {
            ServerLog.diagnostic("prompt state: eviction refused on load: \(refusal)")
        }
        return state.withLock { state in
            state.diskLRU.compactMap {
                guard let entry = state.disk[$0]?.metadata.entry,
                    entry.domain == domain
                else { return nil }
                return entry
            }
        }
    }

    /// Persist a snapshot. Async: the write (SHA-256 + state.bin +
    /// metadata.json) runs on the store's serial disk queue, never on the
    /// caller's actor, so a multi-GiB write does not stall the session.
    func save(
        entry: ServerPromptCacheEntry,
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState? = nil
    ) async -> ServerPromptStateSaveResult {
        await withCheckedContinuation { continuation in
            diskQueue.async { [self] in
                continuation.resume(
                    returning: saveSync(
                        entry: entry, snapshot: snapshot,
                        auxiliary: auxiliary))
            }
        }
    }

    private func saveSync(
        entry: ServerPromptCacheEntry,
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?
    ) -> ServerPromptStateSaveResult {
        precondition(entry.kvPosition == snapshot.descriptor.position)
        if let auxiliary {
            precondition(auxiliary.position == entry.kvPosition)
        }
        let auxiliaryBytes = auxiliary?.payload.count ?? 0
        let totalBytes = snapshot.payload.count + auxiliaryBytes
        var unbacked: [UUID] = []
        var diskError: String?
        if auxiliaryBytes > Self.maximumAuxiliaryBytes {
            diskError =
                "auxiliary state is \(auxiliaryBytes) bytes, over the "
                + "\(Self.maximumAuxiliaryBytes)-byte cap"
        }

        if auxiliaryBytes <= Self.maximumAuxiliaryBytes,
            totalBytes <= configuration.memoryLimitBytes,
            configuration.memoryLimitBytes > 0
        {
            insertMemory(snapshot, auxiliary: auxiliary, id: entry.id)
        }

        if auxiliaryBytes <= Self.maximumAuxiliaryBytes,
            configuration.diskDirectory != nil,
            configuration.diskLimitBytes > 0,
            totalBytes <= configuration.diskLimitBytes
        {
            do {
                try writeDisk(
                    entry: entry, snapshot: snapshot, auxiliary: auxiliary)
            } catch {
                diskError = String(describing: error)
            }
        }

        let memoryEvicted = evictMemoryIfNeeded()
        let diskEviction = evictDiskIfNeeded()
        let diskEvicted = diskEviction.evicted
        if let refusal = diskEviction.refusal {
            // Both are the disk not doing what was asked, so both travel in the one
            // field rather than the later one overwriting the earlier.
            diskError = diskError.map { "\($0) / \(refusal)" } ?? refusal
        }
        for id in memoryEvicted + diskEvicted where !contains(id) {
            if !unbacked.contains(id) { unbacked.append(id) }
        }
        if !contains(entry.id), !unbacked.contains(entry.id) {
            unbacked.append(entry.id)
        }

        let bytes = state.withLock { state in (state.memoryBytes, state.diskBytes) }
        return ServerPromptStateSaveResult(
            unbackedEntryIDs: unbacked,
            diskError: diskError,
            memoryBytes: bytes.0,
            diskBytes: bytes.1)
    }

    func restore(
        entryID: UUID,
        into runner: RealForwardRunner
    ) async throws -> String {
        // S3: the disk read + SHA-256 verification run off the caller's actor
        // (detached task, awaited here); only the Metal-buffer restore stays on
        // the actor.
        let loaded = try await Task.detached(priority: .userInitiated) { [self] in
            try loadSnapshot(entryID: entryID)
        }.value
        do {
            try runner.restoreInferenceState(
                loaded.snapshot, auxiliary: loaded.auxiliary)
            return loaded.tier
        } catch {
            remove(entryIDs: [entryID])
            throw error
        }
    }

    func loadSnapshot(
        entryID: UUID
    ) throws -> (
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?,
        tier: String
    ) {
        if let record = state.withLock({ $0.memory[entryID] }) {
            touchMemory(entryID)
            touchDisk(entryID)
            return (record.snapshot, record.auxiliary, "ram")
        }
        guard let record = state.withLock({ $0.disk[entryID] }) else {
            throw ServerPromptStateStoreError.missing(entryID)
        }
        do {
            // S3: stream the payload in bounded chunks, verifying the SHA-256
            // while copying (single pass over the file) — never materialize a
            // second full copy just to hash it.
            let payload = try readPayloadVerifying(record: record, entryID: entryID)
            let snapshot = InferenceStateSnapshot(
                descriptor: record.metadata.descriptor,
                payload: payload)
            let auxiliary = try readAuxiliaryVerifying(
                record: record, entryID: entryID)
            if snapshot.payload.count + (auxiliary?.payload.count ?? 0)
                <= configuration.memoryLimitBytes,
                configuration.memoryLimitBytes > 0
            {
                insertMemory(snapshot, auxiliary: auxiliary, id: entryID)
                _ = evictMemoryIfNeeded()
            }
            touchDisk(entryID)
            return (snapshot, auxiliary, "ssd")
        } catch {
            remove(entryIDs: [entryID])
            throw error
        }
    }

    func contains(_ entryID: UUID) -> Bool {
        state.withLock { $0.memory[entryID] != nil || $0.disk[entryID] != nil }
    }

    func remove(entryIDs: some Sequence<UUID>) {
        for id in entryIDs {
            state.withLock { state in
                if let record = state.memory.removeValue(forKey: id) {
                    state.memoryBytes -= record.bytes
                }
                state.memoryLRU.removeAll { $0 == id }
            }
            if let refusal = releaseDiskRecord(id: id) {
                ServerLog.diagnostic("prompt state: unlink refused for \(id): \(refusal)")
            }
        }
    }

    /// Give one entry's directory back to the volume, and its bytes back to the ledger
    /// only if the volume took the directory. Subtracting `payloadBytes` before the
    /// unlink that has to free them left a refused eviction counting space that was
    /// still occupied, so the cap believed it had room and stopped evicting.
    private func releaseDiskRecord(id: UUID) -> String? {
        guard let record = state.withLock({ $0.disk[id] }) else {
            state.withLock { $0.diskLRU.removeAll { $0 == id } }
            return nil
        }
        do {
            try fileManager.removeItem(at: record.directory)
        } catch {
            return String(describing: error)
        }
        state.withLock { state in
            if let current = state.disk.removeValue(forKey: id) {
                state.diskBytes -=
                    current.metadata.descriptor.payloadBytes
                    + current.auxiliaryBytes
            }
            state.diskLRU.removeAll { $0 == id }
        }
        return nil
    }

    private func insertMemory(
        _ snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?,
        id: UUID
    ) {
        let record = MemoryRecord(snapshot: snapshot, auxiliary: auxiliary)
        state.withLock { state in
            if let previous = state.memory.updateValue(record, forKey: id) {
                state.memoryBytes -= previous.bytes
            }
            state.memoryBytes += record.bytes
        }
        touchMemory(id)
    }

    private func touchMemory(_ id: UUID) {
        state.withLock { state in
            state.memoryLRU.removeAll { $0 == id }
            state.memoryLRU.append(id)
        }
    }

    func touchDisk(_ id: UUID) {
        let record = state.withLock { state -> DiskRecord? in
            guard var record = state.disk[id] else { return nil }
            record.modificationDate = Date()
            state.disk[id] = record
            state.diskLRU.removeAll { $0 == id }
            state.diskLRU.append(id)
            return record
        }
        guard let record else { return }
        // S34: the mtime only feeds cross-restart LRU ordering; update it on
        // the background disk queue so the caller's actor never blocks on
        // setAttributes.
        let date = record.modificationDate
        let path = record.directory.path
        diskQueue.async { [self] in
            do {
                try self.fileManager.setAttributes(
                    [.modificationDate: date], ofItemAtPath: path)
            } catch {
                // Not fatal: the record keeps its older date, so it is simply an
                // earlier eviction candidate next restart. Say so rather than
                // swallow it -- the checkpoint store reports the same failure
                // (AUD2-007), and this one used to be dropped with `try?`.
                ServerLog.diagnostic(
                    "prompt state: could not record access on \(id): \(error)")
            }
        }
    }

    private func evictMemoryIfNeeded() -> [UUID] {
        state.withLock { state -> [UUID] in
            var evicted: [UUID] = []
            while state.memoryBytes > configuration.memoryLimitBytes,
                let id = state.memoryLRU.first
            {
                state.memoryLRU.removeFirst()
                if let record = state.memory.removeValue(forKey: id) {
                    state.memoryBytes -= record.bytes
                    evicted.append(id)
                }
            }
            return evicted
        }
    }

    private func evictDiskIfNeeded() -> (evicted: [UUID], refusal: String?) {
        var evicted: [UUID] = []
        while let id = oldestEvictableDiskEntry() {
            // Stop at the first refusal rather than evicting the next-oldest entry to
            // pay for space this one never gave up.
            if let refusal = releaseDiskRecord(id: id) { return (evicted, refusal) }
            evicted.append(id)
        }
        return (evicted, nil)
    }

    private func oldestEvictableDiskEntry() -> UUID? {
        state.withLock { state in
            guard state.diskBytes > configuration.diskLimitBytes else { return nil }
            while let id = state.diskLRU.first {
                if state.disk[id] != nil { return id }
                state.diskLRU.removeFirst()
            }
            return nil
        }
    }

}
