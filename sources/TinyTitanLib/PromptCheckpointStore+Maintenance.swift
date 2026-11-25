import Foundation
import TinyTitanCore

extension PromptCheckpointStore {
    private static let stagingPrefix = ".tmp-"
    private static let backupPrefix = ".bak-"

    /// The hash a `.tmp-<hash>-<uuid>`/`.bak-<hash>-<uuid>` leftover names.
    private static func leftoverHash(_ name: String, prefix: String) -> String? {
        guard name.hasPrefix(prefix) else { return nil }
        let rest = name.dropFirst(prefix.count)
        guard let hash = rest.split(separator: "-", maxSplits: 1).first, !hash.isEmpty
        else { return nil }
        return String(hash)
    }

    /// Deletes one directory this store owns and says why. A failure to delete is
    /// reported and the directory stays for a later attempt; it is never counted.
    private func reclaim(_ directory: URL, reason: String) {
        do {
            try fileManager.removeItem(at: directory)
        } catch  where !ServerPromptStateStore.isFileNotFound(error) {
            ServerLog.diagnostic(
                "prompt checkpoints: could not reclaim "
                    + "\(directory.lastPathComponent): \(error)")
            return
        } catch {
            // Already absent on disk.
        }
        ServerLog.diagnostic(
            "prompt checkpoints: reclaimed \(directory.lastPathComponent): \(reason)")
    }

    /// Reads one canonical record directory and refuses it unless every part is
    /// present and matches its own metadata: the metadata decodes and validates,
    /// its key hashes to the directory name, and the payload (and auxiliary, when
    /// declared) is a regular file of exactly the declared size.
    private func diskRecord(
        at directory: URL, expectedHash: String, modificationDate: Date
    ) throws -> DiskRecord {
        let key = PromptCheckpointKey(contentHash: expectedHash)
        let metadataURL = directory.appendingPathComponent(Self.metadataName)
        let data = try BoundedMetadataRead.read(
            fileAt: metadataURL, maxBytes: 16 * 1_048_576)
        let diskMetadata = try JSONDecoder().decode(DiskMetadata.self, from: data)
        guard diskMetadata.version == DiskMetadata.currentVersion else {
            throw PromptCheckpointStoreError.corrupt(
                key, "unsupported metadata version \(diskMetadata.version)")
        }
        guard diskMetadata.metadata.key.contentHash == expectedHash else {
            throw PromptCheckpointStoreError.corrupt(
                key, "metadata does not hash to the record directory")
        }
        try diskMetadata.validate()
        let payloadURL = directory.appendingPathComponent(Self.payloadName)
        let payloadBytes = diskMetadata.metadata.descriptor.payloadBytes
        let payloadValues = try payloadURL.resourceValues(
            forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
        guard payloadValues.isRegularFile == true,
            payloadValues.isSymbolicLink != true,
            payloadValues.fileSize == payloadBytes
        else {
            throw PromptCheckpointStoreError.corrupt(
                key,
                "payload file is \(payloadValues.fileSize.map(String.init) ?? "unreadable") "
                    + "bytes, expected \(payloadBytes)")
        }
        let auxiliaryBytes = diskMetadata.auxiliaryBytes ?? 0
        if auxiliaryBytes > 0 {
            let auxiliaryURL = directory.appendingPathComponent(Self.auxiliaryName)
            let auxiliaryValues = try auxiliaryURL.resourceValues(
                forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
            guard auxiliaryValues.isRegularFile == true,
                auxiliaryValues.isSymbolicLink != true,
                auxiliaryValues.fileSize == auxiliaryBytes
            else {
                throw PromptCheckpointStoreError.corrupt(
                    key,
                    "auxiliary file is "
                        + "\(auxiliaryValues.fileSize.map(String.init) ?? "unreadable") "
                        + "bytes, expected \(auxiliaryBytes)")
            }
        }
        return DiskRecord(
            metadata: diskMetadata.metadata, directory: directory,
            payload: payloadURL, auxiliaryBytes: auxiliaryBytes,
            modificationDate: modificationDate)
    }

    /// Indexes the disk directories once, and only once the listing succeeds.
    ///
    /// Startup is also where crash leftovers are settled. A record directory must
    /// be named for the hash its metadata contains and carry a payload of exactly
    /// the declared size; anything else is quarantined and reclaimed instead of
    /// staying uncounted forever (AUD2-001/002). A `.bak-` left by the rename
    /// swap is restored when the canonical directory is missing -- the crash
    /// window between the two renames -- and deleted as stale otherwise. `.tmp-`
    /// staging is never published: those are writes that did not commit.
    private static let directoryKeys: Set<URLResourceKey> = [
        .isDirectoryKey, .isSymbolicLinkKey, .contentModificationDateKey,
    ]

    func indexDiskIfNeeded() {
        guard !diskIndexed else { return }
        guard let root = configuration.diskDirectory else {
            diskIndexed = true
            return
        }
        // A failed listing is not "nothing to index": leaving the flag clear
        // lets a later call retry, where latching it would strand every
        // record on disk for the store's lifetime.
        guard
            let entries = try? fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: Array(Self.directoryKeys))
        else { return }
        diskIndexed = true
        mergeIntoIndex(settleDiskEntries(entries, root: root))
    }

    /// Validates the listing, reclaims crash debris, restores a recoverable
    /// backup, and returns the records that may be indexed. Split out of
    /// `indexDiskIfNeeded` to keep each function under the length gate.
    private func settleDiskEntries(
        _ entries: [URL], root: URL
    ) -> [(record: DiskRecord, date: Date)] {
        var valid: [String: (record: DiskRecord, date: Date)] = [:]
        var backups: [(hash: String, directory: URL, date: Date)] = []
        var staged: [(directory: URL, reason: String)] = []
        var rejected: [(directory: URL, reason: String)] = []
        for directory in entries {
            guard
                let values = try? directory.resourceValues(
                    forKeys: Self.directoryKeys),
                values.isDirectory == true, values.isSymbolicLink != true
            else { continue }
            let name = directory.lastPathComponent
            let date = values.contentModificationDate ?? .distantPast
            if name.hasPrefix(Self.stagingPrefix) {
                staged.append(
                    (directory, "unpublished staging from an interrupted save"))
                continue
            }
            if name.hasPrefix(Self.backupPrefix) {
                guard let hash = Self.leftoverHash(name, prefix: Self.backupPrefix)
                else {
                    rejected.append((directory, "backup directory names no record hash"))
                    continue
                }
                backups.append((hash, directory, date))
                continue
            }
            if name.hasPrefix(".") {
                rejected.append((directory, "unrecognised hidden directory"))
                continue
            }
            do {
                let record = try diskRecord(
                    at: directory, expectedHash: name, modificationDate: date)
                guard state.disk[record.metadata.key] == nil else { continue }
                valid[name] = (record, date)
            } catch {
                rejected.append((directory, "invalid record: \(error)"))
            }
        }
        // Reclaim rejected canonical directories before a backup for the same
        // hash is considered: the backup is exactly what should replace one.
        for entry in rejected {
            reclaim(entry.directory, reason: entry.reason)
        }
        let indexed = { [self] (hash: String) -> Bool in
            valid[hash] != nil || state.disk.keys.contains { $0.contentHash == hash }
        }
        for backup in backups.sorted(by: { $0.date > $1.date }) {
            if indexed(backup.hash) {
                reclaim(
                    backup.directory,
                    reason: "stale backup of an already indexed record")
                continue
            }
            do {
                let record = try diskRecord(
                    at: backup.directory, expectedHash: backup.hash,
                    modificationDate: backup.date)
                let canonical = root.appendingPathComponent(
                    backup.hash, isDirectory: true)
                guard !fileManager.fileExists(atPath: canonical.path) else {
                    reclaim(
                        backup.directory,
                        reason: "backup whose canonical path already exists")
                    continue
                }
                try fileManager.moveItem(at: backup.directory, to: canonical)
                let restored = DiskRecord(
                    metadata: record.metadata, directory: canonical,
                    payload: canonical.appendingPathComponent(Self.payloadName),
                    auxiliaryBytes: record.auxiliaryBytes,
                    modificationDate: record.modificationDate)
                valid[backup.hash] = (restored, backup.date)
                ServerLog.diagnostic(
                    "prompt checkpoints: restored backup "
                        + "\(backup.directory.lastPathComponent) as \(backup.hash)")
            } catch {
                reclaim(backup.directory, reason: "invalid backup: \(error)")
            }
        }
        for entry in staged {
            reclaim(entry.directory, reason: entry.reason)
        }
        return valid.values.sorted { $0.date < $1.date }
    }

    /// Merges the validated listing into the in-memory index in one date order,
    /// then applies the disk budget. A record `load` found before this index is
    /// already in `state.disk`; appending the listing behind it would make the
    /// most recently loaded record the first eviction candidate.
    private func mergeIntoIndex(_ loaded: [(record: DiskRecord, date: Date)]) {
        // Restart LRU order is the directories' modification dates, not the
        // volume's listing order — the turn store applies the same rule, and
        // a listing order would evict an arbitrary record first.
        var order: [(key: PromptCheckpointKey, date: Date)] =
            state.diskLRU.compactMap { key in
                state.disk[key].map { (key, $0.modificationDate) }
            }
        order.append(
            contentsOf: loaded.map {
                (key: $0.record.metadata.key, date: $0.record.modificationDate)
            })
        order.sort { $0.date < $1.date }
        state.diskLRU = order.map(\.key)
        for entry in loaded {
            state.disk[entry.record.metadata.key] = entry.record
            state.diskBytes +=
                entry.record.metadata.descriptor.payloadBytes
                + entry.record.auxiliaryBytes
        }
        // A restart can index more than the budget allows (the limit may have
        // been lowered); evict immediately rather than waiting for a save
        // that may never come — the turn store does the same on load.
        evictDiskIfNeeded()
    }

    func evictDiskIfNeeded() {
        while state.diskBytes > configuration.diskLimitBytes,
            let oldest = state.diskLRU.first
        {
            guard let record = state.disk.removeValue(forKey: oldest) else {
                state.diskLRU.removeFirst()
                continue
            }
            do {
                try fileManager.removeItem(at: record.directory)
            } catch  where !ServerPromptStateStore.isFileNotFound(error) {
                // Keep the charge: the directory still occupies the disk, and
                // retrying it in this loop cannot succeed.
                state.disk[oldest] = record
                ServerLog.diagnostic(
                    "prompt checkpoints: could not evict "
                        + "\(oldest.contentHash): \(error)")
                return
            } catch {
                // Already absent on disk: release the ledger charge and continue.
            }
            state.diskLRU.removeFirst()
            state.diskBytes -=
                record.metadata.descriptor.payloadBytes + record.auxiliaryBytes
        }
    }

    func remove(_ key: PromptCheckpointKey) {
        if let removed = state.memory.removeValue(forKey: key) {
            state.memoryBytes -=
                removed.snapshot.payload.count
                + (removed.auxiliary?.payload.count ?? 0)
        }
        state.memoryLRU.removeAll { $0 == key }
        guard let record = state.disk[key] else { return }
        do {
            try fileManager.removeItem(at: record.directory)
        } catch  where !ServerPromptStateStore.isFileNotFound(error) {
            // The directory still occupies disk. Dropping the entry would
            // free bytes that are not free and leave nothing to retry.
            ServerLog.diagnostic(
                "prompt checkpoints: could not delete "
                    + "\(key.contentHash): \(error)")
            return
        } catch {
            // Already absent on disk: release the ledger entry.
        }
        state.disk.removeValue(forKey: key)
        state.diskLRU.removeAll { $0 == key }
        state.diskBytes -=
            record.metadata.descriptor.payloadBytes + record.auxiliaryBytes
    }

    func removeAll() {
        // Index first: without it, `state.disk` is empty and nothing is
        // unlinked, so the directories survive a removeAll on a fresh store.
        indexDiskIfNeeded()
        var survivors = State()
        for record in state.disk.values {
            do {
                try fileManager.removeItem(at: record.directory)
            } catch  where !ServerPromptStateStore.isFileNotFound(error) {
                ServerLog.diagnostic(
                    "prompt checkpoints: could not delete "
                        + "\(record.metadata.key.contentHash) during removeAll: "
                        + "\(error)")
                survivors.disk[record.metadata.key] = record
                survivors.diskBytes +=
                    record.metadata.descriptor.payloadBytes
                    + record.auxiliaryBytes
                survivors.diskLRU.append(record.metadata.key)
            } catch {
                // Already absent on disk.
            }
        }
        survivors.diskLRU.sort {
            (survivors.disk[$0]?.modificationDate ?? .distantPast)
                < (survivors.disk[$1]?.modificationDate ?? .distantPast)
        }
        // Memory is process-local and always reclaimable; only the disk
        // records that refused removal survive.
        state = survivors
    }
}
