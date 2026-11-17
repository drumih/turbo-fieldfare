import Foundation
import TinyTitanCore

extension PromptCheckpointStore {
    func writeDisk(
        metadata: PromptCheckpointMetadata,
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?
    ) throws {
        guard let root = configuration.diskDirectory else {
            throw PromptCheckpointStoreError.diskUnavailable
        }
        // The eviction loop at the end charges `state.disk`; without the
        // index, records a previous process left on disk are invisible to it
        // and the first save after a restart evicts nothing.
        indexDiskIfNeeded()
        let directory = root.appendingPathComponent(
            metadata.key.contentHash, isDirectory: true)
        // Write into a temporary directory and swap it into place with a
        // rename: writing payload, auxiliary and metadata straight into the
        // final directory leaves an orphan a crash can produce — files with no
        // (or stale) metadata, invisible to indexing and eviction. The swap
        // is the same discipline the turn store applies (S23 there).
        let staging = root.appendingPathComponent(
            ".tmp-\(metadata.key.contentHash)-\(UUID().uuidString)",
            isDirectory: true)
        try fileManager.createDirectory(
            at: staging, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        let auxiliaryBytes = auxiliary?.payload.count ?? 0
        do {
            let payloadURL = staging.appendingPathComponent(Self.payloadName)
            try snapshot.payload.write(to: payloadURL, options: .atomic)
            try fileManager.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: payloadURL.path)
            let auxiliaryURL = staging.appendingPathComponent(Self.auxiliaryName)
            let auxiliarySHA256: String?
            if let auxiliary {
                try auxiliary.payload.write(to: auxiliaryURL, options: .atomic)
                try fileManager.setAttributes(
                    [.posixPermissions: 0o600], ofItemAtPath: auxiliaryURL.path)
                auxiliarySHA256 = Sha256Verifier.hashData(auxiliary.payload)
            } else {
                auxiliarySHA256 = nil
            }
            let diskMetadata = DiskMetadata(
                version: DiskMetadata.currentVersion,
                metadata: metadata,
                payloadSHA256: Sha256Verifier.hashData(snapshot.payload),
                auxiliaryBytes: auxiliary == nil ? nil : auxiliaryBytes,
                auxiliarySHA256: auxiliarySHA256)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(diskMetadata).write(
                to: staging.appendingPathComponent(Self.metadataName),
                options: .atomic)
            if fileManager.fileExists(atPath: directory.path) {
                let backup = root.appendingPathComponent(
                    ".bak-\(metadata.key.contentHash)-\(UUID().uuidString)",
                    isDirectory: true)
                try fileManager.moveItem(at: directory, to: backup)
                do {
                    try fileManager.moveItem(at: staging, to: directory)
                    try? fileManager.removeItem(at: backup)
                } catch {
                    try? fileManager.moveItem(at: backup, to: directory)
                    throw error
                }
            } else {
                try fileManager.moveItem(at: staging, to: directory)
            }
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
        // Replacing a record charges the payload once, not twice: the old
        // entry's bytes are still in `diskBytes` until they are subtracted.
        if let replaced = state.disk[metadata.key] {
            state.diskBytes -=
                replaced.metadata.descriptor.payloadBytes
                + replaced.auxiliaryBytes
            state.diskLRU.removeAll { $0 == metadata.key }
        }
        state.disk[metadata.key] = DiskRecord(
            metadata: metadata, directory: directory,
            payload: directory.appendingPathComponent(Self.payloadName),
            auxiliaryBytes: auxiliaryBytes,
            modificationDate: Date())
        state.diskLRU.append(metadata.key)
        state.diskBytes += snapshot.payload.count + auxiliaryBytes
        evictDiskIfNeeded()
    }

    func readPayload(
        at url: URL, expectedBytes: Int
    ) throws -> Data {
        guard expectedBytes > 0,
            expectedBytes <= max(Self.maximumCaptureBytes, Self.maximumAuxiliaryBytes)
        else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "payload size \(expectedBytes) is not a valid capture size")
        }
        // Check the file's size before reading it, and read exactly the declared
        // size. Reading `expected + 1` to detect a too-long file allocated the
        // declared buffer first, so a corrupt or swapped multi-GiB record was
        // materialized before anything checked it (AUD2-004).
        let key = PromptCheckpointKey(
            contentHash: url.deletingLastPathComponent().lastPathComponent)
        let values = try url.resourceValues(
            forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true,
            values.isSymbolicLink != true,
            values.fileSize == expectedBytes
        else {
            throw PromptCheckpointStoreError.corrupt(
                key,
                "payload is \(values.fileSize.map(String.init) ?? "unreadable") bytes, "
                    + "expected \(expectedBytes)")
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: expectedBytes) ?? Data()
        guard data.count == expectedBytes else {
            throw PromptCheckpointStoreError.corrupt(
                key,
                "payload is \(data.count) bytes, expected \(expectedBytes)")
        }
        return data
    }
}
