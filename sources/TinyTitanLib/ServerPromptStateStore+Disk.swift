import CryptoKit
import Foundation
import Synchronization
import TinyTitan
import TinyTitanCore

// Disk payload IO for the turn-end prompt-state store.
//
// Split out of `ServerPromptStateStore.swift` (2026-10-22) under the
// 500-line-per-file rule as pure code motion; the methods are an extension
// because they read the store's mutex-guarded record index.
extension ServerPromptStateStore {
    func readPayloadVerifying(
        record: DiskRecord,
        entryID: UUID
    ) throws -> Data {
        let expected = record.metadata.descriptor.payloadBytes
        let handle = try FileHandle(forReadingFrom: record.payload)
        defer { try? handle.close() }
        // Never read past expected+1: a swapped payload larger than the
        // declared size is refused on size, and the bytes it would have taken
        // are never allocated. The declared bytes themselves are still
        // materialized -- the caller needs the payload -- so this bounds the
        // read to what the metadata claims, not to what the file is.
        var data = try handle.read(upToCount: expected + 1) ?? Data()
        while data.count < expected + 1 {
            guard let chunk = try handle.read(upToCount: 1_048_576),
                !chunk.isEmpty
            else { break }
            data.append(chunk)
        }
        guard data.count == expected else {
            throw ServerPromptStateStoreError.corrupt(entryID, "payload size mismatch")
        }
        var hasher = SHA256()
        hasher.update(data: data)
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == record.metadata.payloadSHA256 else {
            throw ServerPromptStateStoreError.corrupt(entryID, "SHA-256 mismatch")
        }
        return data
    }

    func readAuxiliaryVerifying(
        record: DiskRecord,
        entryID: UUID
    ) throws -> InferenceAuxiliaryState? {
        guard let expected = record.metadata.auxiliaryBytes,
            let expectedSHA = record.metadata.auxiliarySHA256
        else { return nil }
        guard expected > 0, expected <= Self.maximumAuxiliaryBytes else {
            throw ServerPromptStateStoreError.corrupt(
                entryID, "auxiliary size \(expected) is out of range")
        }
        let handle = try FileHandle(forReadingFrom: record.auxiliary)
        defer { try? handle.close() }
        let data = try handle.read(upToCount: expected + 1) ?? Data()
        guard data.count == expected else {
            throw ServerPromptStateStoreError.corrupt(
                entryID, "auxiliary payload size mismatch")
        }
        guard Self.sha256(data) == expectedSHA else {
            throw ServerPromptStateStoreError.corrupt(
                entryID, "auxiliary SHA-256 mismatch")
        }
        return InferenceAuxiliaryState(
            position: record.metadata.descriptor.position, payload: data)
    }

    func writeDisk(
        entry: ServerPromptCacheEntry,
        snapshot: InferenceStateSnapshot,
        auxiliary: InferenceAuxiliaryState?
    ) throws {
        guard let root = configuration.diskDirectory else { return }
        let id = entry.id
        let finalDirectory = root.appendingPathComponent(id.uuidString.lowercased())
        let temporaryDirectory = root.appendingPathComponent(
            ".tmp-\(id.uuidString.lowercased())-\(UUID().uuidString.lowercased())")
        try fileManager.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        do {
            let payloadURL = temporaryDirectory.appendingPathComponent(Self.payloadName)
            try snapshot.payload.write(to: payloadURL, options: [.atomic])
            try fileManager.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: payloadURL.path)
            let auxiliaryURL = temporaryDirectory.appendingPathComponent(
                Self.auxiliaryName)
            let auxiliaryBytes = auxiliary?.payload.count ?? 0
            let auxiliarySHA256: String?
            if let auxiliary {
                try auxiliary.payload.write(to: auxiliaryURL, options: [.atomic])
                try fileManager.setAttributes(
                    [.posixPermissions: 0o600],
                    ofItemAtPath: auxiliaryURL.path)
                auxiliarySHA256 = Self.sha256(auxiliary.payload)
            } else {
                auxiliarySHA256 = nil
            }
            let metadata = DiskMetadata(
                version: DiskMetadata.currentVersion,
                entry: entry,
                descriptor: snapshot.descriptor,
                payloadSHA256: Self.sha256(snapshot.payload),
                auxiliaryBytes: auxiliary == nil ? nil : auxiliaryBytes,
                auxiliarySHA256: auxiliarySHA256)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            let metadataURL = temporaryDirectory.appendingPathComponent(Self.metadataName)
            try encoder.encode(metadata).write(to: metadataURL, options: [.atomic])
            try fileManager.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: metadataURL.path)
            // S23: never remove the live directory before the replacement is
            // in place. Swap via renames: move the old directory aside, move
            // the new one into place, then drop the backup. If the swap fails,
            // restore the old directory so the previously persisted record
            // stays loadable instead of being orphaned.
            if fileManager.fileExists(atPath: finalDirectory.path) {
                let backupDirectory = root.appendingPathComponent(
                    ".bak-\(id.uuidString.lowercased())-\(UUID().uuidString.lowercased())")
                try fileManager.moveItem(at: finalDirectory, to: backupDirectory)
                do {
                    try fileManager.moveItem(at: temporaryDirectory, to: finalDirectory)
                    try? fileManager.removeItem(at: backupDirectory)
                } catch {
                    try? fileManager.moveItem(at: backupDirectory, to: finalDirectory)
                    throw error
                }
            } else {
                try fileManager.moveItem(at: temporaryDirectory, to: finalDirectory)
            }
            state.withLock { state in
                if let previous = state.disk[id] {
                    state.diskBytes -=
                        previous.metadata.descriptor.payloadBytes
                        + previous.auxiliaryBytes
                }
                let record = DiskRecord(
                    metadata: metadata,
                    directory: finalDirectory,
                    payload: finalDirectory.appendingPathComponent(Self.payloadName),
                    auxiliary: finalDirectory.appendingPathComponent(
                        Self.auxiliaryName),
                    auxiliaryBytes: auxiliaryBytes,
                    modificationDate: Date())
                state.disk[id] = record
                state.diskBytes += snapshot.payload.count + auxiliaryBytes
            }
            touchDisk(id)
        } catch {
            try? fileManager.removeItem(at: temporaryDirectory)
            throw error
        }
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
