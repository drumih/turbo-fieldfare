import Darwin
import Foundation
import TurboFieldfare
import TurboFieldfareRepackCore

/// Publishes whole conversation directories while retaining the store lock's inode.
/// The journal makes an interrupted multi-directory import rollback on the next attempt.
enum ConversationStoreImport {
    struct Journal: Codable {
        var source: String
        var fingerprints: [String: [String: String]]
    }

    static func run(from source: URL, to destination: URL,
                    destinationAlreadyLocked: Bool = false,
                    afterPublish: (() throws -> Void)? = nil) throws {
        let sourceKind = try Posix.entryKind(source.standardizedFileURL.path)
        guard sourceKind != .absent else { return }
        let destinationKind = try Posix.entryKind(destination.standardizedFileURL.path)
        guard sourceKind == .directory, destinationKind == .directory || destinationKind == .absent else {
            throw ImportError("History source and destination must be regular directories, without symlink roots.")
        }
        let source = source.standardizedFileURL.resolvingSymlinksInPath()
        let destination = destination.standardizedFileURL.resolvingSymlinksInPath()
        guard source != destination else { return }
        guard FileManager.default.fileExists(atPath: source.path) else { return }
        guard !destination.path.hasPrefix(source.path + "/"), !source.path.hasPrefix(destination.path + "/") else {
            throw ImportError("History source and destination must be separate directories.")
        }
        guard let sourceLock = try ConversationStoreLock.acquire(storeRoot: source) else {
            throw ImportError("Close the app using the legacy history before importing it.")
        }
        let destinationLock: ConversationStoreLock?
        if destinationAlreadyLocked { destinationLock = nil } else {
            guard let lock = try ConversationStoreLock.acquire(storeRoot: destination) else {
                throw ImportError("Close the other app using the destination history before importing it.")
            }
            destinationLock = lock
        }
        defer { withExtendedLifetime((sourceLock, destinationLock)) {} }
        let manager = FileManager.default
        let staging = destination.appendingPathComponent(".history-import", isDirectory: true)
        let journalURL = destination.appendingPathComponent(".history-import.json")
        try recover(at: destination, destinationAlreadyLocked: true)
        if manager.fileExists(atPath: staging.path) { try manager.removeItem(at: staging) }
        try manager.createDirectory(at: staging, withIntermediateDirectories: false)
        var journal = Journal(source: source.path, fingerprints: [:])
        do {
            var required: Int64 = 0
            let conversations = try manager.contentsOfDirectory(at: source, includingPropertiesForKeys: nil)
                .filter { UUID(uuidString: $0.lastPathComponent) != nil }
            // Verify every collision before publishing any conversation.
            for conversation in conversations {
                let name = conversation.lastPathComponent
                let files = try fingerprint(conversation)
                let existing = destination.appendingPathComponent(name)
                if manager.fileExists(atPath: existing.path) {
                    guard try fingerprint(existing) == files else {
                        throw ImportError("History contains a conflicting conversation ID \(name). Import into a separate app-data root or resolve the conflict first.")
                    }
                    continue
                }
                for path in files.keys {
                    let values = try conversation.appendingPathComponent(path).resourceValues(forKeys: [.fileSizeKey])
                    required += Int64(values.fileSize ?? 0)
                }
                journal.fingerprints[name] = files
            }
            let available = try destination.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
                .volumeAvailableCapacityForImportantUsage
            guard let available, available >= required else { throw ImportError("Insufficient free space to import history on the app-data volume.") }
            for (name, expected) in journal.fingerprints {
                let copied = staging.appendingPathComponent(name)
                try manager.copyItem(at: source.appendingPathComponent(name), to: copied)
                guard try fingerprint(copied) == expected else { throw ImportError("History copy verification failed for \(name).") }
                try makeDurable(copied, files: expected.keys)
            }
            try Posix.fsyncDirectory(staging.path)
            try Posix.atomicWrite(JSONEncoder().encode(journal), to: journalURL.path, durableIn: destination.path)
            for name in journal.fingerprints.keys.sorted() {
                try manager.moveItem(at: staging.appendingPathComponent(name), to: destination.appendingPathComponent(name))
                try afterPublish?()
            }
            try Posix.fsyncDirectory(staging.path)
            try Posix.fsyncDirectory(destination.path)
            // Completion is durable before abandoning the journal. Exact duplicates
            // are verified again if settings persistence fails after this point.
            let receipt = destination.appendingPathComponent(".history-import-completed.json")
            try Posix.atomicWrite(JSONEncoder().encode(journal), to: receipt.path, durableIn: destination.path)
            try manager.removeItem(at: journalURL)
            try manager.removeItem(at: staging)
            try Posix.fsyncDirectory(destination.path)
        } catch {
            if manager.fileExists(atPath: journalURL.path) {
                do {
                    for (name, expected) in journal.fingerprints {
                        let published = destination.appendingPathComponent(name)
                        if manager.fileExists(atPath: published.path) {
                            guard try fingerprint(published) == expected else {
                                throw ImportError("Published conversation changed during rollback: \(name)")
                            }
                            try manager.removeItem(at: published)
                        }
                    }
                    try manager.removeItem(at: journalURL)
                    try manager.removeItem(at: staging)
                } catch let rollbackError {
                    throw ImportError("History import failed: \(error); rollback also failed: \(rollbackError). Originals and the recovery journal were preserved.")
                }
            }
            throw ImportError("History import failed; originals were preserved: \(error)")
        }
    }

    /// Recovery needs only the destination lock: the source drive may have
    /// disconnected since publication, while originals remain on that drive.
    static func recover(at destination: URL, destinationAlreadyLocked: Bool = false) throws {
        let journalURL = destination.appendingPathComponent(".history-import.json")
        guard try Posix.entryKind(journalURL.path) != .absent else { return }
        guard try Posix.entryKind(destination.path) == .directory else {
            throw ImportError("History recovery refuses a symlink or special-file store root.")
        }
        let lock: ConversationStoreLock?
        if destinationAlreadyLocked { lock = nil } else {
            guard let acquired = try ConversationStoreLock.acquire(storeRoot: destination) else {
                throw ImportError("Close the other app using history before recovering an interrupted import.")
            }
            lock = acquired
        }
        defer { withExtendedLifetime(lock) {} }
        let manager = FileManager.default
        let journal = try JSONDecoder().decode(Journal.self, from: Data(contentsOf: journalURL))
        // Verify the entire rollback before removing anything, retaining changed
        // records and the journal when manual recovery is required.
        for (name, expected) in journal.fingerprints {
            guard UUID(uuidString: name) != nil else { throw ImportError("Invalid history import journal.") }
            let published = destination.appendingPathComponent(name)
            if manager.fileExists(atPath: published.path), try fingerprint(published) != expected {
                throw ImportError("Interrupted import contains changed conversation \(name); preserve it and recover manually.")
            }
        }
        for name in journal.fingerprints.keys {
            let published = destination.appendingPathComponent(name)
            if manager.fileExists(atPath: published.path) { try manager.removeItem(at: published) }
        }
        let staging = destination.appendingPathComponent(".history-import")
        if manager.fileExists(atPath: staging.path) { try manager.removeItem(at: staging) }
        try manager.removeItem(at: journalURL)
        try Posix.fsyncDirectory(destination.path)
    }

    private static func makeDurable(_ root: URL, files: Dictionary<String, String>.Keys) throws {
        var directories: Set<URL> = [root]
        for relative in files {
            let file = root.appendingPathComponent(relative)
            let descriptor = try Posix.openReadNoFollow(file.path)
            do { try Posix.fsync(descriptor, path: file.path) }
            catch { Darwin.close(descriptor); throw error }
            Darwin.close(descriptor)
            var parent = file.deletingLastPathComponent()
            while parent.pathComponents.count >= root.pathComponents.count {
                directories.insert(parent)
                if parent == root { break }
                parent.deleteLastPathComponent()
            }
        }
        for directory in directories.sorted(by: { $0.pathComponents.count > $1.pathComponents.count }) {
            try Posix.fsyncDirectory(directory.path)
        }
    }

    private static func fingerprint(_ root: URL) throws -> [String: String] {
        guard try Posix.entryKind(root.standardizedFileURL.path) == .directory else {
            throw ImportError("History conversation is not a regular directory: \(root.path).")
        }
        let root = root.standardizedFileURL.resolvingSymlinksInPath()
        let keys: [URLResourceKey] = [.isSymbolicLinkKey, .isDirectoryKey, .isRegularFileKey]
        var enumerationError: Error?
        guard let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys,
                                                              errorHandler: { _, error in enumerationError = error; return false }) else {
            throw ImportError("Cannot enumerate history at \(root.path).")
        }
        var result: [String: String] = [:]
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: Set(keys))
            guard values.isSymbolicLink != true else { throw ImportError("History import refuses symlinks: \(url.path).") }
            if values.isDirectory == true { continue }
            guard values.isRegularFile == true else { throw ImportError("History import refuses special files: \(url.path).") }
            let components = url.standardizedFileURL.resolvingSymlinksInPath().pathComponents
            guard components.starts(with: root.pathComponents), components.count > root.pathComponents.count else {
                throw ImportError("History file escaped the source directory: \(url.path)")
            }
            let relative = components.dropFirst(root.pathComponents.count).joined(separator: "/")
            result[relative] = try Sha256Verifier.hashFile(at: url, chunkBytes: 65_536)
        }
        if let enumerationError { throw enumerationError }
        guard result["conversation.json"] != nil, result["transcript.jsonl"] != nil else {
            throw ImportError("History conversation is incomplete: \(root.path); files=\(result.keys.sorted()).")
        }
        return result
    }

    struct ImportError: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }
}
