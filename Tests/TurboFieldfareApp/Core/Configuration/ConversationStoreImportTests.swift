import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct ConversationStoreImportTests {
    private func roots() throws -> (URL, URL, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("history-import-\(UUID())")
        let source = root.appendingPathComponent("legacy/conversations")
        let destination = root.appendingPathComponent("central/conversations")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        return (root, source, destination)
    }

    private func conversation(in root: URL, id: UUID = UUID(), content: String = "record") throws -> UUID {
        let directory = root.appendingPathComponent(id.uuidString)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("images"), withIntermediateDirectories: true)
        try Data("{\"id\":\"\(id)\"}".utf8).write(to: directory.appendingPathComponent("conversation.json"))
        try Data(content.utf8).write(to: directory.appendingPathComponent("transcript.jsonl"))
        try Data(repeating: 42, count: 131_072).write(to: directory.appendingPathComponent("images/picture.png"))
        return id
    }

    @Test func importPreservesOriginalsAttachmentsIDsAndLockInodeAndIsIdempotent() throws {
        let (root, source, destination) = try roots()
        defer { try? FileManager.default.removeItem(at: root) }
        let id = try conversation(in: source)
        let lockURL = destination.appendingPathComponent(ConversationStoreLock.fileName)
        do { let lock = try ConversationStoreLock.acquire(storeRoot: destination); withExtendedLifetime(lock) {} }
        let inode = try FileManager.default.attributesOfItem(atPath: lockURL.path)[.systemFileNumber] as? NSNumber
        try ConversationStoreImport.run(from: source, to: destination)
        try ConversationStoreImport.run(from: source, to: destination)
        #expect(FileManager.default.fileExists(atPath: source.appendingPathComponent(id.uuidString).path))
        #expect(try Data(contentsOf: source.appendingPathComponent("\(id)/images/picture.png")) == Data(contentsOf: destination.appendingPathComponent("\(id)/images/picture.png")))
        #expect(try FileManager.default.attributesOfItem(atPath: lockURL.path)[.systemFileNumber] as? NSNumber == inode)
        #expect(try FileManager.default.contentsOfDirectory(atPath: destination.path).filter { UUID(uuidString: $0) != nil }.count == 1)
    }

    @Test func conflictingIDRejectsWholeImportBeforePublishing() throws {
        let (root, source, destination) = try roots()
        defer { try? FileManager.default.removeItem(at: root) }
        let conflict = try conversation(in: source)
        _ = try conversation(in: destination, id: conflict, content: "different")
        let other = try conversation(in: source)
        #expect(throws: (any Error).self) { try ConversationStoreImport.run(from: source, to: destination) }
        #expect(!FileManager.default.fileExists(atPath: destination.appendingPathComponent(other.uuidString).path))
        #expect(try String(contentsOf: destination.appendingPathComponent("\(conflict)/transcript.jsonl"), encoding: .utf8) == "different")
    }

    @Test func sourceWriterLockRejectsImport() throws {
        let (root, source, destination) = try roots()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try conversation(in: source)
        let lock = try #require(try ConversationStoreLock.acquire(storeRoot: source))
        #expect(throws: (any Error).self) { try ConversationStoreImport.run(from: source, to: destination) }
        withExtendedLifetime(lock) {}
    }

    @Test func publishFailureRollsBackAndRetryDoesNotDuplicate() throws {
        let (root, source, destination) = try roots()
        defer { try? FileManager.default.removeItem(at: root) }
        let id = try conversation(in: source)
        struct Interrupted: Error {}
        #expect(throws: (any Error).self) {
            try ConversationStoreImport.run(from: source, to: destination, afterPublish: { throw Interrupted() })
        }
        #expect(!FileManager.default.fileExists(atPath: destination.appendingPathComponent(id.uuidString).path))
        try ConversationStoreImport.run(from: source, to: destination)
        #expect(FileManager.default.fileExists(atPath: destination.appendingPathComponent(id.uuidString).path))
    }

    @Test func interruptedPublishJournalRecoversBeforeRetrying() throws {
        let (root, source, destination) = try roots()
        defer { try? FileManager.default.removeItem(at: root) }
        let id = try conversation(in: source)
        let name = id.uuidString
        let original = source.appendingPathComponent(name)
        let published = destination.appendingPathComponent(name)
        try FileManager.default.copyItem(at: original, to: published)
        var files: [String: String] = [:]
        for path in ["conversation.json", "transcript.jsonl", "images/picture.png"] {
            files[path] = try Sha256Verifier.hashFile(at: original.appendingPathComponent(path), chunkBytes: 65_536)
        }
        let journal = ConversationStoreImport.Journal(source: source.path, fingerprints: [name: files])
        try JSONEncoder().encode(journal).write(to: destination.appendingPathComponent(".history-import.json"))
        try ConversationStoreImport.run(from: source, to: destination)
        #expect(!FileManager.default.fileExists(atPath: destination.appendingPathComponent(".history-import.json").path))
        #expect(try FileManager.default.contentsOfDirectory(atPath: destination.path).filter { UUID(uuidString: $0) != nil }.count == 1)
        #expect(FileManager.default.fileExists(atPath: original.path))
    }

    @Test func symlinkAttachmentRefusesImport() throws {
        let (root, source, destination) = try roots()
        defer { try? FileManager.default.removeItem(at: root) }
        let id = try conversation(in: source)
        try FileManager.default.createSymbolicLink(at: source.appendingPathComponent("\(id)/images/escape"), withDestinationURL: root)
        #expect(throws: (any Error).self) { try ConversationStoreImport.run(from: source, to: destination) }
        #expect(!FileManager.default.fileExists(atPath: destination.appendingPathComponent(id.uuidString).path))
    }

    @Test func symlinkConversationCannotQualifyAsAnExactDuplicate() throws {
        let (root, source, destination) = try roots()
        defer { try? FileManager.default.removeItem(at: root) }
        let id = try conversation(in: source)
        let existing = destination.appendingPathComponent(id.uuidString)
        try FileManager.default.createSymbolicLink(at: existing, withDestinationURL: source.appendingPathComponent(id.uuidString))
        #expect(throws: (any Error).self) { try ConversationStoreImport.run(from: source, to: destination) }
        #expect(try existing.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true)
    }

    @Test func symlinkHistoryRootIsRefusedBeforeAnyPublication() throws {
        let (root, source, destination) = try roots()
        defer { try? FileManager.default.removeItem(at: root) }
        let id = try conversation(in: source)
        let alias = root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: source)
        #expect(throws: (any Error).self) { try ConversationStoreImport.run(from: alias, to: destination) }
        #expect(!FileManager.default.fileExists(atPath: destination.appendingPathComponent(id.uuidString).path))
    }
}
