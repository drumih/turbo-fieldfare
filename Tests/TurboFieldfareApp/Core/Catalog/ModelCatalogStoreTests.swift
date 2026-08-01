import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelCatalogStoreTests {
    private func makeSupportDirectory(_ name: String) throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catalog-store-\(name)-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private func makeEntry(repoID: String) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Test Model",
            repoID: repoID,
            revision: "main",
            trustTier: .custom,
            recordedIndexSHA256: "abc123",
            approximateDownloadBytes: 1_000,
            installedBytes: 900,
            reserveBytes: 100)
    }

    @Test func returnsEmptyCatalogWhenFileMissing() throws {
        let directory = try makeSupportDirectory("missing")
        defer { try? FileManager.default.removeItem(at: directory) }

        let loaded = ModelCatalogStore.load(inSupportDirectory: directory)
        #expect(loaded.customEntries.isEmpty)
    }

    @Test func roundTripsThroughDisk() throws {
        let directory = try makeSupportDirectory("round-trip")
        defer { try? FileManager.default.removeItem(at: directory) }

        let file = ModelCatalogFile(customEntries: [makeEntry(repoID: "owner/model")])
        try ModelCatalogStore.save(file, inSupportDirectory: directory)
        #expect(ModelCatalogStore.load(inSupportDirectory: directory) == file)
    }

    @Test func backsUpCorruptFileAndFallsBackToEmpty() throws {
        let directory = try makeSupportDirectory("corrupt")
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = ModelCatalogStore.fileURL(inSupportDirectory: directory)
        try Data("{ not json".utf8).write(to: url)

        let loaded = ModelCatalogStore.load(inSupportDirectory: directory)
        #expect(loaded.customEntries.isEmpty)

        let backup = directory.appendingPathComponent(ModelCatalogStore.corruptFileName)
        #expect(FileManager.default.fileExists(atPath: backup.path))
        #expect(FileManager.default.fileExists(atPath: url.path) == false)
    }

    @Test func backsUpStructurallyInvalidFile() throws {
        let directory = try makeSupportDirectory("invalid")
        defer { try? FileManager.default.removeItem(at: directory) }

        let duplicates = ModelCatalogFile(customEntries: [
            makeEntry(repoID: "owner/model"),
            makeEntry(repoID: "owner/model"),
        ])
        let url = ModelCatalogStore.fileURL(inSupportDirectory: directory)
        try JSONEncoder().encode(duplicates).write(to: url)

        #expect(ModelCatalogStore.load(inSupportDirectory: directory).customEntries.isEmpty)
        let backup = directory.appendingPathComponent(ModelCatalogStore.corruptFileName)
        #expect(FileManager.default.fileExists(atPath: backup.path))
    }

    @Test func refusesToSaveInvalidCatalog() throws {
        let directory = try makeSupportDirectory("refuse")
        defer { try? FileManager.default.removeItem(at: directory) }

        let duplicates = ModelCatalogFile(customEntries: [
            makeEntry(repoID: "owner/model"),
            makeEntry(repoID: "owner/model"),
        ])
        #expect(throws: (any Error).self) {
            try ModelCatalogStore.save(duplicates, inSupportDirectory: directory)
        }
    }
}
