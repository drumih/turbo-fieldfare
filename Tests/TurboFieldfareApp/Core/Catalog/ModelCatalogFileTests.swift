import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelCatalogFileTests {
    private func makeEntry(repoID: String,
                           tier: ModelTrustTier = .custom,
                           sha: String? = "abc123") -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Test Model",
            repoID: repoID,
            revision: "main",
            trustTier: tier,
            recordedIndexSHA256: sha,
            approximateDownloadBytes: 1_000,
            installedBytes: 900,
            reserveBytes: 100)
    }

    @Test func roundTripsThroughJSON() throws {
        let file = ModelCatalogFile(customEntries: [makeEntry(repoID: "owner/model")])
        let encoded = try JSONEncoder().encode(file)
        let decoded = try JSONDecoder().decode(ModelCatalogFile.self, from: encoded)
        #expect(decoded == file)
    }

    @Test func identityIsRepositoryID() {
        #expect(makeEntry(repoID: "owner/model").id == "owner/model")
    }

    @Test func rejectsDuplicateRepositoryIDs() {
        let file = ModelCatalogFile(customEntries: [
            makeEntry(repoID: "owner/model"),
            makeEntry(repoID: "owner/model"),
        ])
        #expect(file.isValid() == false)
    }

    @Test func rejectsCuratedEntriesInTheCustomList() {
        let file = ModelCatalogFile(customEntries: [
            makeEntry(repoID: "owner/model", tier: .curated),
        ])
        #expect(file.isValid() == false)
    }

    @Test func rejectsUnknownVersion() {
        var file = ModelCatalogFile(customEntries: [])
        file.version = 99
        #expect(file.isValid() == false)
    }

    @Test func acceptsWellFormedFile() {
        let file = ModelCatalogFile(customEntries: [
            makeEntry(repoID: "owner/one"),
            makeEntry(repoID: "owner/two"),
        ])
        #expect(file.isValid())
    }
}
