import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelCatalogTests {
    private func makeEntry(repoID: String,
                           tier: ModelTrustTier = .custom) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Test Model",
            repoID: repoID,
            revision: "main",
            trustTier: tier,
            recordedIndexSHA256: nil,
            approximateDownloadBytes: 1_000,
            installedBytes: 900,
            reserveBytes: 100)
    }

    @Test func curatedListContainsThePinnedGemmaModel() {
        let pinned = ModelCatalog.curated.first {
            $0.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit"
        }
        #expect(pinned != nil)
        #expect(pinned?.trustTier == .curated)
        #expect(pinned?.recordedIndexSHA256?.isEmpty == false)
    }

    @Test func mergesCuratedAndCustomEntries() {
        let catalog = ModelCatalog(curated: [makeEntry(repoID: "owner/curated", tier: .curated)],
                                   custom: [makeEntry(repoID: "owner/custom")])
        #expect(catalog.entries.count == 2)
    }

    @Test func curatedWinsOnRepositoryIDCollision() {
        let catalog = ModelCatalog(curated: [makeEntry(repoID: "owner/shared", tier: .curated)],
                                   custom: [makeEntry(repoID: "owner/shared")])
        #expect(catalog.entries.count == 1)
        #expect(catalog.entry(forRepoID: "owner/shared")?.trustTier == .curated)
    }

    @Test func listsCuratedEntriesBeforeCustomOnes() {
        let catalog = ModelCatalog(curated: [makeEntry(repoID: "owner/curated", tier: .curated)],
                                   custom: [makeEntry(repoID: "owner/aaa-custom")])
        #expect(catalog.entries.first?.trustTier == .curated)
    }

    @Test func addingCustomRejectsCuratedCollision() {
        let catalog = ModelCatalog(curated: [makeEntry(repoID: "owner/shared", tier: .curated)],
                                   custom: [])
        #expect(throws: ModelCatalog.DuplicateEntry.self) {
            try catalog.addingCustom(makeEntry(repoID: "owner/shared"))
        }
    }

    @Test func addingCustomRejectsExistingCustomEntry() {
        let catalog = ModelCatalog(curated: [], custom: [makeEntry(repoID: "owner/model")])
        #expect(throws: ModelCatalog.DuplicateEntry.self) {
            try catalog.addingCustom(makeEntry(repoID: "owner/model"))
        }
    }

    @Test func addingCustomReturnsCatalogContainingTheNewEntry() throws {
        let catalog = try ModelCatalog(curated: [], custom: [])
            .addingCustom(makeEntry(repoID: "owner/model"))
        #expect(catalog.entry(forRepoID: "owner/model")?.trustTier == .custom)
    }
}
