import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelTrustPolicyTests {
    private func makeEntry(tier: ModelTrustTier,
                           sha: String?) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Test Model",
            repoID: "owner/model",
            revision: "main",
            trustTier: tier,
            recordedIndexSHA256: sha,
            approximateDownloadBytes: 1_000,
            installedBytes: 900,
            reserveBytes: 100)
    }

    @Test func curatedWithMatchingFingerprintIsAllowed() {
        let decision = ModelTrustPolicy.decide(
            entry: makeEntry(tier: .curated, sha: "aaa"),
            observedIndexSHA256: "aaa")
        #expect(decision == .allowCurated)
    }

    @Test func curatedWithMismatchedFingerprintIsRejected() {
        let decision = ModelTrustPolicy.decide(
            entry: makeEntry(tier: .curated, sha: "aaa"),
            observedIndexSHA256: "bbb")
        #expect(decision == .curatedFingerprintMismatch(expected: "aaa", observed: "bbb"))
    }

    @Test func customWithNoRecordedFingerprintNeedsConsent() {
        let decision = ModelTrustPolicy.decide(
            entry: makeEntry(tier: .custom, sha: nil),
            observedIndexSHA256: "bbb")
        #expect(decision == .needsConsent(observed: "bbb"))
    }

    @Test func customWithMatchingRecordedFingerprintIsAllowed() {
        let decision = ModelTrustPolicy.decide(
            entry: makeEntry(tier: .custom, sha: "bbb"),
            observedIndexSHA256: "bbb")
        #expect(decision == .allowCustom)
    }

    @Test func customWithChangedFingerprintIsBlocked() {
        let decision = ModelTrustPolicy.decide(
            entry: makeEntry(tier: .custom, sha: "bbb"),
            observedIndexSHA256: "ccc")
        #expect(decision == .sourceChanged(recorded: "bbb", observed: "ccc"))
    }

    @Test func onlyCuratedEntriesRequireAKnownSource() {
        #expect(ModelTrustPolicy.requiresKnownSource(for: makeEntry(tier: .curated, sha: "a")))
        #expect(ModelTrustPolicy.requiresKnownSource(
            for: makeEntry(tier: .custom, sha: "a")) == false)
    }
}
