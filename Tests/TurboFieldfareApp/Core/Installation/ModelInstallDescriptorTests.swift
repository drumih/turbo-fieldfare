import Foundation
import Testing
import TurboFieldfareRepackCore

@testable import TurboFieldfareAppCore

@Suite struct ModelInstallDescriptorTests {
    private func makeEntry(tier: ModelTrustTier) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Custom Gemma",
            repoID: "owner/gemma-4-26b-a4b-uncensored",
            revision: "abc123",
            trustTier: tier,
            recordedIndexSHA256: "deadbeef",
            approximateDownloadBytes: 2_000,
            installedBytes: 1_500,
            reserveBytes: 500)
    }

    @Test func copiesEveryFieldFromTheCatalogEntry() {
        let descriptor = AppModelInstallDescriptor(entry: makeEntry(tier: .custom))
        #expect(descriptor.displayName == "Custom Gemma")
        #expect(descriptor.repoID == "owner/gemma-4-26b-a4b-uncensored")
        #expect(descriptor.revision == "abc123")
        #expect(descriptor.sourceIndexSHA256 == "deadbeef")
        #expect(descriptor.approximateDownloadBytes == 2_000)
        #expect(descriptor.installedBytes == 1_500)
        #expect(descriptor.reserveBytes == 500)
    }

    @Test func usesAnEmptyFingerprintForAnUnrecordedCustomEntry() {
        let entry = ModelCatalogEntry(
            displayName: "Fresh",
            repoID: "owner/fresh",
            revision: "main",
            trustTier: .custom,
            recordedIndexSHA256: nil,
            approximateDownloadBytes: 1,
            installedBytes: 1,
            reserveBytes: 1)
        #expect(AppModelInstallDescriptor(entry: entry).sourceIndexSHA256 == "")
    }

    @Test func requiredFreeBytesIncludesStagingAndReserve() {
        let descriptor = AppModelInstallDescriptor(entry: makeEntry(tier: .custom))
        let expected = UInt64(1_500) + UInt64(RemoteChunkPolicy.defaultBytes) + UInt64(500)
        #expect(descriptor.requiredFreeBytes == expected)
    }

    @Test func curatedEntriesRequireAKnownSource() {
        #expect(ModelTrustPolicy.requiresKnownSource(for: makeEntry(tier: .curated)))
    }

    @Test func customEntriesDoNotRequireAKnownSource() {
        #expect(ModelTrustPolicy.requiresKnownSource(for: makeEntry(tier: .custom)) == false)
    }

    @Test func theCuratedGemmaEntryMatchesTheLegacyDefaultDescriptor() throws {
        let curated = try #require(ModelCatalog.curated.first)
        let descriptor = AppModelInstallDescriptor(entry: curated)
        #expect(descriptor.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(descriptor.installedBytes == 14_291_921_884)
        #expect(descriptor.approximateDownloadBytes == 14_620_479_420)
    }
}

@Suite struct RepackOptionsForEntryTests {
    private let outputDirectory = URL(fileURLWithPath: "/tmp/model.gturbo", isDirectory: true)

    private func makeEntry(tier: ModelTrustTier) -> ModelCatalogEntry {
        ModelCatalogEntry(
            displayName: "Entry",
            repoID: "owner/model",
            revision: "rev1",
            trustTier: tier,
            recordedIndexSHA256: "abc",
            approximateDownloadBytes: 10,
            installedBytes: 9,
            reserveBytes: 7)
    }

    /// The security-relevant bit: a curated entry must still be checked against
    /// the project's pinned fingerprint, and a custom one must not be — the
    /// whole point of the custom tier is that no such pin exists.
    @Test func curatedEntriesRepackWithKnownSourceRequired() {
        let options = RepackModelInstallerClient.repackOptions(
            entry: makeEntry(tier: .curated),
            outputDirectory: outputDirectory,
            token: nil,
            resume: false)
        #expect(options.requireKnownSource)
    }

    @Test func customEntriesRepackWithoutKnownSourceRequired() {
        let options = RepackModelInstallerClient.repackOptions(
            entry: makeEntry(tier: .custom),
            outputDirectory: outputDirectory,
            token: nil,
            resume: false)
        #expect(options.requireKnownSource == false)
    }

    @Test func optionsCarryTheEntrysRepositoryRevisionAndReserve() {
        let options = RepackModelInstallerClient.repackOptions(
            entry: makeEntry(tier: .custom),
            outputDirectory: outputDirectory,
            token: "hf_token",
            resume: true)
        #expect(options.repoID == "owner/model")
        #expect(options.revision == "rev1")
        #expect(options.outputDir == outputDirectory.path)
        #expect(options.token == "hf_token")
        #expect(options.minFreeReserveBytes == 7)
        #expect(options.resume)
    }
}
