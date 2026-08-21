import Foundation
import Testing

@testable import TurboFieldfareRepackCore

@Suite
struct ModelSourceDescriptorTests {
    /// The legacy `SupportedModelSource` constants are part of the installer's
    /// public surface. They must keep resolving to the default descriptor.
    @Test func legacyAccessorsMatchDefaultDescriptor() {
        let expected = SupportedModelSource.default

        #expect(SupportedModelSource.displayName == expected.displayName)
        #expect(SupportedModelSource.repoID == expected.repoID)
        #expect(SupportedModelSource.revision == expected.revision)
        #expect(SupportedModelSource.sourceIndexSHA256 == expected.sourceIndexSHA256)
        #expect(SupportedModelSource.approximateDownloadBytes == expected.approximateDownloadBytes)
        #expect(SupportedModelSource.installedBytes == expected.installedBytes)
        #expect(SupportedModelSource.reserveBytes == expected.reserveBytes)
    }

    /// The pinned Gemma values are a compatibility contract: a changed repo,
    /// revision, or index digest silently redirects every install.
    @Test func gemmaDescriptorKeepsPinnedIdentity() {
        let gemma = SupportedModelSource.gemma4

        #expect(gemma.key == "gemma4")
        #expect(gemma.family == .gemma4)
        #expect(gemma.displayName == "Gemma 4 26B-A4B IT 4-bit")
        #expect(gemma.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(gemma.revision == "0d77464eeb233a2da68ebf9d7dc4edaac7db956d")
        #expect(gemma.sourceIndexSHA256 ==
                "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13")
        #expect(SupportedModelSource.default == gemma)
    }

    @Test func descriptorLookupResolvesKnownKeysOnly() {
        #expect(SupportedModelSource.descriptor(forKey: "gemma4") == SupportedModelSource.gemma4)
        #expect(SupportedModelSource.descriptor(forKey: "nonexistent") == nil)
        #expect(SupportedModelSource.descriptor(forKey: "") == nil)
    }

    @Test func descriptorKeysAreUnique() {
        let keys = SupportedModelSource.all.map(\.key)
        #expect(Set(keys).count == keys.count)
    }

    /// `knownFingerprints` is derived with `uniqueKeysWithValues`, which traps
    /// on a duplicate repo ID. Keep that failure in the suite, not an install.
    @Test func descriptorRepoIDsAreUnique() {
        let repoIDs = SupportedModelSource.all.map(\.repoID)
        #expect(Set(repoIDs).count == repoIDs.count)
    }

    @Test func everyDescriptorIsRecognisedByFingerprint() {
        for descriptor in SupportedModelSource.all {
            #expect(SourceFingerprint.knownFingerprints[descriptor.repoID] ==
                    descriptor.sourceIndexSHA256)
            #expect(SourceFingerprint.modelID(forIndexSha256: descriptor.sourceIndexSHA256) ==
                    descriptor.repoID)
        }
        #expect(SourceFingerprint.knownFingerprints.count == SupportedModelSource.all.count)
    }

    @Test func installOptionsCarryDescriptorPinning() {
        let descriptor = SupportedModelSource.gemma4
        let directory = URL(fileURLWithPath: "/tmp/turbo-fieldfare-descriptor-test")

        let options = descriptor.installOptions(outputDirectory: directory,
                                                overwrite: true,
                                                token: nil,
                                                resume: true)

        #expect(options.repoID == descriptor.repoID)
        #expect(options.revision == descriptor.revision)
        #expect(options.outputDir == directory.path)
        #expect(options.minFreeReserveBytes == descriptor.reserveBytes)
        #expect(options.requireKnownSource)
        #expect(options.overwrite)
        #expect(options.resume)
    }

    /// The legacy entry point must stay behaviourally identical to the
    /// descriptor call it now forwards to.
    @Test func legacyInstallOptionsMatchDescriptorInstallOptions() {
        let directory = URL(fileURLWithPath: "/tmp/turbo-fieldfare-descriptor-parity")

        let legacy = SupportedModelSource.installOptions(outputDirectory: directory,
                                                         overwrite: false,
                                                         token: "token",
                                                         resume: false)
        let viaDescriptor = SupportedModelSource.default.installOptions(
            outputDirectory: directory,
            overwrite: false,
            token: "token",
            resume: false)

        #expect(legacy.repoID == viaDescriptor.repoID)
        #expect(legacy.revision == viaDescriptor.revision)
        #expect(legacy.outputDir == viaDescriptor.outputDir)
        #expect(legacy.token == viaDescriptor.token)
        #expect(legacy.minFreeReserveBytes == viaDescriptor.minFreeReserveBytes)
        #expect(legacy.requireKnownSource == viaDescriptor.requireKnownSource)
        #expect(legacy.overwrite == viaDescriptor.overwrite)
        #expect(legacy.resume == viaDescriptor.resume)
    }
}
