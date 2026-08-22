import Darwin
import Foundation
import Testing
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

// Nested inside RemotePayloadCopyTests so it inherits that suite's
// serialization: FakeHFURLProtocol keeps its stub state in statics, and
// .serialized only orders tests within one suite, so two top-level suites
// sharing the stub raced and clobbered each other.
extension RemotePayloadCopyTests {
@Suite(.serialized)
struct VisionPackPrepareActivationTests {
    @Test func preparedPackWaitsForRuntimeThenActivatesWithoutDownload() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }

        #expect(RemoteVisionPackInstaller.preparedInstallIsValid(
            outputDirectory: fixture.output.path,
            textModelDirectory: fixture.text.path,
            verifyWeights: true))

        let useDescriptor = try acquireSharedUseLock(fixture)
        #expect(throws: RepackError.self) {
            try activate(fixture)
        }
        #expect(FileManager.default.fileExists(atPath: fixture.paths.partialDirectory))
        #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
        _ = flock(useDescriptor, LOCK_UN)

        try activate(fixture)
        #expect(FileManager.default.fileExists(atPath: fixture.output.path))
        #expect(!FileManager.default.fileExists(atPath: fixture.paths.partialDirectory))
        #expect(!FileManager.default.fileExists(atPath: fixture.paths.checkpointFile))

        #expect(flock(useDescriptor, LOCK_SH | LOCK_NB) == 0)
        _ = try VisionPackVerifier.verify(
            directory: fixture.output,
            textModelDirectory: fixture.text)
        #expect(throws: RepackError.self) {
            try RemoteVisionPackInstaller.removeInstalled(
                outputDirectory: fixture.output.path)
        }
        #expect(FileManager.default.fileExists(atPath: fixture.output.path))
        _ = flock(useDescriptor, LOCK_UN)
        close(useDescriptor)

        try RemoteVisionPackInstaller.removeInstalled(
            outputDirectory: fixture.output.path)
        #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
        #expect(FileManager.default.fileExists(atPath: fixture.text.path))
    }

    @Test(arguments: ["repo", "commit", "index"])
    func activationRejectsStaleCheckpointSourceIdentity(field: String) throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        try fixture.writeCheckpoint(
            resolvedCommit: field == "commit" ? String(repeating: "d", count: 40) : fixture.revision,
            sourceIndexSHA256: field == "index" ? String(repeating: "e", count: 64) : fixture.sourceIndex)

        #expect(throws: RepackError.self) {
            if field == "repo" {
                try RemoteVisionPackInstaller.activatePrepared(
                    outputDirectory: fixture.output.path,
                    textModelDirectory: fixture.text.path,
                    repoID: "fixture/other",
                    requestedRevision: fixture.requestedRevision)
            } else {
                try activate(fixture)
            }
        }
        #expect(FileManager.default.fileExists(atPath: fixture.paths.partialDirectory))
        #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
    }

    @Test func corruptPayloadRequiresRepairInsteadOfActivationRetry() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        let weightsPath = (fixture.paths.partialDirectory as NSString)
            .appendingPathComponent(GTurboVisionFormatV1.weightsFile)
        let descriptor = try Posix.openExistingRW(weightsPath)
        defer { close(descriptor) }
        var byte: UInt8 = 1
        try withUnsafeBytes(of: &byte) {
            try Posix.pwriteAll(
                fd: descriptor,
                path: weightsPath,
                buf: $0.baseAddress!,
                count: 1,
                offset: 0)
        }

        #expect(RemoteVisionPackInstaller.preparedInstallIsValid(
            outputDirectory: fixture.output.path,
            textModelDirectory: fixture.text.path))
        #expect(!RemoteVisionPackInstaller.preparedInstallIsValid(
            outputDirectory: fixture.output.path,
            textModelDirectory: fixture.text.path,
            verifyWeights: true))
        #expect(throws: RepackError.self) { try activate(fixture) }
        #expect(FileManager.default.fileExists(atPath: fixture.paths.partialDirectory))
    }

    @Test(arguments: ["partial-path", "other-path", "other-manifest"])
    func receiptBindingMismatchIsRejected(kind: String) throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        let receiptURL = URL(fileURLWithPath: fixture.paths.partialDirectory)
            .appendingPathComponent(GTurboVisionFormatV1.receiptFile)
        let original = try GTurboVisionReceiptCodec.decode(
            Posix.readBoundedData(
                receiptURL.path,
                maximumBytes: GTurboVisionFormatV1.metadataMaxBytes))
        let receipt = GTurboVisionReceiptV1(
            manifestSha256: kind == "other-manifest"
                ? String(repeating: "f", count: 64)
                : original.manifestSha256,
            companionDirectoryPath: kind == "partial-path"
                ? fixture.paths.partialDirectory
                : kind == "other-path"
                    ? fixture.output.path + ".other"
                    : original.companionDirectoryPath,
            compatibleTextManifestSha256: original.compatibleTextManifestSha256,
            sourceRepoID: original.sourceRepoID,
            sourceRevision: original.sourceRevision,
            verificationTimestamp: original.verificationTimestamp,
            toolVersion: original.toolVersion,
            files: original.files)
        try GTurboVisionReceiptCodec.encode(receipt).write(to: receiptURL)

        #expect(throws: RepackError.self) {
            _ = try VisionPackVerifier.verify(
                directory: URL(fileURLWithPath: fixture.paths.partialDirectory),
                installedDirectory: fixture.output,
                textModelDirectory: fixture.text)
        }
    }

    @Test func changedTextManifestInvalidatesPreparedPack() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        let manifestURL = fixture.text.appendingPathComponent("manifest.json")
        var data = try Data(contentsOf: manifestURL)
        data.append(0x20)
        try data.write(to: manifestURL)

        #expect(!RemoteVisionPackInstaller.preparedInstallIsValid(
            outputDirectory: fixture.output.path,
            textModelDirectory: fixture.text.path))
        #expect(throws: RepackError.self) { try activate(fixture) }
        #expect(FileManager.default.fileExists(atPath: fixture.paths.partialDirectory))
    }

    @Test func rerunFinishesInterruptedPostRenameCommit() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        try Posix.rename(
            from: fixture.paths.partialDirectory,
            to: fixture.paths.finalDirectory)

        try activate(fixture)

        #expect(FileManager.default.fileExists(atPath: fixture.output.path))
        #expect(!FileManager.default.fileExists(atPath: fixture.paths.checkpointFile))
        _ = try VisionPackVerifier.verify(
            directory: fixture.output,
            textModelDirectory: fixture.text)
    }

    @Test func removalClearsInstalledPackAndSavedRepairButNotText() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        try FileManager.default.copyItem(
            atPath: fixture.paths.partialDirectory,
            toPath: fixture.paths.finalDirectory)

        try RemoteVisionPackInstaller.removeInstalled(
            outputDirectory: fixture.output.path)

        #expect(!FileManager.default.fileExists(atPath: fixture.paths.finalDirectory))
        #expect(!FileManager.default.fileExists(atPath: fixture.paths.partialDirectory))
        #expect(!FileManager.default.fileExists(atPath: fixture.paths.checkpointFile))
        #expect(FileManager.default.fileExists(atPath: fixture.text.path))
    }

    @Test(arguments: ["partial-only", "checkpoint-only"])
    func inconsistentSavedStateFailsBeforeNetwork(kind: String) async throws {
        let parent = try temporaryParent("vision-inconsistent")
        defer { try? FileManager.default.removeItem(at: parent) }
        resetFakeHF()
        let output = parent.appendingPathComponent("model.vision.gturbo")
        let paths = try RemoteInstallPaths(outputDirectory: output.path)
        if kind != "checkpoint-only" {
            try FileManager.default.createDirectory(
                atPath: paths.partialDirectory,
                withIntermediateDirectories: true)
        }
        if kind != "partial-only" {
            try RemoteInstallCheckpoint(
                repoID: "fixture/model",
                requestedRevision: "pinned",
                resolvedCommit: String(repeating: "a", count: 40),
                sourceIndexSHA256: String(repeating: "b", count: 64),
                planFingerprint: String(repeating: "c", count: 64),
                totalSourceBytes: 1).write(
                    to: paths.checkpointFile,
                    parentDirectory: paths.parentDirectory)
        }
        let installer = RemoteVisionPackInstaller(options: .init(
            repoID: "fixture/model",
            revision: "pinned",
            textModelDirectory: parent.appendingPathComponent("text.gturbo").path,
            outputDirectory: output.path,
            minFreeReserveBytes: 0,
            overwrite: true,
            resume: true,
            downloadSession: fakeHFSession(),
            baseURL: URL(string: "https://hf.test")!))

        await #expect(throws: RepackError.self) {
            _ = try await installer.prepare()
        }
        #expect(FakeHFURLProtocol.requestCounts.isEmpty)
    }

    /// The partial directory is created before the checkpoint that identifies
    /// its source, so a crash in that window leaves a directory carrying no
    /// recoverable progress. Reporting it as corrupt state blocked even a fresh
    /// install; it has to be reclaimed instead.
    @Test func partialDirectoryWithoutACheckpointIsReclaimed() async throws {
        let parent = try temporaryParent("vision-orphan-partial")
        defer { try? FileManager.default.removeItem(at: parent) }
        resetFakeHF()
        let output = parent.appendingPathComponent("model.vision.gturbo")
        let paths = try RemoteInstallPaths(outputDirectory: output.path)
        try FileManager.default.createDirectory(
            atPath: paths.partialDirectory, withIntermediateDirectories: true)
        try Data("half a download".utf8).write(to: URL(fileURLWithPath:
            (paths.partialDirectory as NSString).appendingPathComponent(".range.tmp")))

        let installer = RemoteVisionPackInstaller(options: .init(
            repoID: "fixture/model",
            revision: "pinned",
            textModelDirectory: parent.appendingPathComponent("text.gturbo").path,
            outputDirectory: output.path,
            minFreeReserveBytes: 0,
            overwrite: true,
            resume: false,
            downloadSession: fakeHFSession(),
            baseURL: URL(string: "https://hf.test")!))

        // It still fails — there is no text model to bind to — but it must get
        // past the saved-state check rather than stopping at it.
        do {
            _ = try await installer.prepare()
            Issue.record("the fixture has no text model, so this cannot succeed")
        } catch let error as RepackError {
            if case .installStateCorrupt = error {
                Issue.record("orphaned partial directory still blocks a fresh install")
            }
        }
    }

    @Test func corruptCheckpointFailsWithoutMutatingPreparedPack() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        try Data("not-json".utf8).write(
            to: URL(fileURLWithPath: fixture.paths.checkpointFile))

        #expect(throws: RepackError.self) { try activate(fixture) }
        #expect(FileManager.default.fileExists(atPath: fixture.paths.partialDirectory))
        #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
    }

    /// The partial directory doubles as download scratch, and the verifier
    /// accepts the four pack entries and nothing else. A crash between the last
    /// write and the installer's own cleanup therefore left a `.range.tmp` or
    /// an interrupted `atomicWrite`'s `.tmp` beside them, and a complete 1.5 GB
    /// download became permanently unactivatable.
    @Test(arguments: [".range.tmp", "manifest.json.tmp", "turbofieldfare-range-x.tmp"])
    func activationReclaimsLeftoverDownloadScratch(name: String) throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        let stray = (fixture.paths.partialDirectory as NSString)
            .appendingPathComponent(name)
        try Data("scratch".utf8).write(to: URL(fileURLWithPath: stray))

        try activate(fixture)
        #expect(FileManager.default.fileExists(atPath: fixture.output.path))
        #expect(!FileManager.default.fileExists(atPath: stray))
        #expect(!FileManager.default.fileExists(atPath: fixture.paths.partialDirectory))
    }

    /// The metadata cache is a directory, not a file, and is re-fetched on
    /// every run, so it is scratch too.
    @Test func activationReclaimsTheMetadataCacheDirectory() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        try FileManager.default.createDirectory(
            atPath: fixture.paths.metadataDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: URL(fileURLWithPath:
            (fixture.paths.metadataDirectory as NSString)
                .appendingPathComponent("config.json")))

        try activate(fixture)
        #expect(FileManager.default.fileExists(atPath: fixture.output.path))
    }

    /// Reclaiming must not touch the pack itself.
    @Test func reclaimingScratchKeepsEveryPackEntry() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        let stray = (fixture.paths.partialDirectory as NSString)
            .appendingPathComponent(".range.tmp")
        try Data("scratch".utf8).write(to: URL(fileURLWithPath: stray))

        let reclaimed = try RemoteVisionPackInstaller.reclaimPartialScratch(
            directory: fixture.paths.partialDirectory)
        #expect(reclaimed == [".range.tmp"])
        let remaining = Set(try FileManager.default.contentsOfDirectory(
            atPath: fixture.paths.partialDirectory))
        #expect(remaining == RemoteVisionPackInstaller.packEntries)
    }

    /// Refusing an install over a pack that is already there used to end the
    /// sentence at "already exists", which is the state a failed repair leaves
    /// people in - with a saved partial sitting right next to it and no way
    /// forward printed.
    @Test func refusingAnExistingPackSaysHowToProceed() async throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        try activate(fixture)
        #expect(FileManager.default.fileExists(atPath: fixture.output.path))

        do {
            _ = try await RemoteVisionPackInstaller(
                options: installOptions(fixture, overwrite: false)).prepare()
            Issue.record("expected the existing pack to refuse the install")
        } catch let error as RepackError {
            let message = error.description
            #expect(message.contains("--overwrite"),
                    "the refusal has to name the flag that continues: \(message)")
        }
    }

    /// The lock that decides activation was only taken at the end, so a loaded
    /// model was discovered after the whole 1.09 GB transfer. Probing it first
    /// turns that into a refusal before the first byte.
    @Test func aLoadedModelIsFoundBeforeTheTransferNotAfterIt() async throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        try activate(fixture)
        let useDescriptor = try acquireSharedUseLock(fixture)
        defer {
            _ = flock(useDescriptor, LOCK_UN)
            close(useDescriptor)
        }

        do {
            _ = try await RemoteVisionPackInstaller(
                options: installOptions(fixture, overwrite: true)).prepare()
            Issue.record("expected the held use lease to refuse the install")
        } catch let error as RepackError {
            guard case .installBusy = error else {
                Issue.record("expected installBusy, got \(error)")
                return
            }
            #expect(error.description.contains("loaded model"))
        }
        // Nothing was downloaded: the pack on disk is untouched and no partial
        // was created beside it.
        #expect(!FileManager.default.fileExists(atPath: fixture.paths.partialDirectory))
        #expect(FileManager.default.fileExists(atPath: fixture.output.path))
    }

    private func installOptions(
        _ fixture: VisionFixture, overwrite: Bool
    ) -> RemoteVisionPackInstallOptions {
        RemoteVisionPackInstallOptions(
            repoID: fixture.repoID,
            revision: fixture.requestedRevision,
            textModelDirectory: fixture.text.path,
            outputDirectory: fixture.output.path,
            overwrite: overwrite)
    }

    private func activate(_ fixture: VisionFixture) throws {
        try RemoteVisionPackInstaller.activatePrepared(
            outputDirectory: fixture.output.path,
            textModelDirectory: fixture.text.path,
            repoID: fixture.repoID,
            requestedRevision: fixture.requestedRevision)
    }

    private func acquireSharedUseLock(_ fixture: VisionFixture) throws -> Int32 {
        let path = fixture.parent.appendingPathComponent(
            ".model.vision.gturbo.use.lock").path
        let descriptor = open(path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0, flock(descriptor, LOCK_SH | LOCK_NB) == 0 else {
            if descriptor >= 0 { close(descriptor) }
            throw RepackError.fileOpenFailed(path: path, errno: errno)
        }
        return descriptor
    }

    private func makeFixture() throws -> VisionFixture {
        let parent = try temporaryParent("vision-prepare-activate")
        let text = parent.appendingPathComponent("model.gturbo", isDirectory: true)
        let output = parent.appendingPathComponent("model.vision.gturbo", isDirectory: true)
        let paths = try RemoteInstallPaths(outputDirectory: output.path)
        try FileManager.default.createDirectory(
            atPath: paths.partialDirectory,
            withIntermediateDirectories: true)

        let textManifest = GTurboManifestV1(
            flags: [:], modelID: "fixture/text", sourceSnapshotHash: "text-snapshot",
            arch: GTurboManifestArchV1(
                hiddenSize: 1, ffnIntermediate: 1, moeIntermediateSize: 1,
                numHeads: 1, numKVHeads: 1, numFullKVHeads: 1,
                headDim: 1, fullHeadDim: 1, vocabSize: 1,
                slidingWindow: 1, finalLogitSoftcap: 1,
                ropeTheta: 1, fullRopeTheta: 1, partialRotaryFactor: 1,
                numLayers: 1, numExperts: 1, topKExperts: 1,
                tieWordEmbeddings: true, attentionKEqV: true,
                hiddenActivation: "gelu", fullAttentionLayerMask: [0]),
            quant: nil, files: [:], expertsPerLayer: 1,
            numLayers: 1, expertStride: 16_384,
            bitWidthOverridesHonored: nil)
        let textManifestData = try GTurboManifestCodec.encode(textManifest)
        try FileManager.default.createDirectory(at: text, withIntermediateDirectories: false)
        try textManifestData.write(to: text.appendingPathComponent("manifest.json"))
        let textSHA = hash(textManifestData)

        let weights = Data(repeating: 0, count: 16_384)
        let processor = Data("{}".utf8)
        let files = [
            GTurboVisionFormatV1.weightsFile: GTurboManifestFileV1(
                size: UInt64(weights.count), sha256: hash(weights)),
            GTurboVisionFormatV1.processorFile: GTurboManifestFileV1(
                size: UInt64(processor.count), sha256: hash(processor)),
        ]
        let revision = String(repeating: "a", count: 40)
        let sourceIndex = String(repeating: "b", count: 64)
        let manifest = GTurboVisionManifestV1(
            modelID: "fixture/model", sourceRevision: revision,
            sourceIndexSha256: sourceIndex,
            processorConfigSha256: hash(processor),
            compatibleTextSourceSnapshotHash: "text-snapshot",
            compatibleTextManifestSha256: textSHA,
            files: files,
            tensors: [.init(
                name: "vision.weight", executionPosition: 0,
                offset: 0, size: 2, shape: [1], dtype: .bf16)])
        let manifestData = try GTurboVisionManifestCodec.encode(manifest)
        let manifestSHA = hash(manifestData)
        let receipt = GTurboVisionReceiptV1(
            manifestSha256: manifestSHA,
            companionDirectoryPath: output.standardizedFileURL.path,
            compatibleTextManifestSha256: textSHA,
            sourceRepoID: "fixture/model", sourceRevision: revision,
            verificationTimestamp: "fixture", toolVersion: "fixture",
            files: files.merging([
                GTurboVisionFormatV1.manifestFile: GTurboManifestFileV1(
                    size: UInt64(manifestData.count), sha256: manifestSHA)
            ]) { _, new in new })
        let partial = URL(fileURLWithPath: paths.partialDirectory)
        try weights.write(to: partial.appendingPathComponent(GTurboVisionFormatV1.weightsFile))
        try processor.write(to: partial.appendingPathComponent(GTurboVisionFormatV1.processorFile))
        try manifestData.write(to: partial.appendingPathComponent(GTurboVisionFormatV1.manifestFile))
        try GTurboVisionReceiptCodec.encode(receipt).write(
            to: partial.appendingPathComponent(GTurboVisionFormatV1.receiptFile))
        let fixture = VisionFixture(
            parent: parent, text: text, output: output, paths: paths,
            repoID: "fixture/model", requestedRevision: "pinned",
            revision: revision, sourceIndex: sourceIndex)
        try fixture.writeCheckpoint(
            resolvedCommit: revision,
            sourceIndexSHA256: sourceIndex)
        return fixture
    }

    private func temporaryParent(_ tag: String) throws -> URL {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(tag)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        return parent
    }

    private func hash(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }
}
}

private struct VisionFixture {
    let parent: URL
    let text: URL
    let output: URL
    let paths: RemoteInstallPaths
    let repoID: String
    let requestedRevision: String
    let revision: String
    let sourceIndex: String

    func writeCheckpoint(resolvedCommit: String, sourceIndexSHA256: String) throws {
        try RemoteInstallCheckpoint(
            repoID: repoID,
            requestedRevision: requestedRevision,
            resolvedCommit: resolvedCommit,
            sourceIndexSHA256: sourceIndexSHA256,
            planFingerprint: String(repeating: "c", count: 64),
            totalSourceBytes: 2).write(
                to: paths.checkpointFile,
                parentDirectory: paths.parentDirectory)
    }

    func remove() {
        try? FileManager.default.removeItem(at: parent)
    }
}
