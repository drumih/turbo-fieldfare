import Darwin
import Foundation
import TurboFieldfareFormat

public struct RemoteVisionPackInstallOptions: Sendable {
    public let repoID: String
    public let revision: String
    public let textModelDirectory: String
    public let outputDirectory: String
    public let token: String?
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let overwrite: Bool
    public let resume: Bool
    public let downloadSession: RemoteDownloadSession
    public let baseURL: URL
    public let rangeRetryAttempts: Int
    public let retryBaseDelayNs: UInt64

    public init(
        repoID: String,
        revision: String,
        textModelDirectory: String,
        outputDirectory: String,
        token: String? = nil,
        rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
        writeTileBytes: Int = WriterCore.tileBytes,
        minFreeReserveBytes: UInt64 = 1_073_741_824,
        overwrite: Bool = false,
        resume: Bool = false,
        downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
        baseURL: URL = URL(string: "https://huggingface.co")!,
        rangeRetryAttempts: Int = 4,
        retryBaseDelayNs: UInt64 = 1_000_000_000
    ) {
        self.repoID = repoID
        self.revision = revision
        self.textModelDirectory = textModelDirectory
        self.outputDirectory = outputDirectory
        self.token = token
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.overwrite = overwrite
        self.resume = resume
        self.downloadSession = downloadSession
        self.baseURL = baseURL
        self.rangeRetryAttempts = rangeRetryAttempts
        self.retryBaseDelayNs = retryBaseDelayNs
    }
}

public struct RemoteVisionPackInstallResult: Sendable {
    public let outputDirectory: String
    public let resolvedCommit: String
    public let tensorCount: Int
    public let weightsBytes: UInt64
    public let remoteBytesToDownload: UInt64
    public let reusedBytes: UInt64
    public let downloadedThisRunBytes: UInt64
    public let peakRSSBytes: UInt64
    public let largestScratchBytes: Int
}

public final class RemoteVisionPackInstaller {
    private let options: RemoteVisionPackInstallOptions
    private let audit: RepackAudit

    /// Exactly what a finished companion pack contains. `VisionPackVerifier`
    /// rejects a directory holding anything else, which is deliberate — but the
    /// same directory doubles as download scratch, so a crash can leave a range
    /// temp file or an interrupted `atomicWrite`'s `.tmp` beside the four real
    /// ones and make an otherwise complete download unverifiable forever.
    static let packEntries: Set<String> = [
        GTurboVisionFormatV1.manifestFile,
        GTurboVisionFormatV1.weightsFile,
        GTurboVisionFormatV1.processorFile,
        GTurboVisionFormatV1.receiptFile,
    ]

    /// Deletes everything in the partial directory that is not part of the pack
    /// itself, and returns what it removed.
    ///
    /// Only ever called while holding the install lock, and only at a point
    /// where no transfer is in flight, so anything it finds is abandoned
    /// scratch. Discarding an in-flight range costs a re-download of that
    /// chunk; the checkpoint, not the scratch file, is what records progress.
    @discardableResult
    static func reclaimPartialScratch(directory: String) throws -> [String] {
        guard try Posix.entryKind(directory) == .directory else { return [] }
        let manager = FileManager.default
        let entries = try manager.contentsOfDirectory(atPath: directory)
        var reclaimed: [String] = []
        for entry in entries where !packEntries.contains(entry) {
            let path = (directory as NSString).appendingPathComponent(entry)
            // Surfaced, not swallowed: a scratch file we cannot delete is the
            // difference between a resumable install and a stuck one.
            try manager.removeItem(atPath: path)
            reclaimed.append(entry)
        }
        if !reclaimed.isEmpty {
            try Posix.fsyncDirectory(directory)
        }
        return reclaimed
    }

    public init(
        options: RemoteVisionPackInstallOptions,
        audit: RepackAudit = RepackAudit()
    ) {
        self.options = options
        self.audit = audit
    }

    public func run(
        progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }
    ) async throws -> RemoteVisionPackInstallResult {
        let result = try await prepare(progress: progress)
        try Self.activatePrepared(
            outputDirectory: options.outputDirectory,
            textModelDirectory: options.textModelDirectory,
            repoID: options.repoID,
            requestedRevision: options.revision)
        return result
    }

    public func prepare(
        progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }
    ) async throws -> RemoteVisionPackInstallResult {
        try validateOptions()
        let lock = try InstallLock.acquire(outputDirectory: options.outputDirectory)
        defer { withExtendedLifetime(lock) {} }
        let paths = lock.paths
        let finalKind = try Posix.entryKind(paths.finalDirectory)
        if finalKind == .directory, !options.overwrite {
            throw RepackError.configurationInvalid(
                detail: "vision companion already exists: \(paths.finalDirectory)"
                    + "; pass --overwrite to replace it, or --resume --overwrite "
                    + "to continue a saved download over it")
        }
        if finalKind == .directory {
            // Activation needs this lock exclusively. Probing it now turns "a
            // loaded model was holding the pack" from a refusal that arrives
            // after the whole 1.09 GB transfer into one that arrives before the
            // first byte. The probe releases immediately, so it does not block
            // readers for the duration of the download, and a model that loads
            // mid-transfer is still caught at activation.
            _ = try VisionPackMutationLock.acquire(
                outputDirectory: paths.finalDirectory)
        }
        var hasPartial = try Posix.entryKind(paths.partialDirectory) == .directory
        let hasCheckpoint = try Posix.entryKind(paths.checkpointFile) == .regular
        if hasPartial, !hasCheckpoint {
            // The partial directory is created before the checkpoint that
            // identifies its source, so a crash in that window leaves a
            // directory nothing can interpret — and, without this, an error on
            // every later attempt including a fresh one. It holds no progress
            // worth keeping, so reclaim it.
            try FileManager.default.removeItem(atPath: paths.partialDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
            hasPartial = false
        }
        guard hasPartial == hasCheckpoint else {
            throw RepackError.installStateCorrupt(
                path: paths.partialDirectory,
                detail: "partial directory and checkpoint must exist together")
        }
        if hasPartial {
            try Self.reclaimPartialScratch(directory: paths.partialDirectory)
        }
        if options.resume {
            guard hasPartial else {
                throw RepackError.installStateMissing(path: paths.checkpointFile)
            }
        } else if hasPartial {
            throw RepackError.installStateIncompatible(
                detail: "saved vision download exists; resume or discard it")
        }
        do {
            return try await runPrepared(paths: paths, progress: progress)
        } catch {
            if !hasCheckpoint,
               (try? Posix.entryKind(paths.checkpointFile)) != .regular {
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            }
            throw error
        }
    }

    public static func inspectPersistentInstall(
        outputDirectory: String,
        repoID: String,
        requestedRevision: String
    ) throws -> RemoteInstallCheckpoint? {
        try RemoteStreamingRepacker.inspectPersistentInstall(
            outputDirectory: outputDirectory,
            repoID: repoID,
            requestedRevision: requestedRevision)
    }

    public static func discardPartial(outputDirectory: String) throws {
        try RemoteStreamingRepacker.discardPartial(outputDirectory: outputDirectory)
    }

    public static func preparedInstallIsValid(
        outputDirectory: String,
        textModelDirectory: String,
        verifyWeights: Bool = false
    ) -> Bool {
        guard let paths = try? RemoteInstallPaths(outputDirectory: outputDirectory),
              (try? Posix.entryKind(paths.partialDirectory)) == .directory,
              (try? Posix.entryKind(paths.checkpointFile)) == .regular else {
            return false
        }
        return (try? VisionPackVerifier.verify(
            directory: URL(fileURLWithPath: paths.partialDirectory, isDirectory: true),
            installedDirectory: URL(fileURLWithPath: paths.finalDirectory, isDirectory: true),
            textModelDirectory: URL(fileURLWithPath: textModelDirectory, isDirectory: true),
            verifyWeights: verifyWeights)) != nil
    }

    public static func activatePrepared(
        outputDirectory: String,
        textModelDirectory: String,
        repoID: String,
        requestedRevision: String,
        onVerifyProgress: ((UInt64, UInt64) throws -> Void)? = nil
    ) throws {
        let lock = try InstallLock.acquire(outputDirectory: outputDirectory)
        defer { withExtendedLifetime(lock) {} }
        let paths = lock.paths
        guard try Posix.entryKind(paths.checkpointFile) == .regular else {
            throw RepackError.installStateMissing(path: paths.checkpointFile)
        }
        let checkpoint = try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
        guard checkpoint.repoID == repoID,
              checkpoint.requestedRevision == requestedRevision else {
            throw RepackError.installStateIncompatible(
                detail: "prepared vision pack belongs to a different source")
        }
        let partialExists = try Posix.entryKind(paths.partialDirectory) == .directory
        if partialExists {
            // A download that finished but crashed before its own cleanup
            // leaves scratch the verifier will reject. Nothing is in flight
            // here — the install lock is held — so it is safe to reclaim.
            try Self.reclaimPartialScratch(directory: paths.partialDirectory)
        }
        let verificationDirectory = partialExists
            ? paths.partialDirectory
            : paths.finalDirectory
        guard try Posix.entryKind(verificationDirectory) == .directory else {
            throw RepackError.installStateMissing(path: paths.partialDirectory)
        }
        // The only cancellable part of activation: everything after it either
        // renames or removes, and neither may be abandoned half done.
        let verification = try VisionPackVerifier.verify(
            directory: URL(fileURLWithPath: verificationDirectory, isDirectory: true),
            installedDirectory: URL(fileURLWithPath: paths.finalDirectory, isDirectory: true),
            textModelDirectory: URL(fileURLWithPath: textModelDirectory, isDirectory: true),
            onHashProgress: onVerifyProgress)
        guard checkpoint.resolvedCommit == verification.sourceRevision,
              checkpoint.sourceIndexSHA256.lowercased()
                == verification.sourceIndexSha256.lowercased() else {
            throw RepackError.installStateIncompatible(
                detail: "prepared vision pack and checkpoint source identity differ")
        }
        if !partialExists {
            try FileManager.default.removeItem(atPath: paths.checkpointFile)
            try Posix.fsyncDirectory(paths.parentDirectory)
            return
        }
        let useLock = try VisionPackMutationLock.acquire(
            outputDirectory: paths.finalDirectory)
        defer { withExtendedLifetime(useLock) {} }
        if try Posix.entryKind(paths.finalDirectory) == .directory {
            try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
            try FileManager.default.removeItem(atPath: paths.partialDirectory)
        } else {
            try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
        }
        try FileManager.default.removeItem(atPath: paths.checkpointFile)
        try Posix.fsyncDirectory(paths.parentDirectory)
    }

    public static func removeInstalled(outputDirectory: String) throws {
        let lock = try InstallLock.acquire(outputDirectory: outputDirectory)
        defer { withExtendedLifetime(lock) {} }
        guard try Posix.entryKind(lock.paths.finalDirectory) == .directory else {
            throw RepackError.installStateMissing(path: lock.paths.finalDirectory)
        }
        let useLock = try VisionPackMutationLock.acquire(
            outputDirectory: lock.paths.finalDirectory)
        defer { withExtendedLifetime(useLock) {} }
        for path in [
            lock.paths.finalDirectory,
            lock.paths.partialDirectory,
            lock.paths.checkpointFile,
        ] {
            if try Posix.entryKind(path) != .absent {
                try FileManager.default.removeItem(atPath: path)
            }
        }
        try Posix.fsyncDirectory(lock.paths.parentDirectory)
    }

    private func runPrepared(
        paths: RemoteInstallPaths,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> RemoteVisionPackInstallResult {
        try Task.checkCancellation()
        let textBinding = try readTextBinding()
        let saved = options.resume
            ? try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
            : nil
        if let saved {
            guard saved.repoID == options.repoID,
                  saved.requestedRevision == options.revision else {
                throw RepackError.installStateIncompatible(
                    detail: "saved vision download belongs to a different source")
            }
        }

        let retryPolicy = RemoteRetryPolicy(
            attempts: options.rangeRetryAttempts,
            baseDelayNs: options.retryBaseDelayNs)
        let remote = HuggingFaceRemoteSource(
            repoID: options.repoID,
            requestedRevision: options.revision,
            resolvedCommit: saved?.resolvedCommit,
            token: options.token,
            downloadSession: options.downloadSession,
            baseURL: options.baseURL,
            tempDirectory: paths.partialDirectory,
            retryPolicy: retryPolicy)
        progress(.downloadingMetadata)
        let snapshot = try await RemoteSnapshotLoader.load(
            remote: remote,
            requireKnownSource: true,
            metadataDirectory: paths.metadataDirectory,
            audit: audit)
        guard SourceFingerprint.modelID(forIndexSha256: snapshot.metadata.indexSha256Hex)
                == options.repoID else {
            throw RepackError.sourceFingerprintRejected(
                path: snapshot.metadata.indexPath,
                sha256: snapshot.metadata.indexSha256Hex)
        }
        let visionPlan = try RepackPlanner.planVisionCompanion(
            meta: snapshot.metadata,
            shardHeaders: snapshot.shardHeaders)
        let rangePlan = try RangeCopyPlanner.plan(
            visionPackPlan: visionPlan,
            outputDirectory: paths.partialDirectory,
            rangeChunkBytes: options.rangeChunkBytes,
            textManifestSha256: textBinding.manifestSha256)
        var checkpoint = saved ?? RemoteInstallCheckpoint(
            repoID: options.repoID,
            requestedRevision: options.revision,
            resolvedCommit: snapshot.resolvedCommit,
            sourceIndexSHA256: snapshot.metadata.indexSha256Hex,
            planFingerprint: rangePlan.canonicalFingerprint,
            totalSourceBytes: rangePlan.remoteBytesToDownload)
        if saved != nil {
            guard checkpoint.resolvedCommit == snapshot.resolvedCommit,
                  checkpoint.totalSourceBytes == rangePlan.remoteBytesToDownload,
                  checkpoint.matches(
                    repoID: options.repoID,
                    requestedRevision: options.revision,
                    sourceIndexSHA256: snapshot.metadata.indexSha256Hex,
                    planFingerprint: rangePlan.canonicalFingerprint) else {
                throw RepackError.installStateIncompatible(
                    detail: "saved vision download source, text binding, or copy plan changed")
            }
            if try outputFileMatches(
                path: paths.partialDirectory,
                size: visionPlan.weightsFileSize) {
                checkpoint.completedRanges = try RemoteStreamingRepacker
                    .validatedCompletedRanges(
                        checkpoint.completedRanges,
                        copies: rangePlan.coalescedCopies,
                        partialDirectory: paths.partialDirectory)
            } else {
                checkpoint.completedRanges = []
                try createWeightsFile(paths: paths, size: visionPlan.weightsFileSize)
            }
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.parentDirectory)
        }

        let reusedDestinationBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.destinationBytes
        }
        let remainingBytes = visionPlan.weightsFileSize > reusedDestinationBytes
            ? visionPlan.weightsFileSize - reusedDestinationBytes
            : 0
        progress(.planning(
            downloadBytes: rangePlan.remoteBytesToDownload,
            outputBytes: visionPlan.weightsFileSize))
        let requirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: remainingBytes + UInt64(options.rangeChunkBytes),
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(requirement))
        try Task.checkCancellation()

        if saved == nil {
            progress(.reservingOutput(bytes: visionPlan.weightsFileSize))
            try createWeightsFile(paths: paths, size: visionPlan.weightsFileSize)
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.parentDirectory)
        }

        let pinned = remote.pinned(commit: snapshot.resolvedCommit)
        let processorInfo = try await pinned.resolveFileInfo(
            filename: GTurboVisionFormatV1.processorFile,
            audit: audit)
        guard processorInfo.resolvedCommit == snapshot.resolvedCommit else {
            throw RepackError.remoteProtocolInvalid(
                detail: "processor config commit differs from model commit")
        }
        let processorPath = (paths.partialDirectory as NSString)
            .appendingPathComponent(GTurboVisionFormatV1.processorFile)
        try await pinned.fetchSmallFile(
            filename: GTurboVisionFormatV1.processorFile,
            info: processorInfo,
            capBytes: GTurboVisionFormatV1.metadataMaxBytes,
            outputPath: processorPath,
            audit: audit)

        let provider = HTTPRangeSourceByteProvider(
            remote: pinned,
            files: snapshot.remoteFiles,
            writeTileBytes: options.writeTileBytes)
        let reusedBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.sourceBytes
        }
        let downloadStart = audit.remoteBytesDownloaded
        progress(.copyingPayload(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: 0,
            totalBytes: rangePlan.remoteBytesToDownload))
        try await provider.copyBatch(
            rangePlan.coalescedCopies,
            completedRangeIDs: Set(checkpoint.completedRanges.map(\.id)),
            partialDirectory: paths.partialDirectory,
            temporaryPath: paths.rangeTemporaryFile,
            audit: audit,
            progress: { downloaded in
                progress(.copyingPayload(
                    reusedBytes: reusedBytes,
                    downloadedThisRunBytes: downloaded,
                    totalBytes: rangePlan.remoteBytesToDownload))
            },
            commit: { completed in
                checkpoint.completedRanges.removeAll { $0.id == completed.id }
                checkpoint.completedRanges.append(completed)
                checkpoint.completedRanges.sort { $0.id < $1.id }
                try checkpoint.write(
                    to: paths.checkpointFile,
                    parentDirectory: paths.parentDirectory)
            })

        let weightsPath = (paths.partialDirectory as NSString)
            .appendingPathComponent(GTurboVisionFormatV1.weightsFile)
        progress(.hashingOutput(GTurboVisionFormatV1.weightsFile))
        let weightsSHA = try Sha256Stream.hashFile(
            path: weightsPath, noCache: true, noFollow: true)
        progress(.hashingOutput(GTurboVisionFormatV1.processorFile))
        let processorData = try Posix.readBoundedData(
            processorPath,
            maximumBytes: GTurboVisionFormatV1.metadataMaxBytes)
        let processorSHA = hash(processorData)
        progress(.finalizing)
        try Task.checkCancellation()
        try writeMetadata(
            plan: visionPlan,
            sourceIndexSha256: snapshot.metadata.indexSha256Hex,
            sourceRevision: snapshot.resolvedCommit,
            textBinding: textBinding,
            processorData: processorData,
            processorSha256: processorSHA,
            weightsSha256: weightsSHA,
            paths: paths)
        // Removes the range scratch and the metadata cache along with anything
        // else left over: the verifier below accepts the four pack entries and
        // nothing besides.
        try Self.reclaimPartialScratch(directory: paths.partialDirectory)
        try Posix.fsyncDirectory(paths.partialDirectory)
        try Task.checkCancellation()
        _ = try VisionPackVerifier.verify(
            directory: URL(fileURLWithPath: paths.partialDirectory, isDirectory: true),
            installedDirectory: URL(fileURLWithPath: paths.finalDirectory, isDirectory: true),
            textModelDirectory: URL(fileURLWithPath: options.textModelDirectory, isDirectory: true))
        try Task.checkCancellation()
        audit.sampleRSS()
        return RemoteVisionPackInstallResult(
            outputDirectory: paths.finalDirectory,
            resolvedCommit: snapshot.resolvedCommit,
            tensorCount: visionPlan.entries.count,
            weightsBytes: visionPlan.weightsFileSize,
            remoteBytesToDownload: rangePlan.remoteBytesToDownload,
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: audit.remoteBytesDownloaded - downloadStart,
            peakRSSBytes: audit.peakRssBytes,
            largestScratchBytes: audit.largestScratchBytes)
    }

    private func validateOptions() throws {
        guard options.rangeChunkBytes > 0,
              options.rangeChunkBytes <= RemoteChunkPolicy.maxBytes else {
            throw RepackError.configurationInvalid(
                detail: "bad range chunk bytes \(options.rangeChunkBytes)")
        }
        guard options.writeTileBytes > 0,
              options.writeTileBytes <= BoundedScratch.defaultLimitBytes else {
            throw RepackError.configurationInvalid(
                detail: "bad write tile bytes \(options.writeTileBytes)")
        }
        guard options.rangeRetryAttempts >= 0 else {
            throw RepackError.configurationInvalid(
                detail: "bad range retry attempts \(options.rangeRetryAttempts)")
        }
    }

    private func readTextBinding() throws -> TextBinding {
        let path = (options.textModelDirectory as NSString)
            .appendingPathComponent(GTurboVisionFormatV1.manifestFile)
        let data = try Posix.readBoundedData(
            path,
            maximumBytes: GTurboVisionFormatV1.metadataMaxBytes)
        let manifest = try GTurboManifestCodec.decode(data)
        guard let source = manifest.sourceSnapshotHash, !source.isEmpty else {
            throw RepackError.configurationInvalid(
                detail: "text manifest has no sourceSnapshotHash")
        }
        return TextBinding(
            sourceSnapshotHash: source,
            manifestSha256: hash(data))
    }

    private func createWeightsFile(paths: RemoteInstallPaths, size: UInt64) throws {
        try Posix.mkdirP(paths.partialDirectory)
        let path = (paths.partialDirectory as NSString)
            .appendingPathComponent(GTurboVisionFormatV1.weightsFile)
        if try Posix.entryKind(path) != .absent {
            try FileManager.default.removeItem(atPath: path)
        }
        let descriptor = try Posix.openCreateRW(path)
        defer { close(descriptor) }
        try Posix.ftruncate(descriptor, path: path, size: size)
        try Posix.fsync(descriptor, path: path)
        try Posix.fsyncDirectory(paths.partialDirectory)
    }

    private func outputFileMatches(path: String, size: UInt64) throws -> Bool {
        let weights = (path as NSString)
            .appendingPathComponent(GTurboVisionFormatV1.weightsFile)
        guard try Posix.entryKind(weights) == .regular else { return false }
        let descriptor = try Posix.openReadNoFollow(weights)
        defer { close(descriptor) }
        return try Posix.fileSize(fd: descriptor, path: weights) == size
    }

    private func writeMetadata(
        plan: VisionPackPlan,
        sourceIndexSha256: String,
        sourceRevision: String,
        textBinding: TextBinding,
        processorData: Data,
        processorSha256: String,
        weightsSha256: String,
        paths: RemoteInstallPaths
    ) throws {
        let files = [
            GTurboVisionFormatV1.weightsFile: GTurboManifestFileV1(
                size: plan.weightsFileSize, sha256: weightsSha256),
            GTurboVisionFormatV1.processorFile: GTurboManifestFileV1(
                size: UInt64(processorData.count), sha256: processorSha256),
        ]
        let tensors = plan.entries.map { entry in
            GTurboVisionTensorRegionV1(
                name: entry.source.name,
                executionPosition: entry.executionPosition,
                offset: entry.fileOffset,
                size: entry.source.sizeBytes,
                shape: entry.source.shape,
                dtype: entry.source.dtype == .u32 ? .u32 : .bf16,
                quantization: entry.quantSpec.map { spec in
                    GTurboVisionQuantizationV1(
                        weightBits: spec.bits,
                        scheme: "affine",
                        scaleType: "bf16",
                        biasType: "bf16",
                        groupSize: entry.groupSize)
                })
        }
        let manifest = GTurboVisionManifestV1(
            modelID: options.repoID,
            sourceRevision: sourceRevision,
            sourceIndexSha256: sourceIndexSha256,
            processorConfigSha256: processorSha256,
            compatibleTextSourceSnapshotHash: textBinding.sourceSnapshotHash,
            compatibleTextManifestSha256: textBinding.manifestSha256,
            files: files,
            tensors: tensors)
        let manifestData = try GTurboVisionManifestCodec.encode(manifest)
        let manifestSHA = hash(manifestData)
        let receiptFiles = files.merging([
            GTurboVisionFormatV1.manifestFile: GTurboManifestFileV1(
                size: UInt64(manifestData.count), sha256: manifestSHA)
        ]) { _, new in new }
        let receipt = GTurboVisionReceiptV1(
            manifestSha256: manifestSHA,
            companionDirectoryPath: paths.finalDirectory,
            compatibleTextManifestSha256: textBinding.manifestSha256,
            sourceRepoID: options.repoID,
            sourceRevision: sourceRevision,
            verificationTimestamp: "source-revision:\(sourceRevision)",
            toolVersion: "TurboFieldfareRepack/remote-vision-v1",
            files: receiptFiles)
        try Posix.atomicWrite(
            manifestData,
            to: (paths.partialDirectory as NSString)
                .appendingPathComponent(GTurboVisionFormatV1.manifestFile),
            durableIn: paths.partialDirectory)
        try Posix.atomicWrite(
            try GTurboVisionReceiptCodec.encode(receipt),
            to: (paths.partialDirectory as NSString)
                .appendingPathComponent(GTurboVisionFormatV1.receiptFile),
            durableIn: paths.partialDirectory)
    }

    private func hash(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }
}

private struct TextBinding: Sendable {
    let sourceSnapshotHash: String
    let manifestSha256: String
}
