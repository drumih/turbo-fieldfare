import Darwin
import Foundation

public struct RemoteVisionSidecarOptions: Sendable {
    public let repoID: String
    public let revision: String
    public let modelDirectory: String
    public let token: String?
    public let requireKnownSource: Bool
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
        modelDirectory: String,
        token: String? = nil,
        requireKnownSource: Bool = false,
        rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
        writeTileBytes: Int = WriterCore.tileBytes,
        minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
        overwrite: Bool = false,
        resume: Bool = false,
        downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
        baseURL: URL = URL(string: "https://huggingface.co")!,
        rangeRetryAttempts: Int = 4,
        retryBaseDelayNs: UInt64 = 1_000_000_000
    ) {
        self.repoID = repoID
        self.revision = revision
        self.modelDirectory = modelDirectory
        self.token = token
        self.requireKnownSource = requireKnownSource
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

public struct RemoteVisionSidecarResult: Sendable {
    public let visionDirectory: String
    public let resolvedCommit: String
    public let entryCount: Int
    public let sourceTensorCount: Int
    public let remoteBytesToDownload: UInt64
    public let reusedBytes: UInt64
    public let downloadedThisRunBytes: UInt64
}

struct VisionSidecarInstallPaths: Sendable, Equatable {
    let modelDirectory: String
    let finalDirectory: String
    let partialDirectory: String
    let checkpointFile: String
    let rangeTemporaryFile: String
    let metadataDirectory: String

    init(modelDirectory: String) {
        self.modelDirectory = modelDirectory
        self.finalDirectory = (modelDirectory as NSString).appendingPathComponent("vision")
        self.partialDirectory = (modelDirectory as NSString).appendingPathComponent(".vision.partial")
        self.checkpointFile = (modelDirectory as NSString).appendingPathComponent(".vision.resume.json")
        self.rangeTemporaryFile = (partialDirectory as NSString).appendingPathComponent(".range.tmp")
        self.metadataDirectory = (partialDirectory as NSString).appendingPathComponent(".remote-metadata")
    }

    func validateEntryTypes() throws {
        try require(finalDirectory, kinds: [.absent, .directory])
        try require(partialDirectory, kinds: [.absent, .directory])
        try require(checkpointFile, kinds: [.absent, .regular])
    }

    private func require(_ path: String, kinds: Set<Posix.EntryKind>) throws {
        let kind = try Posix.entryKind(path)
        guard kinds.contains(kind) else {
            throw RepackError.installPathUnsafe(
                path: path,
                detail: "unexpected \(kind) entry")
        }
    }
}

public final class RemoteVisionSidecarInstaller {
    private let options: RemoteVisionSidecarOptions
    private let audit: RepackAudit

    public init(
        options: RemoteVisionSidecarOptions,
        audit: RepackAudit = RepackAudit()
    ) {
        self.options = options
        self.audit = audit
    }

    public func run(
        progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }
    ) async throws -> RemoteVisionSidecarResult {
        try validateOptions()
        let installLock = try InstallLock.acquire(outputDirectory: options.modelDirectory)
        defer { withExtendedLifetime(installLock) {} }
        let modelDirectory = installLock.paths.finalDirectory
        let paths = VisionSidecarInstallPaths(modelDirectory: modelDirectory)
        try paths.validateEntryTypes()
        let binding = try RootModelSourceBinding.load(modelDirectory: modelDirectory)

        if try Posix.entryKind(paths.finalDirectory) == .directory,
           !options.overwrite,
           !options.resume {
            throw RepackError.configurationInvalid(
                detail: "vision sidecar already exists: \(paths.finalDirectory)")
        }
        let hasPartial = try Posix.entryKind(paths.partialDirectory) == .directory
        let hasCheckpoint = try Posix.entryKind(paths.checkpointFile) == .regular
        guard hasPartial == hasCheckpoint else {
            throw RepackError.installStateCorrupt(
                path: paths.partialDirectory,
                detail: "vision partial directory and checkpoint must exist together")
        }
        if options.resume {
            guard hasPartial else {
                throw RepackError.installStateMissing(path: paths.checkpointFile)
            }
        } else if hasPartial {
            throw RepackError.installStateIncompatible(
                detail: "saved vision download exists; resume or discard it")
        }

        if !hasPartial {
            try Posix.mkdirP(paths.partialDirectory)
            try Posix.fsyncDirectory(paths.modelDirectory)
        }
        do {
            return try await runPrepared(
                paths: paths,
                rootBinding: binding,
                progress: progress)
        } catch {
            if !hasCheckpoint,
               (try? Posix.entryKind(paths.checkpointFile)) != .regular {
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
                try? Posix.fsyncDirectory(paths.modelDirectory)
            }
            throw error
        }
    }

    public static func inspectPersistentInstall(
        modelDirectory: String,
        repoID: String,
        requestedRevision: String
    ) throws -> RemoteInstallCheckpoint? {
        let lock = try InstallLock.acquire(outputDirectory: modelDirectory)
        defer { withExtendedLifetime(lock) {} }
        let paths = VisionSidecarInstallPaths(modelDirectory: lock.paths.finalDirectory)
        try paths.validateEntryTypes()
        let partial = try Posix.entryKind(paths.partialDirectory)
        let checkpoint = try Posix.entryKind(paths.checkpointFile)
        if partial == .absent, checkpoint == .absent { return nil }
        guard partial == .directory, checkpoint == .regular else {
            throw RepackError.installStateCorrupt(
                path: paths.partialDirectory,
                detail: "vision partial directory and checkpoint must exist together")
        }
        let value = try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
        guard value.repoID == repoID,
              value.requestedRevision == requestedRevision else {
            throw RepackError.installStateIncompatible(
                detail: "saved vision download belongs to a different source")
        }
        return value
    }

    public static func discardPartial(modelDirectory: String) throws {
        let lock = try InstallLock.acquire(outputDirectory: modelDirectory)
        defer { withExtendedLifetime(lock) {} }
        let paths = VisionSidecarInstallPaths(modelDirectory: lock.paths.finalDirectory)
        try paths.validateEntryTypes()
        let hasPartial = try Posix.entryKind(paths.partialDirectory) != .absent
        let hasCheckpoint = try Posix.entryKind(paths.checkpointFile) != .absent
        guard hasPartial || hasCheckpoint else {
            throw RepackError.installStateMissing(path: paths.checkpointFile)
        }
        if hasPartial {
            try FileManager.default.removeItem(atPath: paths.partialDirectory)
        }
        if hasCheckpoint {
            try FileManager.default.removeItem(atPath: paths.checkpointFile)
        }
        try Posix.fsyncDirectory(paths.modelDirectory)
    }

    private func runPrepared(
        paths: VisionSidecarInstallPaths,
        rootBinding: RootModelSourceBinding,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> RemoteVisionSidecarResult {
        try Task.checkCancellation()
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
            requireKnownSource: options.requireKnownSource,
            metadataDirectory: paths.metadataDirectory,
            audit: audit)
        let sourceSnapshotHash = "sha256:" + snapshot.metadata.indexSha256Hex
        guard sourceSnapshotHash == rootBinding.sourceSnapshotHash else {
            throw RepackError.installStateIncompatible(
                detail: "vision source snapshot does not match the installed language model")
        }
        if options.requireKnownSource {
            guard SourceFingerprint.modelID(
                forIndexSha256: snapshot.metadata.indexSha256Hex) == options.repoID else {
                throw RepackError.sourceFingerprintRejected(
                    path: snapshot.metadata.indexPath,
                    sha256: snapshot.metadata.indexSha256Hex)
            }
        }

        try Task.checkCancellation()
        let plan = try VisionSidecarPlanner.plan(
            meta: snapshot.metadata,
            shardHeaders: snapshot.shardHeaders,
            outputDirectory: paths.partialDirectory)
        let rangePlan = try RangeCopyPlanner.planVisionSidecar(
            visionPlan: plan,
            rangeChunkBytes: options.rangeChunkBytes)
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
                    detail: "saved vision download source or copy plan changed")
            }
            if try outputFileMatches(plan: plan, rangePlan: rangePlan) {
                checkpoint.completedRanges = try RemoteStreamingRepacker.validatedCompletedRanges(
                    checkpoint.completedRanges,
                    copies: rangePlan.coalescedCopies,
                    partialDirectory: paths.partialDirectory)
            } else {
                checkpoint.completedRanges = []
                try FileManager.default.removeItem(atPath: paths.partialDirectory)
                try Posix.mkdirP(paths.partialDirectory)
                try createOutputFile(plan: plan)
            }
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.modelDirectory)
        }

        let outputBytes = plan.resident.totalSize
        progress(.planning(
            downloadBytes: rangePlan.remoteBytesToDownload,
            outputBytes: outputBytes))
        let reusedDestinationBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.destinationBytes
        }
        let remainingOutputBytes = outputBytes > reusedDestinationBytes
            ? outputBytes - reusedDestinationBytes
            : 0
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.modelDirectory,
            bytes: remainingOutputBytes + UInt64(options.rangeChunkBytes),
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()

        audit.remoteRepoID = options.repoID
        audit.remoteRequestedRevision = options.revision
        audit.remoteResolvedCommit = snapshot.resolvedCommit
        audit.remoteRangeStreamingSupported = true
        audit.remoteGapBytesDownloaded = rangePlan.remoteGapBytesDownloaded
        audit.sourceSnapshotSha256 = snapshot.metadata.indexSha256Hex
        audit.wholeFileHeapBuffers = false

        if saved == nil {
            progress(.reservingOutput(bytes: outputBytes))
            try createOutputFile(plan: plan)
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.modelDirectory)
        }

        let provider = HTTPRangeSourceByteProvider(
            remote: remote.pinned(commit: snapshot.resolvedCommit),
            files: snapshot.remoteFiles,
            writeTileBytes: options.writeTileBytes)
        let reusedBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.sourceBytes
        }
        let payloadDownloadStart = audit.remoteBytesDownloaded
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
            progress: { downloadedBytes in
                progress(.copyingPayload(
                    reusedBytes: reusedBytes,
                    downloadedThisRunBytes: downloadedBytes,
                    totalBytes: rangePlan.remoteBytesToDownload))
            },
            commit: { completed in
                checkpoint.completedRanges.removeAll { $0.id == completed.id }
                checkpoint.completedRanges.append(completed)
                checkpoint.completedRanges.sort { $0.id < $1.id }
                try checkpoint.write(
                    to: paths.checkpointFile,
                    parentDirectory: paths.modelDirectory)
            })

        progress(.hashingOutput("vision/weights.bin"))
        let weightsSize = plan.resident.totalSize
        let weightsSha = try WriterCore.hashEntireFile(
            path: plan.resident.path,
            size: weightsSize,
            audit: audit,
            cancellationCheck: Task.checkCancellation)
        audit.outputFiles.append(.init(
            relativePath: "weights.bin",
            size: weightsSize,
            sha256: weightsSha))

        try? FileManager.default.removeItem(atPath: paths.rangeTemporaryFile)
        try? FileManager.default.removeItem(atPath: paths.metadataDirectory)
        progress(.finalizing)
        try Task.checkCancellation()
        try writeManifestAndReceipt(
            paths: paths,
            rootBinding: rootBinding,
            snapshot: snapshot,
            plan: plan,
            rangePlan: rangePlan,
            weightsSize: weightsSize,
            weightsSha: weightsSha)

        try Task.checkCancellation()
        if try Posix.entryKind(paths.finalDirectory) == .directory {
            try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
            try Posix.fsyncDirectory(paths.modelDirectory)
            try FileManager.default.removeItem(atPath: paths.partialDirectory)
        } else {
            try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
        }
        try Posix.fsyncDirectory(paths.modelDirectory)
        if try Posix.entryKind(paths.checkpointFile) == .regular {
            try FileManager.default.removeItem(atPath: paths.checkpointFile)
            try Posix.fsyncDirectory(paths.modelDirectory)
        }

        return RemoteVisionSidecarResult(
            visionDirectory: paths.finalDirectory,
            resolvedCommit: snapshot.resolvedCommit,
            entryCount: plan.entryCount,
            sourceTensorCount: plan.sourceTensorCount,
            remoteBytesToDownload: rangePlan.remoteBytesToDownload,
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: audit.remoteBytesDownloaded - payloadDownloadStart)
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

    private func createOutputFile(plan: VisionSidecarPlan) throws {
        let descriptor = try ResidentWriter.createAndWriteIndex(
            plan: plan.resident,
            audit: audit)
        try Posix.fsync(descriptor, path: plan.resident.path)
        close(descriptor)
        try Posix.fsyncDirectory(
            (plan.resident.path as NSString).deletingLastPathComponent)
    }

    private func outputFileMatches(
        plan: VisionSidecarPlan,
        rangePlan: RangeCopyPlan
    ) throws -> Bool {
        guard let output = rangePlan.expectedOutputs.first,
              output.relativePath == "weights.bin",
              try Posix.entryKind(plan.resident.path) == .regular else {
            return false
        }
        let descriptor = try Posix.openReadNoFollow(plan.resident.path)
        defer { close(descriptor) }
        guard try Posix.fileSize(fd: descriptor, path: plan.resident.path) == output.size else {
            return false
        }
        let expectedIndex = try ResidentWriter.encodeIndex(plan: plan.resident)
        return try expectedIndex.withUnsafeBytes { expected in
            guard let expectedBase = expected.baseAddress else { return true }
            let scratch = UnsafeMutableRawBufferPointer.allocate(
                byteCount: min(WriterCore.tileBytes, max(1, expected.count)),
                alignment: 16_384)
            defer { scratch.deallocate() }
            var offset = 0
            while offset < expected.count {
                let count = min(scratch.count, expected.count - offset)
                try Posix.preadAll(
                    fd: descriptor,
                    path: plan.resident.path,
                    buf: scratch.baseAddress!,
                    count: count,
                    offset: UInt64(offset))
                guard memcmp(
                    scratch.baseAddress!,
                    expectedBase.advanced(by: offset),
                    count) == 0 else { return false }
                offset += count
            }
            return true
        }
    }

    private func writeManifestAndReceipt(
        paths: VisionSidecarInstallPaths,
        rootBinding: RootModelSourceBinding,
        snapshot: RemoteSnapshot,
        plan: VisionSidecarPlan,
        rangePlan: RangeCopyPlan,
        weightsSize: UInt64,
        weightsSha: String
    ) throws {
        let currentBinding = try RootModelSourceBinding.load(
            modelDirectory: paths.modelDirectory)
        guard currentBinding == rootBinding else {
            throw RepackError.installStateIncompatible(
                detail: "root model manifest changed during vision installation")
        }
        let manifest = VisionSidecarManifest(
            sourceRepoID: options.repoID,
            sourceRevision: snapshot.resolvedCommit,
            sourceSnapshotHash: rootBinding.sourceSnapshotHash,
            rootManifestSha256: rootBinding.manifestSha256,
            residentIndexSha256: rangePlan.residentIndexSha256,
            entryCount: plan.entryCount,
            sourceTensorCount: plan.sourceTensorCount,
            weightsSize: weightsSize,
            weightsSha256: weightsSha)
        let manifestData = try manifest.encode()
        let manifestPath = (paths.partialDirectory as NSString)
            .appendingPathComponent("manifest.json")
        try Posix.atomicWrite(
            manifestData,
            to: manifestPath,
            durableIn: paths.partialDirectory)
        let manifestSha = try Sha256Stream.hashFile(path: manifestPath)
        let receiptData = try VerifiedInstallReceiptWriter.encode(
            outputDir: paths.finalDirectory,
            manifestSha256: manifestSha,
            manifestSize: UInt64(manifestData.count),
            sourceRepoID: options.repoID,
            sourceRevision: snapshot.resolvedCommit,
            files: audit.outputFiles)
        let receiptPath = (paths.partialDirectory as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        try Posix.atomicWrite(
            receiptData,
            to: receiptPath,
            durableIn: paths.partialDirectory)
    }
}
