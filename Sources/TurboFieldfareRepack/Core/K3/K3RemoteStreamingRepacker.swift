import Darwin
import Foundation
import TurboFieldfareFormat

public struct K3RemoteStreamingRepackOptions: Sendable {
    public let repoID: String
    public let revision: String
    public let outputDir: String
    public let token: String?
    public let requireKnownSource: Bool
    public let trunkQuant: K3TrunkQuant
    public let copyAuditPath: String?
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let overwrite: Bool
    public let resume: Bool
    /// A completed verified K3 bundle whose immutable MXFP4 layer files are
    /// APFS-cloned into the new bundle. This avoids downloading or allocating
    /// another physical copy of the routed experts during a trunk upgrade.
    public let reuseExpertsFrom: String?
    public let dryRunSpaceCheck: Bool
    public let downloadSession: RemoteDownloadSession
    public let baseURL: URL
    public let rangeRetryAttempts: Int
    public let retryBaseDelayNs: UInt64

    public init(repoID: String,
                revision: String,
                outputDir: String,
                token: String? = nil,
                requireKnownSource: Bool = false,
                trunkQuant: K3TrunkQuant = .int4,
                copyAuditPath: String? = nil,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                writeTileBytes: Int = WriterCore.tileBytes,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
                overwrite: Bool = false,
                resume: Bool = false,
                reuseExpertsFrom: String? = nil,
                dryRunSpaceCheck: Bool = false,
                downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
                baseURL: URL = URL(string: "https://huggingface.co")!,
                rangeRetryAttempts: Int = 4,
                retryBaseDelayNs: UInt64 = 1_000_000_000) {
        self.repoID = repoID
        self.revision = revision
        self.outputDir = outputDir
        self.token = token
        self.requireKnownSource = requireKnownSource
        self.trunkQuant = trunkQuant
        self.copyAuditPath = copyAuditPath
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.overwrite = overwrite
        self.resume = resume
        self.reuseExpertsFrom = reuseExpertsFrom
        self.dryRunSpaceCheck = dryRunSpaceCheck
        self.downloadSession = downloadSession
        self.baseURL = baseURL
        self.rangeRetryAttempts = rangeRetryAttempts
        self.retryBaseDelayNs = retryBaseDelayNs
    }
}

public struct K3RemoteStreamingRepackResult: Sendable {
    public let outputDir: String
    public let resolvedCommit: String
    let plan: K3RepackPlan
    public let rangeRequestCount: Int
    public let remoteBytesToDownload: UInt64
    public let remoteGapBytesDownloaded: UInt64
    public let remoteRetryCount: UInt64
    public let reusedBytes: UInt64
    public let downloadedThisRunBytes: UInt64
    public let dryRun: Bool
}

/// Everything phase 1 of the K3 driver needs from a source checkpoint.
struct K3Snapshot {
    let indexSha256Hex: String
    let arch: K3ArchInfo
    let shardHeaders: [K3Safetensors.Header]
    let resolvedCommit: String
}

/// Where the K3 driver gets snapshot metadata, payload bytes, and tokenizer
/// sidecars. The shipped implementation is `K3RemoteSnapshotSource`
/// (Hugging Face range requests); tests substitute a local-directory source.
/// The checkpoint/resume machinery below is source-agnostic and shared with
/// the Gemma installer unchanged.
protocol K3SnapshotSource {
    func loadSnapshot(metadataDirectory: String,
                      partialDirectory: String,
                      audit: RepackAudit) async throws -> K3Snapshot
    func makeByteProvider(snapshot: K3Snapshot,
                          writeTileBytes: Int) throws -> SourceByteProvider
    /// Stage the bundled sidecars into `<partialDirectory>/tokenizer`;
    /// returns the staged relative paths (recorded into the manifest).
    func stageTokenizerFiles(metadataDirectory: String,
                             partialDirectory: String,
                             audit: RepackAudit) async throws -> [String]
}

/// Hugging Face implementation of the K3 snapshot source. Mirrors
/// `RemoteSnapshotLoader` with the K3 parsers and a larger index cap (the
/// real K3 index is 59,764,096 bytes — it carries ~497k tensors).
final class K3RemoteSnapshotSource: K3SnapshotSource {
    static let indexCapBytes: UInt64 = 256 * 1024 * 1024
    static let configCapBytes: UInt64 = 16 * 1024 * 1024
    static let tiktokenCapBytes: UInt64 = 64 * 1024 * 1024
    static let tokenizerConfigCapBytes: UInt64 = 16 * 1024 * 1024

    private let repoID: String
    private let requestedRevision: String
    private let token: String?
    private let downloadSession: RemoteDownloadSession
    private let baseURL: URL
    private let retryPolicy: RemoteRetryPolicy
    private let requireKnownSource: Bool
    private var pinned: HuggingFaceRemoteSource?
    private var remoteFiles: [String: RemoteFileInfo] = [:]

    init(options: K3RemoteStreamingRepackOptions) {
        self.repoID = options.repoID
        self.requestedRevision = options.revision
        self.token = options.token
        self.downloadSession = options.downloadSession
        self.baseURL = options.baseURL
        self.retryPolicy = RemoteRetryPolicy(attempts: options.rangeRetryAttempts,
                                             baseDelayNs: options.retryBaseDelayNs)
        self.requireKnownSource = options.requireKnownSource
    }

    func loadSnapshot(metadataDirectory: String,
                      partialDirectory: String,
                      audit: RepackAudit) async throws -> K3Snapshot {
        try Posix.mkdirP(metadataDirectory)
        let remote = HuggingFaceRemoteSource(repoID: repoID,
                                             requestedRevision: requestedRevision,
                                             token: token,
                                             downloadSession: downloadSession,
                                             baseURL: baseURL,
                                             tempDirectory: partialDirectory,
                                             retryPolicy: retryPolicy)
        let indexInfo = try await remote.resolveFileInfo(filename: "model.safetensors.index.json",
                                                         audit: audit)
        let pinned = remote.pinned(commit: indexInfo.resolvedCommit)
        let configInfo = try await pinned.resolveFileInfo(filename: "config.json",
                                                          audit: audit)
        guard configInfo.resolvedCommit == indexInfo.resolvedCommit else {
            throw RepackError.remoteProtocolInvalid(detail: "config commit differs from index commit")
        }

        try await pinned.fetchSmallFile(filename: "model.safetensors.index.json",
                                        info: indexInfo,
                                        capBytes: Self.indexCapBytes,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("model.safetensors.index.json"),
                                        audit: audit)
        try await pinned.fetchSmallFile(filename: "config.json",
                                        info: configInfo,
                                        capBytes: Self.configCapBytes,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("config.json"),
                                        audit: audit)

        let metadata = try K3IndexLoader.load(snapshotDir: metadataDirectory)
        if requireKnownSource,
           metadata.indexSha256Hex != K3SupportedModelSource.sourceIndexSHA256 {
            throw RepackError.sourceFingerprintRejected(path: metadata.indexPath,
                                                        sha256: metadata.indexSha256Hex)
        }
        let arch = try K3ArchInfo.load(configPath: metadata.configPath)

        var files: [String: RemoteFileInfo] = [
            indexInfo.filename: indexInfo,
            configInfo.filename: configInfo,
        ]
        var headers: [K3Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)
        for shard in metadata.shardFilenames {
            try Task.checkCancellation()
            let info = try await pinned.resolveFileInfo(filename: shard, audit: audit)
            guard info.resolvedCommit == indexInfo.resolvedCommit else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "shard \(shard) commit differs from index commit")
            }
            guard info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "shard \(shard) does not advertise byte ranges")
            }
            files[shard] = info
            let prefix = try await pinned.downloadRangeToTempFile(filename: shard,
                                                                  info: info,
                                                                  offset: 0,
                                                                  length: 8,
                                                                  audit: audit)
            defer { try? FileManager.default.removeItem(atPath: prefix.path) }
            let prefixData = try Data(contentsOf: URL(fileURLWithPath: prefix.path))
            guard prefixData.count == 8 else {
                throw RepackError.safetensorsHeaderInvalid(path: shard,
                                                           detail: "short header prefix")
            }
            let headerSize = prefixData.withUnsafeBytes { raw -> UInt64 in
                var value: UInt64 = 0
                for i in 0..<8 { value |= UInt64(raw[i]) << UInt64(i * 8) }
                return value
            }
            if headerSize > K3Safetensors.maxHeaderBytes || headerSize > info.size - 8 {
                throw RepackError.safetensorsHeaderTooLarge(path: shard, size: headerSize)
            }
            let headerFile = try await pinned.downloadRangeToTempFile(filename: shard,
                                                                      info: info,
                                                                      offset: 8,
                                                                      length: Int(headerSize),
                                                                      audit: audit)
            defer { try? FileManager.default.removeItem(atPath: headerFile.path) }
            let headerData = try Data(contentsOf: URL(fileURLWithPath: headerFile.path))
            headers.append(try K3Safetensors.parseHeaderBytes(path: shard,
                                                              fileSize: info.size,
                                                              headerBytes: headerData))
        }

        self.pinned = pinned
        self.remoteFiles = files
        return K3Snapshot(indexSha256Hex: metadata.indexSha256Hex,
                          arch: arch,
                          shardHeaders: headers,
                          resolvedCommit: indexInfo.resolvedCommit)
    }

    func makeByteProvider(snapshot: K3Snapshot,
                          writeTileBytes: Int) throws -> SourceByteProvider {
        guard let pinned else {
            throw RepackError.configurationInvalid(detail: "snapshot not loaded")
        }
        return HTTPRangeSourceByteProvider(remote: pinned,
                                           files: remoteFiles,
                                           writeTileBytes: writeTileBytes)
    }

    func stageTokenizerFiles(metadataDirectory: String,
                             partialDirectory: String,
                             audit: RepackAudit) async throws -> [String] {
        guard let pinned else {
            throw RepackError.configurationInvalid(detail: "snapshot not loaded")
        }
        let tokenizerDir = (partialDirectory as NSString).appendingPathComponent("tokenizer")
        try Posix.mkdirP(tokenizerDir)
        var staged: [String] = []

        let configSrc = (metadataDirectory as NSString).appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configSrc) {
            let dst = (tokenizerDir as NSString).appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: dst) {
                try FileManager.default.removeItem(atPath: dst)
            }
            try FileManager.default.copyItem(atPath: configSrc, toPath: dst)
            staged.append("tokenizer/config.json")
        }

        for (name, cap) in [("tiktoken.model", Self.tiktokenCapBytes),
                            ("tokenizer_config.json", Self.tokenizerConfigCapBytes)] {
            try Task.checkCancellation()
            let info = try await pinned.resolveFileInfo(filename: name, audit: audit)
            let dst = (tokenizerDir as NSString).appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: dst) {
                try FileManager.default.removeItem(atPath: dst)
            }
            try await pinned.fetchSmallFile(filename: name,
                                            info: info,
                                            capBytes: cap,
                                            outputPath: dst,
                                            audit: audit)
            staged.append("tokenizer/\(name)")
        }
        return staged
    }
}

/// The K3 v2 installer. Same state machine as `RemoteStreamingRepacker`
/// (lock -> snapshot -> plan -> checkpoint/resume -> range copy -> finalize
/// -> atomic rename), reusing the shared machinery unchanged
/// (`RemoteInstallCheckpoint`, `SourceByteProvider.copyBatch` + per-range
/// destination digests, `ResidentWriter`, `RangeCopyPlanner` statics,
/// `DiskSpaceChecker`, `VerifiedInstallReceiptWriter`). K3 adds one local
/// phase between the copy and the manifest: quantize the staged BF16 trunk
/// into the resident file, then delete the staging file.
///
/// Resume states:
/// - partial ranges: validate completed-range digests, copy the rest.
/// - all ranges complete, staging present: re-run the quantization phase
///   (idempotent rewrite of the resident quantized regions).
/// - all ranges complete, staging absent: the previous run finished the
///   manifest and died before the rename; verify and promote.
public final class K3RemoteStreamingRepacker {
    private let options: K3RemoteStreamingRepackOptions
    private let source: K3SnapshotSource
    private let audit: RepackAudit
    private let startTime = Date()

    public convenience init(options: K3RemoteStreamingRepackOptions,
                            audit: RepackAudit = RepackAudit()) {
        self.init(options: options,
                  source: K3RemoteSnapshotSource(options: options),
                  audit: audit)
    }

    init(options: K3RemoteStreamingRepackOptions,
         source: K3SnapshotSource,
         audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.source = source
        self.audit = audit
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in })
        async throws -> K3RemoteStreamingRepackResult {
        try validateOptions()
        let installLock = try InstallLock.acquire(outputDirectory: options.outputDir)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        if try Posix.entryKind(paths.finalDirectory) == .directory, !options.overwrite {
            throw RepackError.configurationInvalid(detail:
                "output directory already exists: \(paths.finalDirectory)")
        }
        let hasPartial = try Posix.entryKind(paths.partialDirectory) == .directory
        let hasCheckpoint = try Posix.entryKind(paths.checkpointFile) == .regular
        guard hasPartial == hasCheckpoint else {
            throw RepackError.installStateCorrupt(
                path: paths.partialDirectory,
                detail: "partial directory and checkpoint must exist together")
        }
        if options.resume {
            guard hasPartial else {
                throw RepackError.installStateMissing(path: paths.checkpointFile)
            }
        } else if hasPartial {
            throw RepackError.installStateIncompatible(
                detail: "saved download exists; resume or discard it")
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

    private func runPrepared(paths: RemoteInstallPaths,
                             progress: @escaping @Sendable (ModelInstallProgress) -> Void)
        async throws -> K3RemoteStreamingRepackResult {
        try Task.checkCancellation()
        let saved = options.resume
            ? try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
            : nil
        if let saved {
            guard saved.repoID == options.repoID,
                  saved.requestedRevision == options.revision else {
                throw RepackError.installStateIncompatible(
                    detail: "saved download belongs to a different source")
            }
        }
        progress(.downloadingMetadata)
        let snapshot = try await source.loadSnapshot(
            metadataDirectory: paths.metadataDirectory,
            partialDirectory: paths.partialDirectory,
            audit: audit)
        try Task.checkCancellation()
        let plan = try K3RepackPlanner.plan(arch: snapshot.arch,
                                            shardHeaders: snapshot.shardHeaders,
                                            outputDir: paths.partialDirectory,
                                            trunkQuant: options.trunkQuant)
        let expertReuse = try options.reuseExpertsFrom.map {
            try K3VerifiedExpertReuse.load(sourceDirectory: $0,
                                           plan: plan,
                                           snapshot: snapshot,
                                           repoID: options.repoID)
        }
        let rangePlan = try K3RangeCopyPlanner.plan(repackPlan: plan,
                                                    rangeChunkBytes: options.rangeChunkBytes,
                                                    includeExpertPayloads: expertReuse == nil)
        var checkpoint = saved ?? RemoteInstallCheckpoint(
            repoID: options.repoID,
            requestedRevision: options.revision,
            resolvedCommit: snapshot.resolvedCommit,
            sourceIndexSHA256: snapshot.indexSha256Hex,
            planFingerprint: rangePlan.canonicalFingerprint,
            totalSourceBytes: rangePlan.remoteBytesToDownload)

        var phaseOneComplete = false
        if saved != nil {
            guard checkpoint.resolvedCommit == snapshot.resolvedCommit,
                  checkpoint.totalSourceBytes == rangePlan.remoteBytesToDownload,
                  checkpoint.matches(
                      repoID: options.repoID,
                      requestedRevision: options.revision,
                      sourceIndexSHA256: snapshot.indexSha256Hex,
                      planFingerprint: rangePlan.canonicalFingerprint) else {
                throw RepackError.installStateIncompatible(
                    detail: "saved download source or copy plan changed")
            }
            let stagingPresent = try Posix.entryKind(plan.stagingPath) == .regular
            if !stagingPresent {
                // The staging file is deleted only after every range is
                // committed, the trunk is quantized, and the manifest is
                // written — so a staging-less partial can only be awaiting
                // promotion.
                guard checkpoint.completedRanges.count
                        == rangePlan.coalescedCopies.count else {
                    throw RepackError.installStateCorrupt(
                        path: paths.partialDirectory,
                        detail: "staging file removed before all ranges completed")
                }
                return try promoteFinalized(paths: paths,
                                            plan: plan,
                                            rangePlan: rangePlan,
                                            snapshot: snapshot,
                                            checkpoint: checkpoint,
                                            progress: progress)
            }
            if try outputFilesMatch(plan: plan, rangePlan: rangePlan) {
                checkpoint.completedRanges = try RemoteStreamingRepacker
                    .validatedCompletedRanges(
                        checkpoint.completedRanges,
                        copies: rangePlan.coalescedCopies,
                        partialDirectory: paths.partialDirectory)
            } else {
                checkpoint.completedRanges = []
                try FileManager.default.removeItem(atPath: paths.partialDirectory)
                try Posix.mkdirP(paths.partialDirectory)
                try createOutputFiles(plan: plan, paths: paths, expertReuse: expertReuse)
            }
            phaseOneComplete = checkpoint.completedRanges.count
                == rangePlan.coalescedCopies.count
            try checkpoint.write(to: paths.checkpointFile,
                                 parentDirectory: paths.parentDirectory)
        }

        let outputBytes = plan.outputBytes
        progress(.planning(downloadBytes: rangePlan.remoteBytesToDownload,
                           outputBytes: outputBytes))
        let reusedDestinationBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.destinationBytes
        }
        let cloneReusedBytes = expertReuse?.logicalBytes ?? 0
        let totalFootprint = outputBytes - cloneReusedBytes + plan.stagingSize
        let remainingOutputBytes = totalFootprint > reusedDestinationBytes
            ? totalFootprint - reusedDestinationBytes
            : 0
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: remainingOutputBytes + UInt64(options.rangeChunkBytes),
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()

        audit.remoteRepoID = options.repoID
        audit.remoteRequestedRevision = options.revision
        audit.remoteResolvedCommit = snapshot.resolvedCommit
        audit.remoteRangeStreamingSupported = true
        audit.remoteGapBytesDownloaded = rangePlan.remoteGapBytesDownloaded
        audit.sourceSnapshotSha256 = snapshot.indexSha256Hex
        audit.tensorsDroppedMultimodal = plan.excludedTensorNames
        audit.packedExpertLayoutMode = expertReuse == nil
            ? "identity" : "apfs-clone-verified-experts"

        if options.dryRunSpaceCheck {
            if saved == nil {
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            }
            return K3RemoteStreamingRepackResult(
                outputDir: options.outputDir,
                resolvedCommit: snapshot.resolvedCommit,
                plan: plan,
                rangeRequestCount: rangePlan.coalescedCopies.count,
                remoteBytesToDownload: rangePlan.remoteBytesToDownload,
                remoteGapBytesDownloaded: rangePlan.remoteGapBytesDownloaded,
                remoteRetryCount: audit.remoteRangeRetries,
                reusedBytes: checkpoint.completedRanges.reduce(0) { $0 + $1.sourceBytes }
                    + cloneReusedBytes,
                downloadedThisRunBytes: 0,
                dryRun: true)
        }

        if saved == nil {
            progress(.reservingOutput(bytes: totalFootprint))
            try createOutputFiles(plan: plan, paths: paths, expertReuse: expertReuse)
            try checkpoint.write(to: paths.checkpointFile,
                                 parentDirectory: paths.parentDirectory)
        }

        let reusedBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.sourceBytes
        }
        let payloadDownloadStart = audit.remoteBytesDownloaded
        if !phaseOneComplete {
            let provider = try source.makeByteProvider(
                snapshot: snapshot, writeTileBytes: options.writeTileBytes)
            progress(.copyingPayload(reusedBytes: reusedBytes,
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
                    try checkpoint.write(to: paths.checkpointFile,
                                         parentDirectory: paths.parentDirectory)
                })
        }

        // Phase 2: staged BF16 trunk -> quantized resident regions.
        try quantizeStagedTrunk(plan: plan)

        try recordOutputFile(relativePath: "model_weights.bin",
                             path: plan.resident.path,
                             progress: progress)
        if let expertReuse {
            try expertReuse.recordFiles(
                in: audit, destinationRoot: paths.partialDirectory)
        } else {
            for layer in plan.layers {
                try Task.checkCancellation()
                let rel = "packed_experts/" + (layer.path as NSString).lastPathComponent
                try recordOutputFile(relativePath: rel, path: layer.path, progress: progress)
            }
        }

        let layoutData = try K3GTurboJSON.encodeLayout(plan: plan)
        let layoutPath = ((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts") as NSString)
            .appendingPathComponent("layout.json")
        try writeSmall(path: layoutPath, data: layoutData)
        try recordOutputFile(relativePath: "packed_experts/layout.json",
                             path: layoutPath,
                             progress: progress)

        try Task.checkCancellation()
        let stagedSidecars = try await source.stageTokenizerFiles(
            metadataDirectory: paths.metadataDirectory,
            partialDirectory: paths.partialDirectory,
            audit: audit)
        for rel in stagedSidecars {
            try recordOutputFile(relativePath: rel,
                                 path: (paths.partialDirectory as NSString)
                                     .appendingPathComponent(rel),
                                 progress: progress)
        }

        try? FileManager.default.removeItem(atPath: paths.rangeTemporaryFile)
        try? FileManager.default.removeItem(atPath: paths.metadataDirectory)
        progress(.finalizing)
        try Task.checkCancellation()
        try writeManifestAndReceipt(plan: plan,
                                    snapshot: snapshot,
                                    layoutData: layoutData,
                                    partialDir: paths.partialDirectory)

        try Task.checkCancellation()
        // Point of no return: after the staging file is deleted a re-run can
        // no longer re-quantize, so everything the bundle needs is fsynced
        // first. A crash between here and the rename resumes into
        // `promoteFinalized`.
        try FileManager.default.removeItem(atPath: plan.stagingPath)
        try Posix.fsyncDirectory(paths.partialDirectory)
        try promote(paths: paths)

        try? FileManager.default.removeItem(atPath: paths.checkpointFile)

        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false
        if let auditPath = options.copyAuditPath {
            let data = try audit.toJSONData(outputDir: options.outputDir)
            try Posix.mkdirP((auditPath as NSString).deletingLastPathComponent)
            try data.write(to: URL(fileURLWithPath: auditPath))
        }

        return K3RemoteStreamingRepackResult(
            outputDir: options.outputDir,
            resolvedCommit: snapshot.resolvedCommit,
            plan: plan,
            rangeRequestCount: rangePlan.coalescedCopies.count,
            remoteBytesToDownload: rangePlan.remoteBytesToDownload,
            remoteGapBytesDownloaded: rangePlan.remoteGapBytesDownloaded,
            remoteRetryCount: audit.remoteRangeRetries,
            reusedBytes: reusedBytes + cloneReusedBytes,
            downloadedThisRunBytes: audit.remoteBytesDownloaded - payloadDownloadStart,
            dryRun: false)
    }

    // MARK: - Resume state: finalized but not promoted

    /// The previous run wrote the manifest, deleted the staging file, and
    /// died before the rename. Verify the partial directory holds the
    /// complete bundle, then promote.
    private func promoteFinalized(paths: RemoteInstallPaths,
                                  plan: K3RepackPlan,
                                  rangePlan: RangeCopyPlan,
                                  snapshot: K3Snapshot,
                                  checkpoint: RemoteInstallCheckpoint,
                                  progress: @escaping @Sendable (ModelInstallProgress) -> Void)
        throws -> K3RemoteStreamingRepackResult {
        for expected in rangePlan.expectedOutputs
        where expected.relativePath
                != (plan.stagingPath as NSString).lastPathComponent {
            let path = (paths.partialDirectory as NSString)
                .appendingPathComponent(expected.relativePath)
            guard try Posix.entryKind(path) == .regular else {
                throw RepackError.installStateCorrupt(
                    path: paths.partialDirectory,
                    detail: "finalized install is missing \(expected.relativePath)")
            }
            let fd = try Posix.openReadNoFollow(path)
            let size = try Posix.fileSize(fd: fd, path: path)
            close(fd)
            guard size == expected.size else {
                throw RepackError.installStateCorrupt(
                    path: path,
                    detail: "finalized \(expected.relativePath) size \(size) "
                        + "!= \(expected.size)")
            }
        }
        for rel in ["packed_experts/layout.json", "manifest.json",
                    VerifiedInstallReceiptWriter.fileName] {
            guard try Posix.entryKind((paths.partialDirectory as NSString)
                    .appendingPathComponent(rel)) == .regular else {
                throw RepackError.installStateCorrupt(
                    path: paths.partialDirectory,
                    detail: "finalized install is missing \(rel)")
            }
        }
        progress(.finalizing)
        try promote(paths: paths)
        try? FileManager.default.removeItem(atPath: paths.checkpointFile)
        return K3RemoteStreamingRepackResult(
            outputDir: options.outputDir,
            resolvedCommit: snapshot.resolvedCommit,
            plan: plan,
            rangeRequestCount: rangePlan.coalescedCopies.count,
            remoteBytesToDownload: rangePlan.remoteBytesToDownload,
            remoteGapBytesDownloaded: rangePlan.remoteGapBytesDownloaded,
            remoteRetryCount: audit.remoteRangeRetries,
            reusedBytes: checkpoint.completedRanges.reduce(0) { $0 + $1.sourceBytes },
            downloadedThisRunBytes: 0,
            dryRun: false)
    }

    private func promote(paths: RemoteInstallPaths) throws {
        if try Posix.entryKind(paths.finalDirectory) == .directory {
            try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
        } else {
            try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
        }
    }

    // MARK: - Phase 2: quantize the staged trunk

    private func quantizeStagedTrunk(plan: K3RepackPlan) throws {
        let stagingFD = try Posix.openReadNoFollow(plan.stagingPath)
        defer { close(stagingFD) }
        let residentFD = try Posix.openExistingRW(plan.resident.path)
        defer { close(residentFD) }
        // Stream-through I/O: neither the staging read nor the resident
        // re-read (hashing) should evict the working set.
        _ = fcntl(stagingFD, F_NOCACHE, 1)
        _ = fcntl(residentFD, F_NOCACHE, 1)

        for tensor in plan.tensors {
            try Task.checkCancellation()
            switch tensor.transform {
            case .verbatim, .synthesizeZeros:
                continue
            case .affine(let bits, let stagingOffset):
                try K3TrunkQuantizer.quantizeStagedTensor(
                    stagingFD: stagingFD, stagingPath: plan.stagingPath,
                    stagingOffset: stagingOffset,
                    residentFD: residentFD, residentPath: plan.resident.path,
                    weightOffset: tensor.entry.fileOffset,
                    scaleOffset: tensor.entry.scaleOffset,
                    biasOffset: tensor.entry.biasOffset,
                    rows: tensor.rows, columns: tensor.columns, bits: bits,
                    audit: audit)
            case .widenFP32(let stagingOffset):
                try K3TrunkQuantizer.widenStagedTensor(
                    stagingFD: stagingFD, stagingPath: plan.stagingPath,
                    stagingOffset: stagingOffset,
                    residentFD: residentFD, residentPath: plan.resident.path,
                    residentOffset: tensor.entry.fileOffset,
                    elementCount: tensor.rows * tensor.columns,
                    audit: audit)
            case .truncateFP32(let stagingOffset, let sourceCount, let keepCount):
                try K3TrunkQuantizer.truncateStagedTensor(
                    stagingFD: stagingFD, stagingPath: plan.stagingPath,
                    stagingOffset: stagingOffset,
                    residentFD: residentFD, residentPath: plan.resident.path,
                    residentOffset: tensor.entry.fileOffset,
                    sourceCount: sourceCount, keepCount: keepCount,
                    name: tensor.entry.name,
                    audit: audit)
            }
        }
        try Posix.fsync(residentFD, path: plan.resident.path)
    }

    // MARK: - Output files

    private func createOutputFiles(plan: K3RepackPlan,
                                   paths: RemoteInstallPaths,
                                   expertReuse: K3VerifiedExpertReuse?) throws {
        try Posix.mkdirP((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts"))
        let resident = try ResidentWriter.createAndWriteIndex(plan: plan.resident,
                                                              audit: audit)
        try Posix.fsync(resident, path: plan.resident.path)
        close(resident)
        let staging = try Posix.openCreateRW(plan.stagingPath)
        do {
            try Posix.ftruncate(staging, path: plan.stagingPath, size: plan.stagingSize)
            try Posix.fsync(staging, path: plan.stagingPath)
            close(staging)
        } catch {
            close(staging)
            throw error
        }
        if let expertReuse {
            try expertReuse.cloneExperts(into: plan)
        } else {
            for layer in plan.layers {
                try Task.checkCancellation()
                let descriptor = try Posix.openCreateRW(layer.path)
                try Posix.ftruncate(descriptor, path: layer.path, size: layer.fileSize)
                try Posix.fsync(descriptor, path: layer.path)
                close(descriptor)
            }
        }
        try Posix.fsyncDirectory(paths.partialDirectory)
    }

    private func outputFilesMatch(plan: K3RepackPlan,
                                  rangePlan: RangeCopyPlan) throws -> Bool {
        for output in rangePlan.expectedOutputs {
            let path = ((plan.resident.path as NSString).deletingLastPathComponent
                as NSString).appendingPathComponent(output.relativePath)
            guard try Posix.entryKind(path) == .regular else { return false }
            let descriptor = try Posix.openReadNoFollow(path)
            defer { close(descriptor) }
            guard try Posix.fileSize(fd: descriptor, path: path) == output.size else {
                return false
            }
        }

        let expectedIndex = try ResidentWriter.encodeIndex(plan: plan.resident)
        let descriptor = try Posix.openReadNoFollow(plan.resident.path)
        defer { close(descriptor) }
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: min(WriterCore.tileBytes, max(1, expectedIndex.count)),
            alignment: 16_384)
        defer { scratch.deallocate() }
        return try expectedIndex.withUnsafeBytes { expected in
            var offset = 0
            while offset < expected.count {
                let count = min(scratch.count, expected.count - offset)
                try Posix.preadAll(fd: descriptor,
                                   path: plan.resident.path,
                                   buf: scratch.baseAddress!,
                                   count: count,
                                   offset: UInt64(offset))
                guard memcmp(scratch.baseAddress!,
                             expected.baseAddress!.advanced(by: offset),
                             count) == 0 else { return false }
                offset += count
            }
            return true
        }
    }

    private func recordOutputFile(relativePath: String,
                                  path: String,
                                  progress: @Sendable (ModelInstallProgress) -> Void) throws {
        progress(.hashingOutput(relativePath))
        try Task.checkCancellation()
        let size = try {
            let fd = try Posix.openRead(path)
            defer { close(fd) }
            return try Posix.fileSize(fd: fd, path: path)
        }()
        let sha = try WriterCore.hashEntireFile(path: path,
                                                size: size,
                                                audit: audit,
                                                cancellationCheck: Task.checkCancellation)
        audit.outputFiles.append(.init(relativePath: relativePath, size: size, sha256: sha))
    }

    private func writeSmall(path: String, data: Data) throws {
        try Posix.mkdirP((path as NSString).deletingLastPathComponent)
        try data.write(to: URL(fileURLWithPath: path), options: [.atomic])
        audit.recordWrite(bytes: data.count)
    }

    private func writeManifestAndReceipt(plan: K3RepackPlan,
                                         snapshot: K3Snapshot,
                                         layoutData: Data,
                                         partialDir: String) throws {
        let files = audit.outputFiles.map {
            (relativePath: $0.relativePath, size: $0.size, sha256: $0.sha256)
        }
        let data = try K3GTurboJSON.encodeManifest(
            plan: plan,
            modelID: options.repoID,
            sourceSnapshotHash: "sha256:" + snapshot.indexSha256Hex,
            files: files)
        // Final structural proof before the manifest lands: manifest vs
        // layout cross-validation (dims, MoE layer set, layer file sizes).
        let layout = try GTurboPackedExpertsLayoutCodecV2.decode(layoutData)
        try GTurboV2StructuralValidator.crossValidate(
            manifest: GTurboManifestCodecV2.decode(data), layout: layout)

        let tmp = (partialDir as NSString).appendingPathComponent("manifest.json.tmp")
        let final = (partialDir as NSString).appendingPathComponent("manifest.json")
        try writeSmall(path: tmp, data: data)
        try Posix.rename(from: tmp, to: final)
        let manifestSha = try Sha256Stream.hashFile(path: final)
        let receipt = try VerifiedInstallReceiptWriter.encode(
            outputDir: options.outputDir,
            manifestSha256: manifestSha,
            manifestSize: UInt64(data.count),
            sourceRepoID: options.repoID,
            sourceRevision: snapshot.resolvedCommit,
            files: audit.outputFiles)
        let receiptPath = (partialDir as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        try writeSmall(path: receiptPath, data: receipt)
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
        if let reuseSource = options.reuseExpertsFrom {
            guard options.overwrite else {
                throw RepackError.configurationInvalid(
                    detail: "expert reuse requires atomic overwrite of an existing bundle")
            }
            guard try Posix.entryKind(
                URL(fileURLWithPath: reuseSource).standardizedFileURL.path) == .directory else {
                throw RepackError.configurationInvalid(
                    detail: "expert reuse source does not exist: \(reuseSource)")
            }
        }
    }
}
