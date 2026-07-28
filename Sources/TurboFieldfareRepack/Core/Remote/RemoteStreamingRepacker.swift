import Foundation
public struct RemoteStreamingRepackOptions: Sendable {
    public let repoID: String
    public let revision: String
    public let outputDir: String
    public let token: String?
    public let requireKnownSource: Bool
    public let copyAuditPath: String?
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let overwrite: Bool
    public let resume: Bool
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
                copyAuditPath: String? = nil,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                writeTileBytes: Int = WriterCore.tileBytes,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
                overwrite: Bool = false,
                resume: Bool = false,
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
        self.copyAuditPath = copyAuditPath
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.overwrite = overwrite
        self.resume = resume
        self.dryRunSpaceCheck = dryRunSpaceCheck
        self.downloadSession = downloadSession
        self.baseURL = baseURL
        self.rangeRetryAttempts = rangeRetryAttempts
        self.retryBaseDelayNs = retryBaseDelayNs
    }
}

public struct RemoteStreamingRepackResult: Sendable {
    public let outputDir: String
    public let resolvedCommit: String
    let plan: RepackPlan
    public let rangeRequestCount: Int
    public let remoteBytesToDownload: UInt64
    public let remoteGapBytesDownloaded: UInt64
    public let remoteRetryCount: UInt64
    public let reusedBytes: UInt64
    public let downloadedThisRunBytes: UInt64
    public let dryRun: Bool
}

public final class RemoteStreamingRepacker {
    private let options: RemoteStreamingRepackOptions
    private let audit: RepackAudit
    private let startTime = Date()

    public init(options: RemoteStreamingRepackOptions,
                audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.audit = audit
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws
        -> RemoteStreamingRepackResult {
        try validateOptions()
        let installLock = try InstallLock.acquire(outputDirectory: options.outputDir)
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

    public static func inspectPersistentInstall(
        outputDirectory: String,
        repoID: String,
        requestedRevision: String
    ) throws -> RemoteInstallCheckpoint? {
        let lock = try InstallLock.acquire(outputDirectory: outputDirectory)
        let paths = lock.paths
        let partial = try Posix.entryKind(paths.partialDirectory)
        let checkpoint = try Posix.entryKind(paths.checkpointFile)
        if partial == .absent, checkpoint == .absent { return nil }
        guard partial == .directory, checkpoint == .regular else {
            throw RepackError.installStateCorrupt(
                path: paths.partialDirectory,
                detail: "partial directory and checkpoint must exist together")
        }
        let value = try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
        guard value.repoID == repoID, value.requestedRevision == requestedRevision else {
            throw RepackError.installStateIncompatible(
                detail: "saved download belongs to a different source")
        }
        return value
    }

    public static func discardPartial(outputDirectory: String) throws {
        let lock = try InstallLock.acquire(outputDirectory: outputDirectory)
        let paths = lock.paths
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
        try Posix.fsyncDirectory(paths.parentDirectory)
    }

    private func runPrepared(paths: RemoteInstallPaths,
                             progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> RemoteStreamingRepackResult {
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
        let retryPolicy = RemoteRetryPolicy(attempts: options.rangeRetryAttempts,
                                            baseDelayNs: options.retryBaseDelayNs)
        let remote = HuggingFaceRemoteSource(repoID: options.repoID,
                                             requestedRevision: options.revision,
                                             resolvedCommit: saved?.resolvedCommit,
                                             token: options.token,
                                             downloadSession: options.downloadSession,
                                             baseURL: options.baseURL,
                                             tempDirectory: paths.partialDirectory,
                                             retryPolicy: retryPolicy)
        progress(.downloadingMetadata)
        let snapshot = try await RemoteSnapshotLoader.load(remote: remote,
                                                           requireKnownSource: options.requireKnownSource,
                                                           metadataDirectory: paths.metadataDirectory,
                                                           audit: audit)
        try Task.checkCancellation()
        let plan = try RepackPlanner.plan(meta: snapshot.metadata,
                                          arch: snapshot.arch,
                                          shardHeaders: snapshot.shardHeaders,
                                          outputDir: paths.partialDirectory)
        let rangePlan = try RangeCopyPlanner.plan(repackPlan: plan,
                                                  rangeChunkBytes: options.rangeChunkBytes,
                                                  layoutMode: "identity",
                                                  layoutOrderSha256: nil)
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
                    detail: "saved download source or copy plan changed")
            }
            if try outputFilesMatch(plan: plan, rangePlan: rangePlan) {
                checkpoint.completedRanges = try Self.validatedCompletedRanges(
                    checkpoint.completedRanges,
                    copies: rangePlan.coalescedCopies,
                    partialDirectory: paths.partialDirectory)
            } else {
                checkpoint.completedRanges = []
                try FileManager.default.removeItem(atPath: paths.partialDirectory)
                try Posix.mkdirP(paths.partialDirectory)
                try createOutputFiles(plan: plan, paths: paths)
            }
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.parentDirectory)
        }
        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
        progress(.planning(downloadBytes: rangePlan.remoteBytesToDownload,
                           outputBytes: outputBytes))
        let reusedDestinationBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.destinationBytes
        }
        let remainingOutputBytes = outputBytes > reusedDestinationBytes
            ? outputBytes - reusedDestinationBytes
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
        audit.sourceSnapshotSha256 = snapshot.metadata.indexSha256Hex
        audit.bitWidthOverridesHonored = snapshot.metadata.bitsOverrides.count
        audit.tensorsDroppedMultimodal = plan.excludedMultimodalTensorNames
        audit.packedExpertLayoutMode = "identity"

        if options.dryRunSpaceCheck {
            if saved == nil {
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            }
            return RemoteStreamingRepackResult(outputDir: options.outputDir,
                                               resolvedCommit: snapshot.resolvedCommit,
                                               plan: plan,
                                               rangeRequestCount: rangePlan.coalescedCopies.count,
                                               remoteBytesToDownload: rangePlan.remoteBytesToDownload,
                                               remoteGapBytesDownloaded: rangePlan.remoteGapBytesDownloaded,
                                               remoteRetryCount: audit.remoteRangeRetries,
                                               reusedBytes: checkpoint.completedRanges.reduce(0) {
                                                   $0 + $1.sourceBytes
                                               },
                                               downloadedThisRunBytes: 0,
                                               dryRun: true)
        }

        if saved == nil {
            progress(.reservingOutput(bytes: outputBytes))
            try createOutputFiles(plan: plan, paths: paths)
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.parentDirectory)
        }

        let provider = HTTPRangeSourceByteProvider(remote: remote.pinned(commit: snapshot.resolvedCommit),
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
                    parentDirectory: paths.parentDirectory)
            })

        try recordOutputFile(relativePath: "model_weights.bin",
                             path: plan.resident.path,
                             progress: progress)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let rel = "packed_experts/" + (layer.path as NSString).lastPathComponent
            try recordOutputFile(relativePath: rel, path: layer.path, progress: progress)
        }

        let layoutPath = ((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts") as NSString)
            .appendingPathComponent("layout.json")
        let expertStride = plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0
        let layoutData = try GTurboJSON.encodeLayout(plan: plan, expertStride: expertStride)
        try writeSmall(path: layoutPath, data: layoutData)
        try GTurboLayoutValidator.validate(path: layoutPath, plan: plan)
        try recordOutputFile(relativePath: "packed_experts/layout.json",
                             path: layoutPath,
                             progress: progress)

        try Task.checkCancellation()
        try await copyRemoteMetadataSidecars(snapshot: snapshot,
                                             remote: remote,
                                             partialDir: paths.partialDirectory,
                                             progress: progress)
        try? FileManager.default.removeItem(atPath: paths.rangeTemporaryFile)
        try? FileManager.default.removeItem(atPath: paths.metadataDirectory)
        progress(.finalizing)
        try Task.checkCancellation()
        try writeManifest(plan: plan,
                          partialDir: paths.partialDirectory,
                          metadata: snapshot.metadata,
                          expertStride: expertStride,
                          resolvedCommit: snapshot.resolvedCommit)

        try Task.checkCancellation()
        if try Posix.entryKind(paths.finalDirectory) == .directory {
            try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
        } else {
            try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
        }
        try? FileManager.default.removeItem(atPath: paths.checkpointFile)

        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false
        if let auditPath = options.copyAuditPath {
            let data = try audit.toJSONData(outputDir: options.outputDir)
            try Posix.mkdirP((auditPath as NSString).deletingLastPathComponent)
            try data.write(to: URL(fileURLWithPath: auditPath))
        }

        return RemoteStreamingRepackResult(outputDir: options.outputDir,
                                           resolvedCommit: snapshot.resolvedCommit,
                                           plan: plan,
                                           rangeRequestCount: rangePlan.coalescedCopies.count,
                                           remoteBytesToDownload: rangePlan.remoteBytesToDownload,
                                           remoteGapBytesDownloaded: rangePlan.remoteGapBytesDownloaded,
                                           remoteRetryCount: audit.remoteRangeRetries,
                                           reusedBytes: reusedBytes,
                                           downloadedThisRunBytes:
                                               audit.remoteBytesDownloaded - payloadDownloadStart,
                                           dryRun: false)
    }

    private func validateOptions() throws {
        guard options.rangeChunkBytes > 0,
              options.rangeChunkBytes <= RemoteChunkPolicy.maxBytes else {
            throw RepackError.configurationInvalid(detail: "bad range chunk bytes \(options.rangeChunkBytes)")
        }
        guard options.writeTileBytes > 0,
              options.writeTileBytes <= BoundedScratch.defaultLimitBytes else {
            throw RepackError.configurationInvalid(detail: "bad write tile bytes \(options.writeTileBytes)")
        }
        guard options.rangeRetryAttempts >= 0 else {
            throw RepackError.configurationInvalid(detail:
                "bad range retry attempts \(options.rangeRetryAttempts)")
        }
    }

    private func createOutputFiles(plan: RepackPlan,
                                   paths: RemoteInstallPaths) throws {
        try Posix.mkdirP((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts"))
        let resident = try ResidentWriter.createAndWriteIndex(
            plan: plan.resident,
            audit: audit)
        try Posix.fsync(resident, path: plan.resident.path)
        close(resident)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let descriptor = try Posix.openCreateRW(layer.path)
            try Posix.ftruncate(descriptor, path: layer.path, size: layer.fileSize)
            try Posix.fsync(descriptor, path: layer.path)
            close(descriptor)
        }
        try Posix.fsyncDirectory(paths.partialDirectory)
    }

    private func outputFilesMatch(plan: RepackPlan,
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
                try Posix.preadAll(
                    fd: descriptor,
                    path: plan.resident.path,
                    buf: scratch.baseAddress!,
                    count: count,
                    offset: UInt64(offset))
                guard memcmp(
                    scratch.baseAddress!,
                    expected.baseAddress!.advanced(by: offset),
                    count) == 0 else { return false }
                offset += count
            }
            return true
        }
    }

    static func validatedCompletedRanges(
        _ completed: [RemoteCompletedRange],
        copies: [CoalescedRangeCopy],
        partialDirectory: String
    ) throws -> [RemoteCompletedRange] {
        let copiesByID = Dictionary(uniqueKeysWithValues: copies.map { ($0.id, $0) })
        var valid: [RemoteCompletedRange] = []
        for range in completed {
            guard let copy = copiesByID[range.id],
                  range.sourceBytes == copy.size,
                  range.destinationBytes
                      == copy.destinations.reduce(UInt64(0), { $0 + $1.size }) else {
                throw RepackError.installStateCorrupt(
                    path: partialDirectory,
                    detail: "checkpoint contains an unknown range")
            }
            let digest = try HTTPRangeSourceByteProvider.destinationDigest(
                copy,
                partialDirectory: partialDirectory)
            if digest == range.destinationDigest {
                valid.append(range)
            }
        }
        return valid.sorted { $0.id < $1.id }
    }

    private func recordOutputFile(relativePath: String,
                                  path: String,
                                  progress: @Sendable (ModelInstallProgress) -> Void) throws {
        progress(.hashingOutput(relativePath))
        try Task.checkCancellation()
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        let size = try Posix.fileSize(fd: fd, path: path)
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

    private func copyRemoteMetadataSidecars(snapshot: RemoteSnapshot,
                                           remote: HuggingFaceRemoteSource,
                                           partialDir: String,
                                           progress: @Sendable (ModelInstallProgress) -> Void) async throws {
        let tokenizerDir = (partialDir as NSString).appendingPathComponent("tokenizer")
        for filename in ["config.json"] {
            try Task.checkCancellation()
            let src = (snapshot.metadataDirectory as NSString).appendingPathComponent(filename)
            guard FileManager.default.fileExists(atPath: src) else { continue }
            try Posix.mkdirP(tokenizerDir)
            let dst = (tokenizerDir as NSString).appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: dst) {
                try FileManager.default.removeItem(atPath: dst)
            }
            try FileManager.default.copyItem(atPath: src, toPath: dst)
            try recordOutputFile(relativePath: "tokenizer/\(filename)",
                                 path: dst,
                                 progress: progress)
        }

        let pinned = remote.pinned(commit: snapshot.resolvedCommit)
        let tokenizerFiles: [(name: String, cap: UInt64, required: Bool)] = [
            ("tokenizer.json", 64 * 1024 * 1024, true),
            ("tokenizer_config.json", 4 * 1024 * 1024, true),
            ("special_tokens_map.json", 1 * 1024 * 1024, false),
            ("chat_template.jinja", 4 * 1024 * 1024, false),
            ("chat_template.json", 4 * 1024 * 1024, false),
        ]
        for file in tokenizerFiles {
            try Task.checkCancellation()
            let info: RemoteFileInfo
            do {
                info = try await pinned.resolveFileInfo(filename: file.name, audit: audit)
            } catch {
                if file.required || !isRemoteNotFound(error) {
                    throw error
                }
                continue
            }
            let dst = (tokenizerDir as NSString).appendingPathComponent(file.name)
            try await pinned.fetchSmallFile(filename: file.name,
                                            info: info,
                                            capBytes: file.cap,
                                            outputPath: dst,
                                            audit: audit)
            try recordOutputFile(relativePath: "tokenizer/\(file.name)",
                                 path: dst,
                                            progress: progress)
        }
    }

    private func isRemoteNotFound(_ error: Error) -> Bool {
        if case RepackError.remoteHTTPStatus(_, 404) = error {
            return true
        }
        if case RepackError.remoteHTTPResponse(_, 404, _) = error {
            return true
        }
        return false
    }

    private func writeManifest(plan: RepackPlan,
                               partialDir: String,
                               metadata: IndexLoader.SourceMetadata,
                               expertStride: UInt64,
                               resolvedCommit: String) throws {
        var bits = GTurboJSON.QuantBitWidths(
            embedding: 4,
            attention: 4,
            router: 8,
            sharedExpert: 8,
            routedExpert: 4)
        for e in plan.resident.entries {
            if e.name == "language_model.model.embed_tokens.weight", let s = e.quantSpec {
                bits.embedding = s.bits
            }
            if e.name.hasSuffix(".self_attn.q_proj.weight"), let s = e.quantSpec {
                bits.attention = s.bits
            }
            if e.name.hasSuffix(".router.proj.weight"), let s = e.quantSpec {
                bits.router = s.bits
            }
            if e.name.hasSuffix(".mlp.gate_proj.weight"), let s = e.quantSpec {
                bits.sharedExpert = s.bits
            }
        }
        if let layer = plan.layers.first(where: { !$0.subTensors.isEmpty }),
           let routedBits = layer.subTensors.first?.bitsForWeights {
            bits.routedExpert = routedBits
        }
        let files = audit.outputFiles.map {
            ($0.relativePath, GTurboJSON.FileEntry(size: $0.size, sha256: $0.sha256))
        }
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: plan.matchedModelID ?? "unknown/snapshot",
            sourceSnapshotHash: "sha256:" + metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: expertStride,
            bitWidths: bits)
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
            sourceRevision: resolvedCommit,
            files: audit.outputFiles)
        let receiptPath = (partialDir as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        try writeSmall(path: receiptPath, data: receipt)
    }
}
