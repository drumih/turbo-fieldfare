import Foundation

// The prepared remote repack run: one pass over the remote manifest, weights
// and sidecars, with the resume receipts.
//
// Split out of `RemoteStreamingRepacker.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
extension RemoteStreamingRepacker {

    /// lint:allow-long the install pipeline for one prepared plan: fetch
    /// ranges, verify, write, checkpoint, promote. The stages share the
    /// checkpoint, the byte budget and the progress reporter, and their order
    /// is the resumability contract -- separating them would move that
    /// contract into parameter lists.
    func runPrepared(
        paths: RemoteInstallPaths,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws
        -> RemoteStreamingRepackResult
    {
        try Task.checkCancellation()
        let saved =
            options.resume
            ? try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
            : nil
        if let saved {
            guard saved.repoID == options.repoID,
                saved.requestedRevision == options.revision
            else {
                throw RepackError.installStateIncompatible(
                    detail: "saved download belongs to a different source")
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
            installDraftHead: options.installDraftHead,
            audit: audit)
        try Task.checkCancellation()
        // Files this family carries verbatim alongside the tensor payload.
        // They are standalone files rather than index entries, so their sizes
        // come from the remote before planning; the planner then treats them
        // as ordinary resumable range copies.
        var passthroughFiles: [PassthroughFile] = []
        // The transfer layer resolves a copy's source through this map. These
        // files are not safetensors shards, so the snapshot loader never put
        // them there -- carrying the resolved info forward is what makes them
        // fetchable rather than merely planned.
        var passthroughRemoteInfo: [String: RemoteFileInfo] = [:]
        for requirement in RepackPlanner.passthroughRequirements(
            family: snapshot.arch.family)
        {
            let info: RemoteFileInfo
            do {
                info = try await remote.resolveFileInfo(
                    filename: requirement.name,
                    audit: audit)
            } catch {
                // An absent optional file leaves a runnable install; an absent
                // required one does not, and must not be discovered later.
                if requirement.required || !isRemoteNotFound(error) { throw error }
                continue
            }
            guard info.size <= requirement.capBytes else {
                throw RepackError.remoteFileTooLarge(
                    path: requirement.name,
                    size: info.size,
                    cap: requirement.capBytes)
            }
            passthroughFiles.append(
                PassthroughFile(
                    sourceName: requirement.name,
                    destinationName: requirement.name,
                    size: info.size,
                    required: requirement.required))
            passthroughRemoteInfo[requirement.name] = info
        }
        let plan = try RepackPlanner.plan(
            meta: snapshot.metadata,
            arch: snapshot.arch,
            shardHeaders: snapshot.shardHeaders,
            outputDir: paths.partialDirectory,
            passthroughFiles: passthroughFiles)
        let rangePlan = try RangeCopyPlanner.plan(
            repackPlan: plan,
            rangeChunkBytes: options.rangeChunkBytes,
            layoutMode: "identity",
            layoutOrderSha256: nil)
        var checkpoint =
            saved
            ?? RemoteInstallCheckpoint(
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
                    planFingerprint: rangePlan.canonicalFingerprint)
            else {
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
                // A space check must not destroy a partial download. It issues no
                // requests (see the early return below) and its caller may decide
                // not to proceed, yet this branch used to remove the partial
                // directory and preallocate every output file -- tens or hundreds
                // of gigabytes of transferred data gone to answer "would it fit".
                // The reservation is computed from the plan and the checkpoint,
                // never from what is on disk, so a dry run does not need the files
                // to exist. Every real call passes `dryRunSpaceCheck: false`, so
                // this is the only branch that changes.
                if !options.dryRunSpaceCheck {
                    try FileManager.default.removeItem(atPath: paths.partialDirectory)
                    try Posix.mkdirP(paths.partialDirectory)
                    try createOutputFiles(plan: plan, paths: paths)
                }
            }
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.parentDirectory)
        }
        // The passthrough files are part of the install, not an afterthought:
        // `createOutputFiles` ftruncates each of them to its full size and the
        // range provider writes all of it. Leaving them out here let a
        // Qwen3.8-Flash-Next install pass a reservation that counted the ~66 GiB
        // backbone while the install really needed ~168 GiB, and die with ENOSPC
        // after tens of GiB of transfer. Qwen3.8's n-gram table alone is ~95 GiB
        // against a 256 GiB per-file cap.
        let outputBytes =
            plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
            + plan.passthroughFiles.reduce(UInt64(0)) { $0 + $1.size }
        progress(
            .planning(
                downloadBytes: rangePlan.remoteBytesToDownload,
                outputBytes: outputBytes))
        let reusedDestinationBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.destinationBytes
        }
        let remainingOutputBytes =
            outputBytes > reusedDestinationBytes
            ? outputBytes - reusedDestinationBytes
            : 0
        // The extra chunk budget accounts for the `.range.tmp` staging file
        // (at most one chunk is staged at a time, including on failure paths
        // where the file is unlinked before the error propagates).
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
            return RemoteStreamingRepackResult(
                outputDir: options.outputDir,
                resolvedCommit: snapshot.resolvedCommit,
                plan: plan,
                // Dry run issues no HTTP requests,
                // so report the planned count.
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

        let provider = HTTPRangeSourceByteProvider(
            remote: remote.pinned(commit: snapshot.resolvedCommit),
            files: snapshot.remoteFiles
                .merging(passthroughRemoteInfo) { shard, _ in shard },
            writeTileBytes: options.writeTileBytes)
        let reusedBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.sourceBytes
        }
        let payloadDownloadStart = audit.remoteBytesDownloaded
        progress(
            .copyingPayload(
                reusedBytes: reusedBytes,
                downloadedThisRunBytes: 0,
                totalBytes: rangePlan.remoteBytesToDownload))
        // The checkpoint is rewritten at most once per 16 coalesced ranges
        // (or 64 MiB of payload) instead of after every range. The first
        // commit is still written immediately so an early cancellation keeps
        // its completed ranges, and one final fsynced write after the batch
        // makes the last ranges durable before they are relied on. Resume
        // correctness does not depend on the checkpoint being fresh: the next
        // run re-hashes destination bytes before trusting completedRanges.
        var rangesSinceCheckpointWrite = 0
        var pendingCheckpointBytes: UInt64 = 0
        try await provider.copyBatch(
            rangePlan.coalescedCopies,
            completedRangeIDs: Set(checkpoint.completedRanges.map(\.id)),
            partialDirectory: paths.partialDirectory,
            temporaryPath: paths.rangeTemporaryFile,
            audit: audit,
            progress: { downloadedBytes in
                progress(
                    .copyingPayload(
                        reusedBytes: reusedBytes,
                        downloadedThisRunBytes: downloadedBytes,
                        totalBytes: rangePlan.remoteBytesToDownload))
            },
            commit: { completed in
                checkpoint.completedRanges.removeAll { $0.id == completed.id }
                checkpoint.completedRanges.append(completed)
                checkpoint.completedRanges.sort { $0.id < $1.id }
                rangesSinceCheckpointWrite += 1
                pendingCheckpointBytes += completed.sourceBytes
                // The first commit is written immediately (an early
                // cancellation must keep its completed ranges); afterwards the
                // checkpoint is rewritten at most once every 16 ranges or
                // 64 MiB of payload.
                if rangesSinceCheckpointWrite == 1
                    || rangesSinceCheckpointWrite % 16 == 0
                    || pendingCheckpointBytes >= 64 * 1024 * 1024
                {
                    pendingCheckpointBytes = 0
                    try checkpoint.write(
                        to: paths.checkpointFile,
                        parentDirectory: paths.parentDirectory)
                }
            },
            consumeSource: false)
        try checkpoint.write(
            to: paths.checkpointFile,
            parentDirectory: paths.parentDirectory)

        try recordOutputFile(
            relativePath: "model_weights.bin",
            path: plan.resident.path,
            progress: progress)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let rel = "packed_experts/" + (layer.path as NSString).lastPathComponent
            try recordOutputFile(relativePath: rel, path: layer.path, progress: progress)
        }
        // Recorded so the manifest carries their size and digest, which is
        // what the install receipt attests over.
        for file in plan.passthroughFiles {
            try Task.checkCancellation()
            let path = (paths.partialDirectory as NSString)
                .appendingPathComponent(file.destinationName)
            try recordOutputFile(
                relativePath: file.destinationName,
                path: path, progress: progress)
        }

        let layoutPath =
            ((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts") as NSString)
            .appendingPathComponent("layout.json")
        let expertStride = plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0
        let layoutData = try SSDAIJSON.encodeLayout(plan: plan, expertStride: expertStride)
        try writeSmall(path: layoutPath, data: layoutData)
        try SSDAILayoutValidator.validate(path: layoutPath, plan: plan)
        try recordOutputFile(
            relativePath: "packed_experts/layout.json",
            path: layoutPath,
            progress: progress)

        try Task.checkCancellation()
        // The MTP sidecar deliberately contains only tensors needed by the
        // draft layer. It shares tokenization, embedding and lm_head with the
        // target bundle, so copying tokenizer/config sidecars would be both
        // redundant and a misleading standalone-model contract.
        if plan.arch.family != .qwen36MTP && plan.arch.family != .qwen38flashMTP
            && plan.arch.family != .qwen4flashMTP
        {
            try await copyRemoteMetadataSidecars(
                snapshot: snapshot,
                remote: remote,
                partialDir: paths.partialDirectory,
                progress: progress)
        }
        try? FileManager.default.removeItem(atPath: paths.rangeTemporaryFile)
        try? FileManager.default.removeItem(atPath: paths.metadataDirectory)
        progress(.finalizing)
        try Task.checkCancellation()
        try writeManifest(
            plan: plan,
            partialDir: paths.partialDirectory,
            metadata: snapshot.metadata,
            expertStride: expertStride,
            resolvedCommit: snapshot.resolvedCommit,
            modelIDOverride: nil)

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

        return RemoteStreamingRepackResult(
            outputDir: options.outputDir,
            resolvedCommit: snapshot.resolvedCommit,
            plan: plan,
            // Actual ranged HTTP requests issued
            // this run, counted by the byte
            // provider (retries are separate).
            rangeRequestCount: Int(
                min(
                    audit.remoteRangeRequests,
                    UInt64(Int.max))),
            remoteBytesToDownload: rangePlan.remoteBytesToDownload,
            remoteGapBytesDownloaded: rangePlan.remoteGapBytesDownloaded,
            remoteRetryCount: audit.remoteRangeRetries,
            reusedBytes: reusedBytes,
            downloadedThisRunBytes:
                audit.remoteBytesDownloaded - payloadDownloadStart,
            dryRun: false)
    }
}
