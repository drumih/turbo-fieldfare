import Foundation

// The local-snapshot repack path: validation, planning and the copy pass.
//
// Split out of `RemoteStreamingRepacker.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
extension RemoteStreamingRepacker {

    /// Repack an already-complete local affine safetensors snapshot through
    /// the same planner, file layout, hashing, and trusted-receipt path as a
    /// pinned remote install. Local imports intentionally do not support
    /// resume: the source is already present, so a failed attempt is removed
    /// atomically and can be restarted without network transfer.
    public static func runLocalSnapshot(
        options local: LocalSnapshotRepackOptions,
        audit: RepackAudit = RepackAudit(),
        progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }
    ) async throws -> RemoteStreamingRepackResult {
        // The Inkling 2 checkpoint is planned from its own tensors and config,
        // not through `IndexLoader`, which reads affine quantization only.
        if isInkling2Checkpoint(directory: local.inputSnapshotDir) {
            return try await runLocalInkling2Checkpoint(
                options: local, audit: audit, progress: progress)
        }
        let source = try LocalSnapshotLoader.load(
            directory: local.inputSnapshotDir,
            draftHead: local.draftHead)
        let worker = RemoteStreamingRepacker(
            options: RemoteStreamingRepackOptions(
                repoID: "local/snapshot",
                revision: String(source.metadata.indexSha256Hex.prefix(40)),
                outputDir: local.outputDir,
                requireKnownSource: false,
                rangeChunkBytes: local.rangeChunkBytes,
                writeTileBytes: local.writeTileBytes,
                minFreeReserveBytes: local.minFreeReserveBytes,
                overwrite: local.overwrite),
            audit: audit)
        return try await worker.runLocalPrepared(
            source: source,
            local: local,
            progress: progress)
    }

    func runLocalPrepared(
        source: LocalSnapshot,
        local: LocalSnapshotRepackOptions,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> RemoteStreamingRepackResult {
        try validateOptions()
        try validateLocalModelID(local.modelID)
        let installLock = try InstallLock.acquire(outputDirectory: local.outputDir)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        try validateLocalDestination(paths: paths, overwrite: local.overwrite)
        let (plan, rangePlan, outputBytes) = try prepareLocalPlan(
            source: source, local: local, paths: paths, progress: progress)
        configureLocalAudit(source: source, plan: plan)
        try await executeLocalCopy(
            source: source,
            local: local,
            paths: paths,
            plan: plan,
            rangePlan: rangePlan,
            outputBytes: outputBytes,
            progress: progress)
        return RemoteStreamingRepackResult(
            outputDir: local.outputDir,
            resolvedCommit: String(source.metadata.indexSha256Hex.prefix(40)),
            plan: plan,
            rangeRequestCount: 0,
            remoteBytesToDownload: rangePlan.remoteBytesToDownload,
            remoteGapBytesDownloaded: 0,
            remoteRetryCount: 0,
            reusedBytes: 0,
            downloadedThisRunBytes: rangePlan.remoteBytesToDownload,
            dryRun: false)
    }

    func validateLocalModelID(_ modelID: String) throws {
        guard !modelID.isEmpty,
            modelID.utf8.count <= 256,
            !modelID.contains(where: { $0.isWhitespace })
        else {
            throw RepackError.configurationInvalid(
                detail: "local snapshot model ID must be non-empty and contain no whitespace")
        }
    }

    func validateLocalDestination(
        paths: RemoteInstallPaths,
        overwrite: Bool
    ) throws {
        if try Posix.entryKind(paths.finalDirectory) == .directory,
            !overwrite
        {
            throw RepackError.configurationInvalid(
                detail: "output directory already exists: \(paths.finalDirectory)")
        }
        guard try Posix.entryKind(paths.partialDirectory) == .absent,
            try Posix.entryKind(paths.checkpointFile) == .absent
        else {
            throw RepackError.installStateIncompatible(
                detail: "saved remote download exists; resume or discard it first")
        }
    }

    func prepareLocalPlan(
        source: LocalSnapshot,
        local: LocalSnapshotRepackOptions,
        paths: RemoteInstallPaths,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) throws -> (RepackPlan, RangeCopyPlan, UInt64) {
        progress(.downloadingMetadata)
        // The same non-tensor files the remote path carries. Omitting them
        // here would silently produce an install missing its n-gram table and
        // PLE constants -- an install that loads and is quietly degraded,
        // which is worse than one that fails.
        var passthroughFiles: [PassthroughFile] = []
        var sharedPassthrough: [(name: String, size: UInt64)] = []
        for requirement in RepackPlanner.passthroughRequirements(
            family: source.arch.family)
        {
            let path = (local.inputSnapshotDir as NSString)
                .appendingPathComponent(requirement.name)
            guard
                let attrs = try? FileManager.default
                    .attributesOfItem(atPath: path),
                let size = (attrs[FileAttributeKey.size] as? NSNumber)?.uint64Value
            else {
                if requirement.required {
                    throw RepackError.snapshotFileMissing(
                        path: path,
                        detail: "\(requirement.name) is required by this "
                            + "architecture; the snapshot is incomplete")
                }
                continue
            }
            if local.shareNgramTable && requirement.name == "ngram_table.bin" {
                // Kept out of the copy plan entirely; linked below, once the
                // partial directory exists, and recorded alongside the files
                // that were copied so the manifest and receipt are identical
                // either way.
                sharedPassthrough.append((requirement.name, size))
                continue
            }
            passthroughFiles.append(
                PassthroughFile(
                    sourceName: requirement.name,
                    destinationName: requirement.name,
                    size: size,
                    required: requirement.required))
        }
        let plan = try RepackPlanner.plan(
            meta: source.metadata,
            arch: source.arch,
            shardHeaders: source.shardHeaders,
            outputDir: paths.partialDirectory,
            passthroughFiles: passthroughFiles)
        let rangePlan = try RangeCopyPlanner.plan(
            repackPlan: plan,
            rangeChunkBytes: local.rangeChunkBytes,
            layoutMode: "identity",
            layoutOrderSha256: nil)
        // Same as the remote path above. A passthrough file that is hardlinked
        // rather than copied -- `--share-ngram-table` -- never enters
        // `plan.passthroughFiles`, so it is excluded here for free, which is
        // correct: a hardlink consumes no new blocks.
        let outputBytes =
            plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
            + plan.passthroughFiles.reduce(UInt64(0)) { $0 + $1.size }
        progress(
            .planning(
                downloadBytes: rangePlan.remoteBytesToDownload,
                outputBytes: outputBytes))
        // With `--consume-snapshot` the install grows as its source shards
        // are deleted, so the extra space the copy needs is the install less
        // the bytes those shards will release. The check is still bounded
        // below by the reserve, never skipped.
        var additionalBytes = outputBytes
        if local.consumeSnapshot {
            let releasable = source.metadata.shardFilenames.reduce(UInt64(0)) {
                partial, shard in
                let path = (local.inputSnapshotDir as NSString)
                    .appendingPathComponent(shard)
                let attributes = try? FileManager.default.attributesOfItem(
                    atPath: path)
                let size = (attributes?[.size] as? NSNumber)?.uint64Value ?? 0
                return partial + size
            }
            additionalBytes = outputBytes > releasable ? outputBytes - releasable : 0
        }
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: additionalBytes,
            reserveBytes: local.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()
        return (plan, rangePlan, outputBytes)
    }

    func configureLocalAudit(source: LocalSnapshot, plan: RepackPlan) {
        audit.remoteRepoID = "local/snapshot"
        audit.remoteRequestedRevision = source.metadata.indexSha256Hex
        audit.remoteResolvedCommit = String(source.metadata.indexSha256Hex.prefix(40))
        audit.remoteRangeStreamingSupported = false
        audit.remoteGapBytesDownloaded = 0
        audit.sourceSnapshotSha256 = source.metadata.indexSha256Hex
        audit.bitWidthOverridesHonored = source.metadata.bitsOverrides.count
        audit.tensorsDroppedMultimodal = plan.excludedMultimodalTensorNames
        audit.packedExpertLayoutMode = "identity"
    }

    /// packed_experts/layout.json, validated and recorded; returns the
    /// expert stride the manifest needs.
    func writeAndRecordLayout(
        plan: RepackPlan,
        paths: RemoteInstallPaths,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) throws -> UInt64 {
        let layoutPath =
            ((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts") as NSString)
            .appendingPathComponent("layout.json")
        let expertStride = plan.manifestExpertStride
        let layoutData = try SSDAIJSON.encodeLayout(
            plan: plan,
            expertStride: expertStride)
        try writeSmall(path: layoutPath, data: layoutData)
        try SSDAILayoutValidator.validate(path: layoutPath, plan: plan)
        try recordOutputFile(
            relativePath: "packed_experts/layout.json",
            path: layoutPath,
            progress: progress)
        return expertStride
    }

    /// The shared n-gram table, hard-linked and recorded (see the comment
    /// inside for why it is linked here rather than copied with the plan).
    func linkSharedNgramTable(
        source: LocalSnapshot,
        local: LocalSnapshotRepackOptions,
        paths: RemoteInstallPaths,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) throws {
        // Passthrough files, recorded the way the remote path records
        // them. They are copied into the install either way, but a file
        // the manifest does not list is a file the loader treats as
        // absent: ple_constants.json on disk and missing from `files`
        // fails the install at load with "missing required file".
        // `prepareLocalPlan` kept these out of the copy plan; link them
        // now that the partial directory exists. Deriving the set from the
        // same requirement list both functions read keeps them from
        // disagreeing about what was skipped.
        for requirement in RepackPlanner.passthroughRequirements(
            family: source.arch.family)
        where local.shareNgramTable && requirement.name == "ngram_table.bin" {
            try Task.checkCancellation()
            guard
                let destination = try Self.linkPassthroughFile(
                    named: requirement.name,
                    from: local.inputSnapshotDir,
                    into: paths.partialDirectory)
            else { continue }
            // Digested like any other output. A hardlink is the same bytes, so
            // the receipt attests over it exactly as it would over a copy --
            // sharing changes the disk cost, not the proof.
            try recordOutputFile(
                relativePath: requirement.name,
                path: destination,
                progress: progress)
        }
    }

    /// Hardlink one passthrough file from the snapshot into the partial
    /// directory, or nil when the snapshot does not carry it.
    ///
    /// Hardlink rather than symlink: both installs then hold one inode, so
    /// deleting either leaves the other intact, and the runtime's `F_NOCACHE`
    /// reads are indifferent to the extra link. It needs the two on one
    /// filesystem, which an install and its snapshot are.
    ///
    /// The size check is not ceremony. A truncated or empty table would link
    /// happily and then be read as a table with rows that are not there, so a
    /// zero-length origin is refused rather than shared. The name is a
    /// constant (`ngram_table.bin`) and never comes from a snapshot, so there
    /// is no traversal to guard here.
    public static func linkPassthroughFile(
        named name: String,
        from snapshotDirectory: String,
        into partialDirectory: String
    ) throws -> String? {
        let origin = (snapshotDirectory as NSString).appendingPathComponent(name)
        guard (try? Posix.entryKind(origin)) == .regular else { return nil }
        let destination = (partialDirectory as NSString).appendingPathComponent(name)
        // `entryKind` reports `.absent` for a missing path rather than
        // throwing, so `try?` is `.some(.absent)` and a `!= nil` test is true
        // for a file that is not there.
        if (try? Posix.entryKind(destination)) == .regular {
            try FileManager.default.removeItem(atPath: destination)
        }
        try FileManager.default.linkItem(atPath: origin, toPath: destination)
        let a = try FileManager.default.attributesOfItem(atPath: origin)
        let b = try FileManager.default.attributesOfItem(atPath: destination)
        let sa = (a[FileAttributeKey.size] as? NSNumber)?.uint64Value ?? 0
        let sb = (b[FileAttributeKey.size] as? NSNumber)?.uint64Value ?? 1
        guard sa == sb, sa > 0 else {
            throw RepackError.configurationInvalid(
                detail: "\(name): linked \(sb) bytes, expected \(sa)")
        }
        return destination
    }

    func executeLocalCopy(
        source: LocalSnapshot,
        local: LocalSnapshotRepackOptions,
        paths: RemoteInstallPaths,
        plan: RepackPlan,
        rangePlan: RangeCopyPlan,
        outputBytes: UInt64,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws {
        do {
            progress(.reservingOutput(bytes: outputBytes))
            try createOutputFiles(plan: plan, paths: paths)
            let provider = LocalSourceByteProvider(
                snapshotDirectory: local.inputSnapshotDir,
                writeTileBytes: local.writeTileBytes)
            progress(
                .copyingPayload(
                    reusedBytes: 0,
                    downloadedThisRunBytes: 0,
                    totalBytes: rangePlan.remoteBytesToDownload))
            try await provider.copyBatch(
                rangePlan.coalescedCopies,
                completedRangeIDs: [],
                partialDirectory: paths.partialDirectory,
                temporaryPath: paths.rangeTemporaryFile,
                audit: audit,
                progress: { bytes in
                    progress(
                        .copyingPayload(
                            reusedBytes: 0,
                            downloadedThisRunBytes: bytes,
                            totalBytes: rangePlan.remoteBytesToDownload))
                },
                commit: { _ in },
                consumeSource: local.consumeSnapshot)

            try recordOutputFile(
                relativePath: "model_weights.bin",
                path: plan.resident.path,
                progress: progress)
            for layer in plan.layers where layer.expertsPerLayer > 0 {
                let relative =
                    "packed_experts/"
                    + (layer.path as NSString).lastPathComponent
                try recordOutputFile(
                    relativePath: relative,
                    path: layer.path,
                    progress: progress)
            }
            let expertStride = try writeAndRecordLayout(
                plan: plan, paths: paths,
                progress: progress)
            try linkSharedNgramTable(
                source: source, local: local, paths: paths,
                progress: progress)
            for file in plan.passthroughFiles {
                try Task.checkCancellation()
                let passthroughPath = (paths.partialDirectory as NSString)
                    .appendingPathComponent(file.destinationName)
                try recordOutputFile(
                    relativePath: file.destinationName,
                    path: passthroughPath,
                    progress: progress)
            }
            // Tokenizer assets, copied from the snapshot the way the remote
            // path fetches them from the release. Without these the install
            // loads far enough to look finished and then the server refuses
            // it, because a model with no chat template cannot be prompted.
            //
            // This path only ever built MTP sidecars before, which carry no
            // tokenizer of their own -- they use the target's -- so the gap
            // did not show until a whole model was imported from a local
            // snapshot.
            // A draft head is the exception the comment above describes: it is
            // loaded beside a target and prompted through the target's
            // tokenizer, so requiring one here would refuse a sidecar that is
            // correct. Only a whole model needs its own.
            // An Inkling GGUF's tokenizer and chat template travel in the GGUF
            // header the plan copies as a passthrough file, not as tokenizer
            // files; the Inkling checkpoint carries them as files like any
            // snapshot, plus its tiktoken model and quantization config.
            let ggufHeaderInstall = plan.passthroughFiles.contains {
                $0.destinationName == Inkling2GGUFPlanner.headerSidecarName
            }
            if !local.draftHead && !source.arch.family.isDraftHead && !ggufHeaderInstall {
                try copyLocalTokenizer(
                    snapshotDirectory: local.inputSnapshotDir,
                    partialDirectory: paths.partialDirectory,
                    extraFiles: source.arch.family == .inkling2
                        ? Inkling2CheckpointPlanner.tokenizerExtras : [],
                    record: { relative, path in
                        try recordOutputFile(
                            relativePath: relative,
                            path: path,
                            progress: progress)
                    })
            }
            progress(.finalizing)
            try writeManifest(
                plan: plan,
                partialDir: paths.partialDirectory,
                metadata: source.metadata,
                expertStride: expertStride,
                resolvedCommit: String(source.metadata.indexSha256Hex.prefix(40)),
                modelIDOverride: local.modelID)

            if try Posix.entryKind(paths.finalDirectory) == .directory {
                try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
                try Posix.fsyncDirectory(paths.parentDirectory)
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            } else {
                try Posix.rename(
                    from: paths.partialDirectory,
                    to: paths.finalDirectory)
                try Posix.fsyncDirectory(paths.parentDirectory)
            }
        } catch {
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            throw error
        }
    }
}
