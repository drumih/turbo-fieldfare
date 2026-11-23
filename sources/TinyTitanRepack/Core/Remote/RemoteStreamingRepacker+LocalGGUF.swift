import Foundation
import TinyTitanFormat

// The local-GGUF install path (TT-121): an Inkling 2 GGUF already on disk,
// converted to `.ssdai` through the same copy, layout, manifest and receipt path
// as a local snapshot.
extension RemoteStreamingRepacker {
    /// Installs the Inkling 2 GGUF at `ggufPath`. The file is the only shard; its header's SHA-256 is the
    /// source fingerprint the manifest and receipt record; with
    /// `consumeSnapshot` it is deleted once its last range is copied.
    public static func runLocalGGUF(
        ggufPath: String,
        outputDir: String,
        modelID: String,
        consumeSource: Bool = false,
        overwrite: Bool = false,
        audit: RepackAudit = RepackAudit(),
        progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }
    ) async throws -> RemoteStreamingRepackResult {
        let file = URL(fileURLWithPath: ggufPath).standardizedFileURL
        let directory = file.deletingLastPathComponent().path
        let (header, headerBytes, fileSize) = try readGGUFHeader(at: file.path)
        let ggufSource = Inkling2GGUFPlanner.Source(
            fileName: file.lastPathComponent, header: header, fileSize: fileSize)
        var hasher = Sha256Stream()
        headerBytes.withUnsafeBytes { hasher.update($0) }
        let fingerprint = hasher.finalizeHexString()
        let arch = try Inkling2GGUFArch.archInfo(header)
        let metadata = IndexLoader.SourceMetadata(
            indexPath: file.path, configPath: file.path, indexSha256Hex: fingerprint,
            weightMap: [:], baseBits: 8, baseGroupSize: 32,
            baseMode: SSDAIBlockEncodingV1.ggufQ8_0.rawValue, bitsOverrides: [:],
            shardFilenames: [file.lastPathComponent])
        let local = LocalSnapshotRepackOptions(
            inputSnapshotDir: directory, outputDir: outputDir, modelID: modelID,
            consumeSnapshot: consumeSource, overwrite: overwrite)
        let worker = RemoteStreamingRepacker(
            options: RemoteStreamingRepackOptions(
                repoID: "local/gguf",
                revision: String(fingerprint.prefix(40)),
                outputDir: outputDir,
                requireKnownSource: false,
                rangeChunkBytes: local.rangeChunkBytes,
                writeTileBytes: local.writeTileBytes,
                minFreeReserveBytes: local.minFreeReserveBytes,
                overwrite: overwrite),
            audit: audit)
        return try await worker.runLocalPlanned(
            source: LocalSnapshot(metadata: metadata, arch: arch, shardHeaders: []),
            local: local, repoID: "local/gguf", progress: progress
        ) { outputDir in
            try Inkling2GGUFPlanner.plan(source: ggufSource, outputDir: outputDir)
        }
    }

    /// Installs a source whose plan is built by `makePlan` rather than by
    /// `RepackPlanner` — the Inkling GGUF and checkpoint — through the same
    /// lock, disk check, copy, manifest and receipt path as a local snapshot.
    /// `makePlan` receives the partial directory the plan writes into.
    func runLocalPlanned(
        source: LocalSnapshot,
        local: LocalSnapshotRepackOptions,
        repoID: String,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void,
        makePlan: (String) throws -> RepackPlan
    ) async throws -> RemoteStreamingRepackResult {
        try validateOptions()
        try validateLocalModelID(local.modelID)
        let installLock = try InstallLock.acquire(outputDirectory: local.outputDir)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        try validateLocalDestination(paths: paths, overwrite: local.overwrite)
        progress(.downloadingMetadata)
        let plan = try makePlan(paths.partialDirectory)
        let rangePlan = try RangeCopyPlanner.plan(
            repackPlan: plan, rangeChunkBytes: local.rangeChunkBytes,
            layoutMode: "identity", layoutOrderSha256: nil)
        let outputBytes =
            plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
            + plan.passthroughFiles.reduce(UInt64(0)) { $0 + $1.size }
        progress(
            .planning(downloadBytes: rangePlan.remoteBytesToDownload, outputBytes: outputBytes))
        // A consumed source frees its bytes only after its last range, so the
        // install still needs its whole size free while it is written.
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory, bytes: outputBytes,
            reserveBytes: local.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()
        configureLocalAudit(source: source, plan: plan)
        audit.remoteRepoID = repoID
        try await executeLocalCopy(
            source: source, local: local, paths: paths, plan: plan,
            rangePlan: rangePlan, outputBytes: outputBytes, progress: progress)
        return RemoteStreamingRepackResult(
            outputDir: local.outputDir,
            resolvedCommit: String(source.metadata.indexSha256Hex.prefix(40)),
            plan: plan, rangeRequestCount: 0,
            remoteBytesToDownload: rangePlan.remoteBytesToDownload,
            remoteGapBytesDownloaded: 0, remoteRetryCount: 0, reusedBytes: 0,
            downloadedThisRunBytes: rangePlan.remoteBytesToDownload, dryRun: false)
    }

    /// The header, its bytes (up to the data section) and the file size. Reads
    /// 64 MiB first and widens when the header runs past it.
    static func readGGUFHeader(at path: String) throws -> (GGUFHeaderV3, [UInt8], UInt64) {
        guard let handle = FileHandle(forReadingAtPath: path) else {
            throw RepackError.snapshotFileMissing(path: path, detail: "cannot open the GGUF")
        }
        defer { try? handle.close() }
        let fileSize = try handle.seekToEnd()
        var want = 64 << 20
        while true {
            try handle.seek(toOffset: 0)
            let bytes = [UInt8](try handle.read(upToCount: want) ?? Data())
            do {
                let header = try GGUFHeaderV3(bytes: bytes)
                guard header.dataStart <= UInt64(bytes.count) else {
                    want = Int(header.dataStart)
                    continue
                }
                return (header, Array(bytes.prefix(Int(header.dataStart))), fileSize)
            } catch GGUFHeaderV3.ReadError.needMoreBytes(let atLeast) {
                guard UInt64(want) < fileSize else {
                    throw GGUFHeaderV3.ReadError.invalid("truncated file")
                }
                want = max(atLeast, want * 2)
            }
        }
    }
}
