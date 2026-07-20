import Foundation
import Darwin

public struct RemoteStreamingRepackOptions: Sendable {
    public let outputDir: String
    public let overwrite: Bool
    let repoID: String
    let revision: String
    let token: String?
    let requireKnownSource: Bool
    let rangeChunkBytes: Int
    let minFreeReserveBytes: UInt64
    let retainPartialOnFailure: Bool
    let session: URLSession
    let baseURL: URL
    let rangeRetryAttempts: Int
    let retryBaseDelayNs: UInt64

    init(outputDir: String,
                overwrite: Bool,
                repoID: String = SupportedModelSource.repoID,
                revision: String = SupportedModelSource.revision,
                token: String? = nil,
                requireKnownSource: Bool = true,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                minFreeReserveBytes: UInt64 = SupportedModelSource.reserveBytes,
                retainPartialOnFailure: Bool = true,
                session: URLSession = .shared,
                baseURL: URL = URL(string: "https://huggingface.co")!,
                rangeRetryAttempts: Int = 4,
                retryBaseDelayNs: UInt64 = 1_000_000_000) {
        self.outputDir = outputDir
        self.overwrite = overwrite
        self.repoID = repoID
        self.revision = revision
        self.token = token
        self.requireKnownSource = requireKnownSource
        self.rangeChunkBytes = rangeChunkBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.retainPartialOnFailure = retainPartialOnFailure
        self.session = session
        self.baseURL = baseURL
        self.rangeRetryAttempts = rangeRetryAttempts
        self.retryBaseDelayNs = retryBaseDelayNs
    }
}

public struct RemoteStreamingRepackResult: Sendable {
    public let outputDir: String
    public let resolvedCommit: String
    public let remoteBytesToDownload: UInt64
}

public final class RemoteStreamingRepacker {
    private let options: RemoteStreamingRepackOptions
    private let audit: RepackAudit

    public init(options: RemoteStreamingRepackOptions,
                audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.audit = audit
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws
        -> RemoteStreamingRepackResult {
        try Self.sweepStalePartials(outputDir: options.outputDir, audit: audit)
        if FileManager.default.fileExists(atPath: options.outputDir), !options.overwrite {
            throw RepackError.configurationInvalid(detail:
                "output directory already exists: \(options.outputDir)")
        }

        let partialDir = options.outputDir + ".partial.\(getpid())"
        let metadataDir = (partialDir as NSString).appendingPathComponent(".remote-metadata")
        let tempDir = (partialDir as NSString).appendingPathComponent(".range-tmp")
        try? FileManager.default.removeItem(atPath: partialDir)

        do {
            return try await runPrepared(partialDir: partialDir,
                                         metadataDir: metadataDir,
                                         tempDir: tempDir,
                                         progress: progress)
        } catch {
            if error is CancellationError || !options.retainPartialOnFailure {
                try? FileManager.default.removeItem(atPath: partialDir)
            }
            throw error
        }
    }

    private func runPrepared(partialDir: String,
                             metadataDir: String,
                             tempDir: String,
                             progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> RemoteStreamingRepackResult {
        try Task.checkCancellation()
        let retryPolicy = RemoteRetryPolicy(attempts: options.rangeRetryAttempts,
                                            baseDelayNs: options.retryBaseDelayNs)
        let remote = HuggingFaceRemoteSource(repoID: options.repoID,
                                             requestedRevision: options.revision,
                                             token: options.token,
                                             session: options.session,
                                             baseURL: options.baseURL,
                                             tempDirectory: tempDir,
                                             retryPolicy: retryPolicy)
        progress(.downloadingMetadata)
        let snapshot = try await RemoteSnapshotLoader.load(remote: remote,
                                                           requireKnownSource: options.requireKnownSource,
                                                           metadataDirectory: metadataDir,
                                                           audit: audit)
        try Task.checkCancellation()
        let plan = try RepackPlanner.plan(meta: snapshot.metadata,
                                          arch: snapshot.arch,
                                          shardHeaders: snapshot.shardHeaders,
                                          outputDir: partialDir)
        let rangePlan = try RangeCopyPlanner.plan(repackPlan: plan,
                                                  rangeChunkBytes: options.rangeChunkBytes)
        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
        progress(.planning(downloadBytes: rangePlan.remoteBytesToDownload,
                           outputBytes: outputBytes))
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: partialDir,
            bytes: outputBytes + UInt64(options.rangeChunkBytes),
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()

        progress(.reservingOutput(bytes: outputBytes))
        try Task.checkCancellation()
        try Posix.mkdirP((partialDir as NSString).appendingPathComponent("packed_experts"))
        try Posix.mkdirP(tempDir)

        let residentFd = try ResidentWriter.createAndWriteIndex(plan: plan.resident, audit: audit)
        try Posix.fsync(residentFd, path: plan.resident.path)
        close(residentFd)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            try Posix.mkdirP((layer.path as NSString).deletingLastPathComponent)
            let fd = try Posix.openCreateRW(layer.path)
            try Posix.ftruncate(fd, path: layer.path, size: layer.fileSize)
            try Posix.fsync(fd, path: layer.path)
            close(fd)
        }

        let provider = HTTPRangeSourceByteProvider(remote: remote.pinned(commit: snapshot.resolvedCommit),
                                                   files: snapshot.remoteFiles,
                                                   writeTileBytes: WriterCore.tileBytes)
        progress(.copyingPayload(downloadedBytes: 0,
                                 totalBytes: rangePlan.remoteBytesToDownload))
        try await provider.copyBatch(rangePlan.coalescedCopies, audit: audit) { downloadedBytes in
            progress(.copyingPayload(downloadedBytes: downloadedBytes,
                                     totalBytes: rangePlan.remoteBytesToDownload))
        }

        try recordOutputFile(relativePath: "model_weights.bin",
                             path: plan.resident.path,
                             progress: progress)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let rel = "packed_experts/" + (layer.path as NSString).lastPathComponent
            try recordOutputFile(relativePath: rel, path: layer.path, progress: progress)
        }

        let layoutPath = ((partialDir as NSString).appendingPathComponent("packed_experts") as NSString)
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
                                             partialDir: partialDir,
                                             progress: progress)
        try? FileManager.default.removeItem(atPath: tempDir)
        try? FileManager.default.removeItem(atPath: metadataDir)
        progress(.finalizing)
        try Task.checkCancellation()
        try writeManifest(plan: plan,
                          partialDir: partialDir,
                          metadata: snapshot.metadata,
                          expertStride: expertStride,
                          resolvedCommit: snapshot.resolvedCommit)

        try Task.checkCancellation()
        if FileManager.default.fileExists(atPath: options.outputDir) {
            try FileManager.default.removeItem(atPath: options.outputDir)
        }
        try Posix.rename(from: partialDir, to: options.outputDir)

        return RemoteStreamingRepackResult(outputDir: options.outputDir,
                                           resolvedCommit: snapshot.resolvedCommit,
                                           remoteBytesToDownload: rangePlan.remoteBytesToDownload)
    }

    static func sweepStalePartials(outputDir: String, audit: RepackAudit? = nil) throws {
        let parent = (outputDir as NSString).deletingLastPathComponent
        let searchDir = parent.isEmpty ? "." : parent
        let prefix = (outputDir as NSString).lastPathComponent + ".partial."
        let fm = FileManager.default
        for entry in (try? fm.contentsOfDirectory(atPath: searchDir)) ?? []
        where entry.hasPrefix(prefix) {
            let path = (searchDir as NSString).appendingPathComponent(entry)
            let suffix = String(entry.dropFirst(prefix.count))
            if let pid = Int32(suffix),
               kill(pid, 0) == 0 {
                throw RepackError.configurationInvalid(detail:
                    "another repack (pid \(pid)) appears to own \(path); " +
                    "delete it manually if that process is not a repack")
            }
            try fm.removeItem(atPath: path)
            audit?.stalePartialsRemoved.append(path)
        }
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
                if file.required {
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

    private func writeManifest(plan: RepackPlan,
                               partialDir: String,
                               metadata: IndexLoader.SourceMetadata,
                               expertStride: UInt64,
                               resolvedCommit: String) throws {
        var bits = GTurboJSON.QuantBitWidths(embedding: 4,
                                             attention: 4,
                                             router: 8,
                                             sharedExpert: 8,
                                             routedExpert: 4)
        for e in plan.resident.entries {
            if e.name == "language_model.model.embed_tokens.weight", let s = e.quantSpec { bits.embedding = s.bits }
            if e.name.hasSuffix(".self_attn.q_proj.weight"), let s = e.quantSpec { bits.attention = s.bits }
            if e.name.hasSuffix(".router.proj.weight"), let s = e.quantSpec { bits.router = s.bits }
            if e.name.hasSuffix(".mlp.gate_proj.weight"), let s = e.quantSpec { bits.sharedExpert = s.bits }
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
