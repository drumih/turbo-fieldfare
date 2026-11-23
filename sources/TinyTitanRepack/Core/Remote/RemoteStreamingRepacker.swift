//
//  RemoteStreamingRepacker.swift
//  TinyTitanRepackCore
//
//  The remote/local repack entry points and the orchestration they share.
//

import Foundation

public final class RemoteStreamingRepacker {
    let options: RemoteStreamingRepackOptions
    let audit: RepackAudit
    let startTime = Date()

    public init(
        options: RemoteStreamingRepackOptions,
        audit: RepackAudit = RepackAudit()
    ) {
        self.options = options
        self.audit = audit
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in })
        async throws
        -> RemoteStreamingRepackResult
    {
        try validateOptions()
        let installLock = try InstallLock.acquire(outputDirectory: options.outputDir)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        if try Posix.entryKind(paths.finalDirectory) == .directory, !options.overwrite {
            throw RepackError.configurationInvalid(
                detail:
                    "output directory already exists: \(paths.finalDirectory)")
        }
        let hasPartial = try Posix.entryKind(paths.partialDirectory) == .directory
        var hasCheckpoint = try Posix.entryKind(paths.checkpointFile) == .regular
        if !hasPartial, hasCheckpoint,
            try Posix.entryKind(paths.finalDirectory) == .directory
        {
            // A previous run renamed the partial directory into place but
            // crashed before deleting the checkpoint. The final directory is
            // authoritative: re-verify its completion markers and drop the
            // stale checkpoint instead of throwing installStateCorrupt.
            let manifestKind = try Posix.entryKind(
                (paths.finalDirectory as NSString)
                    .appendingPathComponent("manifest.json"))
            let receiptKind = try Posix.entryKind(
                (paths.finalDirectory as NSString)
                    .appendingPathComponent(VerifiedInstallReceiptWriter.fileName))
            guard manifestKind == .regular, receiptKind == .regular else {
                throw RepackError.installStateCorrupt(
                    path: paths.partialDirectory,
                    detail: "final directory exists without a complete verified install")
            }
            try FileManager.default.removeItem(atPath: paths.checkpointFile)
            try Posix.fsyncDirectory(paths.parentDirectory)
            hasCheckpoint = false
        }
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
                (try? Posix.entryKind(paths.checkpointFile)) != .regular
            {
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            }
            throw error
        }
    }

    public static func discardPartial(outputDirectory: String) throws {
        let lock = try InstallLock.acquire(outputDirectory: outputDirectory)
        defer { withExtendedLifetime(lock) {} }
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

    func validateOptions() throws {
        guard options.rangeChunkBytes >= RemoteChunkPolicy.minBytes,
            options.rangeChunkBytes <= RemoteChunkPolicy.maxBytes
        else {
            throw RepackError.configurationInvalid(
                detail: "range chunk bytes \(options.rangeChunkBytes) outside "
                    + "[\(RemoteChunkPolicy.minBytes), \(RemoteChunkPolicy.maxBytes)]")
        }
        guard options.writeTileBytes > 0,
            options.writeTileBytes <= BoundedScratch.defaultLimitBytes
        else {
            throw RepackError.configurationInvalid(
                detail: "bad write tile bytes \(options.writeTileBytes)")
        }
        guard options.rangeRetryAttempts >= 0 else {
            throw RepackError.configurationInvalid(
                detail:
                    "bad range retry attempts \(options.rangeRetryAttempts)")
        }
    }

    func createOutputFiles(
        plan: RepackPlan,
        paths: RemoteInstallPaths
    ) throws {
        try Posix.mkdirP(
            (paths.partialDirectory as NSString)
                .appendingPathComponent("packed_experts"))
        let resident = try ResidentWriter.createAndWriteIndex(
            plan: plan.resident,
            audit: audit)
        defer { close(resident) }
        try Posix.fsync(resident, path: plan.resident.path)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let descriptor = try Posix.openCreateRW(layer.path)
            defer { close(descriptor) }
            try Posix.ftruncate(descriptor, path: layer.path, size: layer.fileSize)
            try Posix.fsync(descriptor, path: layer.path)
        }
        // Passthrough destinations need the same preallocation: the transfer
        // opens them for writing and does not create them.
        for file in plan.passthroughFiles {
            try Task.checkCancellation()
            let path = (paths.partialDirectory as NSString)
                .appendingPathComponent(file.destinationName)
            let descriptor = try Posix.openCreateRW(path)
            defer { close(descriptor) }
            try Posix.ftruncate(descriptor, path: path, size: file.size)
            try Posix.fsync(descriptor, path: path)
        }
        try Posix.fsyncDirectory(paths.partialDirectory)
    }

    /// Copies tokenizer assets from a local snapshot into the install.
    ///
    /// Mirrors the remote path's list, including which of them are optional:
    /// `tokenizer.json` and `tokenizer_config.json` are required, the chat
    /// template and special-token map are not, and `config.json` is recorded
    /// under `tokenizer/` because that is where the loader looks for it.
    func copyLocalTokenizer(
        snapshotDirectory: String,
        partialDirectory: String,
        extraFiles: [(source: String, name: String)] = [],
        record: (String, String) throws -> Void
    ) throws {
        let tokenizerDir = (partialDirectory as NSString)
            .appendingPathComponent("tokenizer")
        try Posix.mkdirP(tokenizerDir)
        let files: [(name: String, required: Bool)] = [
            ("config.json", true),
            ("tokenizer.json", true),
            ("tokenizer_config.json", true),
            ("special_tokens_map.json", false),
            ("chat_template.jinja", false),
            ("chat_template.json", false),
        ]
        for file in files {
            let source = (snapshotDirectory as NSString)
                .appendingPathComponent(file.name)
            guard (try? Posix.entryKind(source)) == .regular else {
                if file.required {
                    throw RepackError.configurationInvalid(
                        detail: "local snapshot is missing \(file.name); a model "
                            + "imported from a snapshot needs its tokenizer beside it")
                }
                continue
            }
            let destination = (tokenizerDir as NSString)
                .appendingPathComponent(file.name)
            let data = try Posix.readBoundedData(source, maximumBytes: 64 * 1024 * 1024)
            try writeSmall(path: destination, data: data)
            try record("tokenizer/\(file.name)", destination)
        }
        // Optional files a family carries beside its tokenizer; `source` may
        // sit in a subdirectory, `name` is flat under `tokenizer/`.
        for extra in extraFiles {
            let source = (snapshotDirectory as NSString).appendingPathComponent(extra.source)
            guard (try? Posix.entryKind(source)) == .regular else { continue }
            let destination = (tokenizerDir as NSString).appendingPathComponent(extra.name)
            let data = try Posix.readBoundedData(source, maximumBytes: 64 * 1024 * 1024)
            try writeSmall(path: destination, data: data)
            try record("tokenizer/\(extra.name)", destination)
        }
    }

    func outputFilesMatch(
        plan: RepackPlan,
        rangePlan: RangeCopyPlan
    ) throws -> Bool {
        for output in rangePlan.expectedOutputs {
            let path =
                ((plan.resident.path as NSString).deletingLastPathComponent
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
        guard let scratchBase = scratch.baseAddress else {
            throw RepackError.configurationInvalid(
                detail: "the comparison scratch buffer could not be allocated")
        }
        return try expectedIndex.withUnsafeBytes { expected in
            guard let expectedBase = expected.baseAddress else { return false }
            var offset = 0
            while offset < expected.count {
                let count = min(scratch.count, expected.count - offset)
                try Posix.preadAll(
                    fd: descriptor,
                    path: plan.resident.path,
                    buf: scratchBase,
                    count: count,
                    offset: UInt64(offset))
                guard
                    memcmp(
                        scratchBase,
                        expectedBase.advanced(by: offset),
                        count) == 0
                else { return false }
                offset += count
            }
            return true
        }
    }

}
