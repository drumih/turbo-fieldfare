import Foundation
import TinyTitanFormat

// The local Inkling 2 checkpoint install path (TT-121): the original ModelOpt
// NVFP4 snapshot on disk, converted to `.ssdai` through the same copy, layout,
// manifest and receipt path as a local snapshot. `--input-snapshot` routes here
// when the snapshot's `config.json` names the Inkling 2 model type.
extension RemoteStreamingRepacker {
    /// Whether `directory` holds an Inkling 2 checkpoint: its `config.json`
    /// declares the Inkling 2 model type.
    static func isInkling2Checkpoint(directory: String) -> Bool {
        let path = (directory as NSString).appendingPathComponent("config.json")
        guard let data = try? Posix.readBoundedData(path, maximumBytes: 4 << 20),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return root["model_type"] as? String == Inkling2CheckpointArch.modelType
    }

    static func runLocalInkling2Checkpoint(
        options local: LocalSnapshotRepackOptions,
        audit: RepackAudit,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> RemoteStreamingRepackResult {
        guard !local.draftHead, !local.shareNgramTable else {
            throw RepackError.configurationInvalid(
                detail: "an Inkling 2 checkpoint has no draft-head or n-gram install")
        }
        let directory = URL(fileURLWithPath: local.inputSnapshotDir).standardizedFileURL.path
        let source = try loadInkling2Checkpoint(directory: directory)
        var hasher = Sha256Stream()
        let config = try Posix.readBoundedData(
            (directory as NSString).appendingPathComponent("config.json"),
            maximumBytes: 4 << 20)
        config.withUnsafeBytes { hasher.update($0) }
        for shard in source.shards { shard.headerBytes.withUnsafeBytes { hasher.update($0) } }
        let fingerprint = hasher.finalizeHexString()
        let metadata = IndexLoader.SourceMetadata(
            indexPath: (directory as NSString).appendingPathComponent(
                "model.safetensors.index.json"),
            configPath: (directory as NSString).appendingPathComponent("config.json"),
            indexSha256Hex: fingerprint, weightMap: [:], baseBits: 16,
            baseGroupSize: source.arch.quantGroupSize, baseMode: "BF16", bitsOverrides: [:],
            shardFilenames: source.shards.map(\.fileName))
        let worker = RemoteStreamingRepacker(
            options: RemoteStreamingRepackOptions(
                repoID: "local/snapshot",
                revision: String(fingerprint.prefix(40)),
                outputDir: local.outputDir,
                requireKnownSource: false,
                rangeChunkBytes: local.rangeChunkBytes,
                writeTileBytes: local.writeTileBytes,
                minFreeReserveBytes: local.minFreeReserveBytes,
                overwrite: local.overwrite),
            audit: audit)
        return try await worker.runLocalPlanned(
            source: LocalSnapshot(metadata: metadata, arch: source.arch, shardHeaders: []),
            local: local, repoID: "local/snapshot", progress: progress
        ) { outputDir in
            try Inkling2CheckpointPlanner.plan(source: source, outputDir: outputDir)
        }
    }

    /// The checkpoint's shards — the ones `model.safetensors.index.json` maps,
    /// then `mtp.safetensors` when present — and its architecture.
    static func loadInkling2Checkpoint(directory: String) throws
        -> Inkling2CheckpointPlanner.Source
    {
        let index = try Posix.readBoundedData(
            (directory as NSString).appendingPathComponent("model.safetensors.index.json"),
            maximumBytes: 16 << 20)
        guard let root = try JSONSerialization.jsonObject(with: index) as? [String: Any],
            let weightMap = root["weight_map"] as? [String: String], !weightMap.isEmpty
        else {
            throw RepackError.configurationInvalid(
                detail: "model.safetensors.index.json has no weight_map")
        }
        var shards: [Inkling2CheckpointPlanner.Shard] = []
        for name in Set(weightMap.values).sorted() {
            shards.append(try readShard(directory: directory, fileName: name, namePrefix: ""))
        }
        let mtp = (directory as NSString).appendingPathComponent(Inkling2CheckpointPlanner.mtpShard)
        if try Posix.entryKind(mtp) == .regular {
            shards.append(
                try readShard(
                    directory: directory, fileName: Inkling2CheckpointPlanner.mtpShard,
                    namePrefix: Inkling2CheckpointPlanner.mtpPrefix))
        }
        let tied = !shards.contains { shard in
            shard.header.tensors.contains { $0.name == "model.llm.unembed.weight" }
        }
        let config = try Posix.readBoundedData(
            (directory as NSString).appendingPathComponent("config.json"),
            maximumBytes: 4 << 20)
        return Inkling2CheckpointPlanner.Source(
            shards: shards,
            arch: try Inkling2CheckpointArch.archInfo(config: config, tiedEmbeddings: tied))
    }

    private static func readShard(
        directory: String, fileName: String, namePrefix: String
    ) throws -> Inkling2CheckpointPlanner.Shard {
        guard !fileName.isEmpty, !fileName.contains("/"), !fileName.contains("..") else {
            throw RepackError.configurationInvalid(detail: "unsafe shard name \(fileName)")
        }
        let path = (directory as NSString).appendingPathComponent(fileName)
        let descriptor = try Posix.openReadNoFollow(path)
        defer { close(descriptor) }
        let fileSize = try Posix.fileSize(fd: descriptor, path: path)
        guard let handle = FileHandle(forReadingAtPath: path) else {
            throw RepackError.snapshotFileMissing(path: path, detail: "cannot open the shard")
        }
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else {
            throw RepackError.safetensorsHeaderInvalid(path: fileName, detail: "truncated")
        }
        let headerSize = prefix.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        guard headerSize <= Safetensors.maxHeaderBytes, headerSize <= fileSize - 8 else {
            throw RepackError.safetensorsHeaderTooLarge(path: fileName, size: headerSize)
        }
        let body = try handle.read(upToCount: Int(headerSize)) ?? Data()
        guard body.count == Int(headerSize) else {
            throw RepackError.safetensorsHeaderInvalid(path: fileName, detail: "truncated header")
        }
        return Inkling2CheckpointPlanner.Shard(
            fileName: fileName, fileSize: fileSize,
            header: try Safetensors.parseHeaderBytes(
                path: fileName, fileSize: fileSize, headerBytes: body,
                opaqueDTypes: Inkling2CheckpointPlanner.opaqueDTypes),
            headerBytes: prefix + body, namePrefix: namePrefix)
    }
}
