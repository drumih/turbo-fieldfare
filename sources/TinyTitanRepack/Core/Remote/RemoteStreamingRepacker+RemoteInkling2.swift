import Foundation
import TinyTitanFormat

// The remote Inkling 2 install path (TT-121): the GGUF or the original
// checkpoint streamed from Hugging Face. Only headers and small metadata files
// are fetched before planning; every tensor byte then arrives through the same
// ranged copy, checkpoint and resume path as any other remote install, so no
// whole source file is ever staged on disk.
extension RemoteStreamingRepacker {
    func prepareInkling2RemotePlan(
        _ kind: RemoteInkling2Source,
        remote: HuggingFaceRemoteSource,
        paths: RemoteInstallPaths
    ) async throws -> (RemoteSnapshot, RepackPlan, [String: RemoteFileInfo]) {
        try Posix.mkdirP(paths.metadataDirectory)
        let (snapshot, plan): (RemoteSnapshot, RepackPlan)
        switch kind {
        case .gguf(let fileName):
            (snapshot, plan) = try await prepareInkling2GGUF(
                fileName: fileName, remote: remote, paths: paths)
        case .checkpoint:
            (snapshot, plan) = try await prepareInkling2Checkpoint(remote: remote, paths: paths)
        }
        if options.requireKnownSource
            && SourceFingerprint.modelID(forIndexSha256: snapshot.metadata.indexSha256Hex) == nil
        {
            throw RepackError.sourceFingerprintRejected(
                path: snapshot.metadata.indexPath, sha256: snapshot.metadata.indexSha256Hex)
        }
        var matched = plan
        matched.matchedModelID = SourceFingerprint.modelID(
            forIndexSha256: snapshot.metadata.indexSha256Hex)
        // The GGUF's header sidecar is a passthrough copy out of the GGUF
        // itself, which `remoteFiles` already resolves.
        return (snapshot, matched, [:])
    }

    private func prepareInkling2GGUF(
        fileName: String, remote: HuggingFaceRemoteSource, paths: RemoteInstallPaths
    ) async throws -> (RemoteSnapshot, RepackPlan) {
        let info = try await remote.resolveFileInfo(filename: fileName, audit: audit)
        guard info.acceptsRanges else {
            throw RepackError.remoteProtocolInvalid(
                detail: "\(fileName) does not advertise byte ranges")
        }
        let pinned = remote.pinned(commit: info.resolvedCommit)
        var want = UInt64(16 << 20)  // the Q4_K_XL header is 12 MB
        var header: GGUFHeaderV3?
        var headerBytes: [UInt8] = []
        while header == nil {
            let length = min(want, info.size)
            let bytes = [UInt8](
                try await fetchRange(
                    pinned, fileName: fileName, info: info, offset: 0, length: length))
            do {
                let parsed = try GGUFHeaderV3(bytes: bytes)
                if parsed.dataStart > UInt64(bytes.count) {
                    guard parsed.dataStart <= info.size else {
                        throw GGUFHeaderV3.ReadError.invalid("data section past the file")
                    }
                    want = parsed.dataStart
                    continue
                }
                header = parsed
                headerBytes = Array(bytes.prefix(Int(parsed.dataStart)))
            } catch GGUFHeaderV3.ReadError.needMoreBytes(let atLeast) {
                guard length < info.size else {
                    throw GGUFHeaderV3.ReadError.invalid("truncated file")
                }
                want = max(UInt64(atLeast), want * 2)
            }
        }
        guard let header else { throw GGUFHeaderV3.ReadError.invalid("no header") }
        var hasher = Sha256Stream()
        headerBytes.withUnsafeBytes { hasher.update($0) }
        let fingerprint = hasher.finalizeHexString()
        let arch = try Inkling2GGUFArch.archInfo(header)
        let metadata = IndexLoader.SourceMetadata(
            indexPath: fileName, configPath: fileName, indexSha256Hex: fingerprint,
            weightMap: [:], baseBits: 8, baseGroupSize: 32,
            baseMode: SSDAIBlockEncodingV1.ggufQ8_0.rawValue, bitsOverrides: [:],
            shardFilenames: [fileName])
        let plan = try Inkling2GGUFPlanner.plan(
            source: .init(fileName: fileName, header: header, fileSize: info.size),
            outputDir: paths.partialDirectory)
        return (
            RemoteSnapshot(
                metadata: metadata, arch: arch, shardHeaders: [],
                remoteFiles: [fileName: info], resolvedCommit: info.resolvedCommit,
                metadataDirectory: paths.metadataDirectory),
            plan
        )
    }

    private func prepareInkling2Checkpoint(
        remote: HuggingFaceRemoteSource, paths: RemoteInstallPaths
    ) async throws -> (RemoteSnapshot, RepackPlan) {
        let indexName = "model.safetensors.index.json"
        let indexInfo = try await remote.resolveFileInfo(filename: indexName, audit: audit)
        let commit = indexInfo.resolvedCommit
        let pinned = remote.pinned(commit: commit)
        var files: [String: RemoteFileInfo] = [indexName: indexInfo]
        func fetchMetadata(_ name: String, cap: UInt64) async throws -> Data {
            let info =
                name == indexName
                ? indexInfo : try await pinned.resolveFileInfo(filename: name, audit: audit)
            guard info.resolvedCommit == commit else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "\(name) commit differs from the index commit")
            }
            let path = (paths.metadataDirectory as NSString).appendingPathComponent(name)
            try await pinned.fetchSmallFile(
                filename: name, info: info, capBytes: cap, outputPath: path, audit: audit)
            files[name] = info
            return try Posix.readBoundedData(path, maximumBytes: cap)
        }
        let index = try await fetchMetadata(indexName, cap: IndexLoader.maximumIndexBytes)
        let config = try await fetchMetadata("config.json", cap: 1 << 20)
        guard let root = try JSONSerialization.jsonObject(with: index) as? [String: Any],
            let weightMap = root["weight_map"] as? [String: String], !weightMap.isEmpty
        else {
            throw RepackError.configurationInvalid(detail: "\(indexName) has no weight_map")
        }
        var shardNames = Set(weightMap.values).sorted().map { ($0, "") }
        shardNames.append((Inkling2CheckpointPlanner.mtpShard, Inkling2CheckpointPlanner.mtpPrefix))
        var shards: [Inkling2CheckpointPlanner.Shard] = []
        for (name, prefix) in shardNames {
            let info: RemoteFileInfo
            do {
                info = try await pinned.resolveFileInfo(filename: name, audit: audit)
            } catch  where prefix == Inkling2CheckpointPlanner.mtpPrefix && isRemoteNotFound(error)
            {
                continue
            }
            guard info.resolvedCommit == commit, info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "shard \(name) is not a ranged file at the index commit")
            }
            files[name] = info
            shards.append(try await remoteShard(pinned, info: info, namePrefix: prefix))
        }
        var hasher = Sha256Stream()
        config.withUnsafeBytes { hasher.update($0) }
        for shard in shards { shard.headerBytes.withUnsafeBytes { hasher.update($0) } }
        let fingerprint = hasher.finalizeHexString()
        let tied = !shards.contains { shard in
            shard.header.tensors.contains { $0.name == "model.llm.unembed.weight" }
        }
        let source = Inkling2CheckpointPlanner.Source(
            shards: shards,
            arch: try Inkling2CheckpointArch.archInfo(config: config, tiedEmbeddings: tied))
        let metadata = IndexLoader.SourceMetadata(
            indexPath: (paths.metadataDirectory as NSString).appendingPathComponent(indexName),
            configPath: (paths.metadataDirectory as NSString).appendingPathComponent(
                "config.json"),
            indexSha256Hex: fingerprint, weightMap: [:], baseBits: 16,
            baseGroupSize: source.arch.quantGroupSize, baseMode: "BF16", bitsOverrides: [:],
            shardFilenames: shards.map(\.fileName))
        return (
            RemoteSnapshot(
                metadata: metadata, arch: source.arch, shardHeaders: [], remoteFiles: files,
                resolvedCommit: commit, metadataDirectory: paths.metadataDirectory),
            try Inkling2CheckpointPlanner.plan(source: source, outputDir: paths.partialDirectory)
        )
    }

    /// A shard's safetensors header, fetched as two ranges: the size prefix,
    /// then exactly the header it declares.
    private func remoteShard(
        _ pinned: HuggingFaceRemoteSource, info: RemoteFileInfo, namePrefix: String
    ) async throws -> Inkling2CheckpointPlanner.Shard {
        let name = info.filename
        guard info.size >= 8 else {
            throw RepackError.safetensorsHeaderInvalid(path: name, detail: "shorter than 8 bytes")
        }
        let prefix = try await fetchRange(pinned, fileName: name, info: info, offset: 0, length: 8)
        let headerSize = prefix.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        guard headerSize <= Safetensors.maxHeaderBytes, headerSize <= info.size - 8 else {
            throw RepackError.safetensorsHeaderTooLarge(path: name, size: headerSize)
        }
        let body = try await fetchRange(
            pinned, fileName: name, info: info, offset: 8, length: headerSize)
        return Inkling2CheckpointPlanner.Shard(
            fileName: name, fileSize: info.size,
            header: try Safetensors.parseHeaderBytes(
                path: name, fileSize: info.size, headerBytes: body,
                opaqueDTypes: Inkling2CheckpointPlanner.opaqueDTypes),
            headerBytes: prefix + body, namePrefix: namePrefix)
    }

    /// Exactly `length` bytes at `offset`, read back from the range's temp
    /// file with a bound of the length this process requested.
    private func fetchRange(
        _ pinned: HuggingFaceRemoteSource, fileName: String, info: RemoteFileInfo,
        offset: UInt64, length: UInt64
    ) async throws -> Data {
        let file = try await pinned.downloadRangeToTempFile(
            filename: fileName, info: info, offset: offset, length: Int(length), audit: audit)
        defer { try? FileManager.default.removeItem(atPath: file.path) }
        let data = try Posix.readBoundedData(file.path, maximumBytes: length)
        guard UInt64(data.count) == length else {
            throw RepackError.remoteProtocolInvalid(
                detail: "\(fileName): range at \(offset) returned \(data.count) of \(length) bytes")
        }
        return data
    }
}
