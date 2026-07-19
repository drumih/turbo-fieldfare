import Foundation

struct RemoteSnapshot {
    let metadata: IndexLoader.SourceMetadata
    let arch: ArchInfo
    let shardHeaders: [Safetensors.Header]
    let remoteFiles: [String: RemoteFileInfo]
    let resolvedCommit: String
    let metadataDirectory: String
}

enum RemoteSnapshotLoader {
    static func load(remote: HuggingFaceRemoteSource,
                            requireKnownSource: Bool,
                            metadataDirectory: String,
                            audit: RepackAudit? = nil) async throws -> RemoteSnapshot {
        try Posix.mkdirP(metadataDirectory)

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
                                        capBytes: 4 * 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("model.safetensors.index.json"),
                                        audit: audit)
        try await pinned.fetchSmallFile(filename: "config.json",
                                        info: configInfo,
                                        capBytes: 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("config.json"),
                                        audit: audit)

        let metadata = try IndexLoader.load(snapshotDir: metadataDirectory)
        if requireKnownSource && SourceFingerprint.modelID(forIndexSha256: metadata.indexSha256Hex) == nil {
            throw RepackError.sourceFingerprintRejected(path: metadata.indexPath,
                                                        sha256: metadata.indexSha256Hex)
        }
        let arch = try ArchInfo.load(configPath: metadata.configPath)

        var files: [String: RemoteFileInfo] = [
            indexInfo.filename: indexInfo,
            configInfo.filename: configInfo,
        ]
        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)
        for shard in metadata.shardFilenames {
            let info = try await pinned.resolveFileInfo(filename: shard, audit: audit)
            guard info.resolvedCommit == indexInfo.resolvedCommit else {
                throw RepackError.remoteProtocolInvalid(detail: "shard \(shard) commit differs from index commit")
            }
            guard info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(detail: "shard \(shard) does not advertise byte ranges")
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
                throw RepackError.safetensorsHeaderInvalid(path: shard, detail: "short header prefix")
            }
            let headerSize = prefixData.withUnsafeBytes { raw -> UInt64 in
                var value: UInt64 = 0
                for i in 0..<8 {
                    value |= UInt64(raw[i]) << UInt64(i * 8)
                }
                return value
            }
            if headerSize > Safetensors.maxHeaderBytes || headerSize > info.size - 8 {
                throw RepackError.safetensorsHeaderTooLarge(path: shard, size: headerSize)
            }
            let headerFile = try await pinned.downloadRangeToTempFile(filename: shard,
                                                                     info: info,
                                                                     offset: 8,
                                                                     length: Int(headerSize),
                                                                     audit: audit)
            defer { try? FileManager.default.removeItem(atPath: headerFile.path) }
            let headerData = try Data(contentsOf: URL(fileURLWithPath: headerFile.path))
            headers.append(try Safetensors.parseHeaderBytes(path: shard,
                                                            fileSize: info.size,
                                                            headerBytes: headerData))
        }

        return RemoteSnapshot(metadata: metadata,
                              arch: arch,
                              shardHeaders: headers,
                              remoteFiles: files,
                              resolvedCommit: indexInfo.resolvedCommit,
                              metadataDirectory: metadataDirectory)
    }
}
