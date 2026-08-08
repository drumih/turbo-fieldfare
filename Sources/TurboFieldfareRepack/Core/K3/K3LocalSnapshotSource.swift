import Darwin
import Foundation

/// Test support: a `K3SnapshotSource` that reads a checkpoint directory from
/// local disk instead of Hugging Face. The installer never uses it; the K3
/// repack tests and the engine end-to-end test drive the full
/// planner -> range-plan -> checkpoint/resume -> quantize pipeline through it.
/// Lives in the core target because two test targets
/// (`TurboFieldfareRepackTests`, `TurboFieldfareTestsCore`) share it.
final class K3LocalSnapshotSource: K3SnapshotSource {
    let directory: String

    init(directory: String) {
        self.directory = directory
    }

    func loadSnapshot(metadataDirectory: String,
                      partialDirectory: String,
                      audit: RepackAudit) async throws -> K3Snapshot {
        let metadata = try K3IndexLoader.load(snapshotDir: directory)
        let arch = try K3ArchInfo.load(configPath: metadata.configPath)
        var headers: [K3Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)
        for shard in metadata.shardFilenames {
            let path = (directory as NSString).appendingPathComponent(shard)
            let fd = try Posix.openReadNoFollow(path)
            defer { close(fd) }
            let fileSize = try Posix.fileSize(fd: fd, path: path)
            var prefix: UInt64 = 0
            try withUnsafeMutableBytes(of: &prefix) { raw in
                try Posix.preadAll(fd: fd, path: path, buf: raw.baseAddress!,
                                   count: 8, offset: 0)
            }
            let headerSize = UInt64(littleEndian: prefix)
            if headerSize > K3Safetensors.maxHeaderBytes || headerSize > fileSize - 8 {
                throw RepackError.safetensorsHeaderTooLarge(path: shard, size: headerSize)
            }
            var headerData = Data(count: Int(headerSize))
            try headerData.withUnsafeMutableBytes { raw in
                try Posix.preadAll(fd: fd, path: path, buf: raw.baseAddress!,
                                   count: Int(headerSize), offset: 8)
            }
            headers.append(try K3Safetensors.parseHeaderBytes(path: shard,
                                                              fileSize: fileSize,
                                                              headerBytes: headerData))
            audit.recordRead(bytes: 8 + Int(headerSize))
        }
        // A checkpoint identity requires a 40-hex commit string; derive a
        // deterministic one from the index digest.
        return K3Snapshot(indexSha256Hex: metadata.indexSha256Hex,
                          arch: arch,
                          shardHeaders: headers,
                          resolvedCommit: String(metadata.indexSha256Hex.prefix(40)))
    }

    func makeByteProvider(snapshot: K3Snapshot,
                          writeTileBytes: Int) throws -> SourceByteProvider {
        K3LocalShardByteProvider(directory: directory, writeTileBytes: writeTileBytes)
    }

    func stageTokenizerFiles(metadataDirectory: String,
                             partialDirectory: String,
                             audit: RepackAudit) async throws -> [String] {
        let tokenizerDir = (partialDirectory as NSString).appendingPathComponent("tokenizer")
        try Posix.mkdirP(tokenizerDir)
        var staged: [String] = []
        for name in ["config.json", "tiktoken.model", "tokenizer_config.json"] {
            let src = (directory as NSString).appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: src) else { continue }
            let dst = (tokenizerDir as NSString).appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: dst) {
                try FileManager.default.removeItem(atPath: dst)
            }
            try FileManager.default.copyItem(atPath: src, toPath: dst)
            staged.append("tokenizer/\(name)")
        }
        return staged
    }
}

/// Reads coalesced range slices straight out of local shard files and writes
/// them to their destinations, committing per-range destination digests
/// through the exact same checkpoint path the HTTP provider uses.
final class K3LocalShardByteProvider: SourceByteProvider {
    private let directory: String
    private let writeTileBytes: Int

    init(directory: String, writeTileBytes: Int = WriterCore.tileBytes) {
        self.directory = directory
        self.writeTileBytes = writeTileBytes
    }

    func copyBatch(_ copies: [CoalescedRangeCopy],
                   completedRangeIDs: Set<String>,
                   partialDirectory: String,
                   temporaryPath: String,
                   audit: RepackAudit,
                   progress: @escaping @Sendable (UInt64) -> Void,
                   commit: (RemoteCompletedRange) throws -> Void) async throws {
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes, alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var shardFDs: [String: Int32] = [:]
        var outputFDs: [String: Int32] = [:]
        defer {
            shardFDs.values.forEach { close($0) }
            outputFDs.values.forEach { close($0) }
        }
        var downloaded: UInt64 = 0

        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()
            let shardPath = (directory as NSString).appendingPathComponent(copy.shardID)
            let shardFD: Int32
            if let existing = shardFDs[copy.shardID] {
                shardFD = existing
            } else {
                shardFD = try Posix.openReadNoFollow(shardPath)
                shardFDs[copy.shardID] = shardFD
            }
            var touched = Set<String>()
            for destination in copy.destinations {
                let destinationFD: Int32
                if let existing = outputFDs[destination.destinationPath] {
                    destinationFD = existing
                } else {
                    destinationFD = try Posix.openExistingRW(destination.destinationPath)
                    outputFDs[destination.destinationPath] = destinationFD
                }
                touched.insert(destination.destinationPath)
                var remaining = destination.size
                var source = destination.sourceOffset
                var destinationOffset = destination.destinationOffset
                while remaining > 0 {
                    try Task.checkCancellation()
                    let count = min(Int(remaining), scratch.count)
                    try Posix.preadAll(fd: shardFD, path: shardPath,
                                       buf: scratch.baseAddress!,
                                       count: count, offset: source)
                    try Posix.pwriteAll(fd: destinationFD,
                                        path: destination.destinationPath,
                                        buf: scratch.baseAddress!,
                                        count: count, offset: destinationOffset)
                    audit.recordTile(bytes: count)
                    audit.recordRead(bytes: count)
                    audit.recordWrite(bytes: count)
                    remaining -= UInt64(count)
                    source += UInt64(count)
                    destinationOffset += UInt64(count)
                }
            }
            try Task.checkCancellation()
            for path in touched {
                if let descriptor = outputFDs[path] {
                    try Posix.fsync(descriptor, path: path)
                }
            }
            let digest = try HTTPRangeSourceByteProvider.destinationDigest(
                copy, partialDirectory: partialDirectory, scratch: scratch)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(UInt64(0)) { $0 + $1.size }))
            downloaded += copy.size
            audit.remoteRangeRequests += 1
            audit.remoteBytesDownloaded += copy.size
            progress(downloaded)
            try Task.checkCancellation()
        }
    }
}
