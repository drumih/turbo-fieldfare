import Darwin
import Foundation

public protocol SourceByteProvider {
    func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws
}

public final class HTTPRangeSourceByteProvider: SourceByteProvider {
    private let remote: HuggingFaceRemoteSource
    private let files: [String: RemoteFileInfo]
    private let writeTileBytes: Int

    public init(remote: HuggingFaceRemoteSource,
                files: [String: RemoteFileInfo],
                writeTileBytes: Int = WriterCore.tileBytes) {
        self.remote = remote
        self.files = files
        self.writeTileBytes = writeTileBytes
    }

    public func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var outputFDs: [String: Int32] = [:]
        defer { outputFDs.values.forEach { close($0) } }
        var downloaded: UInt64 = 0

        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()
            guard let info = files[copy.shardID] else {
                throw RepackError.configurationInvalid(
                    detail: "missing remote info for \(copy.shardID)")
            }
            if try Posix.entryKind(temporaryPath) != .absent {
                try FileManager.default.removeItem(atPath: temporaryPath)
            }
            let base = downloaded
            let temporary = try await remote.downloadRangeToTempFile(
                filename: copy.shardID,
                info: info,
                offset: copy.sourceOffset,
                length: Int(copy.size),
                targetPath: temporaryPath,
                progress: { bytes in progress(base + bytes) },
                audit: audit)

            audit.remoteRangeRequests += 1
            audit.remoteBytesDownloaded += temporary.byteCount
            audit.largestRemoteTransferBytes = max(
                audit.largestRemoteTransferBytes,
                Int(temporary.byteCount))
            downloaded += temporary.byteCount
            progress(downloaded)

            let sourceFD = try Posix.openReadNoFollow(temporary.path)
            var touched = Set<String>()
            do {
                for destination in copy.destinations {
                    let destinationFD: Int32
                    if let existing = outputFDs[destination.destinationPath] {
                        destinationFD = existing
                    } else {
                        destinationFD = try Posix.openExistingRW(
                            destination.destinationPath)
                        outputFDs[destination.destinationPath] = destinationFD
                    }
                    touched.insert(destination.destinationPath)
                    try copyBytes(
                        sourceFD: sourceFD,
                        sourcePath: temporary.path,
                        destinationFD: destinationFD,
                        destinationPath: destination.destinationPath,
                        sourceOffset: destination.sourceOffset - copy.sourceOffset,
                        destinationOffset: destination.destinationOffset,
                        size: destination.size,
                        scratch: scratch,
                        audit: audit)
                }
                close(sourceFD)
            } catch {
                close(sourceFD)
                throw error
            }

            try Task.checkCancellation()
            for path in touched {
                if let descriptor = outputFDs[path] {
                    try Posix.fsync(descriptor, path: path)
                }
            }
            let digest = try Self.destinationDigest(
                copy,
                partialDirectory: partialDirectory,
                scratch: scratch)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(0) { $0 + $1.size }))
            progress(downloaded)
            try? FileManager.default.removeItem(atPath: temporary.path)
            try Task.checkCancellation()
        }
    }

    public static func destinationDigest(
        _ copy: CoalescedRangeCopy,
        partialDirectory: String,
        scratch suppliedScratch: UnsafeMutableRawBufferPointer? = nil
    ) throws -> String {
        let scratch = suppliedScratch ?? UnsafeMutableRawBufferPointer.allocate(
            byteCount: WriterCore.tileBytes,
            alignment: 16_384)
        defer {
            if suppliedScratch == nil { scratch.deallocate() }
        }

        var digest = DestinationDigest(copy: copy)
        for destination in copy.destinations {
            digest.append(try RangeCopyPlanner.normalizedRelativePath(
                destination.destinationPath,
                root: partialDirectory))
            digest.append(destination.destinationOffset)
            digest.append(destination.sourceOffset - copy.sourceOffset)
            digest.append(destination.size)

            let descriptor = try Posix.openReadNoFollow(destination.destinationPath)
            defer { close(descriptor) }
            var remaining = destination.size
            var offset = destination.destinationOffset
            while remaining > 0 {
                let count = min(Int(remaining), scratch.count)
                try Posix.preadAll(
                    fd: descriptor,
                    path: destination.destinationPath,
                    buf: scratch.baseAddress!,
                    count: count,
                    offset: offset)
                digest.append(UnsafeRawBufferPointer(
                    start: scratch.baseAddress,
                    count: count))
                remaining -= UInt64(count)
                offset += UInt64(count)
            }
        }
        return digest.finalize()
    }

    private func copyBytes(
        sourceFD: Int32,
        sourcePath: String,
        destinationFD: Int32,
        destinationPath: String,
        sourceOffset: UInt64,
        destinationOffset: UInt64,
        size: UInt64,
        scratch: UnsafeMutableRawBufferPointer,
        audit: RepackAudit
    ) throws {
        var remaining = size
        var source = sourceOffset
        var destination = destinationOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let count = min(Int(remaining), scratch.count)
            try Posix.preadAll(
                fd: sourceFD,
                path: sourcePath,
                buf: scratch.baseAddress!,
                count: count,
                offset: source)
            try Posix.pwriteAll(
                fd: destinationFD,
                path: destinationPath,
                buf: scratch.baseAddress!,
                count: count,
                offset: destination)
            audit.recordTile(bytes: count)
            audit.recordRead(bytes: count)
            audit.recordWrite(bytes: count)
            remaining -= UInt64(count)
            source += UInt64(count)
            destination += UInt64(count)
        }
    }
}

private struct DestinationDigest {
    private var stream = Sha256Stream()

    init(copy: CoalescedRangeCopy) {
        append("TurboFieldfare.RemoteRangeDestination.v1")
        append(copy.id)
        append(UInt64(copy.destinations.count))
    }

    mutating func append(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { stream.update($0) }
    }

    mutating func append(_ value: String) {
        let data = Data(value.utf8)
        append(UInt64(data.count))
        data.withUnsafeBytes { stream.update($0) }
    }

    mutating func append(_ bytes: UnsafeRawBufferPointer) {
        stream.update(bytes)
    }

    func finalize() -> String {
        stream.finalizeHexString()
    }
}
