import Foundation
import Darwin

final class HTTPRangeSourceByteProvider {
    private let remote: HuggingFaceRemoteSource
    private let files: [String: RemoteFileInfo]
    private let writeTileBytes: Int

    init(remote: HuggingFaceRemoteSource,
                files: [String: RemoteFileInfo],
                writeTileBytes: Int = WriterCore.tileBytes) {
        self.remote = remote
        self.files = files
        self.writeTileBytes = writeTileBytes
    }

    func copyBatch(_ copies: [CoalescedRangeCopy],
                          audit: RepackAudit,
                          progress: @Sendable (UInt64) -> Void) async throws {
        let scratch = UnsafeMutableRawBufferPointer.allocate(byteCount: writeTileBytes,
                                                             alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var outputFDs: [String: Int32] = [:]
        defer {
            for fd in outputFDs.values { close(fd) }
        }

        var downloadedBytes: UInt64 = 0
        for copy in copies {
            try Task.checkCancellation()
            guard let info = files[copy.shardID] else {
                throw RepackError.configurationInvalid(detail: "missing remote info for \(copy.shardID)")
            }
            let temp = try await remote.downloadRangeToTempFile(filename: copy.shardID,
                                                               info: info,
                                                               offset: copy.sourceOffset,
                                                               length: Int(copy.size),
                                                               audit: audit)
            defer { try? FileManager.default.removeItem(atPath: temp.path) }
            audit.remoteRangeRequests += 1
            audit.remoteBytesDownloaded += temp.byteCount
            audit.largestRemoteTransferBytes = max(audit.largestRemoteTransferBytes, Int(temp.byteCount))
            audit.largestRemotePayloadHeapBytes = max(audit.largestRemotePayloadHeapBytes, scratch.count)
            downloadedBytes += temp.byteCount
            progress(downloadedBytes)

            let srcFd = try Posix.openRead(temp.path)
            defer { close(srcFd) }
            for destination in copy.destinations {
                try Task.checkCancellation()
                let dstFd: Int32
                if let cached = outputFDs[destination.destinationPath] {
                    dstFd = cached
                } else {
                    dstFd = open(destination.destinationPath, O_RDWR)
                    if dstFd < 0 {
                        throw RepackError.fileOpenFailed(path: destination.destinationPath, errno: errno)
                    }
                    outputFDs[destination.destinationPath] = dstFd
                }
                let tempOffset = destination.sourceOffset - copy.sourceOffset
                try copyTempRange(srcFd: srcFd,
                                  srcPath: temp.path,
                                  dstFd: dstFd,
                                  dstPath: destination.destinationPath,
                                  srcOffset: tempOffset,
                                  dstOffset: destination.destinationOffset,
                                  size: destination.size,
                                  scratch: scratch,
                                  audit: audit)
            }
        }
        for (path, fd) in outputFDs {
            try Task.checkCancellation()
            try Posix.fsync(fd, path: path)
        }
    }

    private func copyTempRange(srcFd: Int32,
                               srcPath: String,
                               dstFd: Int32,
                               dstPath: String,
                               srcOffset: UInt64,
                               dstOffset: UInt64,
                               size: UInt64,
                               scratch: UnsafeMutableRawBufferPointer,
                               audit: RepackAudit) throws {
        var remaining = size
        var src = srcOffset
        var dst = dstOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let n = min(Int(remaining), scratch.count)
            try Posix.preadAll(fd: srcFd,
                               path: srcPath,
                               buf: scratch.baseAddress!,
                               count: n,
                               offset: src)
            try Posix.pwriteAll(fd: dstFd,
                                path: dstPath,
                                buf: scratch.baseAddress!,
                                count: n,
                                offset: dst)
            audit.recordTile(bytes: n)
            audit.recordRead(bytes: n)
            audit.recordWrite(bytes: n)
            remaining -= UInt64(n)
            src += UInt64(n)
            dst += UInt64(n)
        }
    }
}
