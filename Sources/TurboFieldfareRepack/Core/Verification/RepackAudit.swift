import Foundation

/// Bounded-copy counters and file hashes used while creating a verified install.
public final class RepackAudit {
    public var sourceBytesRead: UInt64 = 0
    public var outputBytesWritten: UInt64 = 0
    public var intentionalCopyBytes: UInt64 = 0
    public var byteCopyTiles: UInt64 = 0
    public var largestScratchBytes: Int = 0
    public var outputFiles: [OutputFile] = []
    public var remoteBytesDownloaded: UInt64 = 0
    public var remoteRangeRequests: UInt64 = 0
    public var remoteRangeRetries: UInt64 = 0
    public var largestRemoteTransferBytes: Int = 0
    public var largestRemotePayloadHeapBytes: Int = 0
    public var stalePartialsRemoved: [String] = []

    public init() {}

    public struct OutputFile {
        public let relativePath: String
        public let size: UInt64
        public let sha256: String
    }

    public func recordTile(bytes: Int) {
        byteCopyTiles &+= 1
        intentionalCopyBytes &+= UInt64(bytes)
    }

    public func recordWrite(bytes: Int) {
        outputBytesWritten &+= UInt64(bytes)
    }

    public func recordRead(bytes: Int) {
        sourceBytesRead &+= UInt64(bytes)
    }

    public func recordRemoteRetry() {
        remoteRangeRetries &+= 1
    }

}
