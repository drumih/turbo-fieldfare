import Foundation

// The remote repack options and result value types.
//
// Split out of `RemoteStreamingRepacker.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
/// An Inkling 2 source the installer plans from its own tensors rather than
/// through `model.safetensors.index.json` affine metadata (TT-121).
public enum RemoteInkling2Source: Sendable, Equatable {
    /// One GGUF file in the repository; every tensor keeps its GGUF encoding.
    case gguf(fileName: String)
    /// The original ModelOpt NVFP4/MXFP8 checkpoint and its MTP shard.
    case checkpoint
}

public struct RemoteStreamingRepackOptions: Sendable {
    public let repoID: String
    public let revision: String
    public let outputDir: String
    public let token: String?
    public let requireKnownSource: Bool
    public let copyAuditPath: String?
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let overwrite: Bool
    public let resume: Bool
    public let dryRunSpaceCheck: Bool
    public let downloadSession: RemoteDownloadSession
    public let baseURL: URL
    public let rangeRetryAttempts: Int
    public let retryBaseDelayNs: UInt64
    /// Install the repository's MTP draft rather than the model itself.
    ///
    /// Qwen3.8-Flash-Next ships its draft head inside the target's repository,
    /// in its own shard, so the two installs differ by which namespace they
    /// claim rather than by where they come from. Nothing in `config.json`
    /// distinguishes them, which is why this is a choice the caller makes and
    /// not something the loader can infer.
    public let installDraftHead: Bool
    /// Set for an Inkling 2 source, which is planned by its own planner.
    public let inkling2: RemoteInkling2Source?

    public init(
        repoID: String,
        revision: String,
        outputDir: String,
        token: String? = nil,
        requireKnownSource: Bool = false,
        copyAuditPath: String? = nil,
        rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
        writeTileBytes: Int = WriterCore.tileBytes,
        minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
        overwrite: Bool = false,
        resume: Bool = false,
        dryRunSpaceCheck: Bool = false,
        downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
        baseURL: URL = RemoteBaseURL.huggingFace,
        rangeRetryAttempts: Int = 4,
        retryBaseDelayNs: UInt64 = 1_000_000_000,
        installDraftHead: Bool = false,
        inkling2: RemoteInkling2Source? = nil
    ) {
        self.inkling2 = inkling2
        self.repoID = repoID
        self.revision = revision
        self.outputDir = outputDir
        self.installDraftHead = installDraftHead
        self.token = token
        self.requireKnownSource = requireKnownSource
        self.copyAuditPath = copyAuditPath
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.overwrite = overwrite
        self.resume = resume
        self.dryRunSpaceCheck = dryRunSpaceCheck
        self.downloadSession = downloadSession
        self.baseURL = baseURL
        self.rangeRetryAttempts = rangeRetryAttempts
        self.retryBaseDelayNs = retryBaseDelayNs
    }
}

public struct RemoteStreamingRepackResult: Sendable {
    public let outputDir: String
    public let resolvedCommit: String
    let plan: RepackPlan
    /// Number of ranged HTTP requests issued this run. Retried attempts are
    /// not counted here; see `remoteRetryCount`.
    public let rangeRequestCount: Int
    public let remoteBytesToDownload: UInt64
    public let remoteGapBytesDownloaded: UInt64
    public let remoteRetryCount: UInt64
    public let reusedBytes: UInt64
    /// Unique payload bytes transferred this run: each coalesced range's
    /// successful transfer is counted once, so retries are not double-counted
    /// and the value equals `remoteBytesToDownload` on a fresh install. The
    /// live progress callbacks may temporarily report attempt-bytes while a
    /// retried transfer is re-streaming, but this final value is exact.
    public let downloadedThisRunBytes: UInt64
    public let dryRun: Bool
}
