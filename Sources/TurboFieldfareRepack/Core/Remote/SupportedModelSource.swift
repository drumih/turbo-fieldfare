import Foundation

public enum SupportedModelSource {
    public static let displayName = "Gemma 4 26B-A4B IT 4-bit"
    public static let repoID = "mlx-community/gemma-4-26b-a4b-it-4bit"
    public static let revision = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
    public static let sourceIndexSHA256 =
        "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13"
    public static let approximateDownloadBytes: UInt64 = 14_620_479_420
    public static let installedBytes: UInt64 = 14_291_921_884
    public static let reserveBytes: UInt64 = 1_073_741_824

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      retainPartialOnFailure: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            outputDir: outputDirectory.path,
            overwrite: overwrite,
            token: token,
            retainPartialOnFailure: retainPartialOnFailure,
            session: Self.downloadSession)
    }

    /// URLSession.shared applies a 60 s per-request timeout that resets only
    /// when data arrives; a single stalled byte-range on the Hugging Face CDN
    /// (observed twice at ~3.3 GB into the first shard) then aborts the whole
    /// ~14.6 GB install. Allow 5 minutes of silence per range request; the
    /// retry policy still bounds genuinely dead connections.
    private static let downloadSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300
        config.timeoutIntervalForResource = 7 * 24 * 3600
        return URLSession(configuration: config)
    }()
}
