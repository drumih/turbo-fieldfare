import Foundation

public enum SupportedModelSource {
    public static let displayName = "gemma-4-26B-A4B-TurboQuant-MLX-4bit"
    public static let repoID = "majentik/gemma-4-26B-A4B-TurboQuant-MLX-4bit"
    public static let revision = "cc499c86a958ea7f05cffaa91c7e7243240dabbe"
    public static let approximateDownloadBytes: UInt64 = 14_952_958_284
    public static let installedBytes: UInt64 = 14_527_372_034
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
            retainPartialOnFailure: retainPartialOnFailure)
    }
}
