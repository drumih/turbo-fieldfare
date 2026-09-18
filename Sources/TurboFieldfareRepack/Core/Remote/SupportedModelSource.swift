import Foundation

/// Convenience accessors for the default source.
///
/// These forward to `SupportedModelSource.default`; the descriptors in
/// `ModelSourceDescriptor.swift` are the source of truth. Prefer
/// `SupportedModelSource.descriptor(forKey:)` or a concrete descriptor in
/// new code.
public enum SupportedModelSource {
    public static let displayName = SupportedModelSource.default.displayName
    public static let repoID = SupportedModelSource.default.repoID
    public static let revision = SupportedModelSource.default.revision
    public static let sourceIndexSHA256 = SupportedModelSource.default.sourceIndexSHA256
    public static let approximateDownloadBytes =
        SupportedModelSource.default.approximateDownloadBytes
    public static let installedBytes = SupportedModelSource.default.installedBytes
    public static let reserveBytes = SupportedModelSource.default.reserveBytes

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        SupportedModelSource.default.installOptions(outputDirectory: outputDirectory,
                                                    overwrite: overwrite,
                                                    token: token,
                                                    resume: resume)
    }
}
