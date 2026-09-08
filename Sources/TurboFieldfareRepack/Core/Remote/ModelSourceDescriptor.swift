import Foundation

/// Weight-format family of a supported checkpoint.
///
/// The repacker's planning, manifest schema, and install verification all
/// depend on how a checkpoint stores its weights, not on which checkpoint it
/// is. Today every supported source is MLX-quantized Gemma, so this enum has
/// a single case; it exists so that a second family can be added without
/// reworking the call sites that need to ask "which layout is this?".
public enum ModelFamily: String, Sendable, Hashable, CaseIterable {
    case gemma4 = "gemma4"
}

/// Pinned description of one supported upstream checkpoint.
///
/// Adding a model means adding one `ModelSourceDescriptor` instance and its
/// weight-index SHA-256, validated against a fresh upload of the source.
public struct ModelSourceDescriptor: Sendable, Hashable {
    /// Stable identifier for this source.
    public let key: String
    public let family: ModelFamily
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    public init(key: String,
                family: ModelFamily,
                displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                reserveBytes: UInt64) {
        self.key = key
        self.family = family
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.reserveBytes = reserveBytes
    }

    public func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }
}

public extension SupportedModelSource {
    /// Gemma 4 26B-A4B IT 4-bit (MLX community re-pack).
    static let gemma4 = ModelSourceDescriptor(
        key: "gemma4",
        family: .gemma4,
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: 1_073_741_824)

    /// Every checkpoint this build can install.
    static let all: [ModelSourceDescriptor] = [gemma4]

    /// The checkpoint installed when no source is specified.
    static let `default` = gemma4

    /// Look up a descriptor by its `key`, or nil when unknown.
    static func descriptor(forKey key: String) -> ModelSourceDescriptor? {
        all.first { $0.key == key }
    }
}
