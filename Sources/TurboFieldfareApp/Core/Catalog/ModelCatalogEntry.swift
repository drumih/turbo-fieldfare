import Foundation

/// Whether a catalog entry's source was validated by the project or added by
/// the user. Determines both the install-time fingerprint policy and the badge
/// shown in the picker.
public enum ModelTrustTier: String, Codable, Equatable, Sendable {
    case curated
    case custom
}

/// One model the user can install and load.
///
/// `recordedIndexSHA256` carries different meaning per tier: for curated
/// entries it is the fingerprint pinned by the project, for custom entries it
/// is the trust-on-first-use value observed at the first install (nil until
/// then).
public struct ModelCatalogEntry: Codable, Equatable, Sendable, Identifiable {
    public var id: String { repoID }

    public let displayName: String
    public let repoID: String
    public let revision: String
    public let trustTier: ModelTrustTier
    public var recordedIndexSHA256: String?
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    public init(displayName: String,
                repoID: String,
                revision: String,
                trustTier: ModelTrustTier,
                recordedIndexSHA256: String?,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                reserveBytes: UInt64) {
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.trustTier = trustTier
        self.recordedIndexSHA256 = recordedIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.reserveBytes = reserveBytes
    }
}
