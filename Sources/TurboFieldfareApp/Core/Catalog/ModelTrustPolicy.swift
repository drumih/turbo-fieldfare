import Foundation

/// What the installer should do about a source's provenance.
///
/// Local integrity is a separate concern handled by `verified-install.json`,
/// which is written and validated for every install regardless of tier. This
/// type only decides whether the *source* is trusted enough to repack.
public enum ModelTrustDecision: Equatable, Sendable {
    /// Curated entry whose observed index matches the pinned fingerprint.
    case allowCurated
    /// Curated entry whose upstream contents no longer match the pin. Always a
    /// hard stop: either the project's pin is stale or the repository moved.
    case curatedFingerprintMismatch(expected: String, observed: String)
    /// Custom entry being installed for the first time. Needs explicit user
    /// consent, after which the observed value is recorded.
    case needsConsent(observed: String)
    /// Custom entry whose observed index matches what was recorded on the first
    /// install.
    case allowCustom
    /// Custom entry whose upstream contents changed since it was added.
    case sourceChanged(recorded: String, observed: String)
}

public enum ModelTrustPolicy {
    public static func decide(entry: ModelCatalogEntry,
                              observedIndexSHA256: String) -> ModelTrustDecision {
        switch entry.trustTier {
        case .curated:
            guard let expected = entry.recordedIndexSHA256 else {
                return .curatedFingerprintMismatch(expected: "", observed: observedIndexSHA256)
            }
            return expected == observedIndexSHA256
                ? .allowCurated
                : .curatedFingerprintMismatch(expected: expected, observed: observedIndexSHA256)
        case .custom:
            guard let recorded = entry.recordedIndexSHA256 else {
                return .needsConsent(observed: observedIndexSHA256)
            }
            return recorded == observedIndexSHA256
                ? .allowCustom
                : .sourceChanged(recorded: recorded, observed: observedIndexSHA256)
        }
    }

    public static func requiresKnownSource(for entry: ModelCatalogEntry) -> Bool {
        entry.trustTier == .curated
    }
}
