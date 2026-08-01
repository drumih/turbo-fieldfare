import Foundation
import TurboFieldfareRepackCore

/// The set of models the picker offers.
///
/// Curated entries are compiled in, so a new architecture always arrives with
/// the kernels that can run it. On a repository-ID collision the curated entry
/// wins: otherwise re-adding a pinned model as "custom" would silently
/// downgrade it out of the fingerprint-checked tier.
public struct ModelCatalog: Equatable, Sendable {
    public struct DuplicateEntry: Error, Equatable, Sendable {
        public let repoID: String
    }

    public static let curated: [ModelCatalogEntry] = [
        ModelCatalogEntry(
            displayName: SupportedModelSource.displayName,
            repoID: SupportedModelSource.repoID,
            revision: SupportedModelSource.revision,
            trustTier: .curated,
            recordedIndexSHA256: SupportedModelSource.sourceIndexSHA256,
            approximateDownloadBytes: SupportedModelSource.approximateDownloadBytes,
            installedBytes: SupportedModelSource.installedBytes,
            reserveBytes: SupportedModelSource.reserveBytes),
    ]

    public let entries: [ModelCatalogEntry]

    public init(curated: [ModelCatalogEntry] = ModelCatalog.curated,
                custom: [ModelCatalogEntry]) {
        let curatedIDs = Set(curated.map(\.repoID))
        self.entries = curated + custom.filter { !curatedIDs.contains($0.repoID) }
    }

    public func entry(forRepoID repoID: String) -> ModelCatalogEntry? {
        entries.first { $0.repoID == repoID }
    }

    public var customEntries: [ModelCatalogEntry] {
        entries.filter { $0.trustTier == .custom }
    }

    public func addingCustom(_ entry: ModelCatalogEntry) throws -> ModelCatalog {
        guard self.entry(forRepoID: entry.repoID) == nil else {
            throw DuplicateEntry(repoID: entry.repoID)
        }
        return ModelCatalog(curated: entries.filter { $0.trustTier == .curated },
                            custom: customEntries + [entry])
    }
}
