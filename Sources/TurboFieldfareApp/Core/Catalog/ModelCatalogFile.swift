import Foundation

/// On-disk shape of the user-added part of the catalog. Curated entries are
/// compiled in and never written here, so a user editing this file by hand can
/// only affect their own additions.
public struct ModelCatalogFile: Codable, Equatable, Sendable {
    public static let currentVersion = 1
    public static let fileName = "catalog.json"

    public var version: Int
    public var customEntries: [ModelCatalogEntry]

    public init(version: Int = ModelCatalogFile.currentVersion,
                customEntries: [ModelCatalogEntry] = []) {
        self.version = version
        self.customEntries = customEntries
    }

    public func isValid() -> Bool {
        guard version == Self.currentVersion else { return false }
        guard customEntries.allSatisfy({ $0.trustTier == .custom }) else { return false }
        let ids = customEntries.map(\.repoID)
        return Set(ids).count == ids.count
    }
}
