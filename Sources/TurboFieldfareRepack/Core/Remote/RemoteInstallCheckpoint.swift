import Foundation

public struct RemoteCompletedRange: Codable, Sendable, Equatable {
    public let id: String
    public let destinationDigest: String
    public let sourceBytes: UInt64
    public let destinationBytes: UInt64
}

public struct RemoteInstallCheckpoint: Codable, Sendable, Equatable {
    public static let schemaVersion = 1
    public static let maximumBytes: UInt64 = 8 * 1024 * 1024

    public let schema: Int
    public let repoID: String
    public let requestedRevision: String
    public let resolvedCommit: String
    public let sourceIndexSHA256: String
    public let planFingerprint: String
    public let totalSourceBytes: UInt64
    public var completedRanges: [RemoteCompletedRange]

    public init(repoID: String,
                requestedRevision: String,
                resolvedCommit: String,
                sourceIndexSHA256: String,
                planFingerprint: String,
                totalSourceBytes: UInt64,
                completedRanges: [RemoteCompletedRange] = []) {
        self.schema = Self.schemaVersion
        self.repoID = repoID
        self.requestedRevision = requestedRevision
        self.resolvedCommit = resolvedCommit
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.planFingerprint = planFingerprint
        self.totalSourceBytes = totalSourceBytes
        self.completedRanges = completedRanges
    }

    public static func load(from path: String) throws -> Self {
        let data = try Posix.readBoundedData(path, maximumBytes: maximumBytes)
        do {
            let checkpoint = try JSONDecoder().decode(Self.self, from: data)
            try checkpoint.validate(path: path)
            return checkpoint
        } catch let error as RepackError {
            throw error
        } catch {
            throw RepackError.installStateCorrupt(path: path, detail: "\(error)")
        }
    }

    public func write(to path: String, parentDirectory: String) throws {
        try validate(path: path)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        guard data.count <= Self.maximumBytes else {
            throw RepackError.installStateCorrupt(
                path: path,
                detail: "checkpoint exceeds \(Self.maximumBytes)-byte cap")
        }
        try Posix.atomicWrite(data, to: path, durableIn: parentDirectory)
    }

    public func matches(repoID: String,
                        requestedRevision: String,
                        sourceIndexSHA256: String,
                        planFingerprint: String) -> Bool {
        self.repoID == repoID
            && self.requestedRevision == requestedRevision
            && self.sourceIndexSHA256 == sourceIndexSHA256
            && self.planFingerprint == planFingerprint
    }

    public func validatedDestinationBytes(maximum: UInt64, path: String) throws -> UInt64 {
        try validate(path: path)
        var total: UInt64 = 0
        for range in completedRanges {
            let sum = total.addingReportingOverflow(range.destinationBytes)
            guard !sum.overflow, sum.partialValue <= maximum else {
                throw RepackError.installStateCorrupt(
                    path: path,
                    detail: "completed destination bytes exceed the installed model")
            }
            total = sum.partialValue
        }
        return total
    }

    private func validate(path: String) throws {
        guard schema == Self.schemaVersion,
              !repoID.isEmpty,
              !requestedRevision.isEmpty,
              resolvedCommit.count == 40,
              sourceIndexSHA256.count == 64,
              planFingerprint.count == 64,
              totalSourceBytes > 0 else {
            throw RepackError.installStateCorrupt(
                path: path,
                detail: "invalid checkpoint identity")
        }
        let ids = completedRanges.map(\.id)
        guard Set(ids).count == ids.count,
              completedRanges.allSatisfy({
                  !$0.id.isEmpty
                      && $0.destinationDigest.count == 64
                      && $0.sourceBytes > 0
                      && $0.destinationBytes > 0
              }) else {
            throw RepackError.installStateCorrupt(
                path: path,
                detail: "invalid completed range")
        }
        var sourceTotal: UInt64 = 0
        var destinationTotal: UInt64 = 0
        for range in completedRanges {
            let nextSource = sourceTotal.addingReportingOverflow(range.sourceBytes)
            let nextDestination = destinationTotal.addingReportingOverflow(
                range.destinationBytes)
            guard !nextSource.overflow,
                  nextSource.partialValue <= totalSourceBytes,
                  !nextDestination.overflow else {
                throw RepackError.installStateCorrupt(
                    path: path,
                    detail: "completed range byte totals are invalid")
            }
            sourceTotal = nextSource.partialValue
            destinationTotal = nextDestination.partialValue
        }
    }
}
