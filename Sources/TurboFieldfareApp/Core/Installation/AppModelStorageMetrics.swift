import Foundation

public struct AppModelStorageMetrics: Equatable, Sendable {
    public let logicalBytes: UInt64
    public let allocatedBytes: UInt64
    public let availableBytes: UInt64?

    public static func measure(at directory: URL) throws -> Self {
        let keys: Set<URLResourceKey> = [
            .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
            .totalFileAllocatedSizeKey,
        ]
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]) else {
            throw CocoaError(.fileReadUnknown)
        }
        var logical: UInt64 = 0
        var allocated: UInt64 = 0
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: keys)
            guard values.isRegularFile == true, values.isSymbolicLink != true else {
                continue
            }
            logical += UInt64(max(0, values.fileSize ?? 0))
            allocated += UInt64(max(0, values.totalFileAllocatedSize ?? 0))
        }
        let volume = try directory.deletingLastPathComponent().resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return Self(
            logicalBytes: logical,
            allocatedBytes: allocated,
            availableBytes: volume.volumeAvailableCapacityForImportantUsage.map(UInt64.init))
    }
}
