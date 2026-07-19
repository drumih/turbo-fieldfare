import Foundation
import Darwin

public struct DiskSpaceRequirement: Equatable, Sendable {
    public let path: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(path: String, requiredBytes: UInt64, availableBytes: UInt64) {
        self.path = path
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }

    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public enum DiskSpaceChecker {
    public static func assess(path: String,
                              bytes: UInt64,
                              reserveBytes: UInt64 = 1 * 1024 * 1024 * 1024) throws
        -> DiskSpaceRequirement {
        let requestedDirectory = directoryForProbe(path)
        let probeDirectory = nearestExistingDirectory(requestedDirectory)
        return try requirement(path: probeDirectory,
                               bytes: bytes,
                               reserveBytes: reserveBytes)
    }

    public static func requireAvailable(path: String,
                                        bytes: UInt64,
                                        reserveBytes: UInt64 = 1 * 1024 * 1024 * 1024) throws -> DiskSpaceRequirement {
        let dir = directoryForProbe(path)
        try Posix.mkdirP(dir)
        let result = try requirement(path: dir, bytes: bytes, reserveBytes: reserveBytes)
        guard result.canInstall else {
            throw RepackError.diskSpaceInsufficient(path: dir,
                                                    required: result.requiredBytes,
                                                    available: result.availableBytes)
        }
        return result
    }

    private static func requirement(path: String,
                                    bytes: UInt64,
                                    reserveBytes: UInt64) throws -> DiskSpaceRequirement {
        var st = statfs()
        if statfs(path, &st) != 0 {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        let available = UInt64(st.f_bavail) * UInt64(st.f_bsize)
        let required = bytes + reserveBytes
        return DiskSpaceRequirement(path: path,
                                    requiredBytes: required,
                                    availableBytes: available)
    }

    private static func nearestExistingDirectory(_ path: String) -> String {
        var url = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
        while !FileManager.default.fileExists(atPath: url.path) {
            let parent = url.deletingLastPathComponent()
            if parent.path == url.path { return url.path }
            url = parent
        }
        return url.path
    }

    private static func directoryForProbe(_ path: String) -> String {
        let ns = path as NSString
        let ext = ns.pathExtension
        if ext.isEmpty {
            return path
        }
        return ns.deletingLastPathComponent
    }
}
