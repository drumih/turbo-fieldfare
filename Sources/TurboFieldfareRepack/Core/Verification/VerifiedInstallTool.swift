import Foundation
import Darwin
import TurboFieldfareFormat

public struct VerifyInstallOptions: Sendable {
    public let inputGTurbo: String

    public init(inputGTurbo: String) {
        self.inputGTurbo = inputGTurbo
    }
}

public struct VerifyInstallResult: Sendable {
    public let receiptPath: String
    public let fileCount: Int
    public let bytesVerified: UInt64
    public let unexpectedEntries: [String]
}

public enum VerifiedInstallTool {
    public static let metadataMaxBytes: UInt64 = 16 * 1024 * 1024
    public static let manifestMaxBytes: UInt64 = 4 * 1024 * 1024
    public static let layoutMaxBytes: UInt64 = 16 * 1024 * 1024

    public static func run(options: VerifyInstallOptions) throws -> VerifyInstallResult {
        let root = URL(fileURLWithPath: options.inputGTurbo).standardizedFileURL
        let access = try GTurboDirectoryAccess(rootPath: root.path)
        let manifestFD = try access.openFile("manifest.json")
        defer { close(manifestFD) }
        _ = fcntl(manifestFD, F_NOCACHE, 1)
        let manifestData = try access.readMetadata(
            fileDescriptor: manifestFD, relativePath: "manifest.json",
            maxBytes: manifestMaxBytes)
        let manifestSize = UInt64(manifestData.count)
        let manifestSha = hashMetadata(manifestData)
        let manifest = try loadManifest(data: manifestData)

        let layoutRelativePath = "packed_experts/layout.json"
        guard let layoutManifestEntry = manifest.files[layoutRelativePath] else {
            throw RepackError.configurationInvalid(
                detail: "manifest missing \(layoutRelativePath)")
        }
        let layoutFD = try access.openFile(layoutRelativePath)
        defer { close(layoutFD) }
        _ = fcntl(layoutFD, F_NOCACHE, 1)
        let layoutData = try access.readMetadata(
            fileDescriptor: layoutFD, relativePath: layoutRelativePath,
            maxBytes: layoutMaxBytes)
        let layoutSize = UInt64(layoutData.count)
        let layoutSha = hashMetadata(layoutData)
        let layout = try loadLayout(data: layoutData)
        try validatePackedExpertLayout(manifest: manifest, layout: layout)
        guard layoutSize == layoutManifestEntry.size else {
            throw RepackError.configurationInvalid(
                detail: "\(layoutRelativePath) size \(layoutSize) != manifest \(layoutManifestEntry.size)")
        }
        guard layoutSha.lowercased() == layoutManifestEntry.sha256.lowercased() else {
            throw RepackError.configurationInvalid(detail: "\(layoutRelativePath) SHA mismatch")
        }

        var files: [RepackAudit.OutputFile] = []
        files.reserveCapacity(manifest.files.count)
        var bytesVerified = manifestSize
        for relativePath in manifest.files.keys.sorted() {
            guard let entry = manifest.files[relativePath] else { continue }
            let actualSize: UInt64
            let actualSha: String
            if relativePath == layoutRelativePath {
                actualSize = layoutSize
                actualSha = layoutSha
            } else {
                (actualSize, actualSha) = try inspectFile(
                    access: access, relativePath: relativePath)
            }
            guard actualSize == entry.size else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != manifest \(entry.size)")
            }
            guard actualSha.lowercased() == entry.sha256.lowercased() else {
                throw RepackError.configurationInvalid(detail: "\(relativePath) SHA mismatch")
            }
            bytesVerified = try addingVerifiedBytes(bytesVerified, actualSize)
            files.append(RepackAudit.OutputFile(relativePath: relativePath,
                                                size: actualSize,
                                                sha256: actualSha))
        }
        let unexpectedEntries = try findUnexpectedEntries(access: access, manifest: manifest)

        let receiptData = try VerifiedInstallReceiptWriter.encode(
            outputDir: root.path,
            manifestSha256: manifestSha,
            manifestSize: manifestSize,
            sourceRepoID: nil,
            sourceRevision: manifest.sourceSnapshotHash,
            toolVersion: "TurboFieldfareRepack verify-install",
            files: files)
        let receiptPath = root.appendingPathComponent(VerifiedInstallReceiptWriter.fileName).path
        try access.atomicWrite(receiptData, to: VerifiedInstallReceiptWriter.fileName)
        return VerifyInstallResult(receiptPath: receiptPath,
                                   fileCount: files.count + 1,
                                   bytesVerified: bytesVerified,
                                   unexpectedEntries: unexpectedEntries)
    }

    private static func hashMetadata(_ data: Data) -> String {
        var hasher = Sha256Stream()
        data.withUnsafeBytes { hasher.update($0) }
        return hasher.finalizeHexString()
    }

    private static func inspectFile(access: GTurboDirectoryAccess,
                                    relativePath: String) throws -> (UInt64, String) {
        let fd = try access.openFile(relativePath)
        defer { close(fd) }
        let size = try access.fileSize(
            fileDescriptor: fd, relativePath: relativePath)
        let sha = try access.hash(
            fileDescriptor: fd, relativePath: relativePath, noCache: true)
        return (size, sha)
    }

    package static func addingVerifiedBytes(_ current: UInt64,
                                            _ next: UInt64) throws -> UInt64 {
        let (result, overflow) = current.addingReportingOverflow(next)
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail: "verified byte count exceeds UInt64")
        }
        return result
    }

    private static func loadManifest(data: Data) throws -> GTurboManifestV1 {
        do {
            return try GTurboManifestCodec.decode(data)
        } catch {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid: \(error)")
        }
    }

    private static func loadLayout(data: Data) throws -> GTurboPackedExpertsLayoutV1 {
        do {
            return try GTurboPackedExpertsLayoutCodec.decode(data)
        } catch {
            throw RepackError.configurationInvalid(detail: "packed_experts/layout.json invalid: \(error)")
        }
    }

    private static func validatePackedExpertLayout(manifest: GTurboManifestV1,
                                                   layout: GTurboPackedExpertsLayoutV1) throws {
        do { try GTurboV1StructuralValidator.crossValidate(manifest: manifest, layout: layout) }
        catch {
            throw RepackError.configurationInvalid(
                detail: "packed expert layout does not match manifest: \(error)")
        }
        let expectedLayerSize = UInt64(layout.expertsPerLayer) * layout.expertStride
        for layer in layout.layers {
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw RepackError.configurationInvalid(detail: "manifest missing \(relativePath)")
            }
            guard manifestEntry.size == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) manifest size \(manifestEntry.size) != \(expectedLayerSize)")
            }
            for (index, expert) in layer.experts.enumerated() {
                let expertID = expert.expert ?? index
                guard expert.offset <= expectedLayerSize,
                      expert.size <= expectedLayerSize - expert.offset else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) range exceeds file size")
                }
            }
        }
    }

    private static func findUnexpectedEntries(access: GTurboDirectoryAccess,
                                              manifest: GTurboManifestV1) throws -> [String] {
        let declaredFiles = Set(manifest.files.keys)
            .union(["manifest.json", VerifiedInstallReceiptWriter.fileName])
        var allowed = declaredFiles
        for path in declaredFiles {
            var parts = path.split(separator: "/").map(String.init)
            while parts.count > 1 {
                _ = parts.removeLast()
                allowed.insert(parts.joined(separator: "/"))
            }
        }
        allowed.insert("tokenizer")
        let scanDepth = max(16, declaredFiles.lazy.map {
            $0.split(separator: "/", omittingEmptySubsequences: false).count
        }.max() ?? 1)

        var unexpected: [String] = []
        for rel in try access.relativeEntries(maxDepth: scanDepth) {
            // Every path component, not just a literal top-level `.DS_Store`:
            // a directory that went through Finder, SMB or an archive comes back
            // with AppleDouble `._x` files, `.icloud` placeholders and a Spotlight
            // index, none of which make a cryptographically valid install invalid.
            if rel.split(separator: "/").contains(where: {
                GTurboVisionFormatV1.isSidecarEntry(String($0))
            }) { continue }
            if rel == "tokenizer" || rel.hasPrefix("tokenizer/") { continue }
            if !allowed.contains(rel) {
                unexpected.append(rel)
            }
        }
        return unexpected.sorted()
    }
}
