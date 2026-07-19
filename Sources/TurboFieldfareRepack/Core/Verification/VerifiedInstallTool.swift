import Foundation
import Darwin

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

    public static func run(options: VerifyInstallOptions) throws -> VerifyInstallResult {
        let root = URL(fileURLWithPath: options.inputGTurbo).standardizedFileURL
        let manifestPath = root.appendingPathComponent("manifest.json").path
        let manifestSize = try fileSize(path: manifestPath, relativePath: "manifest.json")
        let manifestSha = try Sha256Stream.hashFile(path: manifestPath, noCache: true)
        let manifest = try loadManifest(path: manifestPath)
        try validatePackedExpertLayout(root: root, manifest: manifest)

        var files: [RepackAudit.OutputFile] = []
        files.reserveCapacity(manifest.files.count)
        var bytesVerified = manifestSize
        for relativePath in manifest.files.keys.sorted() {
            guard let entry = manifest.files[relativePath] else { continue }
            let path = root.appendingPathComponent(relativePath).path
            let actualSize = try fileSize(path: path, relativePath: relativePath)
            guard actualSize == entry.size else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != manifest \(entry.size)")
            }
            let actualSha = try Sha256Stream.hashFile(path: path, noCache: true)
            guard actualSha.lowercased() == entry.sha256.lowercased() else {
                throw RepackError.configurationInvalid(detail: "\(relativePath) SHA mismatch")
            }
            bytesVerified &+= actualSize
            files.append(RepackAudit.OutputFile(relativePath: relativePath,
                                                size: actualSize,
                                                sha256: actualSha))
        }
        let unexpectedEntries = try findUnexpectedEntries(root: root, manifest: manifest)

        let receiptData = try VerifiedInstallReceiptWriter.encode(
            outputDir: root.path,
            manifestSha256: manifestSha,
            manifestSize: manifestSize,
            sourceRepoID: nil,
            sourceRevision: manifest.sourceSnapshotHash,
            toolVersion: "TurboFieldfareRepack verify-install",
            files: files)
        let receiptPath = root.appendingPathComponent(VerifiedInstallReceiptWriter.fileName).path
        try receiptData.write(to: URL(fileURLWithPath: receiptPath), options: .atomic)
        return VerifyInstallResult(receiptPath: receiptPath,
                                   fileCount: files.count + 1,
                                   bytesVerified: bytesVerified,
                                   unexpectedEntries: unexpectedEntries)
    }

    private struct ManifestFileEntry: Decodable {
        let size: UInt64
        let sha256: String
    }

    private struct Manifest: Decodable {
        let files: [String: ManifestFileEntry]
        let expertsPerLayer: Int
        let numLayers: Int
        let expertStride: UInt64
        let sourceSnapshotHash: String?
    }

    private struct PackedExpertsLayout: Decodable {
        let expertStride: UInt64
        let numLayers: Int
        let expertsPerLayer: Int
        let layers: [Layer]
    }

    private struct Layer: Decodable {
        let layer: Int
        let file: String
        let experts: [Expert]
    }

    private struct Expert: Decodable {
        let expert: Int?
        let offset: UInt64
        let size: UInt64
    }

    private static func loadManifest(path: String) throws -> Manifest {
        do {
            let data = try loadMetadataJSON(path: path, relativePath: "manifest.json")
            return try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid: \(error)")
        }
    }

    private static func loadLayout(path: String) throws -> PackedExpertsLayout {
        do {
            let data = try loadMetadataJSON(path: path, relativePath: "packed_experts/layout.json")
            return try JSONDecoder().decode(PackedExpertsLayout.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(detail: "packed_experts/layout.json invalid: \(error)")
        }
    }

    private static func loadMetadataJSON(path: String, relativePath: String) throws -> Data {
        let size = try fileSize(path: path, relativePath: relativePath)
        guard size <= metadataMaxBytes else {
            throw RepackError.configurationInvalid(
                detail: "\(relativePath) size \(size) exceeds metadata cap \(metadataMaxBytes)")
        }
        return try Data(contentsOf: URL(fileURLWithPath: path))
    }

    private static func validatePackedExpertLayout(root: URL, manifest: Manifest) throws {
        let layoutRelativePath = "packed_experts/layout.json"
        guard manifest.files[layoutRelativePath] != nil else {
            throw RepackError.configurationInvalid(detail: "manifest missing \(layoutRelativePath)")
        }
        let layout = try loadLayout(path: root.appendingPathComponent(layoutRelativePath).path)
        let pageSize = UInt64(getpagesize())
        guard layout.expertStride == manifest.expertStride,
              layout.numLayers == manifest.numLayers,
              layout.expertsPerLayer == manifest.expertsPerLayer else {
            throw RepackError.configurationInvalid(detail: "packed expert layout dimensions mismatch manifest")
        }
        guard layout.expertStride % pageSize == 0 else {
            throw RepackError.configurationInvalid(
                detail: "expertStride \(layout.expertStride) is not page-aligned")
        }
        guard layout.layers.count == layout.numLayers else {
            throw RepackError.configurationInvalid(detail: "packed expert layout layer count mismatch")
        }
        let expectedLayerSize = UInt64(layout.expertsPerLayer) * layout.expertStride
        for layer in layout.layers {
            guard layer.layer >= 0 && layer.layer < layout.numLayers else {
                throw RepackError.configurationInvalid(detail: "packed expert layer index out of range")
            }
            guard layer.experts.count == layout.expertsPerLayer else {
                throw RepackError.configurationInvalid(
                    detail: "packed_experts/\(layer.file) expert count mismatch")
            }
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw RepackError.configurationInvalid(detail: "manifest missing \(relativePath)")
            }
            guard manifestEntry.size == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) manifest size \(manifestEntry.size) != \(expectedLayerSize)")
            }
            let actualSize = try fileSize(path: root.appendingPathComponent(relativePath).path,
                                          relativePath: relativePath)
            guard actualSize == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(expectedLayerSize)")
            }
            var seenExperts = Set<Int>()
            for (index, expert) in layer.experts.enumerated() {
                let expertID = expert.expert ?? index
                guard expertID >= 0 && expertID < layout.expertsPerLayer else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert id out of range")
                }
                guard seenExperts.insert(expertID).inserted else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) duplicate expert \(expertID)")
                }
                guard expert.size == layout.expertStride else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) size mismatch")
                }
                guard expert.offset % pageSize == 0 else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) offset is not page-aligned")
                }
                guard expert.offset <= actualSize,
                      expert.size <= actualSize - expert.offset else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) range exceeds file size")
                }
            }
        }
    }

    private static func fileSize(path: String, relativePath: String) throws -> UInt64 {
        var st = stat()
        guard stat(path, &st) == 0 else {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        guard st.st_size >= 0 else {
            throw RepackError.configurationInvalid(detail: "\(relativePath) has negative file size")
        }
        return UInt64(st.st_size)
    }

    private static func findUnexpectedEntries(root: URL, manifest: Manifest) throws -> [String] {
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

        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsPackageDescendants]) else {
            return []
        }

        var unexpected: [String] = []
        for case let url as URL in enumerator {
            let rel = relativePath(for: url, root: root)
            if rel == ".DS_Store" { continue }
            if rel == "tokenizer" || rel.hasPrefix("tokenizer/") { continue }
            if !allowed.contains(rel) {
                unexpected.append(rel)
            }
        }
        return unexpected.sorted()
    }

    private static func relativePath(for url: URL, root: URL) -> String {
        let rootPathRaw = root.standardizedFileURL.path
        let rootPath = rootPathRaw.hasSuffix("/") ? rootPathRaw : rootPathRaw + "/"
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(rootPath) else { return url.lastPathComponent }
        return String(path.dropFirst(rootPath.count))
    }
}
