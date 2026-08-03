import Foundation

struct VisionSidecarManifest: Codable, Sendable, Equatable {
    static let magic = "GTURBO-VISION"
    static let schemaVersion = 1
    static let kind = "gemma4-vision-sidecar"
    static let maximumBytes: UInt64 = 4 * 1024 * 1024

    struct FileEntry: Codable, Sendable, Equatable {
        let size: UInt64
        let sha256: String
    }

    let magic: String
    let schemaVersion: Int
    let kind: String
    let sourceRepoID: String
    let sourceRevision: String
    let sourceSnapshotHash: String
    let rootManifestSha256: String
    let residentIndexSha256: String
    let entryCount: Int
    let sourceTensorCount: Int
    let files: [String: FileEntry]

    init(
        sourceRepoID: String,
        sourceRevision: String,
        sourceSnapshotHash: String,
        rootManifestSha256: String,
        residentIndexSha256: String,
        entryCount: Int,
        sourceTensorCount: Int,
        weightsSize: UInt64,
        weightsSha256: String
    ) {
        self.magic = Self.magic
        self.schemaVersion = Self.schemaVersion
        self.kind = Self.kind
        self.sourceRepoID = sourceRepoID
        self.sourceRevision = sourceRevision
        self.sourceSnapshotHash = sourceSnapshotHash
        self.rootManifestSha256 = rootManifestSha256
        self.residentIndexSha256 = residentIndexSha256
        self.entryCount = entryCount
        self.sourceTensorCount = sourceTensorCount
        self.files = [
            "weights.bin": FileEntry(size: weightsSize, sha256: weightsSha256)
        ]
    }

    func encode() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    static func load(path: String) throws -> Self {
        let data = try Posix.readBoundedData(path, maximumBytes: maximumBytes)
        do {
            let manifest = try JSONDecoder().decode(Self.self, from: data)
            try manifest.validate(path: path)
            return manifest
        } catch let error as RepackError {
            throw error
        } catch {
            throw RepackError.configurationInvalid(
                detail: "vision manifest invalid at \(path): \(error)")
        }
    }

    func validate(path: String) throws {
        guard magic == Self.magic,
              schemaVersion == Self.schemaVersion,
              kind == Self.kind,
              !sourceRepoID.isEmpty,
              sourceRevision.count == 40,
              sourceRevision.allSatisfy(\.isHexDigit),
              sourceSnapshotHash.hasPrefix("sha256:"),
              sourceSnapshotHash.count == "sha256:".count + 64,
              sourceSnapshotHash.dropFirst("sha256:".count).allSatisfy(\.isHexDigit),
              rootManifestSha256.count == 64,
              rootManifestSha256.allSatisfy(\.isHexDigit),
              residentIndexSha256.count == 64,
              residentIndexSha256.allSatisfy(\.isHexDigit),
              entryCount > 0,
              sourceTensorCount >= entryCount,
              files.count == 1,
              let weights = files["weights.bin"],
              weights.size > 0,
              weights.sha256.count == 64 else {
            throw RepackError.configurationInvalid(
                detail: "vision manifest identity is invalid at \(path)")
        }
    }
}

struct RootModelSourceBinding: Sendable, Equatable {
    static let maximumManifestBytes: UInt64 = 16 * 1024 * 1024

    let sourceSnapshotHash: String
    let manifestSha256: String

    static func load(modelDirectory: String) throws -> Self {
        let path = (modelDirectory as NSString).appendingPathComponent("manifest.json")
        guard try Posix.entryKind(modelDirectory) == .directory,
              try Posix.entryKind(path) == .regular else {
            throw RepackError.configurationInvalid(
                detail: "a completed .gturbo model is required before installing vision")
        }
        let data = try Posix.readBoundedData(path, maximumBytes: maximumManifestBytes)
        let root: [String: Any]
        do {
            guard let decoded = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw RepackError.configurationInvalid(detail: "root manifest is not an object")
            }
            root = decoded
        } catch let error as RepackError {
            throw error
        } catch {
            throw RepackError.configurationInvalid(detail: "root manifest is invalid: \(error)")
        }
        guard root["magic"] as? String == GTurboJSON.magic,
              let sourceHash = root["sourceSnapshotHash"] as? String,
              sourceHash.hasPrefix("sha256:"),
              sourceHash.count == "sha256:".count + 64 else {
            throw RepackError.configurationInvalid(
                detail: "root manifest has no valid source snapshot binding")
        }
        return RootModelSourceBinding(
            sourceSnapshotHash: sourceHash,
            manifestSha256: try Sha256Stream.hashFile(path: path))
    }
}

public struct VerifyVisionSidecarResult: Sendable, Equatable {
    public let manifestPath: String
    public let weightsBytesVerified: UInt64
    public let entryCount: Int
}

public enum VisionSidecarVerifier {
    public static func run(modelDirectory: String) throws -> VerifyVisionSidecarResult {
        let root = URL(fileURLWithPath: modelDirectory).standardizedFileURL
        let binding = try RootModelSourceBinding.load(modelDirectory: root.path)
        let vision = root.appendingPathComponent("vision")
        guard try Posix.entryKind(vision.path) == .directory else {
            throw RepackError.configurationInvalid(detail: "vision sidecar is not installed")
        }
        let manifestPath = vision.appendingPathComponent("manifest.json").path
        let manifest = try VisionSidecarManifest.load(path: manifestPath)
        guard manifest.sourceSnapshotHash == binding.sourceSnapshotHash,
              manifest.rootManifestSha256 == binding.manifestSha256 else {
            throw RepackError.installStateIncompatible(
                detail: "vision sidecar belongs to a different root model")
        }
        guard let expected = manifest.files["weights.bin"] else {
            throw RepackError.configurationInvalid(detail: "vision manifest misses weights.bin")
        }
        let weightsPath = vision.appendingPathComponent("weights.bin").path
        let descriptor = try Posix.openReadNoFollow(weightsPath)
        defer { close(descriptor) }
        let size = try Posix.fileSize(fd: descriptor, path: weightsPath)
        guard size == expected.size else {
            throw RepackError.configurationInvalid(detail: "vision weights size mismatch")
        }
        let sha = try Sha256Stream.hashFile(path: weightsPath, noCache: true)
        guard sha.lowercased() == expected.sha256.lowercased() else {
            throw RepackError.configurationInvalid(detail: "vision weights SHA mismatch")
        }
        let indexSha = try hashResidentIndex(
            descriptor: descriptor,
            path: weightsPath,
            fileSize: size)
        guard indexSha.lowercased() == manifest.residentIndexSha256.lowercased() else {
            throw RepackError.configurationInvalid(detail: "vision resident index SHA mismatch")
        }
        try validateReceipt(visionDirectory: vision.path, manifest: manifest)
        return VerifyVisionSidecarResult(
            manifestPath: manifestPath,
            weightsBytesVerified: size,
            entryCount: manifest.entryCount)
    }

    private struct Receipt: Decodable {
        let schemaVersion: Int
        let manifestSha256: String
        let modelDirectoryPath: String
        let sourceRepoID: String?
        let sourceRevision: String?
        let files: [String: VisionSidecarManifest.FileEntry]
    }

    private static func validateReceipt(
        visionDirectory: String,
        manifest: VisionSidecarManifest
    ) throws {
        let receiptPath = (visionDirectory as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        let data = try Posix.readBoundedData(receiptPath, maximumBytes: 4 * 1024 * 1024)
        let receipt: Receipt
        do {
            receipt = try JSONDecoder().decode(Receipt.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(detail: "vision receipt invalid: \(error)")
        }
        let manifestPath = (visionDirectory as NSString).appendingPathComponent("manifest.json")
        let manifestSha = try Sha256Stream.hashFile(path: manifestPath, noCache: true)
        let manifestDescriptor = try Posix.openReadNoFollow(manifestPath)
        let manifestSize: UInt64
        do {
            manifestSize = try Posix.fileSize(fd: manifestDescriptor, path: manifestPath)
            close(manifestDescriptor)
        } catch {
            close(manifestDescriptor)
            throw error
        }
        guard receipt.schemaVersion == 1 else {
            throw RepackError.configurationInvalid(
                detail: "vision receipt has an unsupported schema")
        }
        guard receipt.manifestSha256.lowercased() == manifestSha.lowercased() else {
            throw RepackError.configurationInvalid(
                detail: "vision receipt manifest SHA mismatch")
        }
        let expectedDirectory = try Posix.physicalPath(visionDirectory)
        let receiptDirectory = try Posix.physicalPath(receipt.modelDirectoryPath)
        guard receiptDirectory == expectedDirectory else {
            throw RepackError.configurationInvalid(
                detail: "vision receipt directory mismatch")
        }
        guard receipt.sourceRepoID == manifest.sourceRepoID,
              receipt.sourceRevision == manifest.sourceRevision else {
            throw RepackError.configurationInvalid(
                detail: "vision receipt source mismatch")
        }
        guard Set(receipt.files.keys) == Set(["weights.bin", "manifest.json"]) else {
            throw RepackError.configurationInvalid(
                detail: "vision receipt file set mismatch")
        }
        guard receipt.files["weights.bin"] == manifest.files["weights.bin"] else {
            throw RepackError.configurationInvalid(
                detail: "vision receipt weights metadata mismatch")
        }
        guard receipt.files["manifest.json"]?.size == manifestSize,
              receipt.files["manifest.json"]?.sha256.lowercased()
                == manifestSha.lowercased() else {
            throw RepackError.configurationInvalid(
                detail: "vision receipt manifest metadata mismatch")
        }
    }

    private static func hashResidentIndex(
        descriptor: Int32,
        path: String,
        fileSize: UInt64
    ) throws -> String {
        var prefix = [UInt8](repeating: 0, count: 8)
        try prefix.withUnsafeMutableBytes { raw in
            try Posix.preadAll(
                fd: descriptor,
                path: path,
                buf: raw.baseAddress!,
                count: raw.count,
                offset: 0)
        }
        let indexSize = prefix.enumerated().reduce(UInt64(0)) {
            $0 | (UInt64($1.element) << UInt64($1.offset * 8))
        }
        guard indexSize >= UInt64(GTurboBinary.indexHeaderBytes),
              indexSize <= UInt64(BoundedScratch.defaultLimitBytes),
              indexSize <= fileSize else {
            throw RepackError.configurationInvalid(
                detail: "vision resident index size \(indexSize) is invalid")
        }
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: min(WriterCore.tileBytes, Int(indexSize)),
            alignment: 16_384)
        defer { scratch.deallocate() }
        var hasher = Sha256Stream()
        var offset: UInt64 = 0
        while offset < indexSize {
            let count = min(scratch.count, Int(indexSize - offset))
            try Posix.preadAll(
                fd: descriptor,
                path: path,
                buf: scratch.baseAddress!,
                count: count,
                offset: offset)
            hasher.update(UnsafeRawBufferPointer(
                start: scratch.baseAddress,
                count: count))
            offset += UInt64(count)
        }
        return hasher.finalizeHexString()
    }
}
