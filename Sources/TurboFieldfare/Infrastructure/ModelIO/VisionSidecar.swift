import Darwin
import Foundation
import Metal

public enum VisionSidecarIntegrityPolicy: Sendable, Equatable {
    /// Hash the complete sidecar before opening it. This is the safe default
    /// when no independently trusted installation receipt is available.
    case fullSHA256
    /// Validate metadata and file size only. Callers must use this only after
    /// validating the install receipt through a trusted application boundary.
    case trustedSizeOnly
}

public enum VisionSidecarError: Error, CustomStringConvertible, Equatable {
    case missingSidecar
    case invalidManifest(String)
    case incompatibleModel(expected: String, actual: String)
    case incompatibleSnapshot(expected: String, actual: String)
    case invalidWeights(String)
    case missingTensor(String)
    case invalidTensor(name: String, detail: String)
    case allocationFailed(bytes: Int)
    case cancelled

    public var description: String {
        switch self {
        case .missingSidecar:
            return "the optional Gemma 4 vision sidecar is not installed"
        case .invalidManifest(let detail):
            return "invalid vision sidecar manifest: \(detail)"
        case .incompatibleModel(let expected, let actual):
            return "vision sidecar model \(actual) does not match \(expected)"
        case .incompatibleSnapshot(let expected, let actual):
            return "vision sidecar snapshot \(actual) does not match \(expected)"
        case .invalidWeights(let detail):
            return "invalid vision sidecar weights: \(detail)"
        case .missingTensor(let name):
            return "vision sidecar is missing tensor \(name)"
        case .invalidTensor(let name, let detail):
            return "vision tensor \(name) is invalid: \(detail)"
        case .allocationFailed(let bytes):
            return "could not allocate \(bytes) bytes for bounded vision weight streaming"
        case .cancelled:
            return "vision encoding was cancelled"
        }
    }
}

public struct VisionSidecarFileEntry: Codable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct VisionSidecarManifest: Codable, Equatable, Sendable {
    public let magic: String
    public let schemaVersion: Int
    public let kind: String
    public let sourceRepoID: String
    public let sourceRevision: String
    public let sourceSnapshotHash: String
    public let files: [String: VisionSidecarFileEntry]
    public let entryCount: Int
    public let sourceTensorCount: Int

    public init(magic: String,
                schemaVersion: Int,
                kind: String,
                sourceRepoID: String,
                sourceRevision: String,
                sourceSnapshotHash: String,
                files: [String: VisionSidecarFileEntry],
                entryCount: Int,
                sourceTensorCount: Int) {
        self.magic = magic
        self.schemaVersion = schemaVersion
        self.kind = kind
        self.sourceRepoID = sourceRepoID
        self.sourceRevision = sourceRevision
        self.sourceSnapshotHash = sourceSnapshotHash
        self.files = files
        self.entryCount = entryCount
        self.sourceTensorCount = sourceTensorCount
    }
}

/// A validated, non-resident handle to the optional Gemma 4 vision payload.
///
/// Opening the handle reads only the small binary index. Tensor bytes enter a
/// caller-owned, fixed-size Metal buffer through `pread`; the complete 1+ GiB
/// tower is never materialized in the Swift heap or mapped as one Metal buffer.
public final class VisionSidecar: @unchecked Sendable {
    public static let directoryName = "vision"
    public static let manifestName = "manifest.json"
    public static let weightsName = "weights.bin"

    public let manifest: VisionSidecarManifest
    public let weightsURL: URL
    public let maximumTensorBytes: Int

    private let index: ResidentIndex
    private let fd: Int32
    private let readLock = NSLock()

    public static func isInstalled(in modelDirectory: URL) -> Bool {
        let vision = modelDirectory.appendingPathComponent(directoryName, isDirectory: true)
        return FileManager.default.fileExists(
            atPath: vision.appendingPathComponent(manifestName).path)
            && FileManager.default.fileExists(
                atPath: vision.appendingPathComponent(weightsName).path)
    }

    public init(modelDirectory: URL,
                integrityPolicy: VisionSidecarIntegrityPolicy = .fullSHA256) throws {
        let rootManifest = try ManifestReader.load(
            directoryURL: modelDirectory,
            expecting: .gemma4_26B_A4B)
        let visionDirectory = modelDirectory.appendingPathComponent(
            Self.directoryName,
            isDirectory: true)
        let manifestURL = visionDirectory.appendingPathComponent(Self.manifestName)
        let weightsURL = visionDirectory.appendingPathComponent(Self.weightsName)

        guard FileManager.default.fileExists(atPath: manifestURL.path),
              FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw VisionSidecarError.missingSidecar
        }
        let manifestData = try Self.readBoundedMetadata(manifestURL, maximumBytes: 1 << 20)
        let decoded: VisionSidecarManifest
        do {
            decoded = try JSONDecoder().decode(VisionSidecarManifest.self, from: manifestData)
        } catch {
            throw VisionSidecarError.invalidManifest("JSON decode failed: \(error)")
        }
        try Self.validateManifest(decoded)
        guard decoded.sourceRepoID == rootManifest.modelID else {
            throw VisionSidecarError.incompatibleModel(
                expected: rootManifest.modelID,
                actual: decoded.sourceRepoID)
        }
        if let expectedSnapshot = rootManifest.sourceSnapshotHash,
           decoded.sourceSnapshotHash != expectedSnapshot {
            throw VisionSidecarError.incompatibleSnapshot(
                expected: expectedSnapshot,
                actual: decoded.sourceSnapshotHash)
        }
        guard let fileEntry = decoded.files[Self.weightsName] else {
            throw VisionSidecarError.invalidManifest("files.weights.bin is missing")
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: weightsURL.path)
        guard let actualNumber = attributes[.size] as? NSNumber else {
            throw VisionSidecarError.invalidWeights("file size is unavailable")
        }
        let actualSize = actualNumber.uint64Value
        guard actualSize == fileEntry.size else {
            throw VisionSidecarError.invalidWeights(
                "weights.bin size \(actualSize) does not match manifest \(fileEntry.size)")
        }
        if integrityPolicy == .fullSHA256 {
            do {
                try Sha256Verifier.verifyFile(
                    at: weightsURL,
                    named: "vision/weights.bin",
                    expectedHex: fileEntry.sha256)
            } catch {
                throw VisionSidecarError.invalidWeights("SHA-256 verification failed")
            }
        }

        let parsedIndex: ResidentIndex
        do {
            parsedIndex = try ResidentIndexReader.load(fileURL: weightsURL)
        } catch {
            throw VisionSidecarError.invalidWeights("binary index: \(error)")
        }
        guard parsedIndex.entries.count == decoded.entryCount,
              parsedIndex.header.entryCount == UInt64(decoded.entryCount) else {
            throw VisionSidecarError.invalidWeights(
                "index entry count \(parsedIndex.entries.count) does not match manifest \(decoded.entryCount)")
        }
        let indexedSize = parsedIndex.header.indexSize + parsedIndex.header.residentSize
        guard indexedSize == actualSize else {
            throw VisionSidecarError.invalidWeights(
                "index accounts for \(indexedSize) bytes but file contains \(actualSize)")
        }
        guard parsedIndex.entries.values.allSatisfy({ entry in
            entry.fileOffset >= parsedIndex.header.indexSize
                && entry.fileOffset <= actualSize
                && entry.sizeBytes <= actualSize - entry.fileOffset
                && (entry.scaleSize == 0
                    || (entry.scaleOffset >= parsedIndex.header.indexSize
                        && entry.scaleOffset <= actualSize
                        && entry.scaleSize <= actualSize - entry.scaleOffset))
                && (entry.biasSize == 0
                    || (entry.biasOffset >= parsedIndex.header.indexSize
                        && entry.biasOffset <= actualSize
                        && entry.biasSize <= actualSize - entry.biasOffset))
        }) else {
            throw VisionSidecarError.invalidWeights("one or more tensor ranges are outside weights.bin")
        }

        let openedFD = open(weightsURL.path, O_RDONLY)
        guard openedFD >= 0 else {
            throw VisionSidecarError.invalidWeights("open failed with errno \(errno)")
        }
        self.manifest = decoded
        self.weightsURL = weightsURL
        self.index = parsedIndex
        self.fd = openedFD
        self.maximumTensorBytes = parsedIndex.entries.values.reduce(0) {
            max($0, Int($1.sizeBytes))
        }
    }

    deinit {
        close(fd)
    }

    public var tensorNames: [String] {
        index.entries.keys.sorted()
    }

    public func tensor(named name: String) throws -> VisionTensorDescriptor {
        guard let entry = index.entries[name] else {
            throw VisionSidecarError.missingTensor(name)
        }
        return VisionTensorDescriptor(entry: entry)
    }

    /// Reads the selected tensor and its optional affine companions directly
    /// into a caller-owned shared Metal buffer. No tensor-sized Swift `Data`
    /// object is created.
    @discardableResult
    func readTensor(named name: String,
                    into buffer: MTLBuffer,
                    at destinationOffset: Int = 0,
                    includeAffineCompanions: Bool = false) throws -> VisionLoadedTensor {
        guard let entry = index.entries[name] else {
            throw VisionSidecarError.missingTensor(name)
        }
        let weightBytes = Int(entry.sizeBytes)
        let scaleBytes = includeAffineCompanions ? Int(entry.scaleSize) : 0
        let biasBytes = includeAffineCompanions ? Int(entry.biasSize) : 0
        let scaleDestination = Self.align(destinationOffset + weightBytes, to: 16)
        let biasDestination = Self.align(scaleDestination + scaleBytes, to: 16)
        let requiredEnd = includeAffineCompanions
            ? biasDestination + biasBytes
            : destinationOffset + weightBytes
        guard destinationOffset >= 0, requiredEnd <= buffer.length else {
            throw VisionSidecarError.allocationFailed(bytes: requiredEnd)
        }

        readLock.lock()
        defer { readLock.unlock() }
        try Self.readFull(fd: fd,
                          fileOffset: entry.fileOffset,
                          destination: buffer.contents().advanced(by: destinationOffset),
                          count: weightBytes)
        if includeAffineCompanions, scaleBytes > 0 {
            try Self.readFull(fd: fd,
                              fileOffset: entry.scaleOffset,
                              destination: buffer.contents().advanced(by: scaleDestination),
                              count: scaleBytes)
        }
        if includeAffineCompanions, biasBytes > 0 {
            try Self.readFull(fd: fd,
                              fileOffset: entry.biasOffset,
                              destination: buffer.contents().advanced(by: biasDestination),
                              count: biasBytes)
        }
        return VisionLoadedTensor(
            descriptor: VisionTensorDescriptor(entry: entry),
            buffer: buffer,
            weightOffset: destinationOffset,
            scaleOffset: scaleBytes > 0 ? scaleDestination : nil,
            biasOffset: biasBytes > 0 ? biasDestination : nil,
            totalBytes: requiredEnd - destinationOffset)
    }

    func readTensorSlice(named name: String,
                         sourceByteOffset: Int,
                         count: Int,
                         into buffer: MTLBuffer,
                         at destinationOffset: Int = 0) throws {
        guard let entry = index.entries[name] else {
            throw VisionSidecarError.missingTensor(name)
        }
        guard sourceByteOffset >= 0,
              count >= 0,
              sourceByteOffset <= Int(entry.sizeBytes),
              count <= Int(entry.sizeBytes) - sourceByteOffset else {
            throw VisionSidecarError.invalidTensor(
                name: name,
                detail: "slice [\(sourceByteOffset), \(sourceByteOffset + count)) is out of range")
        }
        guard destinationOffset >= 0,
              destinationOffset <= buffer.length,
              count <= buffer.length - destinationOffset else {
            throw VisionSidecarError.allocationFailed(bytes: destinationOffset + count)
        }
        readLock.lock()
        defer { readLock.unlock() }
        try Self.readFull(
            fd: fd,
            fileOffset: entry.fileOffset + UInt64(sourceByteOffset),
            destination: buffer.contents().advanced(by: destinationOffset),
            count: count)
    }

    private static func validateManifest(_ manifest: VisionSidecarManifest) throws {
        guard manifest.magic == "GTURBO-VISION" else {
            throw VisionSidecarError.invalidManifest("magic must be GTURBO-VISION")
        }
        guard manifest.schemaVersion == 1 else {
            throw VisionSidecarError.invalidManifest("unsupported schemaVersion \(manifest.schemaVersion)")
        }
        guard manifest.kind == "gemma4-vision-sidecar" else {
            throw VisionSidecarError.invalidManifest("unsupported kind \(manifest.kind)")
        }
        guard manifest.sourceRevision.count == 40,
              manifest.sourceRevision.allSatisfy({ $0.isHexDigit }) else {
            throw VisionSidecarError.invalidManifest("sourceRevision must be a 40-character commit hash")
        }
        guard manifest.sourceSnapshotHash.hasPrefix("sha256:"),
              manifest.sourceSnapshotHash.count == 71 else {
            throw VisionSidecarError.invalidManifest("sourceSnapshotHash must be sha256:<64 hex>")
        }
        guard manifest.entryCount > 0, manifest.sourceTensorCount >= manifest.entryCount else {
            throw VisionSidecarError.invalidManifest("invalid tensor counts")
        }
        guard let weights = manifest.files[weightsName],
              weights.size > 0,
              weights.sha256.count == 64,
              weights.sha256.allSatisfy({ $0.isHexDigit }) else {
            throw VisionSidecarError.invalidManifest("invalid files.weights.bin metadata")
        }
    }

    private static func readBoundedMetadata(_ url: URL, maximumBytes: UInt64) throws -> Data {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attributes[.size] as? NSNumber,
              number.uint64Value <= maximumBytes else {
            throw VisionSidecarError.invalidManifest("metadata exceeds \(maximumBytes) bytes")
        }
        return try Data(contentsOf: url, options: [.mappedIfSafe])
    }

    private static func readFull(fd: Int32,
                                 fileOffset: UInt64,
                                 destination: UnsafeMutableRawPointer,
                                 count: Int) throws {
        var completed = 0
        while completed < count {
            let got = pread(fd,
                            destination.advanced(by: completed),
                            count - completed,
                            off_t(fileOffset + UInt64(completed)))
            if got < 0 {
                if errno == EINTR { continue }
                throw VisionSidecarError.invalidWeights("pread failed with errno \(errno)")
            }
            guard got > 0 else {
                throw VisionSidecarError.invalidWeights(
                    "short read at offset \(fileOffset + UInt64(completed))")
            }
            completed += got
        }
    }

    static func align(_ value: Int, to alignment: Int) -> Int {
        precondition(alignment > 0 && alignment.nonzeroBitCount == 1)
        return (value + alignment - 1) & ~(alignment - 1)
    }
}

public struct VisionTensorDescriptor: Sendable, Equatable {
    public let name: String
    public let dtype: UInt8
    public let dimensions: [Int]
    public let sizeBytes: Int
    public let scaleSizeBytes: Int
    public let biasSizeBytes: Int

    fileprivate init(entry: ResidentIndexEntry) {
        self.name = entry.name
        self.dtype = entry.dtype
        self.dimensions = [entry.shape.0, entry.shape.1, entry.shape.2, entry.shape.3]
            .map(Int.init)
            .filter { $0 > 0 }
        self.sizeBytes = Int(entry.sizeBytes)
        self.scaleSizeBytes = Int(entry.scaleSize)
        self.biasSizeBytes = Int(entry.biasSize)
    }
}

struct VisionLoadedTensor {
    let descriptor: VisionTensorDescriptor
    let buffer: MTLBuffer
    let weightOffset: Int
    let scaleOffset: Int?
    let biasOffset: Int?
    let totalBytes: Int
}
