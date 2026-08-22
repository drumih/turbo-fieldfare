import Darwin
import Foundation
import Metal
import TurboFieldfareFormat

public enum VisionPackError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalidTextModelPath(String)
    case packNotFound(String)
    case invalidMetadata(String)
    case incompatibleTextArtifact
    case invalidReceipt(String)
    case invalidDirectoryEntries(String)
    case fileSizeMismatch(file: String, expected: UInt64, actual: UInt64)
    case regionExceedsDevice(name: String, bytes: UInt64, maximum: UInt64)

    public var description: String {
        switch self {
        case let .invalidTextModelPath(path):
            "text model path is not a .gturbo directory: \(path)"
        case let .packNotFound(path):
            "no vision companion pack at \(path)"
        case let .invalidMetadata(detail):
            "vision companion metadata is invalid: \(detail)"
        case .incompatibleTextArtifact:
            "vision companion does not match the loaded text artifact"
        case let .invalidReceipt(detail):
            "vision companion receipt is invalid: \(detail)"
        case let .invalidDirectoryEntries(difference):
            "vision companion directory does not match the v1 contract: \(difference)"
        case let .fileSizeMismatch(file, expected, actual):
            "\(file) size \(actual) does not match manifest size \(expected)"
        case let .regionExceedsDevice(name, bytes, maximum):
            "vision tensor \(name) region \(bytes) exceeds Metal maximum \(maximum)"
        }
    }
}

public enum VisionPackLocation {
    public static func companionURL(forTextModel textModelURL: URL) throws -> URL {
        let standardized = textModelURL.standardizedFileURL
        let name = standardized.lastPathComponent
        guard name.hasSuffix(".gturbo"), name.count > ".gturbo".count else {
            throw VisionPackError.invalidTextModelPath(standardized.path)
        }
        let stem = name.dropLast(".gturbo".count)
        return standardized.deletingLastPathComponent()
            .appendingPathComponent("\(stem).vision.gturbo", isDirectory: true)
    }
}

final class VisionWeightStore {
    let directoryURL: URL
    let manifest: GTurboVisionManifestV1
    private let directory: GTurboModelDirectory
    private let weightsFD: Int32

    static func open(
        directoryURL: URL,
        compatibleTextSourceSnapshotHash: String,
        compatibleTextManifestSha256: String
    ) throws -> VisionWeightStore {
        // "Metadata is invalid" for a pack that was never installed sends the
        // reader looking for corruption. Absence is its own answer.
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: directoryURL.standardizedFileURL.path,
            isDirectory: &isDirectory), isDirectory.boolValue else {
            throw VisionPackError.packNotFound(directoryURL.standardizedFileURL.path)
        }
        let directory: GTurboModelDirectory
        do { directory = try GTurboModelDirectory(rootURL: directoryURL) }
        catch { throw VisionPackError.invalidMetadata("\(error)") }

        let manifestData: Data
        let receiptData: Data
        do {
            manifestData = try directory.readMetadata(
                GTurboVisionFormatV1.manifestFile,
                maxBytes: GTurboVisionFormatV1.metadataMaxBytes)
            receiptData = try directory.readMetadata(
                GTurboVisionFormatV1.receiptFile,
                maxBytes: GTurboVisionFormatV1.metadataMaxBytes)
        } catch {
            throw VisionPackError.invalidMetadata("\(error)")
        }

        let manifest: GTurboVisionManifestV1
        let receipt: GTurboVisionReceiptV1
        do {
            manifest = try GTurboVisionManifestCodec.decode(manifestData)
            receipt = try GTurboVisionReceiptCodec.decode(receiptData)
        } catch {
            throw VisionPackError.invalidMetadata("\(error)")
        }
        guard manifest.compatibleTextSourceSnapshotHash
                == compatibleTextSourceSnapshotHash,
              manifest.compatibleTextManifestSha256.lowercased()
                == compatibleTextManifestSha256.lowercased() else {
            throw VisionPackError.incompatibleTextArtifact
        }

        let manifestSHA = Sha256Verifier.hashData(manifestData)
        let receiptDirectory = canonicalVisionDirectoryPath(URL(
            fileURLWithPath: receipt.companionDirectoryPath,
            isDirectory: true))
        let openedDirectory = canonicalVisionDirectoryPath(directoryURL)
        guard receipt.manifestSha256.lowercased() == manifestSHA,
              receipt.compatibleTextManifestSha256.lowercased()
                == compatibleTextManifestSha256.lowercased(),
              receipt.sourceRepoID == manifest.modelID,
              receipt.sourceRevision == manifest.sourceRevision,
              receiptDirectory == openedDirectory else {
            throw VisionPackError.invalidReceipt("path or manifest binding mismatch")
        }
        let manifestEntry = receipt.files[GTurboVisionFormatV1.manifestFile]
        guard manifestEntry?.size == UInt64(manifestData.count),
              manifestEntry?.sha256.lowercased() == manifestSHA else {
            throw VisionPackError.invalidReceipt("manifest entry mismatch")
        }
        for (path, file) in manifest.files {
            guard let receiptFile = receipt.files[path],
                  receiptFile.size == file.size,
                  receiptFile.sha256.lowercased() == file.sha256.lowercased() else {
                throw VisionPackError.invalidReceipt("payload entry mismatch for \(path)")
            }
        }

        let expectedEntries = Set([
            GTurboVisionFormatV1.manifestFile,
            GTurboVisionFormatV1.weightsFile,
            GTurboVisionFormatV1.processorFile,
            GTurboVisionFormatV1.receiptFile,
        ])
        let actualEntries: Set<String>
        do { actualEntries = try directory.basenames() }
        catch { throw VisionPackError.invalidMetadata("\(error)") }
        // Ignoring what the OS puts there on its own, exactly as the installer's
        // verifier does. Requiring an exact match meant opening the pack folder
        // in Finder once left `.DS_Store` behind and every later load failed as
        // an invalid pack, reporting "image support is unavailable" for a pack
        // that installed and verified cleanly.
        let entries = actualEntries.filter { !GTurboVisionFormatV1.isSidecarEntry($0) }
        guard entries == expectedEntries else {
            throw VisionPackError.invalidDirectoryEntries(
                GTurboVisionFormatV1.entryDifference(
                    expected: expectedEntries, actual: entries))
        }

        for (path, file) in manifest.files {
            let actual: UInt64
            do { actual = try directory.fileSize(path) }
            catch { throw VisionPackError.invalidMetadata("\(error)") }
            guard actual == file.size else {
                throw VisionPackError.fileSizeMismatch(
                    file: path, expected: file.size, actual: actual)
            }
        }
        // Both sides of the old check came out of the same manifest, so it was
        // self-consistent by construction and could only catch a mis-built
        // pack. The processor config is small and is the one payload file whose
        // content changes behaviour, so its bytes are hashed here — the weights
        // are covered by the receipt and their size, as before.
        let processorPath = GTurboVisionFormatV1.processorFile
        guard let processorEntry = manifest.files[processorPath] else {
            throw VisionPackError.invalidMetadata("manifest has no \(processorPath)")
        }
        guard manifest.processorConfigSha256.lowercased()
                == processorEntry.sha256.lowercased() else {
            throw VisionPackError.invalidMetadata("processor hash binding mismatch")
        }
        let processorFD: Int32
        do { processorFD = try directory.openFile(processorPath) }
        catch { throw VisionPackError.invalidMetadata("\(error)") }
        defer { close(processorFD) }
        do {
            try Sha256Verifier.verifyFile(
                fileDescriptor: processorFD,
                named: processorPath,
                expectedHex: processorEntry.sha256)
        } catch {
            throw VisionPackError.invalidMetadata(
                "\(processorPath) does not match its manifest hash")
        }

        let weightsFD: Int32
        do { weightsFD = try directory.openFile(GTurboVisionFormatV1.weightsFile) }
        catch { throw VisionPackError.invalidMetadata("\(error)") }
        return VisionWeightStore(directoryURL: directoryURL.standardizedFileURL,
                                 directory: directory, manifest: manifest,
                                 weightsFD: weightsFD)
    }

    private init(directoryURL: URL, directory: GTurboModelDirectory,
                 manifest: GTurboVisionManifestV1, weightsFD: Int32) {
        self.directoryURL = directoryURL
        self.directory = directory
        self.manifest = manifest
        self.weightsFD = weightsFD
    }

    deinit { close(weightsFD) }

    func mapRegion(executionPosition: Int,
                           device: MTLDevice) throws -> ResidentBuffer {
        guard manifest.tensors.indices.contains(executionPosition) else {
            throw VisionPackError.invalidMetadata(
                "missing execution position \(executionPosition)")
        }
        let region = manifest.tensors[executionPosition]
        guard region.size <= UInt64(device.maxBufferLength) else {
            throw VisionPackError.regionExceedsDevice(
                name: region.name, bytes: region.size,
                maximum: UInt64(device.maxBufferLength))
        }
        return try ResidentBuffer(
            fileURL: directoryURL.appendingPathComponent(region.file),
            fileOffset: region.offset, residentSize: region.size,
            device: device, fileDescriptor: weightsFD)
    }

    func mapRegion(tensorNames: [String],
                           device: MTLDevice) throws -> VisionMappedWeightRegion {
        let byName = Dictionary(uniqueKeysWithValues: manifest.tensors.map { ($0.name, $0) })
        let tensors = try tensorNames.map { name -> GTurboVisionTensorRegionV1 in
            guard let tensor = byName[name] else {
                throw VisionPackError.invalidMetadata("missing tensor \(name)")
            }
            return tensor
        }
        guard let start = tensors.map(\.offset).min(),
              let end = tensors.map({ $0.offset + $0.size }).max(),
              end > start else {
            throw VisionPackError.invalidMetadata("empty mapped tensor group")
        }
        let length = end - start
        guard length <= UInt64(device.maxBufferLength) else {
            throw VisionPackError.regionExceedsDevice(
                name: tensorNames.first ?? "group", bytes: length,
                maximum: UInt64(device.maxBufferLength))
        }
        let resident = try ResidentBuffer(
            fileURL: directoryURL.appendingPathComponent(GTurboVisionFormatV1.weightsFile),
            fileOffset: start, residentSize: length,
            device: device, fileDescriptor: weightsFD)
        let offsets = Dictionary(uniqueKeysWithValues: tensors.map {
            ($0.name, Int($0.offset - start))
        })
        return VisionMappedWeightRegion(resident: resident,
                                        tensors: tensors,
                                        offsets: offsets)
    }

    func tensorNames(withPrefix prefix: String) -> [String] {
        manifest.tensors.lazy.filter { $0.name.hasPrefix(prefix) }.map(\.name)
    }
}

private func canonicalVisionDirectoryPath(_ url: URL) -> String {
    let standardized = url.standardizedFileURL
    return standardized.deletingLastPathComponent()
        .resolvingSymlinksInPath()
        .appendingPathComponent(standardized.lastPathComponent, isDirectory: true)
        .standardizedFileURL.path
}

struct VisionMappedWeightRegion {
    let resident: ResidentBuffer
    let tensors: [GTurboVisionTensorRegionV1]
    let offsets: [String: Int]

    var buffer: MTLBuffer { resident.buffer }

    func offset(of name: String) throws -> Int {
        guard let value = offsets[name] else {
            throw VisionPackError.invalidMetadata("tensor \(name) is outside mapped region")
        }
        return value
    }
}
