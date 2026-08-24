import Darwin
import Foundation
import TurboFieldfareFormat

public struct VisionPackWriteOptions: Sendable {
    public let inputDirectory: String
    public let textModelDirectory: String
    public let outputDirectory: String
    public let sourceRevision: String
    public let overwrite: Bool

    public init(inputDirectory: String, textModelDirectory: String,
                outputDirectory: String, sourceRevision: String,
                overwrite: Bool = false) {
        self.inputDirectory = inputDirectory
        self.textModelDirectory = textModelDirectory
        self.outputDirectory = outputDirectory
        self.sourceRevision = sourceRevision
        self.overwrite = overwrite
    }
}

public struct VisionPackWriteResult: Sendable {
    public let tensorCount: Int
    public let sourcePayloadBytes: UInt64
    public let weightsFileSize: UInt64
    public let manifestSha256: String
    public let peakRSSBytes: UInt64
    public let largestScratchBytes: Int
}

public enum VisionPackWriter {
    public static func write(
        options: VisionPackWriteOptions,
        audit: RepackAudit = RepackAudit()
    ) throws -> VisionPackWriteResult {
        let meta = try IndexLoader.load(snapshotDir: options.inputDirectory)
        guard let modelID = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex),
              modelID == "mlx-community/gemma-4-26b-a4b-it-4bit" else {
            throw RepackError.sourceFingerprintRejected(
                path: meta.indexPath, sha256: meta.indexSha256Hex)
        }

        var headers: [Safetensors.Header] = []
        for filename in meta.shardFilenames {
            let path = (options.inputDirectory as NSString).appendingPathComponent(filename)
            headers.append(try Safetensors.parseHeader(path: path))
        }
        let plan = try RepackPlanner.planVisionCompanion(meta: meta, shardHeaders: headers)

        let textManifestPath = (options.textModelDirectory as NSString)
            .appendingPathComponent(GTurboVisionFormatV1.manifestFile)
        let textManifestData = try Posix.readBoundedData(
            textManifestPath, maximumBytes: GTurboVisionFormatV1.metadataMaxBytes)
        let textManifest = try GTurboManifestCodec.decode(textManifestData)
        guard let textSnapshot = textManifest.sourceSnapshotHash,
              !textSnapshot.isEmpty else {
            throw RepackError.configurationInvalid(
                detail: "text manifest has no sourceSnapshotHash")
        }
        let textManifestSHA = Sha256Stream.hash(data: textManifestData)

        let processorSource = (options.inputDirectory as NSString)
            .appendingPathComponent(GTurboVisionFormatV1.processorFile)
        let processorData = try Posix.readBoundedData(
            processorSource,
            maximumBytes: GTurboVisionFormatV1.metadataMaxBytes,
            followCallerSymlink: true)
        let processorSHA = Sha256Stream.hash(data: processorData)

        let installLock = try InstallLock.acquire(
            outputDirectory: options.outputDirectory)
        defer { withExtendedLifetime(installLock) {} }
        let finalURL = URL(
            fileURLWithPath: installLock.paths.finalDirectory,
            isDirectory: true)
        let partial = installLock.paths.partialDirectory
        let fm = FileManager.default
        if fm.fileExists(atPath: partial) {
            guard options.overwrite else {
                throw RepackError.configurationInvalid(
                    detail: "output exists: \(partial); pass --overwrite")
            }
            try fm.removeItem(atPath: partial)
        }
        if fm.fileExists(atPath: finalURL.path), !options.overwrite {
            throw RepackError.configurationInvalid(
                detail: "output exists: \(finalURL.path); pass --overwrite")
        }
        try Posix.mkdirP(partial)
        do {
            let weightsPath = (partial as NSString)
                .appendingPathComponent(GTurboVisionFormatV1.weightsFile)
            let weightsFD = try Posix.openCreateRW(weightsPath)
            defer { close(weightsFD) }
            try Posix.ftruncate(weightsFD, path: weightsPath, size: plan.weightsFileSize)

            var mappings: [String: MmapHandle] = [:]
            for entry in plan.entries {
                let mapping: MmapHandle
                if let existing = mappings[entry.source.shardPath] {
                    mapping = existing
                } else {
                    mapping = try MmapHandle(path: entry.source.shardPath)
                    mappings[entry.source.shardPath] = mapping
                }
                try WriterCore.pwriteTensorRegion(
                    srcShard: mapping,
                    srcAbsoluteOffset: entry.source.absoluteOffset,
                    size: entry.source.sizeBytes,
                    dstFd: weightsFD,
                    dstPath: weightsPath,
                    dstOffset: entry.fileOffset,
                    audit: audit)
            }
            try Posix.fsync(weightsFD, path: weightsPath)
            mappings.removeAll()

            let processorPath = (partial as NSString)
                .appendingPathComponent(GTurboVisionFormatV1.processorFile)
            try Posix.atomicWrite(processorData, to: processorPath, durableIn: partial)

            let weightsSHA = try WriterCore.hashEntireFile(
                path: weightsPath, size: plan.weightsFileSize, audit: audit)
            let files = [
                GTurboVisionFormatV1.weightsFile: GTurboManifestFileV1(
                    size: plan.weightsFileSize, sha256: weightsSHA),
                GTurboVisionFormatV1.processorFile: GTurboManifestFileV1(
                    size: UInt64(processorData.count), sha256: processorSHA),
            ]
            let tensors = plan.entries.map { entry in
                GTurboVisionTensorRegionV1(
                    name: entry.source.name,
                    executionPosition: entry.executionPosition,
                    offset: entry.fileOffset,
                    size: entry.source.sizeBytes,
                    shape: entry.source.shape,
                    dtype: entry.source.dtype == .u32 ? .u32 : .bf16,
                    quantization: entry.quantSpec.map { spec in
                        GTurboVisionQuantizationV1(
                            weightBits: spec.bits, scheme: "affine",
                            scaleType: "bf16", biasType: "bf16",
                            groupSize: entry.groupSize)
                    })
            }
            let manifest = GTurboVisionManifestV1(
                modelID: modelID,
                sourceRevision: options.sourceRevision,
                sourceIndexSha256: meta.indexSha256Hex,
                processorConfigSha256: processorSHA,
                compatibleTextSourceSnapshotHash: textSnapshot,
                compatibleTextManifestSha256: textManifestSHA,
                files: files,
                tensors: tensors)
            let manifestData = try GTurboVisionManifestCodec.encode(manifest)
            let manifestPath = (partial as NSString)
                .appendingPathComponent(GTurboVisionFormatV1.manifestFile)
            try Posix.atomicWrite(manifestData, to: manifestPath, durableIn: partial)
            let manifestSHA = Sha256Stream.hash(data: manifestData)

            let receiptFiles = files.merging([
                GTurboVisionFormatV1.manifestFile: GTurboManifestFileV1(
                    size: UInt64(manifestData.count), sha256: manifestSHA)
            ]) { _, new in new }
            let receipt = GTurboVisionReceiptV1(
                manifestSha256: manifestSHA,
                companionDirectoryPath: finalURL.path,
                compatibleTextManifestSha256: textManifestSHA,
                sourceRepoID: modelID,
                sourceRevision: options.sourceRevision,
                verificationTimestamp: "source-revision:\(options.sourceRevision)",
                toolVersion: "TurboFieldfareRepack/vision-v1",
                files: receiptFiles)
            let receiptData = try GTurboVisionReceiptCodec.encode(receipt)
            let receiptPath = (partial as NSString)
                .appendingPathComponent(GTurboVisionFormatV1.receiptFile)
            try Posix.atomicWrite(receiptData, to: receiptPath, durableIn: partial)
            try Posix.fsyncDirectory(partial)
            let useLock = try VisionPackMutationLock.acquire(
                outputDirectory: finalURL.path)
            defer { withExtendedLifetime(useLock) {} }
            if try Posix.entryKind(finalURL.path) == .directory {
                try Posix.renameSwap(partial, finalURL.path)
                try Posix.fsyncDirectory(finalURL.deletingLastPathComponent().path)
                try fm.removeItem(atPath: partial)
            } else {
                try Posix.rename(from: partial, to: finalURL.path)
                try Posix.fsyncDirectory(finalURL.deletingLastPathComponent().path)
            }

            audit.sampleRSS()
            return VisionPackWriteResult(
                tensorCount: plan.entries.count,
                sourcePayloadBytes: plan.sourcePayloadBytes,
                weightsFileSize: plan.weightsFileSize,
                manifestSha256: manifestSHA,
                peakRSSBytes: audit.peakRssBytes,
                largestScratchBytes: audit.largestScratchBytes)
        } catch {
            throw error
        }
    }
}

private extension Sha256Stream {
    static func hash(data: Data) -> String {
        var hasher = Sha256Stream()
        data.withUnsafeBytes { hasher.update($0) }
        return hasher.finalizeHexString()
    }
}
