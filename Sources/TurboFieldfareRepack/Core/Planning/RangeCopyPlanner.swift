import Foundation
import TurboFieldfareFormat

public struct RangeCopy: Sendable, Equatable {
    public let shardID: String
    public let sourceOffset: UInt64
    public let size: UInt64
    public let destinationPath: String
    public let destinationOffset: UInt64

    public init(shardID: String,
                sourceOffset: UInt64,
                size: UInt64,
                destinationPath: String,
                destinationOffset: UInt64) {
        self.shardID = shardID
        self.sourceOffset = sourceOffset
        self.size = size
        self.destinationPath = destinationPath
        self.destinationOffset = destinationOffset
    }
}

public struct CoalescedRangeCopy: Sendable, Equatable {
    public let id: String
    public let shardID: String
    public let sourceOffset: UInt64
    public let size: UInt64
    public let destinations: [RangeCopy]
}

public struct RemoteExpectedOutput: Codable, Sendable, Equatable {
    public let relativePath: String
    public let size: UInt64
}

public struct RangeCopyPlan: Sendable {
    public let scalarCopies: [RangeCopy]
    public let coalescedCopies: [CoalescedRangeCopy]
    public let remoteBytesToDownload: UInt64
    public let remoteGapBytesDownloaded: UInt64
    public let canonicalFingerprint: String
    public let residentIndexSha256: String
    public let expectedOutputs: [RemoteExpectedOutput]
}

public enum RangeCopyPlanner {
    static func plan(
        visionPackPlan: VisionPackPlan,
        outputDirectory: String,
        rangeChunkBytes: Int,
        textManifestSha256: String
    ) throws -> RangeCopyPlan {
        let weightsPath = (outputDirectory as NSString)
            .appendingPathComponent(GTurboVisionFormatV1.weightsFile)
        let copies = visionPackPlan.entries.map { entry in
            RangeCopy(
                shardID: entry.source.shardPath,
                sourceOffset: entry.source.absoluteOffset,
                size: entry.source.sizeBytes,
                destinationPath: weightsPath,
                destinationOffset: entry.fileOffset)
        }
        try validateDestinationIntervals(copies, outputRoot: outputDirectory)
        let coalesced = try coalesce(copies: copies, rangeChunkBytes: rangeChunkBytes)
        let expectedOutputs = [RemoteExpectedOutput(
            relativePath: GTurboVisionFormatV1.weightsFile,
            size: visionPackPlan.weightsFileSize)]
        // A separate domain so a vision plan can never share a fingerprint with
        // a text plan that happens to coalesce to the same ranges.
        let fingerprint = try canonicalFingerprint(
            domain: "TurboFieldfare.RemoteVisionInstallPlan.v1",
            copies: coalesced,
            outputRoot: outputDirectory,
            rangeChunkBytes: rangeChunkBytes,
            layoutMode: "vision-execution-order",
            layoutOrderSha256: textManifestSha256,
            residentIndexSha256: "",
            expectedOutputs: expectedOutputs)
        let downloaded = coalesced.reduce(UInt64(0)) { $0 + $1.size }
        return RangeCopyPlan(
            scalarCopies: copies,
            coalescedCopies: coalesced,
            remoteBytesToDownload: downloaded,
            remoteGapBytesDownloaded: downloaded - visionPackPlan.sourcePayloadBytes,
            canonicalFingerprint: fingerprint,
            residentIndexSha256: "",
            expectedOutputs: expectedOutputs)
    }

    static func plan(repackPlan: RepackPlan,
                     rangeChunkBytes: Int,
                     layoutMode: String = "identity",
                     layoutOrderSha256: String? = nil) throws -> RangeCopyPlan {
        var copies: [RangeCopy] = []
        copies.reserveCapacity(repackPlan.resident.entries.count * 3)

        for entry in repackPlan.resident.entries {
            copies.append(RangeCopy(shardID: entry.sourceWeight.shardPath,
                                    sourceOffset: entry.sourceWeight.absoluteOffset,
                                    size: entry.sizeBytes,
                                    destinationPath: entry.fileOffsetPath(in: repackPlan.resident),
                                    destinationOffset: entry.fileOffset))
            if let scales = entry.sourceScales {
                copies.append(RangeCopy(shardID: scales.shardPath,
                                        sourceOffset: scales.absoluteOffset,
                                        size: entry.scaleSize,
                                        destinationPath: repackPlan.resident.path,
                                        destinationOffset: entry.scaleOffset))
            }
            if let biases = entry.sourceBiases {
                copies.append(RangeCopy(shardID: biases.shardPath,
                                        sourceOffset: biases.absoluteOffset,
                                        size: entry.biasSize,
                                        destinationPath: repackPlan.resident.path,
                                        destinationOffset: entry.biasOffset))
            }
        }

        for layer in repackPlan.layers where layer.expertsPerLayer > 0 {
            for expert in 0..<layer.expertsPerLayer {
                let blobBase = UInt64(layer.physicalRank(for: expert)) * layer.expertStride
                for slice in layer.subTensors {
                    copies.append(RangeCopy(
                        shardID: slice.sourceTensor.shardPath,
                        sourceOffset: slice.sourceTensor.absoluteOffset
                            + UInt64(expert) * slice.sourceOffsetPerExpert,
                        size: slice.sizeInExpertBlob,
                        destinationPath: layer.path,
                        destinationOffset: blobBase + slice.offsetInExpertBlob))
                }
            }
        }

        try validateDestinationIntervals(copies, outputRoot: outputRoot(for: repackPlan))
        let coalesced = try coalesce(copies: copies, rangeChunkBytes: rangeChunkBytes)
        let indexData = try ResidentWriter.encodeIndex(plan: repackPlan.resident)
        let indexSha = hashData(indexData)
        let expectedOutputs = expectedOutputList(for: repackPlan)
        let fingerprint = try canonicalFingerprint(
            copies: coalesced,
            outputRoot: outputRoot(for: repackPlan),
            rangeChunkBytes: rangeChunkBytes,
            layoutMode: layoutMode,
            layoutOrderSha256: layoutOrderSha256,
            residentIndexSha256: indexSha,
            expectedOutputs: expectedOutputs)
        let downloaded = coalesced.reduce(UInt64(0)) { $0 + $1.size }
        let copied = copies.reduce(UInt64(0)) { $0 + $1.size }
        return RangeCopyPlan(scalarCopies: copies,
                             coalescedCopies: coalesced,
                             remoteBytesToDownload: downloaded,
                             remoteGapBytesDownloaded: downloaded - copied,
                             canonicalFingerprint: fingerprint,
                             residentIndexSha256: indexSha,
                             expectedOutputs: expectedOutputs)
    }

    public static func coalesce(copies: [RangeCopy],
                                rangeChunkBytes: Int) throws -> [CoalescedRangeCopy] {
        guard rangeChunkBytes > 0 else {
            throw RepackError.configurationInvalid(detail: "rangeChunkBytes must be positive")
        }
        let sorted = splitLargeCopies(copies, rangeChunkBytes: rangeChunkBytes).sorted {
            if $0.shardID != $1.shardID { return $0.shardID < $1.shardID }
            return $0.sourceOffset < $1.sourceOffset
        }
        var out: [CoalescedRangeCopy] = []
        var currentShard: String?
        var currentStart: UInt64 = 0
        var currentEnd: UInt64 = 0
        var currentDestinations: [RangeCopy] = []

        func flush() {
            guard let shard = currentShard else { return }
            out.append(CoalescedRangeCopy(id: "",
                                          shardID: shard,
                                          sourceOffset: currentStart,
                                          size: currentEnd - currentStart,
                                          destinations: currentDestinations))
        }

        for copy in sorted where copy.size > 0 {
            let copyEnd = copy.sourceOffset + copy.size
            if currentShard == nil {
                currentShard = copy.shardID
                currentStart = copy.sourceOffset
                currentEnd = copyEnd
                currentDestinations = [copy]
                continue
            }
            let proposedStart = currentStart
            let proposedEnd = max(currentEnd, copyEnd)
            let canMerge = currentShard == copy.shardID
                && proposedEnd >= proposedStart
                && proposedEnd - proposedStart <= UInt64(rangeChunkBytes)
            if canMerge {
                currentEnd = proposedEnd
                currentDestinations.append(copy)
            } else {
                flush()
                currentShard = copy.shardID
                currentStart = copy.sourceOffset
                currentEnd = copyEnd
                currentDestinations = [copy]
            }
        }
        flush()
        return out.enumerated().map { index, copy in
            CoalescedRangeCopy(
                id: String(format: "range-%08d", index),
                shardID: copy.shardID,
                sourceOffset: copy.sourceOffset,
                size: copy.size,
                destinations: copy.destinations.sorted(by: destinationOrder))
        }
    }

    private static func splitLargeCopies(_ copies: [RangeCopy],
                                         rangeChunkBytes: Int) -> [RangeCopy] {
        let limit = UInt64(rangeChunkBytes)
        var out: [RangeCopy] = []
        for copy in copies {
            var remaining = copy.size
            var src = copy.sourceOffset
            var dst = copy.destinationOffset
            while remaining > 0 {
                let n = min(remaining, limit)
                out.append(RangeCopy(shardID: copy.shardID,
                                     sourceOffset: src,
                                     size: n,
                                     destinationPath: copy.destinationPath,
                                     destinationOffset: dst))
                remaining -= n
                src += n
                dst += n
            }
        }
        return out
    }

    private static func outputRoot(for plan: RepackPlan) -> String {
        (plan.resident.path as NSString).deletingLastPathComponent
    }

    private static func expectedOutputList(for plan: RepackPlan) -> [RemoteExpectedOutput] {
        var outputs = [
            RemoteExpectedOutput(relativePath: "model_weights.bin",
                                 size: plan.resident.totalSize)
        ]
        outputs.append(contentsOf: plan.layers
            .filter { $0.expertsPerLayer > 0 }
            .map {
                RemoteExpectedOutput(
                    relativePath: "packed_experts/" + ($0.path as NSString).lastPathComponent,
                    size: $0.fileSize)
            })
        return outputs.sorted { $0.relativePath < $1.relativePath }
    }

    static func validateDestinationIntervals(_ copies: [RangeCopy],
                                             outputRoot: String) throws {
        let sorted = try copies.map { copy in
            (try normalizedRelativePath(copy.destinationPath, root: outputRoot), copy)
        }.sorted {
            if $0.0 != $1.0 { return $0.0 < $1.0 }
            return $0.1.destinationOffset < $1.1.destinationOffset
        }
        var previousPath: String?
        var previousEnd: UInt64 = 0
        for (path, copy) in sorted {
            guard copy.destinationOffset <= UInt64.max - copy.size else {
                throw RepackError.configurationInvalid(
                    detail: "destination range overflows \(path)")
            }
            if previousPath == path, copy.destinationOffset < previousEnd {
                throw RepackError.configurationInvalid(
                    detail: "overlapping destination ranges in \(path)")
            }
            previousPath = path
            previousEnd = copy.destinationOffset + copy.size
        }
    }

    static func canonicalFingerprint(
        domain: String = "TurboFieldfare.RemoteInstallPlan.v1",
        copies: [CoalescedRangeCopy],
        outputRoot: String,
        rangeChunkBytes: Int,
        layoutMode: String,
        layoutOrderSha256: String?,
        residentIndexSha256: String,
        expectedOutputs: [RemoteExpectedOutput]
    ) throws -> String {
        var writer = FingerprintWriter(domain: domain)
        writer.append(UInt64(GTurboJSON.versionMajor))
        writer.append(UInt64(GTurboJSON.versionMinor))
        writer.append(UInt64(rangeChunkBytes))
        writer.append(layoutMode)
        writer.append(layoutOrderSha256 ?? "")
        writer.append(residentIndexSha256)
        writer.append(UInt64(expectedOutputs.count))
        for output in expectedOutputs {
            writer.append(output.relativePath)
            writer.append(output.size)
        }
        writer.append(UInt64(copies.count))
        for copy in copies {
            writer.append(copy.id)
            writer.append(copy.shardID.precomposedStringWithCanonicalMapping)
            writer.append(copy.sourceOffset)
            writer.append(copy.size)
            writer.append(UInt64(copy.destinations.count))
            for destination in copy.destinations {
                writer.append(try normalizedRelativePath(
                    destination.destinationPath,
                    root: outputRoot))
                writer.append(destination.destinationOffset)
                writer.append(destination.sourceOffset - copy.sourceOffset)
                writer.append(destination.size)
            }
        }
        return writer.finalize()
    }

    static func normalizedRelativePath(_ path: String, root: String) throws -> String {
        let normalizedRoot = root.hasSuffix("/") ? String(root.dropLast()) : root
        let normalizedPath = path
        let prefix = normalizedRoot.hasSuffix("/") ? normalizedRoot : normalizedRoot + "/"
        guard normalizedPath.hasPrefix(prefix) else {
            throw RepackError.configurationInvalid(
                detail: "destination \(normalizedPath) is outside partial output \(normalizedRoot)")
        }
        let relative = String(normalizedPath.dropFirst(prefix.count))
            .precomposedStringWithCanonicalMapping
        let components = relative.split(separator: "/", omittingEmptySubsequences: false)
        guard !relative.isEmpty,
              !relative.hasPrefix("/"),
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw RepackError.configurationInvalid(
                detail: "invalid relative destination path \(relative)")
        }
        return relative
    }

    private static func destinationOrder(_ lhs: RangeCopy, _ rhs: RangeCopy) -> Bool {
        if lhs.destinationPath != rhs.destinationPath {
            return lhs.destinationPath < rhs.destinationPath
        }
        if lhs.destinationOffset != rhs.destinationOffset {
            return lhs.destinationOffset < rhs.destinationOffset
        }
        return lhs.sourceOffset < rhs.sourceOffset
    }
}

private extension ResidentEntry {
    func fileOffsetPath(in plan: ResidentFilePlan) -> String {
        plan.path
    }
}

private struct FingerprintWriter {
    private var stream = Sha256Stream()

    init(domain: String) {
        append(domain)
    }

    mutating func append(_ value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { stream.update($0) }
    }

    mutating func append(_ value: String) {
        let data = Data(value.utf8)
        append(UInt64(data.count))
        data.withUnsafeBytes { stream.update($0) }
    }

    func finalize() -> String {
        stream.finalizeHexString()
    }
}

private func hashData(_ data: Data) -> String {
    var stream = Sha256Stream()
    data.withUnsafeBytes { stream.update($0) }
    return stream.finalizeHexString()
}
