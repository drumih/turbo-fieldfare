import Foundation

struct RangeCopy: Sendable, Equatable {
    let shardID: String
    let sourceOffset: UInt64
    let size: UInt64
    let destinationPath: String
    let destinationOffset: UInt64

    init(shardID: String,
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

struct CoalescedRangeCopy: Sendable, Equatable {
    let shardID: String
    let sourceOffset: UInt64
    let size: UInt64
    let destinations: [RangeCopy]
}

struct RangeCopyPlan: Sendable {
    let coalescedCopies: [CoalescedRangeCopy]
    let remoteBytesToDownload: UInt64
}

enum RangeCopyPlanner {
    static func plan(repackPlan: RepackPlan,
                            rangeChunkBytes: Int) throws -> RangeCopyPlan {
        var copies: [RangeCopy] = []
        copies.reserveCapacity(repackPlan.resident.entries.count * 3)

        for entry in repackPlan.resident.entries {
            copies.append(RangeCopy(shardID: entry.sourceWeight.shardPath,
                                    sourceOffset: entry.sourceWeight.absoluteOffset,
                                    size: entry.sizeBytes,
                                    destinationPath: repackPlan.resident.path,
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
                let blobBase = UInt64(expert) * layer.expertStride
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

        let coalesced = try coalesce(copies: copies, rangeChunkBytes: rangeChunkBytes)
        let downloaded = coalesced.reduce(UInt64(0)) { $0 + $1.size }
        return RangeCopyPlan(coalescedCopies: coalesced,
                             remoteBytesToDownload: downloaded)
    }

    static func coalesce(copies: [RangeCopy],
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
            out.append(CoalescedRangeCopy(shardID: shard,
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
        return out
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
}
