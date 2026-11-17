import Foundation
import Metal
import TinyTitanCore

// Segment-length, payload-append and restore for inference-state snapshots.
//
// Split out of `KVCacheManager.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. Members whose
// callers stay behind widened from `private` to internal.
extension KVCacheManager {

    func snapshotSegmentLengths(at snapshotPosition: Int) throws -> [Int] {
        guard snapshotPosition > 0, snapshotPosition <= maxContext else {
            throw InferenceStateSnapshotError.invalidPosition(snapshotPosition)
        }
        var lengths: [Int] = []
        lengths.reserveCapacity(config.numLayers * 2)
        // One sequence's worth: the snapshot is a prefix of a single slot and is
        // captured/restored at slot 0's region base (offset 0), which is what the
        // prompt cache keys on. A batched slot's payload is a later phase.
        for layer in 0..<config.numLayers where kinds[layer] != .linear {
            // A streamed layer's arena spans `maxContext`; its window is not
            // the snapshot's source.
            let storedTokens =
                streamsLayer(layer)
                ? min(snapshotPosition, maxContext)
                : min(snapshotPosition, capacityTokens[layer])
            let (length, overflow) = storedTokens.multipliedReportingOverflow(
                by: strides[layer])
            guard !overflow else { throw InferenceStateSnapshotError.integerOverflow }
            lengths.append(length)
            lengths.append(length)
        }
        return lengths
    }

    func appendSnapshotPayload(
        to payload: inout Data,
        segmentLengths: [Int]
    ) throws {
        let expected = try snapshotSegmentLengths(at: position)
        guard segmentLengths == expected else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        var segment = 0
        for layer in 0..<config.numLayers where kinds[layer] != .linear {
            let kLength = segmentLengths[segment]
            if streamsLayer(layer) {
                appendStreamedSnapshot(
                    layer: layer, length: kLength, to: &payload)
                segment += 2
                continue
            }
            payload.append(
                kBuffers[layer].contents().assumingMemoryBound(to: UInt8.self),
                count: kLength)
            segment += 1
            let vLength = segmentLengths[segment]
            payload.append(
                vBuffers[layer].contents().assumingMemoryBound(to: UInt8.self),
                count: vLength)
            segment += 1
        }
    }

    func restoreSnapshot(
        position snapshotPosition: Int,
        segmentLengths: [Int],
        bytes: UnsafeRawBufferPointer,
        offset: inout Int
    ) throws {
        // Grow to the snapshot's position *before* computing the expected
        // lengths, because those lengths are a function of capacity:
        // `snapshotSegmentLengths` records `min(position, capacity)`. The saving
        // runner had grown to hold its prefix; a fresh receiver's full-attention
        // layers start at `initialCapacityTokens` (8192), so any snapshot past
        // that produced a different set of lengths and was refused as
        // `invalidLayout`. The restore could therefore never succeed for a long
        // prefix — which is the case the disk tier exists for, and why the
        // feature appeared simply not to work.
        try reserve(tokens: snapshotPosition)
        let expected = try snapshotSegmentLengths(at: snapshotPosition)
        guard segmentLengths == expected else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        reset()
        var segment = 0
        for layer in 0..<config.numLayers where kinds[layer] != .linear {
            let kLength = segmentLengths[segment]
            if streamsLayer(layer) {
                guard offset >= 0, kLength >= 0,
                    offset <= bytes.count - 2 * kLength,
                    let source = bytes.baseAddress?.advanced(by: offset)
                else {
                    throw InferenceStateSnapshotError.invalidLayout
                }
                try restoreStreamedSnapshot(
                    layer: layer, source: source, length: kLength)
                offset += 2 * kLength
                segment += 2
                continue
            }
            try copySnapshotSegment(
                bytes: bytes,
                offset: &offset,
                length: kLength,
                destination: kBuffers[layer])
            segment += 1
            let vLength = segmentLengths[segment]
            try copySnapshotSegment(
                bytes: bytes,
                offset: &offset,
                length: vLength,
                destination: vBuffers[layer])
            segment += 1
        }
        positions[0] = snapshotPosition
    }

    private func copySnapshotSegment(
        bytes: UnsafeRawBufferPointer,
        offset: inout Int,
        length: Int,
        destination: MTLBuffer
    ) throws {
        guard length <= destination.length,
            offset >= 0,
            length >= 0,
            offset <= bytes.count - length,
            let source = bytes.baseAddress?.advanced(by: offset)
        else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        memcpy(destination.contents(), source, length)
        offset += length
    }
}
