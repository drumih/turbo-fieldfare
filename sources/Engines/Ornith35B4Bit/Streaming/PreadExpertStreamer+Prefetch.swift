import Foundation
import TinyTitanFormat
import Metal
import Synchronization
import TinyTitanCore

// Speculative expert reads: the ring, and the compressed arm that decodes into
// its destinations.
//
// Split out of `PreadExpertStreamer+Residency.swift` under the 500-line-per-file
// rule when compressed prefetch landed. Pure code motion plus the compressed
// branch, which decodes into the ring buffer so adoption stays the `memcpy` the
// raw path uses.
/// Ring buffers handed to the scheduler.
/// unchecked-invariant: the buffers are retained by the ring until the operation
/// completes, and each is written by exactly one speculative decode.
final class RANSPrefetchBuffers: @unchecked Sendable {
    let values: [MTLBuffer]

    init(_ values: [MTLBuffer]) {
        self.values = values
    }
}

extension PreadExpertStreamer {
    /// A rANS layer's speculation decodes on the GPU into the ring's buffers,
    /// one layer ahead, so adoption replaces a demand decode.
    public func beginPrefetch(experts: [Int], buffers: [MTLBuffer]) throws
        -> ExpertLoadOperation
    {
        guard let resources = ransResources else {
            return try beginPrefetch(
                experts: experts, destinations: buffers.map { $0.contents() })
        }
        guard experts.count == buffers.count else {
            throw ModelError.internalInconsistency(
                detail: "prefetch experts and buffers differ in count")
        }
        let operation = ExpertLoadOperation()
        let safeBuffers = RANSPrefetchBuffers(buffers)
        ExpertIOScheduler.shared.submit(priority: .speculative) {
            [self, operation, safeBuffers] in
            operation.markInFlight()
            do {
                try decodeRANSInto(
                    experts: experts, outputs: safeBuffers.values,
                    offsets: [UInt64](repeating: 0, count: experts.count),
                    resources: resources, firstStage: nil)
                operation.finish(.success(()))
            } catch {
                operation.finish(.failure(error))
            }
        }
        return operation
    }

    /// Starts a bounded, raw speculative read. The destination buffers are not
    /// cache slots, so an incorrect prediction cannot evict an authoritative
    /// expert. Demand work is always scheduled at higher priority.
    public func beginPrefetch(
        experts: [Int],
        destinations: [UnsafeMutableRawPointer]
    ) throws
        -> ExpertLoadOperation
    {
        guard experts.count == destinations.count else {
            throw ModelError.internalInconsistency(
                detail: "prefetch experts and destinations differ in count")
        }
        // Diagnostic kill switch: a finished-failure operation is never
        // adopted, so the ring falls back to demand reads.
        if ProcessInfo.processInfo.environment["TINYTITAN_NO_PREFETCH"] == "1" {
            let operation = ExpertLoadOperation()
            operation.markInFlight()
            operation.finish(
                .failure(
                    ModelError.internalInconsistency(
                        detail: "prefetch disabled by TINYTITAN_NO_PREFETCH")))
            return operation
        }
        // A rANS layer's payload is not at raw offsets, and the ring stages
        // raw pointers rather than slot buffers, so speculation has no decode
        // path here. Failing the operation keeps its slots from being adopted:
        // a completed-but-unwritten buffer would copy garbage into a slot,
        // which is exactly the corruption this guard prevents.
        if ransResources != nil {
            let operation = ExpertLoadOperation()
            operation.markInFlight()
            operation.finish(
                .failure(
                    ModelError.internalInconsistency(
                        detail: "rANS prefetch is not implemented")))
            return operation
        }
        // A compressed layer prefetches by decoding into the destination
        // buffers, so adoption stays the same memcpy the raw path uses. The
        // decode happens on this scheduler thread, off the decode critical path,
        // which is where the cost has to be if it is to be free -- and it runs
        // in parallel across the predicted experts. Serially, one top-k set is
        // ~15 MB at ~0.9 GB/s = ~17 ms, longer than the ~6 ms a layer takes, so
        // the ring never finished in time and adoption measured 0.06 per token
        // against the raw path's 21.75.
        if let frames = compressedFrames {
            let operation = ExpertLoadOperation()
            // The same wrapper the raw path uses: raw pointers are not `Sendable`
            // and the scheduler's closure is.
            let safeDestinations = PrefetchDestinations(destinations)
            ExpertIOScheduler.shared.submit(
                priority: .speculative
            ) { [self, operation, safeDestinations] in
                operation.markInFlight()
                do {
                    try decodePrefetchedExperts(
                        experts, frames: frames, into: safeDestinations.values)
                    operation.finish(.success(()))
                } catch {
                    operation.finish(.failure(error))
                }
            }
            return operation
        }
        // A tiered layer's cold experts are shorter than the stride. The
        // prefetch must read their real extent and expand them, or the slot
        // holds 2-bit bytes that the 4-bit kernels then read as garbage.
        let plan = try experts.map { expert -> (expert: Int, offset: UInt64, size: Int) in
            guard expert >= 0 && expert < layout.expertsPerLayer else {
                throw ModelError.internalInconsistency(detail: "invalid prefetched expert")
            }
            let size =
                layout.expertSizes.flatMap {
                    $0.indices.contains(expert) ? Int($0[expert]) : nil
                } ?? Int(layout.expertStride)
            return (
                expert,
                layout.streamOffset + layout.expertOffset(layer: 0, expert: expert),
                size
            )
        }
        let safeDestinations = PrefetchDestinations(destinations)
        let operation = ExpertLoadOperation()
        ExpertIOScheduler.shared.submit(priority: .speculative) {
            [self, operation, safeDestinations] in
            operation.markInFlight()
            do {
                if let boundedReader {
                    try boundedReader.fetch(
                        offsets: plan.map(\.offset),
                        byteCounts: plan.map(\.size),
                        into: safeDestinations.values)
                } else {
                    for (entry, destination) in zip(plan, safeDestinations.values) {
                        try readFull(
                            into: destination, fileOffset: entry.offset,
                            count: entry.size)
                    }
                }
                if let maps = layout.expertTensors {
                    for (index, entry) in plan.enumerated() {
                        guard maps.indices.contains(entry.expert),
                            !maps[entry.expert].isEmpty
                        else { continue }
                        if ProcessInfo.processInfo.environment["TINYTITAN_EXPAND_TRACE"] == "1" {
                            FileHandle.standardError.write(
                                Data(
                                    "[expand] site=prefetch layer=\((layout.path as NSString).lastPathComponent) expert=\(entry.expert)\n"
                                        .utf8))
                        }
                        TwoBitExpand.expandExpertInPlace(
                            slot: safeDestinations.values[index],
                            tensors: maps[entry.expert])
                    }
                }
                operation.finish(.success(()))
            } catch {
                operation.finish(.failure(error))
            }
        }
        return operation
    }

    /// Decode a predicted expert set into the ring's buffers.
    ///
    /// One storage call for the whole set, then every block decodes in
    /// parallel: this is the same path the demand wave uses, and it matters for
    /// the same reason -- the ring's reads are issued from a scheduler thread
    /// and a per-expert fetch would queue behind the reader's batch lock.
    private func decodePrefetchedExperts(
        _ experts: [Int],
        frames: [[ExpertBlobIndex.Blob]],
        into destinations: [UnsafeMutableRawPointer]
    ) throws {
        try decodeCompressedSet(
            experts: experts, frames: frames, into: destinations,
            reader: prefetchReader)
    }

}
