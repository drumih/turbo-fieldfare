import Darwin
import Foundation
import Synchronization
import TinyTitanFormat
import Metal
import TinyTitanCore

// The Metal, bounded and cached read paths.
//
// Split out of `PreadExpertStreamer.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// Members whose callers stay behind widened from `private` to internal.
extension PreadExpertStreamer {

    func beginEventDrivenMetalReads(
        _ plan: ExpertCachePlan,
        reader: MetalExpertReader,
        token: ExpertIOCompletionToken?
    ) throws -> ExpertLoadOperation {
        guard let token,
            let stagingLease = metalStagingPool?.tryAcquire(count: plan.misses.count)
        else {
            throw ModelError.internalInconsistency(
                detail: "event-driven Metal I/O staging ring is unavailable")
        }
        let transfer = try makeMetalStagingTransfer(plan: plan, stagingLease: stagingLease)
        let operation = ExpertLoadOperation(
            completionToken: token,
            eventCoordinator: eventCoordinator,
            // MTLIO writes the status word and signals the event in command
            // order. Its handler records terminal state but never wakes the
            // decode task to encode a fixup.
            backendSignalsEvent: true,
            metalStagingTransfer: transfer,
            requiresGPUFinalization: true)
        operation.markInFlight()
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        do {
            try beginMetalReads(
                plan,
                reader: reader,
                destinations: stagingLease.buffers,
                destinationOffsets: [Int](repeating: 0, count: stagingLease.buffers.count),
                completionToken: token
            ) { [self, operation] result in
                switch result {
                case .success:
                    // Cache slots remain LOADING. The runner publishes
                    // RESIDENT only after its event-gated blit completes.
                    finishPlanExecution(
                        plan,
                        succeeded: true,
                        elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                    operation.finish(.success(()))
                case .failure(let error):
                    finishPlanExecution(
                        plan,
                        succeeded: false,
                        elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                    operation.finish(.failure(error))
                }
            }
        } catch {
            finishPlanExecution(plan, succeeded: false, elapsedNanos: 0)
            operation.releaseStagingTransfer()
            operation.finish(.failure(error))
        }
        return operation
    }

    func executeMetalReads(
        _ plan: ExpertCachePlan,
        reader: MetalExpertReader
    ) throws {
        var offsets: [UInt64] = []
        var destinations: [MTLBuffer] = []
        offsets.reserveCapacity(plan.misses.count)
        destinations.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            offsets.append(try fileOffset(plan: plan, index: index))
            destinations.append(slotBuffers[plan.assignedSlots[index]])
        }
        try reader.fetch(
            offsets: offsets,
            into: destinations,
            byteCount: Int(layout.expertStride),
            destinationOffsets: plan.misses.map {
                Int(slotBufferOffsets[plan.assignedSlots[$0]])
            })
    }

    func beginMetalReads(
        _ plan: ExpertCachePlan,
        reader: MetalExpertReader,
        destinations explicitDestinations: [MTLBuffer]? = nil,
        destinationOffsets explicitDestinationOffsets: [Int]? = nil,
        completionToken: ExpertIOCompletionToken?,
        completion: @escaping @Sendable (Result<Void, any Error>) -> Void
    ) throws {
        var offsets: [UInt64] = []
        var destinations: [MTLBuffer] = []
        offsets.reserveCapacity(plan.misses.count)
        destinations.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            offsets.append(try fileOffset(plan: plan, index: index))
            destinations.append(slotBuffers[plan.assignedSlots[index]])
        }
        let finalDestinations = explicitDestinations ?? destinations
        let finalDestinationOffsets =
            explicitDestinationOffsets
            ?? plan.misses.map {
                Int(slotBufferOffsets[plan.assignedSlots[$0]])
            }
        try reader.beginFetch(
            offsets: offsets,
            into: finalDestinations,
            byteCount: Int(layout.expertStride),
            destinationOffsets: finalDestinationOffsets,
            completionToken: completionToken,
            completion: completion)
    }

    func makeMetalStagingTransfer(
        plan: ExpertCachePlan,
        stagingLease: MetalExpertStagingLease
    ) throws -> MetalExpertStagingTransfer {
        var destinations: [MTLBuffer] = []
        var destinationOffsets: [Int] = []
        destinations.reserveCapacity(plan.misses.count)
        destinationOffsets.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            guard slot >= 0, slot < slotBuffers.count else {
                stagingLease.release()
                throw ModelError.internalInconsistency(
                    detail: "Metal I/O staging transfer references an invalid cache slot")
            }
            destinations.append(slotBuffers[slot])
            destinationOffsets.append(Int(slotBufferOffsets[slot]))
        }
        return MetalExpertStagingTransfer(
            lease: stagingLease,
            destinations: destinations,
            destinationOffsets: destinationOffsets,
            byteCount: Int(layout.expertStride))
    }

    func executeBoundedReads(
        _ plan: ExpertCachePlan,
        reader: ParallelExpertReader,
        firstStage: (() -> Void)? = nil
    ) throws {
        var offsets: [UInt64] = []
        var destinations: [UnsafeMutableRawPointer] = []
        offsets.reserveCapacity(plan.misses.count)
        destinations.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            offsets.append(try fileOffset(plan: plan, index: index))
            destinations.append(slotPointers[plan.assignedSlots[index]])
        }
        // Split blob reads: the blob is gate|up|down contiguous, so the front
        // range is everything phase 1 consumes and the tail is what phase 2
        // adds. Residency still publishes only after the tail lands, and the
        // assembled slot is the same bytes a whole-region fetch writes (pinned
        // by splitMissesResidentWithIdenticalBytes). The Metal staging path
        // keeps its whole-region contract -- this split exists for the bounded
        // pread backend, which is the default one.
        // A tiered layer stores cold experts shorter than the slot capacity, so
        // the split-read and whole-stride paths below would read past an
        // expert's payload. When sizes vary, read exactly each expert's bytes
        // through the range fetch and skip the phase-1 split (its split point
        // is per-width, not per-layer).
        let sizesAreUniform = layout.hasUniformExpertSizes
        if !sizesAreUniform, let sizes = layout.expertSizes {
            let counts = plan.misses.map { miss -> Int in
                let expert = plan.experts[miss]
                return sizes.indices.contains(expert)
                    ? Int(sizes[expert]) : Int(layout.expertStride)
            }
            try reader.fetch(offsets: offsets, byteCounts: counts, into: destinations)
            if let maps = layout.expertTensors {
                for (index, miss) in plan.misses.enumerated() {
                    let expert = plan.experts[miss]
                    guard maps.indices.contains(expert), !maps[expert].isEmpty else {
                        continue
                    }
                    let tag = "bounded"
                    if ProcessInfo.processInfo.environment["TINYTITAN_EXPAND_TRACE"] == "1" {
                        FileHandle.standardError.write(
                            Data(
                                "[expand] site=\(tag) layer=\((layout.path as NSString).lastPathComponent) expert=\(expert)\n"
                                    .utf8))
                    }
                    TwoBitExpand.expandExpertInPlace(
                        slot: destinations[index], tensors: maps[expert])
                    if let directory = ProcessInfo.processInfo
                        .environment["TINYTITAN_DUMP_SLOT"]
                    {
                        let name =
                            (layout.path as NSString).lastPathComponent
                        let data = Data(
                            bytes: destinations[index],
                            count: Int(layout.expertStride))
                        try? data.write(
                            to: URL(
                                fileURLWithPath:
                                    "\(directory)/\(name)-e\(expert).bin"))
                    }
                }
            }
            return
        }
        if let phase1 = layout.phase1Bytes {
            let frontBytes = Int(phase1)
            try reader.fetch(
                offsets: offsets,
                byteCounts: [Int](repeating: frontBytes, count: offsets.count),
                into: destinations)
            firstStage?()
            let tails = destinations.map { $0.advanced(by: frontBytes) }
            try reader.fetch(
                offsets: offsets.map { $0 + phase1 },
                byteCounts: [Int](
                    repeating: Int(layout.expertStride) - frontBytes,
                    count: offsets.count),
                into: tails)
        } else {
            try reader.fetch(offsets: offsets, into: destinations)
        }
    }

    func executeCachedPreads(
        _ plan: ExpertCachePlan,
        parallel: Bool
    ) throws {
        if parallel {
            let firstError = Mutex<Error?>(nil)
            // parallelFor, not concurrentPerform: this runs on the expert-I/O
            // scheduler's worker threads, where concurrentPerform starves the
            // GCD pool and deadlocks a full-chunk prefill (see ParallelFor).
            parallelFor(iterations: plan.misses.count) { offset in
                do {
                    try self.readPlanMiss(plan, index: plan.misses[offset])
                } catch {
                    firstError.withLock { if $0 == nil { $0 = error } }
                }
            }
            if let error = firstError.withLock({ $0 }) { throw error }
            return
        }
        for index in plan.misses { try readPlanMiss(plan, index: index) }
    }

    /// Decode every miss of a compressed plan into its slots, eight experts a
    /// wave; a miss set no larger than that is a single wave.
    ///
    /// Each wave is handed to `decodeCompressedSet`, which reads it in one
    /// storage call and decodes every block in parallel -- fronts, then the
    /// tails. The first stage is published from the wave that ends at the last
    /// miss, because the consumer commits GPU work for every miss slot once it
    /// fires.
    /// Measured on Ornith 4-bit: four-experts-at-a-time with per-frame ranges
    /// took 5-13 ms to read and 1.8 ms per expert to decode, against a raw path
    /// that reads 3-4 experts at 3.4 GB/s in ~2 ms.
    func executeCompressedMisses(
        _ plan: ExpertCachePlan,
        firstStage: (() -> Void)? = nil
    ) throws {
        guard let frames = compressedFrames else { return }
        let wave = 8
        var start = 0
        while start < plan.misses.count {
            let end = min(start + wave, plan.misses.count)
            let misses = Array(plan.misses[start..<end])
            try decodeCompressedSet(
                experts: misses.map { plan.experts[$0] },
                frames: frames,
                into: misses.map { slotPointers[plan.assignedSlots[$0]] },
                firstStage: end == plan.misses.count ? firstStage : nil)
            start = end
        }
    }

    func readPlanMiss(_ plan: ExpertCachePlan, index: Int) throws {
        let expert = plan.experts[index]
        let payload =
            layout.expertSizes.flatMap {
                $0.indices.contains(expert) ? $0[expert] : nil
            } ?? layout.expertStride
        let slot = plan.assignedSlots[index]
        try readFull(
            into: slotPointers[slot],
            fileOffset: try fileOffset(plan: plan, index: index),
            count: Int(payload))
        if let maps = layout.expertTensors,
            maps.indices.contains(expert), !maps[expert].isEmpty
        {
            if ProcessInfo.processInfo.environment["TINYTITAN_EXPAND_TRACE"] == "1" {
                FileHandle.standardError.write(
                    Data(
                        "[expand] site=cached layer=\((layout.path as NSString).lastPathComponent) expert=\(expert)\n"
                            .utf8))
            }
            TwoBitExpand.expandExpertInPlace(
                slot: slotPointers[slot], tensors: maps[expert])
        }
    }

    func fileOffset(plan: ExpertCachePlan, index: Int) throws -> UInt64 {
        let regionOffset = layout.expertOffset(
            layer: plan.layer,
            expert: plan.experts[index])
        // Checked for the same reason: a wrapped sum here would read as "in
        // range" and the pread below would take an offset past the end of the
        // mapped file. With this and the open-time check, every sum in this type
        // that mixes a layout offset with a region is known not to wrap.
        // A tiered layer's cold experts are shorter than the slot stride, so
        // the bound is the expert's own payload, not the stride; using the
        // stride rejects the last expert of a compact file.
        let expert = plan.experts[index]
        let payload =
            layout.expertSizes.flatMap {
                $0.indices.contains(expert) ? $0[expert] : nil
            } ?? layout.expertStride
        let (regionEnd, regionOverflow) =
            regionOffset
            .addingReportingOverflow(payload)
        guard !regionOverflow, regionEnd <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        return layout.streamOffset + regionOffset
    }

}
