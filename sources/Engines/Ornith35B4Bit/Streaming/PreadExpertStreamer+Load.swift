import Foundation
import Synchronization
import TinyTitanFormat
import Metal
import TinyTitanCore

// Reading one expert into a slot, raw or compressed.
//
// Split out of `PreadExpertStreamer.swift` under the 500-line-per-file rule when
// the compressed path landed. Pure code motion: the declarations and their
// access are unchanged, and the class has no `private` storage these touch.
extension PreadExpertStreamer {
    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64)
    {
        // K12: slot selection and fill share one critical section so the
        // round-robin path never lands on a slot a concurrent plan reserved
        // (`loading`) and no fill can interleave with another pread.
        cacheLock.lock()
        defer { cacheLock.unlock() }
        var candidate = nextSlot
        var scanned = 0
        while (slotState[candidate] == .loading || slotPinCount[candidate] > 0)
            && scanned < slotCount
        {
            candidate = (candidate + 1) % slotCount
            scanned += 1
        }
        guard scanned < slotCount else {
            throw ModelError.expertCacheUnplaceable(
                detail: "all \(slotCount) expert-cache slots are loading or pinned")
        }
        nextSlot = (candidate + 1) % slotCount
        return try loadExpertUnlocked(layer: layer, expert: expert, slot: candidate)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64)
    {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        // K12: the pread fill and the slot bookkeeping share one critical
        // section so a concurrent plan/execute or another load cannot write
        // into this slot while the pread is in flight.
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard slotState[slot] != .loading, slotPinCount[slot] == 0 else {
            throw ModelError.expertCacheUnplaceable(
                detail: "expert-cache slot \(slot) is loading or pinned")
        }
        return try loadExpertUnlocked(layer: layer, expert: expert, slot: slot)
    }

    /// Fill `slot` with `expert` and update bookkeeping. Callers hold
    /// `cacheLock`.
    func loadExpertUnlocked(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64)
    {
        // A compressed layer is addressed through its index, not by
        // `n * stride`, so the stride-shaped range check does not describe it:
        // the frames are laid out back to back and are shorter than the stride.
        // The guard for that path is each frame's own end, checked below.
        // A tiered layer's cold experts are shorter than the slot stride, so
        // the read and its bound use the expert's own payload.
        let payload: UInt64 =
            layout.expertSizes.flatMap {
                $0.indices.contains(expert) ? $0[expert] : nil
            } ?? layout.expertStride
        let regionOffset: UInt64
        if ransResources != nil {
            // The payload is not at raw offsets; the sidecar's block ranges are
            // checked when the layer opens.
            regionOffset = 0
        } else if let frames = compressedFrames {
            let expertFrames = frames[expert]
            let end = expertFrames.map { $0.offset + UInt64($0.length) }.max() ?? 0
            guard end <= layout.streamSize else {
                throw StreamerError.offsetOutOfRange(end)
            }
            regionOffset = 0
        } else {
            regionOffset = layout.expertOffset(layer: layer, expert: expert)
            // Checked for the same reason: a wrapped sum here would read as "in
            // range" and the pread below would take an offset past the end of the
            // mapped file. With this and the open-time check, every sum in this
            // type that mixes a layout offset with a region is known not to wrap.
            let (regionEnd, regionOverflow) =
                regionOffset
                .addingReportingOverflow(payload)
            guard !regionOverflow, regionEnd <= layout.streamSize else {
                throw StreamerError.offsetOutOfRange(regionOffset)
            }
        }
        slotGeneration[slot] &+= 1
        let previousExpert = slotExpert[slot]
        if previousExpert >= 0 {
            publishResidencyUnlocked(
                expert: previousExpert,
                slot: slot,
                state: ExpertResidencyEntry.empty,
                generation: slotGeneration[slot])
        }
        slotExpert[slot] = expert
        slotState[slot] = .loading
        publishResidencyUnlocked(
            expert: expert,
            slot: slot,
            state: ExpertResidencyEntry.loading,
            generation: slotGeneration[slot])
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        do {
            if ransResources != nil {
                try decodeRANSExpert(expert: expert, slot: slot)
            } else if let frames = compressedFrames {
                try readCompressedExpert(
                    frames: frames[expert], slot: slot)
            } else {
                try readFull(
                    into: slotPointers[slot],
                    fileOffset: layout.streamOffset + regionOffset,
                    count: Int(payload))
                if let maps = layout.expertTensors,
                    maps.indices.contains(expert), !maps[expert].isEmpty
                {
                    if ProcessInfo.processInfo.environment["TINYTITAN_EXPAND_TRACE"] == "1" {
                        FileHandle.standardError.write(
                            Data(
                                "[expand] site=single layer=\((layout.path as NSString).lastPathComponent) expert=\(expert)\n"
                                    .utf8))
                    }
                    TwoBitExpand.expandExpertInPlace(
                        slot: slotPointers[slot], tensors: maps[expert])
                }
            }
            slotState[slot] = .resident
            publishResidencyUnlocked(
                expert: expert,
                slot: slot,
                state: ExpertResidencyEntry.resident,
                generation: slotGeneration[slot])
            slotLastUse[slot] = useClock
            recordSuccessfulLoadsUnlocked(
                experts: [expert], elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
        } catch {
            slotState[slot] = .empty
            slotExpert[slot] = -1
            publishResidencyUnlocked(
                expert: expert,
                slot: slot,
                state: ExpertResidencyEntry.empty,
                generation: slotGeneration[slot])
            throw error
        }
        return (slotBuffers[slot], slotBufferOffsets[slot], layout.expertStride)
    }

}
