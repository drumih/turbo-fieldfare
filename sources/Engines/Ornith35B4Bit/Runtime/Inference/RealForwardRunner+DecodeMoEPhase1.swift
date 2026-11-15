import Foundation
import Metal
import TinyTitanCore

/// Decode-time routed MoE phase-1 encodes: the full and subset phase-1
/// dispatches and the early miss fixup that commits phase 1 over the miss
/// slots while the tail (down) fetch is still in flight.
///
/// Split out of `RealForwardRunner+DecodeMoE.swift` (2026-10-14) under the
/// 500-line-per-file rule as near-pure code motion: the bodies moved from
/// nested functions closing over the layer's locals to methods taking them
/// as parameters (`routedOffsets`, `d`, `f`, `topK`, and the fixup's
/// hit/miss slot state). No behaviour change; every declaration they touch
/// was already internal, so no access widening was needed.
extension RealForwardRunner {

    func encodeDecodeRoutedPhase1Full(
        _ cb: MTLCommandBuffer,
        argBuf: MTLBuffer,
        routedBufs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        try moe.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: cb,
            routedArgBuffer: argBuf,
            routedBlobs: routedBufs,
            routedOffsets: routedOffsets,
            x: routedX,
            acts: moeActs,
            d: d,
            f: f,
            topK: topK,
            ioStatus: ioStatus,
            ioStatusOffset: ioStatusOffset)
    }

    func encodeDecodeRoutedPhase1Subset(
        _ cb: MTLCommandBuffer,
        argBuf: MTLBuffer,
        routedBufs: [MTLBuffer],
        activeSlots: MTLBuffer,
        activeSlotIndices: [UInt32],
        activeCount: UInt32,
        routedOffsets: MoEExpertOffsets,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        try moe.encodeRoutedPersistentPhase1SubsetU16Load(
            commandBuffer: cb,
            routedArgBuffer: argBuf,
            routedBlobs: routedBufs,
            routedOffsets: routedOffsets,
            x: routedX,
            acts: moeActs,
            activeSlots: activeSlots,
            activeSlotIndices: activeSlotIndices,
            activeCount: activeCount,
            d: d,
            f: f,
            topK: topK,
            ioStatus: ioStatus,
            ioStatusOffset: ioStatusOffset)
    }

    /// Phase 1 over the hit slots of a mixed hit/miss plan, encoded ahead of
    /// the miss fetch so the resident experts run while the misses stream.
    /// Fills the hit-split scratch buffers and returns the split argument
    /// buffer plus the uncommitted command buffer, or nil when the plan is
    /// not a mixed hit/miss layer.
    func encodeDecodePlannedHitPhase1(
        plan: RoutedExpertFetchPlan,
        phase1HitSlots: [UInt32],
        routedOffsets: MoEExpertOffsets,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) throws -> (argBuf: MTLBuffer, cb: MTLCommandBuffer)? {
        guard plan.hits > 0, !plan.misses.isEmpty, !phase1HitSlots.isEmpty else { return nil }
        let plannedBlobs = try model.routedExpertBuffers(for: plan)
        for blob in plannedBlobs {
            decodeHitSplitRoutedBufsScratch.append(blob.buffer)
            decodeHitSplitRoutedOffsetsScratch.append(Int(blob.offset))
        }
        guard
            let argBuf = moe.makeRoutedArgumentBuffer(
                routedBlobs: decodeHitSplitRoutedBufsScratch,
                topK: topK,
                routedBufferOffsets: decodeHitSplitRoutedOffsetsScratch)
        else { return nil }
        writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try encodeDecodeRoutedPhase1Subset(
            cb,
            argBuf: argBuf,
            routedBufs: decodeHitSplitRoutedBufsScratch,
            activeSlots: moeHitActiveSlots,
            activeSlotIndices: phase1HitSlots,
            activeCount: UInt32(phase1HitSlots.count),
            routedOffsets: routedOffsets,
            d: d,
            f: f,
            topK: topK)
        return (argBuf, cb)
    }

    /// Encodes and commits phase 1 over the miss slots as soon as the
    /// load's front range (gate+up) is resident, so it executes on the GPU
    /// while the tail (down) fetch is still in flight. Slot resources are
    /// known from the reservation, so the argument buffer is built before
    /// any wait; its bytes are consumed only by commands committed after
    /// `firstStageCompletion` resumed. Returns the committed buffer and its
    /// argument buffer, or nil when the load has no first stage or there
    /// are no miss slots.
    func encodeDecodeEarlyMissFixup(
        load: RoutedExpertLoadOperation,
        phase1MissSlots: [UInt32],
        phase1HitSplitArgBuf: MTLBuffer?,
        routedOffsets: MoEExpertOffsets,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        completionClock: CommandCompletionClock?
    ) async throws -> (cb: MTLCommandBuffer, argBuf: MTLBuffer)? {
        guard load.storage.publishesFirstStage, !phase1MissSlots.isEmpty else { return nil }
        try await load.storage.firstStageCompletion()
        let planBlobs = try model.routedExpertBuffers(for: load.plan)
        var planBufs: [MTLBuffer] = []
        var planOffsets: [Int] = []
        planBufs.reserveCapacity(planBlobs.count)
        planOffsets.reserveCapacity(planBlobs.count)
        for blob in planBlobs {
            planBufs.append(blob.buffer)
            planOffsets.append(Int(blob.offset))
        }
        let argBuf =
            phase1HitSplitArgBuf
            ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: planBufs,
                topK: topK,
                routedBufferOffsets: planOffsets)
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
        if phase1HitSplitArgBuf != nil {
            try encodeDecodeRoutedPhase1Subset(
                cb,
                argBuf: argBuf,
                routedBufs: planBufs,
                activeSlots: moeMissActiveSlots,
                activeSlotIndices: phase1MissSlots,
                activeCount: UInt32(phase1MissSlots.count),
                routedOffsets: routedOffsets,
                d: d,
                f: f,
                topK: topK)
        } else {
            try encodeDecodeRoutedPhase1Full(
                cb,
                argBuf: argBuf,
                routedBufs: planBufs,
                routedOffsets: routedOffsets,
                d: d,
                f: f,
                topK: topK)
        }
        completionClock?.track(cb)
        cb.commit()
        totalHitFixupLayers &+= 1
        return (cb, argBuf)
    }
}
