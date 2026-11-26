import Foundation
import Metal
import TinyTitanCore

/// Decode-time routed MoE: the shared expert, the hit/fixup expert dispatch,
/// and the slot bookkeeping the stage publishes.
///
/// Split out of `RealForwardRunner+Decode.swift` (2026-09-28) under the
/// 500-line-per-file rule as pure code motion: same bodies, no behaviour
/// change, and no declaration needed widening.
extension RealForwardRunner {

    /// Routed-expert stage of one decode layer: top-k readback, expert fetch,
    /// phase-1/phase-2 encode, and the deferred completion hand-off.
    ///
    /// lint:allow-long one pipeline whose phases share the fetch plan, the
    /// argument buffer and the slot scratch; the layer trace at the end reports
    /// timings from every phase, so splitting it would mean threading those
    /// back out purely to shorten a function.
    func encodeDecodeRoutedMoE(
        layer L: Int,
        position: Int,
        sharedProj: LayerSharedExpertProjections,
        attnCB: MTLCommandBuffer,
        tailCB: MTLCommandBuffer,
        sharedCB: MTLCommandBuffer,
        overlapCompletionClock: CommandCompletionClock?,
        pending pendingRoutedCommand: inout PendingRoutedCommand?,
        bodyStart tBodyStart: UInt64,
        cb1Start tCb1Start: UInt64,
        waitMark tWait: UInt64,
        waitNanos: UInt64,
        previousRoutedMicros prevRoutedUs: Double,
        predictedNextLayer: [Int],
        predictedNextLayerWeights: [Float] = [],
        hidden residualHidden: MTLBuffer? = nil,
        hiddenOffset: Int = 0
    ) async throws {
        let D = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let readbackStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let idxPtr = outIndices.contents().bindMemory(
            to: UInt32.self,
            capacity: cfg.topKExperts)
        decodeExpertsScratch.removeAll(keepingCapacity: true)
        decodeExpertsScratch.reserveCapacity(cfg.topKExperts)
        for i in 0..<cfg.topKExperts {
            decodeExpertsScratch.append(min(Int(idxPtr[i]), cfg.numExperts - 1))
        }
        totalRouterReadbackNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - readbackStarted
        let experts = decodeExpertsScratch
        recordRouteTrace(layer: L, position: position, experts: experts)

        let routedOffsets = try model.routedExpertOffsets(layer: L)
        let topK = UInt32(cfg.topKExperts)
        let canUsePlannedFetch = cfg.topKExperts <= moe.maxStreamedExperts
        let residentBeforePlan =
            prefetchTraceFD >= 0
            ? try model.routedExpertResidentIDs(layer: L) : []
        let readyPrefetches = predictivePrefetch?.readyBuffers(layer: L, experts: experts) ?? [:]
        let cachePlanStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let plannedFetch =
            canUsePlannedFetch
            ? try model.planRoutedExperts(
                layer: L, experts: experts, prefetched: readyPrefetches)
            : nil
        totalCachePlanNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cachePlanStarted
        if !readyPrefetches.isEmpty {
            predictivePrefetch?.consume(layer: L, experts: Set(readyPrefetches.keys))
            totalPrefetchAdopted &+= UInt64(readyPrefetches.count)
        }
        let missesForTrace =
            plannedFetch.map { plan in
                plan.misses.map { experts[$0] }
            } ?? experts
        recordPrefetchTrace(
            layer: L, position: position, experts: experts,
            misses: missesForTrace, resident: residentBeforePlan,
            nextLayerPrediction: predictedNextLayer,
            next2LayerPrediction: lastPredictedNext2Layer,
            nextLayerWeights: predictedNextLayerWeights)
        let expertLease = try plannedFetch.map { try model.pinRoutedExperts(for: $0) }
        // v4.2 Phase B: once slots and generations are reserved and pinned,
        // submit real storage immediately. Hit partitioning, argument binding,
        // and command encoding below now overlap the reader queue.
        let shouldSubmitImmediately =
            expertIOSubmission == .immediate
            || expertIOSynchronization == .event
        let plannedLoad =
            shouldSubmitImmediately
            ? try plannedFetch.map {
                try model.beginFetchRoutedExperts(
                    plan: $0,
                    eventDriven: expertIOSynchronization == .event && !$0.misses.isEmpty)
            }
            : nil
        var transferredExpertLease = false
        var phase1HitCB: MTLCommandBuffer?
        // The early miss fixup: when the load splits its reads, phase 1 over
        // the miss slots is encoded and committed after the front range
        // (gate+up) lands, and runs on the GPU while the tail (down) fetch is
        // still in flight. `routedCB` below then carries phase 2 only.
        var phase1MissCB: MTLCommandBuffer?
        var earlyFixupArgBuf: MTLBuffer?
        defer {
            if !transferredExpertLease {
                // A thrown fetch/encode must not make a slot evictable while
                // an already-committed phase-1 command is still reading it.
                if let expertLease {
                    if let phase1HitCB { try? waitForCompletion(phase1HitCB) }
                    if let phase1MissCB { try? waitForCompletion(phase1MissCB) }
                    expertLease.release()
                } else {
                    expertLease?.release()
                }
            }
        }
        var phase1HitSplitArgBuf: MTLBuffer?
        decodeHitSplitRoutedBufsScratch.removeAll(keepingCapacity: true)
        decodeHitSplitRoutedOffsetsScratch.removeAll(keepingCapacity: true)
        decodeHitSlotsScratch.removeAll(keepingCapacity: true)
        decodeMissSlotsScratch.removeAll(keepingCapacity: true)

        if let plan = plannedFetch,
            decodeExpertExecution == .hitFixup
                || decodeExpertExecution == .gpuResidency
        {
            if decodeExpertExecution == .gpuResidency {
                try classifyGPUResidencyForDecode(plan: plan)
            } else {
                DecodeExpertPartition.populate(
                    topK: cfg.topKExperts,
                    missIndices: plan.misses,
                    hits: &decodeHitSlotsScratch,
                    misses: &decodeMissSlotsScratch)
            }
        }
        // Capture the populated arrays. Capturing them before `populate` made
        // empty value-semantic snapshots and silently disabled hit/fixup.
        let phase1HitSlots = decodeHitSlotsScratch
        let phase1MissSlots = decodeMissSlotsScratch

        if let plan = plannedFetch,
            let plannedHit = try encodeDecodePlannedHitPhase1(
                plan: plan,
                phase1HitSlots: phase1HitSlots,
                routedOffsets: routedOffsets,
                d: D,
                f: FmoE,
                topK: topK)
        {
            phase1HitSplitArgBuf = plannedHit.argBuf
            phase1HitCB = plannedHit.cb
        }

        if let cb = phase1HitCB {
            overlapCompletionClock?.track(cb)
            cb.commit()
        }
        let missCount = plannedFetch?.misses.count ?? experts.count
        let completionClock = missCount > 0 ? overlapCompletionClock : nil
        let expectedOverlapCompletions = phase1HitCB == nil ? 1 : 2
        if plannedLoad == nil && rdadviseEnabled && rdadvisePolicyMode != .off {
            try issueDecodeRDAdvice(
                layer: L, position: position, plannedFetch: plannedFetch,
                missCount: missCount, experts: experts)
        }

        // The ring is issued here, before the demand fetch awaits, rather than
        // after it. It is aimed at the next layer, whose prediction is already
        // in hand from this layer's router probe, so every millisecond the
        // demand path spends fetching and decoding is lead time the ring can
        // use. Issued at the old point -- after the await -- a compressed
        // layer's read-and-decode (~3 ms for a predicted set) never finished
        // before the next layer was planned, so adoption measured 0.11/token
        // against the raw path's 21.75.
        if let predictivePrefetch, L + 1 < cfg.numLayers {
            // The ring is one read a layer, aimed at the next layer, from the
            // profile's depth -- unless `TINYTITAN_PREFETCH_LAYER_DEPTH` names
            // this link, which is the layer-selective arm §5al screened and left
            // for a live paired run to settle. Every other allocation of the
            // same budget measured here lost: a second rank of the same layer
            // (per-expert operations, -3.3%), the same read aimed two layers
            // ahead (-2.8%), a second horizon beside it (-1.4%), the probe-weight
            // margin gate (-4.2%) and a lower disk tier (-0.6%). Each is recorded
            // in docs/qwen38-prefetch-predictor-study.md and Lever 5/8 of
            // benchmark/internal-speeds/v2-qwen38-4bit-telemetry.txt.
            let target = L + 1
            let prediction = predictedNextLayer
            let resident = Set(try model.routedExpertResidentIDs(layer: target))
            // The whole ranked prediction goes in; `begin` drops the experts
            // that are already resident or already in flight and then fills
            // whatever ring slots are free, in rank order.
            //
            // Truncating to top-M here first (v4.3) spent the budget before
            // the residency filter ran. About 85% of the top-M predictions are
            // already cached, so the filter starved the ring instead of aiming
            // it: on a 191-position qwen38 trace, top-4 issued 0.94 reads per
            // layer and covered 23% of the demand misses, where filtering
            // first issues 2.59 and covers 46%. The ring size stays the cap on
            // reads in flight; only the order of cap and filter changed.
            try predictivePrefetch.begin(
                model: model, layer: target,
                experts: prediction,
                resident: resident, currentLayer: L)
        }
        // Routed-expert pread — overlaps the shared MLP GPU work above.
        let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let blobs: [TensorView]
        var completedStorageNanos: UInt64?
        let eventLoad = plannedLoad.flatMap { operation -> RoutedExpertLoadOperation? in
            operation.storage.completionToken == nil ? nil : operation
        }
        if let eventLoad {
            // Slot resources and offsets are known from the reservation. Their
            // bytes are consumed only after the shared-event wait encoded
            // below, so no successful completion has to resume this task.
            blobs = try model.routedExpertBuffers(for: eventLoad.plan)
            totalExpertIOHostWaitsAvoided &+= 1
        } else if let plannedFetch, plannedFetch.misses.isEmpty {
            // An all-hit layer has already pinned its current slot generations.
            // Do not manufacture a completed storage operation and an async
            // continuation only to retrieve the same cache views.
            blobs = try model.routedExpertBuffers(for: plannedFetch)
            totalExpertIOHostWaitsAvoided &+= 1
        } else if let plannedLoad {
            totalExpertIOHostWaits &+= plannedLoad.plan.misses.isEmpty ? 0 : 1
            if let fixup = try await encodeDecodeEarlyMissFixup(
                load: plannedLoad,
                phase1MissSlots: phase1MissSlots,
                phase1HitSplitArgBuf: phase1HitSplitArgBuf,
                routedOffsets: routedOffsets,
                d: D,
                f: FmoE,
                topK: topK,
                completionClock: completionClock)
            {
                phase1MissCB = fixup.cb
                earlyFixupArgBuf = fixup.argBuf
            }
            blobs = try await plannedLoad.completion()
            totalIOQueueNanos &+= plannedLoad.storage.submissionToStartNanos
            completedStorageNanos = plannedLoad.storage.completedNanos
        } else if let plannedFetch {
            // The production deferred schedule still uses the split operation
            // so queueing and completion remain observable. It deliberately
            // begins here, after the independent hit work is committed.
            let deferredLoad = try model.beginFetchRoutedExperts(plan: plannedFetch)
            totalExpertIOHostWaits &+= plannedFetch.misses.isEmpty ? 0 : 1
            if let fixup = try await encodeDecodeEarlyMissFixup(
                load: deferredLoad,
                phase1MissSlots: phase1MissSlots,
                phase1HitSplitArgBuf: phase1HitSplitArgBuf,
                routedOffsets: routedOffsets,
                d: D,
                f: FmoE,
                topK: topK,
                completionClock: completionClock)
            {
                phase1MissCB = fixup.cb
                earlyFixupArgBuf = fixup.argBuf
            }
            blobs = try await deferredLoad.completion()
            totalIOQueueNanos &+= deferredLoad.storage.submissionToStartNanos
            completedStorageNanos = deferredLoad.storage.completedNanos
        } else {
            blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
        }
        let layerIo =
            eventLoad == nil
            ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart : 0
        if eventLoad == nil { totalIoNanos &+= layerIo }
        if missCount > 0 && eventLoad == nil {
            totalMissIoNanos &+= layerIo
            // The early miss fixup is tracked in the same clock, so the
            // expected completion count grows with it. If it has not completed
            // by this query, `latest` returns nil and the overlap credit is
            // withheld rather than guessed.
            let overlapExpected = expectedOverlapCompletions + (phase1MissCB != nil ? 1 : 0)
            if let latest = completionClock?.latest(expected: overlapExpected) {
                let overlapEnd = max(tIoStart, latest)
                if overlapEnd < tIoStart + layerIo {
                    totalExposedIoNanos &+= tIoStart + layerIo - overlapEnd
                }
            }
        }
        decodeRoutedBufsScratch.removeAll(keepingCapacity: true)
        decodeRoutedOffsetsScratch.removeAll(keepingCapacity: true)
        for blob in blobs {
            decodeRoutedBufsScratch.append(blob.buffer)
            decodeRoutedOffsetsScratch.append(Int(blob.offset))
        }
        let routedBufs = decodeRoutedBufsScratch
        let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // The phase-2 reduce already folded the shared branch (h1Buf
        // as its residual); the tail is a plain residual add.
        let gTail: (MTLCommandBuffer) throws -> Void = { [self] cb in
            try encodeResidualExitDecode(
                commandBuffer: cb,
                hidden: residualHidden ?? hidden, hiddenOffset: hiddenOffset,
                delta: h2Buf,
                sublayer: .mlp, layer: L)
        }
        guard let routedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let ioToken = eventLoad?.storage.completionToken
        let ioStatus = ioToken.map { ($0.status, $0.statusOffset) }
        if let token = ioToken {
            routedCB.encodeWaitForEvent(token.event, value: token.value)
        }
        if let stagingTransfer = eventLoad?.storage.metalStagingTransfer {
            // The compute command references cache slots only after it has
            // waited for the MTLIO staging event. This is deliberately a GPU
            // blit, not a CPU memcpy or a completion-handler submission.
            try stagingTransfer.encodeCopy(commandBuffer: routedCB)
        }
        let splitArgBuf =
            phase1MissCB == nil
                && phase1HitCB != nil && !phase1MissSlots.isEmpty
            ? phase1HitSplitArgBuf
            : nil
        let argBuf =
            splitArgBuf
            ?? earlyFixupArgBuf
            ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs,
                topK: topK,
                routedBufferOffsets: decodeRoutedOffsetsScratch)
        if splitArgBuf != nil {
            totalHitFixupLayers &+= 1
            writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
            try encodeDecodeRoutedPhase1Subset(
                routedCB,
                argBuf: argBuf,
                routedBufs: routedBufs,
                activeSlots: moeMissActiveSlots,
                activeSlotIndices: phase1MissSlots,
                activeCount: UInt32(phase1MissSlots.count),
                routedOffsets: routedOffsets,
                d: D,
                f: FmoE,
                topK: topK,
                ioStatus: ioStatus?.0,
                ioStatusOffset: ioStatus?.1 ?? 0)
        } else if phase1MissCB == nil {
            // With an early miss fixup, phase 1 (hit subset and miss fixup)
            // is already committed and running; this buffer carries phase 2
            // and the tail only. Same-queue FIFO order puts it after the
            // fixup commit.
            try encodeDecodeRoutedPhase1Full(
                routedCB,
                argBuf: argBuf,
                routedBufs: routedBufs,
                routedOffsets: routedOffsets,
                d: D,
                f: FmoE,
                topK: topK,
                ioStatus: ioStatus?.0,
                ioStatusOffset: ioStatus?.1 ?? 0)
        }
        try moe.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: routedCB,
            routedArgBuffer: argBuf,
            routedBlobs: routedBufs,
            routedOffsets: routedOffsets,
            acts: moeActs,
            routingWeights: outWeights,
            residual: h1Buf,
            y: h2Buf,
            d: D,
            f: FmoE,
            topK: topK,
            ioStatus: ioStatus?.0,
            ioStatusOffset: ioStatus?.1 ?? 0)
        try gTail(routedCB)
        routedCB.commit()
        if missCount > 0, let completed = completedStorageNanos, completed > 0 {
            let submitted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if submitted >= completed {
                totalIOCompletionToFixupSubmitNanos &+= submitted - completed
            }
        }
        guard pendingRoutedCommand == nil else {
            // The pipeline drains the previous layer's routed CB before
            // queuing the next, so this is a logic error, not a user
            // condition — but it must fail the generation, not trap.
            throw ModelError.internalInconsistency(
                detail: "routed command-buffer pipeline not drained before queuing the next layer")
        }
        pendingRoutedCommand = PendingRoutedCommand(
            cb: routedCB,
            sharedCB: sharedCB,
            phase1HitCB: phase1HitCB,
            phase1MissCB: phase1MissCB,
            expertLease: expertLease,
            storageOperation: eventLoad,
            overlapCompletionClock: eventLoad == nil ? nil : overlapCompletionClock,
            expectedOverlapCompletions: expectedOverlapCompletions,
            kernelRole:
                phase1MissCB != nil
                ? "moe_phase2_routed"
                : splitArgBuf == nil
                    ? "moe_phase1_2_routed"
                    : "moe_phase1_miss_fixup_phase2",
            encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
        transferredExpertLease = true
        totalBodyNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tBodyStart
        if ProcessInfo.processInfo.environment["TINYTITAN_LAYER_TRACE"] != nil,
            position < 3 || position % 16 == 0
        {
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let attnUs = (attnCB.gpuEndTime - attnCB.gpuStartTime) * 1_000_000
            let tailUs = (tailCB.gpuEndTime - tailCB.gpuStartTime) * 1_000_000
            emitDecodeMoELayerTrace(
                position: position, layer: L,
                bodyUs: (now - tBodyStart) / 1000, waitUs: waitNanos / 1000,
                ioUs: layerIo / 1000, cb1Us: (tWait - tCb1Start) / 1000,
                cb2Us: (now - tCb2Start) / 1000,
                attnUs: Int(attnUs), tailUs: Int(tailUs), routedUs: Int(prevRoutedUs))
        }
    }
}
