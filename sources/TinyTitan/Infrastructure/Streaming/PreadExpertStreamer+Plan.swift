import Darwin
import Foundation
import Synchronization
import Metal

// Planning an expert cache plan and executing its IO.
//
// Split out of `PreadExpertStreamer.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// Members whose callers stay behind widened from `private` to internal.
extension PreadExpertStreamer {

    func makeExpertCachePlan(
        layer: Int,
        experts: [Int],
        avoidingSlots rawAvoidingSlots: Set<Int>,
        prefetched: [Int: UnsafeMutableRawPointer],
        recordsUse: Bool = true
    )
        -> ExpertCachePlan?
    {
        // `recordsUse: false` is the startup warm-up (`planWarmupExperts`):
        // the same placement, but no LFU score and no hit/miss statistics,
        // because no token routed to these experts.
        // K10: too few slots for the requested expert set is a recoverable
        // placement failure, not a programming error, and both entry points are
        // already built to handle it -- `planExpertsCached` turns nil into
        // `expertCacheUnplaceable`, and `planExpertsCachedIfPossible` returns
        // nil, which the prefill tile scheduler reads as "no plan available"
        // and falls back on. A trap here aborted the process instead, on a
        // user-selectable configuration: `--expert-cache-slots 8` against
        // Qwen3.8-Flash-Next, which routes top-10 experts, whose prefill tiles
        // can carry up to 16 live experts.
        guard experts.count <= slotCount else { return nil }
        let avoidingSlots = Set(rawAvoidingSlots.filter { $0 >= 0 && $0 < slotCount })

        cacheLock.lock()
        defer { cacheLock.unlock() }

        let clock = useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)
        // Loading slots are not valid hits and cannot be reassigned.
        for slot in 0..<slotCount where slotState[slot] == .loading {
            reserved[slot] = true
        }

        for index in experts.indices {
            for slot in 0..<slotCount
            where !reserved[slot] && slotState[slot] == .resident
                && slotExpert[slot] == experts[index]
            {
                assignedSlots[index] = slot
                reserved[slot] = true
                break
            }
        }
        for slot in avoidingSlots where !reserved[slot] {
            reserved[slot] = true
        }

        let candidateMisses = experts.indices.filter { assignedSlots[$0] == -1 }
        let evictable = (0..<slotCount)
            .filter { !reserved[$0] && slotState[$0] != .loading && slotPinCount[$0] == 0 }
            .sorted { shouldEvictSlot($0, before: $1) }
        guard candidateMisses.count <= evictable.count else { return nil }

        useClock = clock
        for expert in experts where recordsUse && expert >= 0 && expert < expertUseCount.count {
            expertUseCount[expert] &+= 1
        }
        for slot in assignedSlots where slot >= 0 {
            slotLastUse[slot] = clock
        }
        var misses: [Int] = []
        var adoptedPrefetches: [Int] = []
        for (offset, index) in candidateMisses.enumerated() {
            let slot = evictable[offset]
            if slotState[slot] == .resident { statisticsEvictions &+= 1 }
            let previousExpert = slotExpert[slot]
            assignedSlots[index] = slot
            reserved[slot] = true
            slotGeneration[slot] &+= 1
            slotExpert[slot] = experts[index]
            slotLastUse[slot] = clock
            slotState[slot] = .loading
            if previousExpert >= 0 {
                publishResidencyUnlocked(
                    expert: previousExpert,
                    slot: slot,
                    state: ExpertResidencyEntry.empty,
                    generation: slotGeneration[slot])
            }
            publishResidencyUnlocked(
                expert: experts[index],
                slot: slot,
                state: ExpertResidencyEntry.loading,
                generation: slotGeneration[slot])
            if let source = prefetched[experts[index]] {
                memcpy(slotPointers[slot], source, Int(layout.expertStride))
                slotState[slot] = .resident
                publishResidencyUnlocked(
                    expert: experts[index],
                    slot: slot,
                    state: ExpertResidencyEntry.resident,
                    generation: slotGeneration[slot])
                adoptedPrefetches.append(experts[index])
            } else {
                misses.append(index)
            }
        }

        recordPrefetchAdoptionsUnlocked(adoptedPrefetches)
        if recordsUse {
            recordPlanStatisticsUnlocked(requested: experts.count, misses: misses.count)
        }

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            assignedGenerations: assignedSlots.map { slotGeneration[$0] },
            misses: misses,
            hits: experts.count - misses.count,
            layer: layer)
    }

    /// The startup warm-up's placement: the plan a decode routing exactly
    /// these experts would get from the same cache state, minus the LFU bump
    /// and the hit/miss statistics. Nil when they cannot all be placed.
    /// Layer 0 as in `Model.planRoutedExperts`: a streamer's file offsets are
    /// relative to its own layer.
    public func planWarmupExperts(_ experts: [Int]) -> ExpertCachePlan? {
        makeExpertCachePlan(
            layer: 0, experts: experts, avoidingSlots: [], prefetched: [:],
            recordsUse: false)
    }

    /// One routed plan's hit/miss accounting. Caller holds `cacheLock`.
    func recordPlanStatisticsUnlocked(requested: Int, misses: Int) {
        statisticsPlans &+= 1
        statisticsRequestedExperts &+= UInt64(requested)
        statisticsHits &+= UInt64(requested - misses)
        statisticsMisses &+= UInt64(misses)
        statisticsPeakLoadingSlots = max(
            statisticsPeakLoadingSlots,
            slotState.count(where: { $0 == .loading }))
    }

    public func executeExpertCachePlan(
        _ plan: ExpertCachePlan,
        firstStage: (() -> Void)? = nil
    ) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)]
    {
        precondition(
            plan.experts.count <= slotCount,
            "expert cache plan exceeds slot count")
        precondition(
            plan.assignedSlots.count == plan.experts.count,
            "expert cache plan slot count mismatch")
        precondition(
            plan.assignedGenerations.count == plan.experts.count,
            "expert cache plan generation count mismatch")

        if ProcessInfo.processInfo.environment["TINYTITAN_FETCH_TRACE"] == "1" {
            FileHandle.standardError.write(
                Data(
                    "[plan] file=\((layout.path as NSString).lastPathComponent) layer=\(plan.layer) misses=\(plan.misses.count) experts=\(plan.experts.count)\n"
                        .utf8))
        }
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var succeeded = false
        defer {
            finishPlanExecution(
                plan,
                succeeded: succeeded,
                elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
        }

        if !plan.misses.isEmpty {
            // A compressed layer has no byte range per expert: every backend
            // below copies a range, and a frame's bytes are not the weights.
            // The decode goes straight into the same slots instead, so the
            // residency and buffer bookkeeping below is unchanged.
            if ransResources != nil {
                try executeRANSPlan(plan, firstStage: firstStage)
            } else if compressedFrames != nil {
                try executeCompressedMisses(plan, firstStage: firstStage)
            } else if let metalReader {
                try executeMetalReads(plan, reader: metalReader)
            } else if let boundedReader {
                try executeBoundedReads(plan, reader: boundedReader, firstStage: firstStage)
            } else {
                // Always parallel when there is more than one miss: the
                // serial alternative (TINYTITAN_PARALLEL_IO=0) measured a wash
                // (+0.4%, 2/3) and is gone.
                try executeCachedPreads(plan, parallel: plan.misses.count > 1)
            }
            try markPlanMissesResident(plan)
        }

        succeeded = true
        return expertCachePlanBuffers(plan)
    }

    /// Submits the plan to the persistent storage service and returns before
    /// any read has to complete. Reserved generations are already pinned by
    /// the caller, so the destination pointers remain valid for the operation.
    public func beginExpertCachePlan(
        _ plan: ExpertCachePlan,
        eventDriven: Bool = false
    ) throws -> ExpertLoadOperation {
        let token: ExpertIOCompletionToken?
        if eventDriven {
            guard let eventCoordinator else {
                throw ModelError.internalInconsistency(
                    detail: "event-driven expert I/O requested without a shared event")
            }
            token = try eventCoordinator.reserve()
        } else {
            token = nil
        }
        guard !plan.misses.isEmpty else {
            let operation = ExpertLoadOperation(
                completionToken: token,
                eventCoordinator: eventCoordinator,
                backendSignalsEvent: false)
            operation.finish(.success(()))
            return operation
        }
        // Compressed layers decode on the CPU, so they take the scheduler path
        // even when Metal I/O is configured: handing MTLIO a frame offset would
        // copy frame bytes into a slot.
        if ransResources != nil {
            return submitRANSPlan(plan, token: token)
        }
        if compressedFrames != nil {
            // `publishesFirstStage` is what lets the runner encode phase 1 for
            // the misses while the tails decode: it awaits `firstStageCompletion`
            // and re-encodes the fixup. Without it the whole read-and-decode sat
            // in front of the GPU -- measured as 0% io overlap and ~3 ms a layer.
            // The phase split exists in this format precisely so this can fire.
            let operation = ExpertLoadOperation(
                completionToken: token,
                eventCoordinator: eventCoordinator,
                backendSignalsEvent: false,
                publishesFirstStage: layout.phase1Bytes != nil)
            ExpertIOScheduler.shared.submit { [self, operation] in
                operation.markInFlight()
                do {
                    _ = try executeExpertCachePlan(
                        plan, firstStage: operation.publishFirstStage)
                    operation.finish(.success(()))
                } catch {
                    operation.finish(.failure(error))
                }
            }
            return operation
        }
        if let metalReader {
            if eventDriven {
                return try beginEventDrivenMetalReads(
                    plan, reader: metalReader, token: token)
            }
            let operation = ExpertLoadOperation(
                completionToken: token,
                eventCoordinator: eventCoordinator,
                backendSignalsEvent: false)
            operation.markInFlight()
            let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            do {
                try beginMetalReads(
                    plan,
                    reader: metalReader,
                    // A native MTLIO signal cross-queued with a waiting compute
                    // buffer deadlocked on the qualification M3. Keep Metal I/O
                    // nonblocking, but bridge its completion handler through
                    // the same proven coordinator used by bounded pread.
                    completionToken: nil
                ) { [self, operation] result in
                    switch result {
                    case .success:
                        do {
                            try markPlanMissesResident(plan)
                            finishPlanExecution(
                                plan,
                                succeeded: true,
                                elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                            operation.finish(.success(()))
                        } catch {
                            finishPlanExecution(
                                plan,
                                succeeded: false,
                                elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
                            operation.finish(.failure(error))
                        }
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
                operation.finish(.failure(error))
            }
            return operation
        }
        let operation = ExpertLoadOperation(
            completionToken: token,
            eventCoordinator: eventCoordinator,
            backendSignalsEvent: false,
            publishesFirstStage:
                boundedReader != nil && layout.splitsPhase1)
        ExpertIOScheduler.shared.submit { [self, operation] in
            operation.markInFlight()
            do {
                _ = try executeExpertCachePlan(plan, firstStage: operation.publishFirstStage)
                operation.finish(.success(()))
            } catch {
                operation.finish(.failure(error))
            }
        }
        return operation
    }

    /// Submits a rANS plan: phase 1 is published when the front blocks land, so
    /// the runner can encode gate|up while the tails decode.
    private func submitRANSPlan(
        _ plan: ExpertCachePlan, token: ExpertIOCompletionToken?
    ) -> ExpertLoadOperation {
        let operation = ExpertLoadOperation(
            completionToken: token,
            eventCoordinator: eventCoordinator,
            backendSignalsEvent: false,
            publishesFirstStage: true)
        ExpertIOScheduler.shared.submit { [self, operation] in
            operation.markInFlight()
            do {
                _ = try executeExpertCachePlan(
                    plan, firstStage: operation.publishFirstStage)
                operation.finish(.success(()))
            } catch {
                operation.finish(.failure(error))
            }
        }
        return operation
    }
}
