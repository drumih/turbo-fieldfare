import Darwin
import Foundation
import Metal
import Testing
@testable import TinyTitanCore

@testable import TinyTitan

extension PreadExpertStreamerTests {
    @Test func expertIOBackendEnvironmentDefaultsAndFailsClosed() throws {
        #expect(try ExpertIOBackend.environmentValue([:]) == .pread)
        #expect(
            try ExpertIOBackend.environmentValue([
                "TINYTITAN_EXPERT_IO_BACKEND": "metal"
            ]) == .metal)
        #expect(throws: (any Error).self) {
            try ExpertIOBackend.environmentValue([
                "TINYTITAN_EXPERT_IO_BACKEND": "unknown"
            ])
        }
    }

    @Test func cachedBatchWithoutExecutorLoadsTaggedBytes() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

        let results = try streamer.loadExpertsCached(experts: [3, 1, 2])
        for (index, result) in results.enumerated() {
            let expert = [3, 1, 2][index]
            let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
            #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
        }
    }

    @Test func adviseExpertsDoesNotChangeLoadedBytes() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
        let experts = [0, 2, 3]

        let advice = streamer.adviseExperts(experts: experts)
        #expect(advice.requested == experts.count)
        #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
            #expect(advice.failed == 0)
        #else
            #expect(advice.failed == experts.count)
        #endif

        let results = try streamer.loadExpertsCached(experts: experts)
        for (index, result) in results.enumerated() {
            let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
            #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
        }
    }

    @Test func adviseExpertMissesSkipsResidentSlots() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

        _ = try streamer.loadExpertsCached(experts: [0])
        let advice = streamer.adviseExpertMisses(experts: [0, 1, 2])

        #expect(advice.requested == 2)
        #expect(advice.calls == 1)
        #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
            #expect(advice.failed == 0)
        #else
            #expect(advice.failed == 1)
        #endif
    }

    @Test func plannedCacheLoadExecutesSameMisses() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

        _ = try streamer.loadExpertsCached(experts: [0])
        let experts = [0, 1, 2]
        let plan = try streamer.planExpertsCached(experts: experts)

        #expect(plan.hits == 1)
        #expect(plan.misses.map { experts[$0] } == [1, 2])

        let results = try streamer.executeExpertCachePlan(plan)
        for (index, result) in results.enumerated() {
            let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
            #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
        }
    }

    @Test func residentSnapshotExcludesLoadingEntries() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

        _ = try streamer.loadExpertsCached(experts: [1, 3])
        #expect(streamer.residentExperts() == [1, 3])

        let plan = try streamer.planExpertsCached(experts: [2])
        #expect(plan.misses == [0])
        #expect(streamer.residentExperts() == [1, 3])

        _ = try streamer.executeExpertCachePlan(plan)
        #expect(streamer.residentExperts() == [1, 2, 3])
    }

    @Test func submittedCacheLoadCompletesWithoutCallerExecutingReads() async throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let context = try MetalContext()
        let coordinator = try #require(ExpertIOEventCoordinator(device: context.device))
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: context.device,
            slotCount: 4, eventCoordinator: coordinator)
        let plan = try streamer.planExpertsCached(experts: [3, 1, 2])

        let operation = try streamer.beginExpertCachePlan(plan, eventDriven: true)
        try await operation.completion()

        #expect(operation.state == .completed)
        let token = try #require(operation.completionToken)
        #expect(token.event.signaledValue >= token.value)
        for (index, result) in streamer.expertCachePlanBuffers(plan).enumerated() {
            let got = Self.bytes(
                of: result.buffer, offset: result.offset,
                count: Self.expertStride)
            #expect(got.allSatisfy { $0 == Self.tagByte(plan.experts[index]) })
        }
    }

    @Test func plannedCacheBuffersExposeReservedSlotsBeforeExecute() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

        _ = try streamer.loadExpertsCached(experts: [0])
        let experts = [0, 1, 2]
        let plan = try streamer.planExpertsCached(experts: experts)
        let reserved = streamer.expertCachePlanBuffers(plan)

        let hitBytes = Self.bytes(of: reserved[0].buffer, offset: 0, count: Self.expertStride)
        #expect(hitBytes.allSatisfy { $0 == Self.tagByte(0) })

        let executed = try streamer.executeExpertCachePlan(plan)
        for i in 0..<experts.count {
            #expect(reserved[i].buffer === executed[i].buffer)
            let got = Self.bytes(of: executed[i].buffer, offset: 0, count: Self.expertStride)
            #expect(got.allSatisfy { $0 == Self.tagByte(experts[i]) })
        }
    }

    @Test func plannedCacheAvoidsInFlightSlotsForHitsAndMisses() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

        let warmed = try streamer.loadExpertsCached(experts: [0, 1])
        let plan = try streamer.planExpertsCached(
            experts: [0, 2],
            avoidingSlots: [0, 1])

        #expect(plan.assignedSlots == [0, 2])
        #expect(plan.hits == 1)
        #expect(plan.misses == [1])

        let executed = try streamer.executeExpertCachePlan(plan)
        for (index, expert) in plan.experts.enumerated() {
            let got = Self.bytes(of: executed[index].buffer, offset: 0, count: Self.expertStride)
            #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
        }

        let avoidedBytes = Self.bytes(of: warmed[0].buffer, offset: 0, count: Self.expertStride)
        #expect(avoidedBytes.allSatisfy { $0 == Self.tagByte(0) })
    }

    @Test func plannedCacheReturnsNilWhenMissesCannotAvoidInFlightSlots() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

        _ = try streamer.loadExpertsCached(experts: [0, 1])
        let plan = streamer.planExpertsCachedIfPossible(
            experts: [0, 2, 3, 4],
            avoidingSlots: [0, 1])

        #expect(plan == nil)
    }

    @Test func pinnedGenerationsCannotBeEvictedUntilReleased() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

        _ = try streamer.loadExpertsCached(experts: [0, 1])
        let residentPlan = try streamer.planExpertsCached(experts: [0, 1])
        let lease = try streamer.pin(residentPlan)

        #expect(streamer.statistics().pinnedSlots == 2)
        #expect(streamer.planExpertsCachedIfPossible(experts: [2]) == nil)

        lease.release()
        #expect(streamer.statistics().pinnedSlots == 0)
        let replacement = try streamer.planExpertsCached(experts: [2])
        _ = try streamer.executeExpertCachePlan(replacement)
        #expect(streamer.statistics().residentSlots == 2)
    }

    @Test func streamingStatisticsCountHitsMissesReloadsAndBoundLatency() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

        _ = try streamer.loadExpertsCached(experts: [0, 1])
        _ = try streamer.loadExpertsCached(experts: [0, 2])
        _ = try streamer.loadExpertsCached(experts: [0, 1])
        let statistics = streamer.statistics()

        #expect(statistics.plans == 3)
        #expect(statistics.requestedExperts == 6)
        #expect(statistics.hits == 2)
        #expect(statistics.misses == 4)
        #expect(statistics.readOperations == 4)
        #expect(statistics.bytesRead == 4 * UInt64(Self.expertStride))
        #expect(statistics.evictions == 2)
        #expect(statistics.reloads == 1)
        #expect(statistics.loadBatches == 3)
        #expect(statistics.latencyHistogram.reduce(0, +) == 3)
        #expect(statistics.loadLatencyPercentile(0.5) > 0)
        #expect(statistics.loadingSlots == 0)
    }

    @Test func failedPlannedReadReleasesLoadingSlot() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)
        let plan = try streamer.planExpertsCached(experts: [3])

        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: Self.streamOffset + UInt64(Self.expertStride))
        try handle.close()
        #expect(throws: (any Error).self) {
            _ = try streamer.executeExpertCachePlan(plan)
        }
        #expect(streamer.statistics().loadingSlots == 0)
    }

    @Test func concurrentPinnedPlansNeverExposeOverwrittenBytes() async throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

        try await withThrowingTaskGroup(of: Void.self) { group in
            for worker in 0..<2 {
                group.addTask {
                    for iteration in 0..<50 {
                        let expert = (worker * 2 + iteration) % Self.numExperts
                        let plan = try streamer.planExpertsCached(experts: [expert])
                        let lease = try streamer.pin(plan)
                        defer { lease.release() }
                        let result = try streamer.executeExpertCachePlan(plan)[0]
                        let bytes = Self.bytes(
                            of: result.buffer, offset: result.offset, count: Self.expertStride)
                        guard bytes.allSatisfy({ $0 == Self.tagByte(expert) }) else {
                            throw ModelError.internalInconsistency(
                                detail: "pinned expert slot was overwritten")
                        }
                    }
                }
            }
            try await group.waitForAll()
        }
        #expect(streamer.statistics().pinnedSlots == 0)
        #expect(streamer.statistics().loadingSlots == 0)
    }

    /// The prefill-to-decode transition. Under LFU a prefill's leftovers carry
    /// counts no decode expert can match, so decode evicts its own working set
    /// first; zeroing the counts lets the leftovers go first instead.
    @Test func resetExpertUseCountsLetsDecodeEvictPrefillLeftovers() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device

        func run(reset: Bool) throws -> (afterThird: [Int], reloads: UInt64) {
            let streamer = try PreadExpertStreamer(
                layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)
            // "Prefill": experts 0 and 1 planned three times each.
            for _ in 0..<3 { _ = try streamer.loadExpertsCached(experts: [0, 1]) }
            if reset { streamer.resetExpertUseCounts() }
            // "Decode": 2 evicts one leftover; 3 must evict the other, not 2.
            _ = try streamer.loadExpertsCached(experts: [2])
            _ = try streamer.loadExpertsCached(experts: [3])
            let afterThird = streamer.residentExperts().sorted()
            _ = try streamer.loadExpertsCached(experts: [2])
            return (afterThird, streamer.statistics().reloads)
        }

        let polluted = try run(reset: false)
        #expect(
            !polluted.afterThird.contains(2),
            "LFU pollution: 3 evicted 2 (count 1) over a leftover (count 3)")
        #expect(polluted.reloads == 1)

        let cleared = try run(reset: true)
        #expect(cleared.afterThird == [2, 3])
        #expect(cleared.reloads == 0)
    }

    /// A saturating cap clamps only the eviction *price*, never the accumulated
    /// count. Two residents whose counts both exceed the cap tie, and recency
    /// then decides -- so a hot expert that decode has stopped using hands its slot
    /// back instead of defending it with everything it did since the model loaded.
    ///
    /// Expert 0 is used five times and then left alone; expert 1 is used three
    /// times, most recently. Their counts order one way (1 < 0) and their last-use
    /// orders the other (0 is the older slot), which is the only shape in which a
    /// cap can change a victim: uncapped LFU evicts expert 1, a cap of 3 evicts
    /// expert 0. A brand-new expert always prices at 1 and always loses, capped or
    /// not, so it cannot be the fixture.
    @Test func saturatingUseCountCapPricesRecentDemandWithoutClampingTheCount() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device

        func run(cap: Int?) throws -> (resident: [Int], useCountOfHotterExpert: Int) {
            let streamer = try PreadExpertStreamer(
                layout: Self.makeLayout(path: url.path), device: device, slotCount: 2,
                expertUseCountCap: cap)
            for _ in 0..<5 { _ = try streamer.loadExpertsCached(experts: [0]) }
            for _ in 0..<3 { _ = try streamer.loadExpertsCached(experts: [1]) }
            _ = try streamer.loadExpertsCached(experts: [2])
            return (streamer.residentExperts().sorted(), streamer.expertUseCount[0])
        }

        // Shipped LFU: 5 vs 3 counts, so the recent-but-cooler expert 1 is the victim.
        let uncapped = try run(cap: nil)
        #expect(uncapped.resident == [0, 2])

        // Both residents now price at the cap, the ranks merge, and the tie falls to
        // last use -- the stale hot expert 0 gives up the slot. The count itself is
        // untouched, which is what keeps this a price and not a rescaling (§5y (6)).
        let capped = try run(cap: 3)
        #expect(capped.resident == [1, 2])
        #expect(capped.useCountOfHotterExpert == 5)
    }

    /// The cap has to clamp the count on *both* sides of the comparator, which a
    /// victim-set assertion cannot show: the planner sorts two candidates, and a
    /// two-element sort makes one comparison in one direction, so a one-sided
    /// clamp leaves the eviction outcome unchanged while leaving the ordering
    /// relation inconsistent. Probing `shouldEvictSlot` in both directions pins
    /// the relation itself.
    ///
    /// Same state as the test above and no eviction yet: expert 0 planned five
    /// times, expert 1 planned three times, both resident.
    @Test func useCountCapMergesRanksInBothComparatorDirections() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device

        func build(cap: Int?) throws -> PreadExpertStreamer {
            let streamer = try PreadExpertStreamer(
                layout: Self.makeLayout(path: url.path),
                device: device,
                slotCount: 2,
                expertUseCountCap: cap)
            for _ in 0..<5 { _ = try streamer.loadExpertsCached(experts: [0]) }
            for _ in 0..<3 { _ = try streamer.loadExpertsCached(experts: [1]) }
            return streamer
        }

        // Shipped LFU: slot 1's expert has the lower count, in either direction.
        let uncapped = try build(cap: nil)
        #expect(!uncapped.shouldEvictSlot(0, before: 1))
        #expect(uncapped.shouldEvictSlot(1, before: 0))

        // Capped, both price the same, so recency decides in either direction:
        // slot 0 is the stale one and goes first.
        let capped = try build(cap: 3)
        #expect(capped.shouldEvictSlot(0, before: 1))
        #expect(!capped.shouldEvictSlot(1, before: 0))
    }

}
