import Testing

@testable import TinyTitan
import TinyTitanCore

/// The slot default is derived from a RAM budget and the model's own expert
/// stride, so one rule serves every quantisation.
///
/// The budget is 8 GiB because expert reads bypass the page cache, leaving the slot
/// cache as the only cache: a routing trace measured 131 distinct experts per layer
/// over a 128-token window, and 128 slots is the first budget that holds it.
/// Measured bounded, 4-bit, short prompt: 8.73 tok/s at 16 slots against 18.91 at
/// 128 — that curve is a cliff rather than a slope, so an under-budget cache is
/// expensive. The note that claimed the ordering *inverts* under the page-cache
/// reader no longer reproduces and was removed: swept on an 8 GB mini, 8 / 16 / 24 /
/// 32 / 40 slots give 6.319 / 6.769 / 7.395 / 8.138 / 8.552 tok/s, monotonic, so
/// more slots is right under either reader.
@Suite struct ExpertCacheBudgetTests {
    static let layers = 40
    static let stride4 = UInt64(1_769_472)
    static let stride6 = UInt64(2_555_904)
    static let stride8 = UInt64(3_342_336)

    @Test func defaultBudgetLandsOnTheMeasuredOptimumPerQuant() {
        // 4-bit holds the working set at 128 slots (8.44 GiB).
        #expect(
            RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: Self.stride4, layers: Self.layers) == 128)
        // 8-bit cannot: 128 slots would be 15.94 GiB and measured 1.22 tok/s
        // thrashing on a 24 GB machine, so the budget caps it at 64.
        #expect(
            RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: Self.stride8, layers: Self.layers) == 64)
    }

    @Test func everyResultIsASupportedSlotCount() {
        for stride in [Self.stride4, Self.stride6, Self.stride8] {
            for budget in [1 << 28, 1 << 30, 1 << 32, 1 << 34] {
                let slots = RuntimeConfiguration.expertCacheSlots(
                    expertStrideBytes: stride, layers: Self.layers,
                    budgetBytes: budget)
                #expect(
                    RuntimeConfiguration.allowedExpertCacheSlots.contains(slots),
                    "stride \(stride) budget \(budget) gave \(slots)")
            }
        }
    }

    @Test func biggerBudgetNeverShrinksTheCache() {
        var previous = 0
        for budget in [1 << 27, 1 << 28, 1 << 29, 1 << 30, 1 << 31, 1 << 32] {
            let slots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: Self.stride4, layers: Self.layers,
                budgetBytes: budget)
            #expect(slots >= previous, "budget \(budget) went backwards")
            previous = slots
        }
    }

    /// A degenerate manifest must not produce a zero or negative slot count --
    /// that would reach the streamer as an empty cache.
    /// 8-bit at the default budget must stay inside RAM. 128 slots is 15.94 GiB,
    /// which measured 1.22 tok/s against 5.21 at 64 -- a cliff, not a slope.
    @Test func eightBitDefaultStaysBelowTheThrashingPoint() {
        let slots = RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: Self.stride8, layers: Self.layers)
        let bytes = Double(slots) * Double(Self.stride8) * Double(Self.layers)
        #expect(
            bytes / 1_073_741_824 < 12.0,
            "8-bit default would reserve \(bytes / 1_073_741_824) GiB")
    }

    @Test func degenerateInputsFallBackToTheSmallestSupportedCount() {
        #expect(
            RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: 0, layers: Self.layers) == 8)
        #expect(
            RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: Self.stride4, layers: 0) == 8)
    }

    /// The budget must be honoured within one step of the allowed ladder, or the
    /// footprint promise is empty.
    @Test func chosenCountStaysNearTheRequestedBudget() {
        let budget = 1 << 30
        for stride in [Self.stride4, Self.stride8] {
            let slots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: stride, layers: Self.layers, budgetBytes: budget)
            let actual = Double(slots) * Double(stride) * Double(Self.layers)
            #expect(
                actual <= Double(budget) * 1.15,
                "stride \(stride): \(actual / 1_073_741_824) GiB exceeds budget")
        }
    }

    /// `--ram-budget` names the whole process, so the cache is the remainder of
    /// (target - resident floor). These are the qwen38 4-bit geometry and weight
    /// file, i.e. the install the 2/4/6/8/10/12/16 GB sweep was taken on:
    /// 2,768,896 B stride x 48 layers = 0.1238 GiB a slot, weights 3.217 GiB.
    @Test func processTargetsSubtractTheResidentFloor() {
        let stride = UInt64(2_768_896)
        let layers = 48
        let floor = RuntimeConfiguration.residentFloorBytes(
            residentWeightBytes: 3_454_121_216)
        // Weights plus the measured 512 MiB runtime reserve.
        #expect(abs(Double(floor) / 1_073_741_824 - 3.717) < 0.01)

        let expected: [(targetGB: Int, slots: Int)] = [
            (2, 8),  // below the floor: the smallest cache is already too much
            (4, 8),  // still below it
            (6, 16),
            (8, 32),
            (10, 48),
            (12, 64),
            (16, 96),
        ]
        for (targetGB, expectedSlots) in expected {
            let slots = RuntimeConfiguration.expertCacheSlotsFitting(
                expertStrideBytes: stride, layers: layers,
                cacheBytes: (targetGB << 30) - floor)
            #expect(slots == expectedSlots, "\(targetGB)G gave \(slots) slots")
            #expect(RuntimeConfiguration.allowedExpertCacheSlots.contains(slots))
            // The estimate the server prints must not exceed the target, except
            // where the target is under the floor and no cache can be small
            // enough.
            let estimate =
                Double(floor)
                + Double(slots) * Double(stride) * Double(layers)
            if targetGB >= 6 {
                #expect(
                    estimate <= Double(targetGB << 30),
                    "\(targetGB)G estimated \(estimate / 1_073_741_824) GiB")
            }
        }
    }

    /// A target with no room left, or an unreadable model, must still produce a
    /// usable cache rather than zero slots.
    @Test func aTargetWithNoRoomFallsBackToTheSmallestRung() {
        let stride = UInt64(2_768_896)
        #expect(
            RuntimeConfiguration.expertCacheSlotsFitting(
                expertStrideBytes: stride, layers: 48, cacheBytes: -1) == 8)
        #expect(
            RuntimeConfiguration.expertCacheSlotsFitting(
                expertStrideBytes: stride, layers: 48, cacheBytes: 0) == 8)
        #expect(
            RuntimeConfiguration.expertCacheSlotsFitting(
                expertStrideBytes: 0, layers: 0, cacheBytes: 8 << 30) == 8)
    }
}

/// `--ram-budget` is the user-facing knob, so its parser has to accept what people
/// type and reject what they mistype -- a silently misparsed size would produce a
/// tiny cache and a 2.2x throughput loss with no error.
@Suite struct BudgetParsingTests {
    @Test func acceptsTheUnitsUsersType() {
        #expect(RuntimeConfiguration.parseBudgetBytes("8G") == 8 << 30)
        #expect(RuntimeConfiguration.parseBudgetBytes("2g") == 2 << 30)
        #expect(RuntimeConfiguration.parseBudgetBytes("512M") == 512 << 20)
        #expect(RuntimeConfiguration.parseBudgetBytes("8GiB") == 8 << 30)
        #expect(RuntimeConfiguration.parseBudgetBytes("1024KB") == 1024 << 10)
        #expect(RuntimeConfiguration.parseBudgetBytes(" 4G ") == 4 << 30)
        #expect(RuntimeConfiguration.parseBudgetBytes("1073741824") == 1 << 30)
    }

    @Test func acceptsFractionalSizes() {
        #expect(RuntimeConfiguration.parseBudgetBytes("1.5G") == Int(1.5 * Double(1 << 30)))
        #expect(RuntimeConfiguration.parseBudgetBytes("0.5G") == 1 << 29)
    }

    @Test func rejectsWhatWouldSilentlyShrinkTheCache() {
        for bad in ["", " ", "bogus", "G", "-2G", "0", "0G", "abcG", "2X", "2GG"] {
            #expect(
                RuntimeConfiguration.parseBudgetBytes(bad) == nil,
                "\(bad.debugDescription) should not parse")
        }
    }

    /// The knob has to move the outcome, and monotonically.
    @Test func budgetDrivesTheSlotCount() {
        let stride = UInt64(1_769_472)
        let one = RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: stride, layers: 40, budgetBytes: 1 << 30)
        let four = RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: stride, layers: 40, budgetBytes: 4 << 30)
        let eight = RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: stride, layers: 40, budgetBytes: 8 << 30)
        #expect(one == 16)
        #expect(four == 64)
        #expect(eight == 128)
        #expect(one < four && four < eight)
    }
}

/// The per-family decode tuning, and the clamp that keeps it honest on a
/// machine smaller than the one it was measured on.
///
/// Measured 2026-08-30, 512-token prose, fresh process per run, interleaved:
/// qwen38flash 4-bit 5.735 -> 6.957 tok/s with 12 GiB + depth 1; qwen36 8-bit
/// 11.994 -> 12.659 with depth 1; qwen36 4-bit regressed 23.234 -> 22.248, so
/// it ships without prefetch.
@Suite struct DecodeTuningTests {
    static let qwen38Stride = UInt64(2_768_896)
    static let qwen38Layers = 48

    @Test func qwen38FlashGetsTheWiderCacheAndPrefetch() {
        let tuning = RuntimeConfiguration.decodeTuning(
            family: "qwen38flash",
            weightBits: 4)
        #expect(tuning.expertCacheBudgetBytes == 12 << 30)
        #expect(tuning.prefetchDepth == 1)
        // 12 GiB has to actually land on 96 slots for this payload, which is
        // the whole point of the entry.
        #expect(
            RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: Self.qwen38Stride, layers: Self.qwen38Layers,
                budgetBytes: tuning.expertCacheBudgetBytes) == 96)
    }

    @Test func prefetchShipsOnlyWhereItMeasuredFaster() {
        // 8-bit streams twice the bytes, so expert I/O is a large enough share
        // of the token for a speculative read to pay.
        #expect(
            RuntimeConfiguration.decodeTuning(
                family: .qwen36, weightBits: 8
            ).prefetchDepth == 1)
        #expect(
            RuntimeConfiguration.decodeTuning(
                family: .qwen36MTP, weightBits: 8
            ).prefetchDepth == 1)
        // At 4-bit the same family spends ~7 ms of a ~44 ms token on expert
        // I/O; prefetch measured -3.9% (Ornith) and -4.2% (Qwen 3.6).
        #expect(
            RuntimeConfiguration.decodeTuning(
                family: .qwen36, weightBits: 4
            ).prefetchDepth == 0)
        #expect(
            RuntimeConfiguration.decodeTuning(
                family: .qwen36MTP, weightBits: 4
            ).prefetchDepth == 0)
    }

    @Test func the35BFamiliesKeepTheirEstablishedBudget() {
        for bits in [4, 8] {
            #expect(
                RuntimeConfiguration.decodeTuning(
                    family: .qwen36, weightBits: bits
                ).expertCacheBudgetBytes
                    == RuntimeConfiguration.defaultExpertCacheBudgetBytes)
        }
    }

    /// The ceiling is a third of physical memory, floored at the tuned budget:
    /// a machine with at least the 24 GiB the budgets were measured on keeps
    /// `wanted`, and only a smaller machine is cut to a third.
    ///
    /// The third exists for the small machine. A half handed an 8 GB Mac mini
    /// 64 slots - 4.22 GiB of cache at a 70.8 MB slot - and it paged: swap grew
    /// 855 -> 1610 MB and throughput fell to 5.576 tok/s, against a flat swap
    /// and 7.289 tok/s at the 40 slots a third picks. The third is the tuned
    /// value in disguise: the budgets above were measured on a 24 GiB machine,
    /// and a third of 24 GiB is exactly the 8 GiB
    /// `defaultExpertCacheBudgetBytes`.
    ///
    /// The floor exists for the large one. Before the QSA indexer grew on
    /// demand (45f55bb) it pre-allocated 0.94 GiB of full-context key buffers,
    /// and qwen38's tuned 12 GiB - 96 slots - paged at the shipped 262144
    /// context even on the 24 GB machine it was tuned on (3.052/3.201 tok/s
    /// mirrored, against 4.481-4.551 at the clamped 64). With the indexer
    /// grown on demand the same 96 slots fit: 5.156 tok/s at full context,
    /// greedy output byte-identical to the 64-slot run. What the machine above
    /// the tune was paying for the small machine's safety is therefore gone,
    /// and the cut is kept only where it buys something.
    @Test func budgetIsClampedToAFractionOfWhatTheMachineCanHold() {
        let wanted = 12 << 30
        // 32 GiB: at or above the 24 GiB tune, so the 12 GiB aimed at
        // qwen38flash passes through untouched. This assertion used to expect
        // (32<<30)/3; the clamp it encoded predates the QSA grow-on-demand
        // fix, and 45f55bb's measurement is why it changed.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                wanted, physicalMemory: 32 << 30) == 12 << 30)
        // 16 GiB: a third is 5 GiB, which is a harsher cut than the half it
        // replaces (8 GiB) - stated rather than hidden, because it is the case
        // this test used to assert the other way.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                wanted, physicalMemory: 16 << 30) == (16 << 30) / 3)
        // The case that was broken: an 8 GB mini, on the 35B 4-bit geometry.
        // 70.8 MB a slot, so a third is 2.67 GiB and must land on 40 slots -
        // the measured optimum - where a half landed on a higher rung and paged.
        let mini = RuntimeConfiguration.affordableExpertCacheBudget(
            wanted, physicalMemory: 8 << 30)
        #expect(mini == (8 << 30) / 3)
        #expect(
            RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: 1_769_472, layers: 40, budgetBytes: mini) == 40)
        // And the clamp has to change the slot count, not just the number.
        #expect(
            RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: Self.qwen38Stride, layers: Self.qwen38Layers,
                budgetBytes: RuntimeConfiguration.affordableExpertCacheBudget(
                    wanted, physicalMemory: 16 << 30)) == 40)
    }

    /// The floor is inclusive at the 24 GiB the budgets were tuned on, and the
    /// third still applies below it.
    @Test func theFloorStartsAtTheMachineTheBudgetsWereTunedOn() {
        let wanted = 12 << 30
        // 24 GiB: the tuning machine itself keeps the tuned budget.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                wanted, physicalMemory: 24 << 30) == wanted)
        // Just below it the third still cuts, and cuts harder than the floor
        // would: 20 GiB keeps only 6.67 GiB.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                wanted, physicalMemory: 20 << 30) == (20 << 30) / 3)
        // The 35B families are unaffected either way: their 8 GiB budget is a
        // third of 24 GiB exactly, so floor and clamp agree on it.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                8 << 30, physicalMemory: 24 << 30) == 8 << 30)
    }

    /// Below the tune the ceiling only ever cuts: a machine smaller than the one
    /// the budgets were measured on gets a third of what it has, never more.
    ///
    /// This test used to assert "the clamp never grows a budget" on the 8 GiB
    /// case at 128 GiB of RAM. That premise belonged to the floor-only rule
    /// (3ed7102) and `aboveTheTuneTheBudgetGrowsWithTheMachine` now asserts the
    /// opposite there, deliberately: a 35B family's 8 GiB on a 128 GiB Mac is
    /// memory the tune left unused, and the same third that protects the small
    /// machine is what the big one can hold. What stays forbidden is growth
    /// *below* the tune, which is what this now pins.
    @Test func clampNeverGrowsABudgetBelowTheTune() {
        // 16 GiB: a third is 5.33 GiB, so the 8 GiB budget is cut, not grown.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                8 << 30, physicalMemory: 16 << 30) == (16 << 30) / 3)
        // The 8 GB mini, the case that set the rule.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                8 << 30, physicalMemory: 8 << 30) == (8 << 30) / 3)
        // Zero and unreadable RAM hand back the tuned number rather than
        // inventing a larger one.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                8 << 30, physicalMemory: 0) == 8 << 30)
        // At the tune itself a third is 8 GiB, so the floor and the ceiling
        // agree and nothing grows.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                8 << 30, physicalMemory: 24 << 30) == 8 << 30)
    }

    /// Above the tune class the budget climbs with the machine: the same
    /// fraction that protects a small Mac is the headroom a big one paid for.
    /// The third is the LMCache lesson applied at home -- size the resident
    /// tier to the machine's RAM, not to a tune pinned on one machine class --
    /// and it never breaks the half-of-RAM wired-cache line, because a third
    /// of anything is below half of it.
    @Test func aboveTheTuneTheBudgetGrowsWithTheMachine() {
        let wanted = 12 << 30
        // 48 GiB: a third (16 GiB) beats the tuned 12 GiB, so it wins.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                wanted, physicalMemory: 48 << 30) == 16 << 30)
        // 64 GiB: 21.33 GiB. On qwen38 4-bit (129.6 MB a slot) that is 160
        // slots against the 96 the tune pinned it to.
        let sixtyFour = RuntimeConfiguration.affordableExpertCacheBudget(
            wanted, physicalMemory: 64 << 30)
        #expect(sixtyFour == (64 << 30) / 3)
        #expect(
            RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: Self.qwen38Stride, layers: Self.qwen38Layers,
                budgetBytes: sixtyFour) == 160)
        // 128 GiB: a third is 42.67 GiB, past the ladder's 256-slot top rung.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                wanted, physicalMemory: 128 << 30) == (128 << 30) / 3)
        #expect(
            RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: Self.qwen38Stride, layers: Self.qwen38Layers,
                budgetBytes: (128 << 30) / 3) == 256)
        // The tune machine itself is the boundary: max(12 GiB, 8 GiB) is the
        // tuned 12 GiB, so nothing changes there.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                wanted, physicalMemory: 24 << 30) == wanted)
        // 32 GiB sits between: a third is 10.67 GiB, so the 12 GiB tune wins.
        #expect(
            RuntimeConfiguration.affordableExpertCacheBudget(
                wanted, physicalMemory: 32 << 30) == 12 << 30)
        // And the growth stays inside the wired-cache half: a third never
        // exceeds half of the same machine.
        for physical in [24 << 30, 48 << 30, 64 << 30, 128 << 30] {
            let budget = RuntimeConfiguration.affordableExpertCacheBudget(
                wanted, physicalMemory: UInt64(physical))
            #expect(
                budget <= physical / 2,
                "\(physical >> 30) GiB machine: \(budget) exceeds the half")
        }
    }
}
