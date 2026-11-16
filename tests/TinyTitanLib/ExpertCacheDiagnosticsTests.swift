import Foundation
import Testing

@testable import TinyTitanLib
@testable import TinyTitanCore

/// The `ram profile` / `ram target` lines are the operator's only record of what
/// the engine allocated, so they must name the rung in effect rather than the
/// rung the derivation preferred.
///
/// Geometry used throughout: 0.125 GiB per slot, 3.00 GiB resident floor, so 8
/// slots is `cache=1.00G` and 16 is `cache=2.00G`.
@Suite
struct ExpertCacheDiagnosticsTests {
    private static let slotBytes = 134_217_728.0
    private static let residentFloor = 3_221_225_472

    @Test func profileLineNamesTheAppliedSlotsWhenAnOverrideIsInEffect() {
        let line = ExpertCacheDiagnostics.profileLine(
            appliedSlots: 8,
            derivedSlots: 16,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(
            line == """
                TinyTitan ram profile cache=1.00G slots=8 derived_slots=16 \
                resident_floor=3.00G estimate=4.00G \
                (cache budget, not a process target)
                """)
    }

    @Test func profileLineIsUnchangedWhenNothingOverridesTheDerivation() {
        let line = ExpertCacheDiagnostics.profileLine(
            appliedSlots: 16,
            derivedSlots: 16,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(
            line == """
                TinyTitan ram profile cache=2.00G slots=16 resident_floor=3.00G \
                estimate=5.00G (cache budget, not a process target)
                """)
    }

    @Test func targetLineNamesTheAppliedSlotsWhenAnOverrideIsInEffect() {
        let lines = ExpertCacheDiagnostics.targetLines(
            targetBytes: 6_442_450_944,
            appliedSlots: 8,
            derivedSlots: 16,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(
            lines == [
                """
                TinyTitan ram target=6.00G cache=1.00G slots=8 derived_slots=16 \
                resident_floor=3.00G estimate=4.00G
                """
            ])
    }

    @Test func targetLineWarnsWhenEvenTheSmallestCacheExceedsTheNamedTarget() {
        let lines = ExpertCacheDiagnostics.targetLines(
            targetBytes: 3_758_096_384,
            appliedSlots: 8,
            derivedSlots: 8,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(lines.count == 2)
        #expect(
            lines[1] == """
                TinyTitan ram warning: 3.50G is below this install's 4.00G floor \
                (3.00G resident + the 8-slot minimum cache); \
                the cache is already at its smallest.
                """)
    }

    /// The `--ram 12G --expert-cache-slots 96` shape measured in §5r: the
    /// override pins more than the target affords, so the gap is the operator's
    /// own number and "the cache is already at its smallest" would be false.
    @Test func targetWarningBlamesTheOverrideWhenItPinsTheCacheAboveTheTarget() {
        let lines = ExpertCacheDiagnostics.targetLines(
            targetBytes: 6_442_450_944,
            appliedSlots: 64,
            derivedSlots: 24,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(lines.count == 2)
        #expect(
            lines[1] == """
                TinyTitan ram warning: 6.00G is below this install's 11.00G footprint \
                (3.00G resident + the 64-slot cache named by --expert-cache-slots); \
                the target alone allows 24 slots, so the gap is the override, not the install.
                """)
    }

    @Test func noWarningWhenTheAppliedFootprintFitsTheNamedTarget() {
        let lines = ExpertCacheDiagnostics.targetLines(
            targetBytes: 6_442_450_944,
            appliedSlots: 16,
            derivedSlots: 16,
            cacheBytesPerSlot: Self.slotBytes,
            residentFloorBytes: Self.residentFloor)

        #expect(lines.count == 1)
        #expect(!lines[0].contains("derived_slots"))
    }

    /// `benchmark/expert_cache_slots.py:48` parses this exact shape out of the
    /// CLI's stderr, and `docs/plan-qwen38-decode-12gb.md:238` names the flag
    /// that turns it on. Nothing emitted it after 0cdba76 removed the CLI's
    /// copy, so the sweep silently measured nothing.
    @Test func decodeIOLineKeepsTheShapeTheBenchmarkParses() {
        let line = ExpertCacheDiagnostics.decodeIOLine(
            hits: 900, misses: 100, bytes: 1_288_490_189, tokens: 400)

        #expect(
            line == "[decode expert io] hits 900 misses 100 (90.0% hit) 1.20 GiB"
                + " = 3.1 MiB/token")
    }

    /// A request that decoded no tokens has no per-token figure to report, and
    /// an empty counter set must print 0.0% rather than a NaN.
    @Test func decodeIOLineOmitsThePerTokenFigureWithNoTokens() {
        let line = ExpertCacheDiagnostics.decodeIOLine(
            hits: 0, misses: 0, bytes: 0, tokens: 0)

        #expect(line == "[decode expert io] hits 0 misses 0 (0.0% hit) 0.00 GiB")
    }
}
