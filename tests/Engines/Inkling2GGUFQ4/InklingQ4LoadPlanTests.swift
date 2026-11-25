import Foundation
import Testing

@testable import Inkling2GGUFQ4

/// How `--ram-budget` becomes a context window and expert slots (TT-121
/// phase 2), on the Q4 install's own geometry: 1,820 MB resident, one slot in
/// every layer 123.7 MB (20 NVFP4 layers at 4,489,216 bytes, 4 Q8_0 layers at
/// 8,470,528), 32 KiB of KV a position, 128 experts, top-8, a 1M window.
struct InklingQ4LoadPlanTests {
    static let geometry = InklingQ4LoadPlan.Geometry(
        residentBytes: 1_819_979_776, bytesPerSlot: 20 * 4_489_216 + 4 * 8_470_528,
        bytesPerPosition: 32_768, experts: 128, expertsPerToken: 8, modelContext: 1_048_576)
    static let workingSet = 17 << 30
    static let gib = 1 << 30

    @Test func fourGigabytesAtAFourKWindowHoldsFourteenSlots() throws {
        let plan = try InklingQ4LoadPlan.make(
            geometry: Self.geometry, requested: 4_096, ramBudget: 4 * Self.gib,
            explicitSlots: nil, workingSet: Self.workingSet)
        // 4,294,967,296 - 1,819,979,776 - 536,870,912 - 4,096 x 32,768 =
        // 1,803,894,272 bytes for slots; / 123,666,432 = 14.
        #expect(plan.context == 4_096)
        #expect(plan.slots == 14)
        #expect(plan.plannedBytes <= 4 * Self.gib)
    }

    @Test func aLargeWindowTakesTheBudgetBeforeSlotsAboveTheMinimum() throws {
        let plan = try InklingQ4LoadPlan.make(
            geometry: Self.geometry, requested: 32_768, ramBudget: 4 * Self.gib,
            explicitSlots: nil, workingSet: Self.workingSet)
        // After the weights, reserve and 8 minimum slots, 948,363,552 bytes
        // hold 28,941 positions: 28,672 on the 1,024 grid.
        #expect(plan.context == 28_672)
        #expect(plan.slots == 8)
        #expect(plan.plannedBytes <= 4 * Self.gib)
    }

    @Test func noBudgetUsesTheDefaultExpertMemory() throws {
        let plan = try InklingQ4LoadPlan.make(
            geometry: Self.geometry, requested: 32_768, ramBudget: nil, explicitSlots: nil,
            workingSet: Self.workingSet)
        #expect(plan.context == 32_768)
        // 8 GiB / 123,666,432 bytes a slot.
        #expect(plan.slots == 69)
    }

    @Test func explicitSlotsWinButStayWithinTheExpertCount() throws {
        let plan = try InklingQ4LoadPlan.make(
            geometry: Self.geometry, requested: 4_096, ramBudget: nil, explicitSlots: 512,
            workingSet: Self.workingSet)
        #expect(plan.slots == 128)
    }

    @Test func aBudgetBelowTheMinimumIsRefusedWithTheMinimum() {
        #expect(throws: InklingQ4LoadPlan.Error.self) {
            _ = try InklingQ4LoadPlan.make(
                geometry: Self.geometry, requested: 4_096, ramBudget: 3 * Self.gib,
                explicitSlots: nil, workingSet: Self.workingSet)
        }
    }

    @Test func anImplicitWindowUnderABudgetCapsAtThirtyTwoKForSlots() throws {
        let plan = try InklingQ4LoadPlan.make(
            geometry: Self.geometry, requested: 262_144, ramBudget: 12 * Self.gib,
            explicitSlots: nil, workingSet: Self.workingSet, maxContextExplicit: false)
        #expect(plan.context == InklingQ4LoadPlan.defaultBudgetedContext)
        #expect(plan.slots > 60)
    }

    @Test func perSlotBytesReflectsWindowKVAndFixedState() throws {
        let geometry = InklingQ4LoadPlan.Geometry(
            residentBytes: Self.geometry.residentBytes, bytesPerSlot: Self.geometry.bytesPerSlot,
            bytesPerPosition: 32_768, fixedSlotBytes: 120_000_000,
            experts: 128, expertsPerToken: 8, modelContext: 1_048_576)
        let plan = try InklingQ4LoadPlan.make(
            geometry: geometry, requested: 32_768, ramBudget: 12 * Self.gib,
            explicitSlots: nil, workingSet: Self.workingSet)
        #expect(plan.perSlotBytes == 32_768 * 32_768 + 120_000_000)
    }
}
