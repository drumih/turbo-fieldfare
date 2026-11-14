import Foundation
import Metal
import Testing
import TinyTitan

@testable import TinyTitanLib
@testable import TinyTitanServerCore
import TinyTitanCore

/// The batched-serving headroom rule: a worst-case per-slot size computed from
/// the architecture, and a width clamped so the slots fit beside the wired
/// expert cache.
@Suite struct BatchedMemoryBudgetTests {

    private let config = ArchConfig.qwen36_35B_A3B

    /// The formula must agree with what the manager allocates. A manager whose
    /// capacity is the whole context is the worst case, so its buffer lengths
    /// are the number to match rather than a second copy of the same arithmetic.
    @Test func kvWorstCaseMatchesTheAllocatedLayout() throws {
        let ctx = try MetalContext()
        let maxContext = 128
        let slots = 3
        let kv = try KVCacheManager(
            device: ctx.device, config: config,
            maxContext: maxContext, slots: slots,
            fp16RingEnabled: false,
            slidingWindow: config.slidingWindow,
            maxPrefillChunkTokens: 128)
        var allocated = 0
        for layer in 0..<config.numLayers where kv.layerKind(layer) != .linear {
            allocated += kv.bufferLength(layer: layer) * 2  // K and V
        }
        let predicted = KVCacheManager.worstCaseBytes(
            config: config, maxContext: maxContext, precision: .fp16,
            slots: slots, fp16RingEnabled: false,
            slidingWindow: config.slidingWindow, maxPrefillChunkTokens: 128)
        #expect(predicted == allocated)
    }

    @Test func gdnWorstCaseMatchesTheAllocatedLayout() throws {
        let ctx = try MetalContext()
        let slots = 3
        let gdn = try GDNStateManager(device: ctx.device, config: config, slots: slots)
        var allocated = 0
        for layer in 0..<config.numLayers where gdn.isLinear(layer: layer) {
            allocated +=
                gdn.stateBuffer(layer: layer).length
                + gdn.convTailBuffer(layer: layer).length
        }
        #expect(GDNStateManager.worstCaseBytes(config: config, slots: slots) == allocated)
    }

    @Test func widthIsClampedToTheBudgetButNeverBelowOne() {
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 4, perSlotBytes: 100, budgetBytes: 300) == 3)
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 4, perSlotBytes: 100, budgetBytes: 1_000) == 4)
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 4, perSlotBytes: 100, budgetBytes: 0) == 1,
            "a model too large for a second slot still serves one")
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 1, perSlotBytes: 100, budgetBytes: 0) == 1)
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 4, perSlotBytes: 0, budgetBytes: 1_000) == 4,
            "an unknown size must not silently pin the width to one")
    }

    @Test func budgetLeavesTheWiredExpertCacheOutOfTheKVHeadroom() {
        let physical: UInt64 = 24 * 1_073_741_824
        #expect(
            BatchedMemoryBudget.slotBudgetBytes(
                physicalMemory: physical, expertCacheBudgetBytes: 0) == Int(physical / 2))
        #expect(
            BatchedMemoryBudget.slotBudgetBytes(
                physicalMemory: physical, expertCacheBudgetBytes: 4 * 1_073_741_824)
                == Int(physical / 2) - 4 * 1_073_741_824)
        #expect(
            BatchedMemoryBudget.slotBudgetBytes(
                physicalMemory: physical, expertCacheBudgetBytes: Int(physical)) == 0)
    }

    /// A dense family allocates no expert cache, so its profile's budget for one
    /// must not be taken off the KV headroom. A MoE family's must.
    @Test func aDenseModelHoldsBackNoExpertCache() {
        #expect(
            BatchedMemoryBudget.expertCacheHeldBack(numExperts: 0, configured: 8 * 1_073_741_824)
                == 0)
        #expect(
            BatchedMemoryBudget.expertCacheHeldBack(numExperts: 512, configured: 8 * 1_073_741_824)
                == 8 * 1_073_741_824)
        #expect(BatchedMemoryBudget.expertCacheHeldBack(numExperts: 512, configured: -1) == 0)
    }

    /// A shipped install at the advertised context must fit: the clamp exists
    /// for the models that do not, not to shrink the common case.
    @Test func aShippedModelFitsItsWorstCaseAtTheAdvertisedContext() {
        let perSlot = BatchedMemoryBudget.perSlotBytes(
            config: config, maxContext: 262_144, precision: .int8,
            fp16RingEnabled: false, slidingWindow: config.slidingWindow,
            maxPrefillChunkTokens: 128, vocab: config.vocabSize)
        #expect(perSlot > 0)
        let budget = BatchedMemoryBudget.slotBudgetBytes(
            physicalMemory: 24 * 1_073_741_824,
            expertCacheBudgetBytes: 4 * 1_073_741_824)
        #expect(
            BatchedMemoryBudget.effectiveSlots(
                requested: 4, perSlotBytes: perSlot, budgetBytes: budget) >= 1)
    }

    /// The host tier trades VRAM for host RAM, not for nothing: a bounded
    /// server must count the window and the arena together.
    @Test func hostKVCountsTheWindowAndTheArena() {
        let maxContext = 262_144
        let window = 65_536
        let streamed = BatchedMemoryBudget.perSlotBytes(
            config: config, maxContext: maxContext, precision: .int8,
            fp16RingEnabled: false, slidingWindow: config.slidingWindow,
            maxPrefillChunkTokens: 128, vocab: config.vocabSize,
            hostKVWindowTokens: window)
        let metal = KVCacheManager.worstCaseBytes(
            config: config, maxContext: maxContext, precision: .int8,
            fp16RingEnabled: false, slidingWindow: config.slidingWindow,
            maxPrefillChunkTokens: 128, windowTokens: window)
        let host = KVCacheManager.worstCaseHostArenaBytes(
            config: config, maxContext: maxContext, precision: .int8)
        #expect(host > 0)
        #expect(streamed > metal)
        #expect(
            streamed >= metal + host,
            "the per-slot budget must include the authoritative arena")
    }

    @Test func aPerSequenceStateModelIsServedOneAtATime() {
        #expect(
            BatchedMemoryBudget.effectiveRequestedSlots(
                requested: 4, hasPerSequenceState: true) == 1,
            "QSA/PLE state is not slot-keyed; two sequences would cross-talk")
        #expect(
            BatchedMemoryBudget.effectiveRequestedSlots(
                requested: 4, hasPerSequenceState: false) == 4)
        #expect(
            BatchedMemoryBudget.effectiveRequestedSlots(
                requested: 1, hasPerSequenceState: true) == 1)
    }
}
