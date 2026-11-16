import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// Tests `GDNStateManager`'s per-slot layout. The delta-rule state and conv
/// tail are fixed-size per sequence, so batching multiplies them by `slots` and
/// puts each sequence's copy in its own region; at one slot every region is at
/// offset 0 exactly as before.
@Suite struct GDNStateManagerTests {

    private let config = ArchConfig.qwen36_35B_A3B

    private func makeManager(slots: Int) throws -> (MetalContext, GDNStateManager) {
        let ctx = try MetalContext()
        let gdn = try GDNStateManager(device: ctx.device, config: config, slots: slots)
        return (ctx, gdn)
    }

    /// A linear layer to address; the config interleaves them with KV layers.
    private func linearLayer(_ gdn: GDNStateManager) -> Int? {
        (0..<config.numLayers).first(where: { gdn.isLinear(layer: $0) })
    }

    /// One slot is exactly the pre-slot layout: region base 0 and buffers sized
    /// for a single sequence.
    @Test func oneSlotKeepsTheOriginalLayout() throws {
        let (_, gdn) = try makeManager(slots: 1)
        guard let layer = linearLayer(gdn) else {
            Issue.record("config has no linear-attention layer")
            return
        }
        #expect(gdn.slots == 1)
        #expect(gdn.stateOffset(layer: layer, slot: 0) == 0)
        #expect(gdn.convTailOffset(layer: layer, slot: 0) == 0)
        #expect(gdn.stateBuffer(layer: layer).length == gdn.stateBytesPerLayer)
        #expect(gdn.convTailBuffer(layer: layer).length == gdn.convTailBytesPerLayer)
    }

    /// Each slot owns a disjoint region, one per-slot size apart.
    @Test func slotsGetDisjointRegions() throws {
        let (_, gdn) = try makeManager(slots: 3)
        guard let layer = linearLayer(gdn) else {
            Issue.record("config has no linear-attention layer")
            return
        }
        #expect(gdn.stateBuffer(layer: layer).length == 3 * gdn.stateBytesPerLayer)
        #expect(gdn.convTailBuffer(layer: layer).length == 3 * gdn.convTailBytesPerLayer)
        for slot in 0..<3 {
            #expect(
                gdn.stateOffset(layer: layer, slot: slot)
                    == slot * gdn.stateBytesPerLayer)
            #expect(
                gdn.convTailOffset(layer: layer, slot: slot)
                    == slot * gdn.convTailBytesPerLayer)
            let state = gdn.stateSlot(layer: layer, slot: slot)
            let tail = gdn.convTailSlot(layer: layer, slot: slot)
            #expect(state.buffer === gdn.stateBuffer(layer: layer))
            #expect(state.offset == slot * gdn.stateBytesPerLayer)
            #expect(tail.offset == slot * gdn.convTailBytesPerLayer)
        }
    }

    /// Resetting one slot zeroes only that slot's state and tail.
    @Test func resettingOneSlotClearsOnlyThatSlot() throws {
        let (_, gdn) = try makeManager(slots: 2)
        guard let layer = linearLayer(gdn) else {
            Issue.record("config has no linear-attention layer")
            return
        }
        let state = gdn.stateBuffer(layer: layer)
        let tail = gdn.convTailBuffer(layer: layer)
        memset(state.contents(), 0xFF, state.length)
        memset(tail.contents(), 0xFF, tail.length)

        gdn.reset(slot: 1)

        let statePtr = state.contents().bindMemory(to: UInt8.self, capacity: state.length)
        for i in 0..<gdn.stateBytesPerLayer {
            #expect(statePtr[i] == 0xFF, "slot 0 state byte \(i) was cleared")
            #expect(
                statePtr[gdn.stateBytesPerLayer + i] == 0,
                "slot 1 state byte \(i) survived the reset")
        }
        let tailPtr = tail.contents().bindMemory(to: UInt8.self, capacity: tail.length)
        for i in 0..<gdn.convTailBytesPerLayer {
            #expect(tailPtr[i] == 0xFF, "slot 0 tail byte \(i) was cleared")
            #expect(
                tailPtr[gdn.convTailBytesPerLayer + i] == 0,
                "slot 1 tail byte \(i) survived the reset")
        }
    }
}
