import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// Tests `KVCacheManager` FP16 shape, growth, separate K/V storage, ring,
/// and reset semantics against the Qwen 3.6 config. Qwen has no
/// sliding-window layers (mask values are 1 = full and 2 = linear), so the
/// FP16 ring never engages and linear layers carry no per-token K/V rows.
@Suite struct KVCacheManagerTests {

    private let config = ArchConfig.qwen36_35B_A3B

    private func makeManager(
        maxContext: Int,
        slots: Int = 1,
        fp16RingEnabled: Bool = false
    ) throws -> (MetalContext, KVCacheManager) {
        let ctx = try MetalContext()
        let kv = try KVCacheManager(
            device: ctx.device,
            config: config,
            maxContext: maxContext,
            slots: slots,
            fp16RingEnabled: fp16RingEnabled,
            slidingWindow: config.slidingWindow,
            maxPrefillChunkTokens: 128)
        return (ctx, kv)
    }

    /// A prefix longer than the initial capacity restores into a fresh manager.
    ///
    /// `snapshotSegmentLengths` records `min(position, capacity)`, so the lengths
    /// are a function of capacity — and a fresh receiver's full-attention layers
    /// start at `initialCapacityTokens` (8192). Before the restore grew to the
    /// snapshot's position, any prefix past 8192 produced different lengths on
    /// the two sides and was refused as `invalidLayout`, so the disk tier could
    /// not restore the long conversations it exists for.
    ///
    /// The assertion is on the *growth*, not on a completed restore: with an empty
    /// payload the copy fails either way (and both failures are the same
    /// `invalidLayout` case, so a test cannot tell them apart), while the capacity
    /// moving to the snapshot's position is exactly what the fix does and exactly
    /// what was missing.
    @Test func aSnapshotPastTheInitialCapacityGrowsTheReceiver() throws {
        let target = 8_500
        let (_, saver) = try makeManager(maxContext: 9_000)
        // Mirrors the runner: reserve before advancing (`advance` never grows).
        try saver.reserve(tokens: target)
        saver.advance(by: target)
        let lengths = try saver.snapshotSegmentLengths(at: target)

        let (_, receiver) = try makeManager(maxContext: 9_000)
        #expect(
            receiver.capacity(layer: 3) == KVCacheManager.initialCapacityTokens,
            "precondition: a fresh manager starts at the initial capacity")
        let empty = Data()
        var offset = 0
        // The copy is expected to fail on the empty payload; what matters is that
        // the length check ran against a capacity grown to the snapshot.
        _ = try? empty.withUnsafeBytes { bytes in
            try receiver.restoreSnapshot(
                position: target, segmentLengths: lengths,
                bytes: bytes, offset: &offset)
        }
        #expect(
            receiver.capacity(layer: 3) >= target,
            Comment(
                rawValue:
                    "the restore must grow to the snapshot's position before comparing "
                    + "lengths; without that any prefix past "
                    + "\(KVCacheManager.initialCapacityTokens) is refused"))
    }

    /// A short prefix round-trips through save and restore.
    ///
    /// The save/restore path had no test at all, which is how the capacity
    /// asymmetry above survived: nothing exercised it.
    @Test func aSnapshotRoundTripsThroughRestore() throws {
        let (_, saver) = try makeManager(maxContext: 128)
        saver.advance(by: 100)
        let lengths = try saver.snapshotSegmentLengths(at: 100)
        var payload = Data()
        try saver.appendSnapshotPayload(to: &payload, segmentLengths: lengths)
        // Full-attention layers only; each contributes K and V.
        #expect(payload.count == lengths.reduce(0, +))
        // Full-attention layers only, and each contributes K and V.
        let fullLayers = (0..<config.numLayers)
            .filter { saver.layerKind($0) != .linear }.count
        #expect(payload.count == fullLayers * 2 * 100 * saver.stride(layer: 3))

        let (_, receiver) = try makeManager(maxContext: 128)
        var offset = 0
        try payload.withUnsafeBytes { bytes in
            try receiver.restoreSnapshot(
                position: 100, segmentLengths: lengths,
                bytes: bytes, offset: &offset)
        }
        #expect(receiver.position == 100)
        #expect(offset == payload.count)
    }

    @Test func strideAndBufferSizes_matchConfig() throws {
        let (_, kv) = try makeManager(maxContext: 128)

        // Full: numFullKVHeads(2) * fullHeadDim(256) * 2 = 1024 B/token.
        // Linear layers carry no per-token K/V storage at all.
        #expect(kv.kRange(layer: 3, start: 0, count: 1).stride == 2 * 256 * 2)
        #expect(kv.keyBuffer(layer: 3, validTokenCount: 0).length == 128 * 1024)
        #expect(kv.layerKind(0) == .linear)
        #expect(kv.stride(layer: 0) == 0)
        #expect(kv.capacity(layer: 0) == 0)
    }

    @Test func linearGrowth_tracksAdvance() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.position == 0)
        for n in 1...100 {
            kv.advance()
            #expect(kv.position == n)
        }
    }

    /// Full layers run k_norm + RoPE on K while V runs the no-scale norm
    /// without RoPE, so they require separate cache slots.
    @Test func fullLayer_separatesKAndVBuffers() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        let k = kv.keyBuffer(layer: 3, validTokenCount: 0)
        let v = kv.valueBuffer(layer: 3, validTokenCount: 0)
        #expect(k !== v, "full-layer K and V must NOT alias")
        let ks = kv.kSlot(layer: 3, position: 3)
        let vs = kv.vSlot(layer: 3, position: 3)
        #expect(ks.buffer !== vs.buffer, "full-layer K/V slots must NOT alias")
        // Offsets are still per-position-strided in both buffers.
        #expect(ks.offset == vs.offset)
    }

    @Test func slotOffsets_areLinear() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.kSlot(layer: 3, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 3, position: 3).offset == 3 * 1024)
        #expect(kv.vSlot(layer: 3, position: 7).offset == 7 * 1024)
    }

    @Test func fp16Ring_neverEngagesWithoutSWALayers() throws {
        let (_, kv) = try makeManager(
            maxContext: 4096,
            fp16RingEnabled: true)

        #expect(kv.fp16RingEnabled)
        // Full layers stay linear; no SWA layer exists to cap.
        #expect(kv.capacity(layer: 3) == 4096)
        #expect(kv.ringCapacity(layer: 3) == 0)
        #expect(kv.keyBuffer(layer: 3, validTokenCount: 0).length == 4096 * 1024)
        // Linear layers keep no KV rows even with the ring enabled.
        #expect(kv.capacity(layer: 0) == 0)
        #expect(kv.ringCapacity(layer: 0) == 0)
    }

    @Test func fp16Ring_slotOffsetsNeverWrap() throws {
        let (_, kv) = try makeManager(
            maxContext: 128,
            fp16RingEnabled: true)

        // No SWA layer wraps; full-layer slots stay linear within maxContext.
        #expect(kv.kSlot(layer: 3, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 3, position: 127).offset == 127 * 1024)
        #expect(kv.vSlot(layer: 3, position: 35).offset == 35 * 1024)
    }

    @Test func rangeSlotsHaveLinearOffsets() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        let fullStride = kv.kRange(layer: 3, start: 0, count: 1).stride

        let k = kv.kRange(layer: 3, start: 7, count: 3)
        let v = kv.vRange(layer: 7, start: 11, count: 5)

        #expect(k.offset == 7 * fullStride)
        #expect(k.stride == fullStride)
        #expect(v.offset == 11 * fullStride)
        #expect(v.stride == fullStride)
        #expect(k.buffer === kv.keyBuffer(layer: 3, validTokenCount: 0))
        #expect(v.buffer === kv.valueBuffer(layer: 7, validTokenCount: 0))
    }

    @Test func advanceByCountTracksCursor() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 31)
        #expect(kv.position == 31)
        kv.advance(by: 0)
        #expect(kv.position == 31)
        kv.advance()
        #expect(kv.position == 32)
    }

    @Test func reset_clearsPosition() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        for _ in 0..<100 { kv.advance() }
        #expect(kv.position == 100)
        kv.reset()
        #expect(kv.position == 0)
        // Cursor reusable after reset.
        kv.advance()
        #expect(kv.position == 1)
    }

    @Test func speculativeRewindMovesOnlyTheLogicalCursor() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 17)
        let slotBefore = kv.kSlot(layer: 3, position: 12)
        try kv.rewind(to: 12)
        #expect(kv.position == 12)
        let slotAfter = kv.kSlot(layer: 3, position: 12)
        #expect(slotBefore.buffer === slotAfter.buffer)
        #expect(slotBefore.offset == slotAfter.offset)
        #expect(throws: InferenceStateSnapshotError.self) {
            try kv.rewind(to: 13)
        }
    }

    /// One slot is exactly the pre-slot layout: region base 0 and a store sized
    /// for one sequence. This is the invariant that keeps the existing
    /// single-sequence path byte-identical, so it is asserted rather than
    /// assumed.
    @Test func oneSlotKeepsTheOriginalLayout() throws {
        let (_, kv) = try makeManager(maxContext: 128, slots: 1)
        let stride = kv.stride(layer: 3)
        #expect(kv.slots == 1)
        #expect(kv.bufferLength(layer: 3) == kv.capacity(layer: 3) * stride)
        #expect(kv.kSlot(layer: 3, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 3, position: 5).offset == 5 * stride)
        #expect(kv.keyView(layer: 3, slot: 0, validTokenCount: 0).offset == 0)
        #expect(kv.keyView(layer: 3, validTokenCount: 7).offset == 0)
    }

    /// Each slot owns a disjoint region of every layer buffer, one capacity
    /// apart, so a batched decode step cannot alias two sequences' rows.
    @Test func slotsGetDisjointRegions() throws {
        let (_, kv) = try makeManager(maxContext: 128, slots: 4)
        let capacity = kv.capacity(layer: 3)
        let stride = kv.stride(layer: 3)
        #expect(kv.bufferLength(layer: 3) == 4 * capacity * stride)
        for slot in 0..<4 {
            let expected = (slot * capacity + 3) * stride
            #expect(kv.kSlot(layer: 3, position: 3, slot: slot).offset == expected)
            #expect(kv.vSlot(layer: 3, position: 3, slot: slot).offset == expected)
            #expect(
                kv.keyView(layer: 3, slot: slot, validTokenCount: 3).offset
                    == slot * capacity * stride)
        }
    }

    /// Slot cursors advance, rewind and reset independently.
    @Test func slotCursorsAreIndependent() throws {
        let (_, kv) = try makeManager(maxContext: 128, slots: 2)
        kv.advance(by: 7)
        kv.advance(slot: 1, by: 3)
        #expect(kv.position == 7)
        #expect(kv.position(slot: 0) == 7)
        #expect(kv.position(slot: 1) == 3)

        kv.reset(slot: 1)
        #expect(kv.position(slot: 1) == 0)
        #expect(kv.position(slot: 0) == 7, "resetting one slot must not move another")

        try kv.rewind(slot: 0, to: 2)
        #expect(kv.position(slot: 0) == 2)
        #expect(kv.position(slot: 1) == 0)
    }

    /// Growth reallocates one shared store, so it must carry every slot's live
    /// rows to each slot's new region — not just the one that asked to grow.
    @Test func growthPreservesEverySlotsRows() throws {
        let (_, kv) = try makeManager(maxContext: 9_000, slots: 2)
        let stride = kv.stride(layer: 3)
        let before = kv.capacity(layer: 3)
        // Give each slot one live row: growth copies `min(capacity, cursor)`
        // rows per slot, so a marker in a slot with a zero cursor is correctly
        // dropped rather than carried.
        kv.advance(slot: 0, by: 1)
        kv.advance(slot: 1, by: 1)
        let s0 = kv.kSlot(layer: 3, position: 0, slot: 0)
        let s1 = kv.kSlot(layer: 3, position: 0, slot: 1)
        #expect(s1.offset - s0.offset == before * stride)
        memset(s0.buffer.contents() + s0.offset, 0xA0, 1)
        memset(s1.buffer.contents() + s1.offset, 0xB0, 1)

        try kv.reserve(tokens: 8_500, slot: 0)
        let after = kv.capacity(layer: 3)
        #expect(after > before, "precondition: the reserve grew the store")
        let r0 = kv.kSlot(layer: 3, position: 0, slot: 0)
        let r1 = kv.kSlot(layer: 3, position: 0, slot: 1)
        #expect(r1.offset - r0.offset == after * stride)
        #expect(r0.buffer.contents().load(fromByteOffset: r0.offset, as: UInt8.self) == 0xA0)
        #expect(r1.buffer.contents().load(fromByteOffset: r1.offset, as: UInt8.self) == 0xB0)
    }

}
