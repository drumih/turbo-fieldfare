import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// Tests `KVCacheManager` FP16 shape, growth, separate K/V storage, ring,
/// and reset semantics against the Gemma 4 config.
@Suite struct KVCacheManagerTests {

    private let config = ArchConfig.gemma4_26B_A4B

    private func makeManager(maxContext: Int,
                             fp16RingEnabled: Bool = false,
                             maxPrefillChunkTokens: Int = 128,
                             fp16RingCapacityOverride: Int? = nil) throws -> (MetalContext, KVCacheManager) {
        let ctx = try MetalContext()
        let kv = try KVCacheManager(device: ctx.device,
                                    config: config,
                                    maxContext: maxContext,
                                    fp16RingEnabled: fp16RingEnabled,
                                    slidingWindow: config.slidingWindow,
                                    maxPrefillChunkTokens: maxPrefillChunkTokens,
                                    fp16RingCapacityOverride: fp16RingCapacityOverride)
        return (ctx, kv)
    }

    /// The chunk floor `RealForwardRunner` applies in production, which is what
    /// makes the ring 1,304 rows rather than the 1,152 a bare 128-token chunk
    /// would give.
    private var productionChunkTokens: Int { VisionConfig().maximumPooledTokens }


    @Test func strideAndBufferSizes_matchConfig() throws {
        let (_, kv) = try makeManager(maxContext: 128)

        // SWA: numKVHeads(8) * headDim(256) * 2 = 4096 B/token.
        // Full: numFullKVHeads(2) * fullHeadDim(512) * 2 = 2048 B/token.
        #expect(kv.kRange(layer: 0, start: 0, count: 1).stride == 8 * 256 * 2)
        #expect(kv.kRange(layer: 5, start: 0, count: 1).stride == 2 * 512 * 2)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0).length == 128 * 4096)
        #expect(kv.keyBuffer(layer: 5, validTokenCount: 0).length == 128 * 2048)
    }

    @Test func linearGrowth_tracksAdvance() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.position == 0)
        for n in 1...100 {
            kv.advance()
            #expect(kv.position == n)
        }
    }

    /// Full layers share the raw k_proj output, then diverge: K runs k_norm +
    /// RoPE while V runs no-scale v_norm without RoPE. They therefore require
    /// separate cache slots.
    @Test func fullLayer_separatesKAndVBuffers() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        let k = kv.keyBuffer(layer: 5, validTokenCount: 0)
        let v = kv.valueBuffer(layer: 5, validTokenCount: 0)
        #expect(k !== v, "full-layer K and V must NOT alias")
        let ks = kv.kSlot(layer: 5, position: 3)
        let vs = kv.vSlot(layer: 5, position: 3)
        #expect(ks.buffer !== vs.buffer, "full-layer K/V slots must NOT alias")
        // Offsets are still per-position-strided in both buffers.
        #expect(ks.offset == vs.offset)
    }

    @Test func swaLayer_hasSeparateKVBuffers() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0)
                !== kv.valueBuffer(layer: 0, validTokenCount: 0))
    }

    @Test func slotOffsets_areLinear() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.kSlot(layer: 0, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 0, position: 3).offset == 3 * 4096)
        #expect(kv.vSlot(layer: 5, position: 7).offset == 7 * 2048)
    }

    @Test func fp16Ring_capsSWALayersAndLeavesFullLayersLinear() throws {
        let (_, kv) = try makeManager(maxContext: 4096,
                                      fp16RingEnabled: true)

        #expect(kv.fp16RingEnabled)
        #expect(kv.capacity(layer: 0) == 1152)
        #expect(kv.ringCapacity(layer: 0) == 1152)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0).length == 1152 * 4096)
        #expect(kv.capacity(layer: 5) == 4096)
        #expect(kv.ringCapacity(layer: 5) == 0)
        #expect(kv.keyBuffer(layer: 5, validTokenCount: 0).length == 4096 * 2048)
    }

    @Test func fp16Ring_shortSessionCapsSWAToMaxContext() throws {
        let (_, kv) = try makeManager(maxContext: 256,
                                      fp16RingEnabled: true)

        #expect(kv.fp16RingEnabled)
        #expect(kv.capacity(layer: 0) == 256)
        #expect(kv.ringCapacity(layer: 0) == 256)
        #expect(kv.keyBuffer(layer: 0, validTokenCount: 0).length == 256 * 4096)
        #expect(kv.capacity(layer: 5) == 256)
        #expect(kv.ringCapacity(layer: 5) == 0)
        #expect(kv.keyBuffer(layer: 5, validTokenCount: 0).length == 256 * 2048)
    }

    @Test func fp16Ring_slotOffsetsWrapOnlyForSWALayers() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      fp16RingEnabled: true,
                                      fp16RingCapacityOverride: 32)

        #expect(kv.kSlot(layer: 0, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 0, position: 31).offset == 31 * 4096)
        #expect(kv.kSlot(layer: 0, position: 32).offset == 0)
        #expect(kv.vSlot(layer: 0, position: 35).offset == 3 * 4096)

        #expect(kv.kSlot(layer: 5, position: 35).offset == 35 * 2048)
        #expect(kv.vSlot(layer: 5, position: 35).offset == 35 * 2048)
    }

    @Test func fp16Ring_rangesMustNotWrap() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      fp16RingEnabled: true,
                                      fp16RingCapacityOverride: 32)

        let k = kv.kRange(layer: 0, start: 28, count: 4)
        #expect(k.offset == 28 * 4096)
        let v = kv.vRange(layer: 0, start: 32, count: 3)
        #expect(v.offset == 0)
    }

    @Test func rangeSlotsHaveLinearOffsets() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        let swaStride = kv.kRange(layer: 0, start: 0, count: 1).stride
        let fullStride = kv.vRange(layer: 5, start: 0, count: 1).stride

        let k = kv.kRange(layer: 0, start: 7, count: 3)
        let v = kv.vRange(layer: 5, start: 11, count: 5)

        #expect(k.offset == 7 * swaStride)
        #expect(k.stride == swaStride)
        #expect(v.offset == 11 * fullStride)
        #expect(v.stride == fullStride)
        #expect(k.buffer === kv.keyBuffer(layer: 0, validTokenCount: 0))
        #expect(v.buffer === kv.valueBuffer(layer: 5, validTokenCount: 0))
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

    @Test func rewindMovesCursorWithinRingSlack() throws {
        let (_, kv) = try makeManager(maxContext: 4_096, fp16RingEnabled: true)
        kv.advance(by: 2_000)
        #expect(kv.maxRewindTokens == 128)
        kv.rewind(to: 1_990)
        #expect(kv.position == 1_990)
    }

    @Test func rewindIsUnboundedForLinearStorage() throws {
        let (_, kv) = try makeManager(maxContext: 64)
        kv.advance(by: 50)
        #expect(kv.maxRewindTokens == 64)
        kv.rewind(to: 1)
        #expect(kv.position == 1)
    }

    @Test func rewindBoundsAgainstHighWaterMarkNotCursor() throws {
        let (_, kv) = try makeManager(maxContext: 4_096, fp16RingEnabled: true)
        kv.advance(by: 1_280)
        kv.rewind(to: 1_160)
        #expect(kv.highWaterPosition == 1_280)
        kv.rewind(to: 1_155)
        #expect(kv.position == 1_155)
        kv.advance(by: 200)
        #expect(kv.highWaterPosition == 1_355)
    }

    @Test func rewindShortSessionRingNeverWrapsSoDepthIsFree() throws {
        let (_, kv) = try makeManager(maxContext: 16, fp16RingEnabled: true)
        kv.advance(by: 10)
        #expect(kv.maxRewindTokens == 16)
        kv.rewind(to: 2)
        #expect(kv.position == 2)
    }

    /// The ceiling is the checkpoint's `max_position_embeddings`, and it is
    /// checked before the first `makeBuffer`: at 262,145 the allocation loop
    /// would ask for 5.4 GiB of full-layer KV for positions the model has no
    /// RoPE frequencies for.
    @Test func contextAboveModelMaximumIsRefusedBeforeAllocation() throws {
        do {
            _ = try makeManager(maxContext: 262_145, fp16RingEnabled: true)
            Issue.record("262,145 was accepted above the model maximum 262,144")
        } catch let error as ModelError {
            #expect(error == .contextExceedsModel(requested: 262_145, maximum: 262_144))
            #expect(error.description.contains("262145"))
            #expect(error.description.contains("262144"))
        }
    }

    /// The model maximum itself constructs. Gated on host memory: the full
    /// layers reserve 5.4 GiB of address space, which is not something CI or
    /// the 8 GB dev Mac should be asked for.
    @Test func modelMaximumContextAllocates() throws {
        guard ProcessInfo.processInfo.physicalMemory >= 16 << 30 else { return }
        let (_, kv) = try makeManager(maxContext: 262_144,
                                      fp16RingEnabled: true,
                                      maxPrefillChunkTokens: productionChunkTokens)
        #expect(kv.capacity(layer: 5) == 262_144)
        #expect(kv.capacity(layer: 0)
                == ContextAdmission.productionSlidingRingRows(config: config,
                                                              maxContext: 262_144))
        #expect(kv.capacity(layer: 0) == 1_304)
        #expect(kv.bufferLength(layer: 5) == 262_144 * 2_048)
    }

    private func writeMarker(_ kv: KVCacheManager, position: Int, value: UInt16) {
        for layer in 0..<config.numLayers {
            let key = kv.kSlot(layer: layer, position: position)
            let value16 = kv.vSlot(layer: layer, position: position)
            key.buffer.contents().advanced(by: key.offset).storeBytes(of: value, as: UInt16.self)
            value16.buffer.contents().advanced(by: value16.offset)
                .storeBytes(of: value &+ 1, as: UInt16.self)
        }
    }

    private func readMarkers(_ kv: KVCacheManager, layer: Int, position: Int) -> (UInt16, UInt16) {
        let key = kv.kSlot(layer: layer, position: position)
        let value = kv.vSlot(layer: layer, position: position)
        return (key.buffer.contents().advanced(by: key.offset).load(as: UInt16.self),
                value.buffer.contents().advanced(by: value.offset).load(as: UInt16.self))
    }

    /// A prefix far past the ring slack comes back intact after a longer
    /// continuation and a whole other conversation overwrote it, which is the
    /// case `rewind` cannot serve.
    @Test func prefixSnapshotSurvivesOverwritesThatRewindCannotUndo() throws {
        let (_, kv) = try makeManager(maxContext: 4_096,
                                      fp16RingEnabled: true,
                                      maxPrefillChunkTokens: productionChunkTokens)
        let prefixLength = 2_000
        for position in 0..<prefixLength {
            writeMarker(kv, position: position, value: UInt16(position % 60_000))
            kv.advance()
        }
        let snapshot = try kv.snapshotPrefix()
        #expect(snapshot.position == prefixLength)
        // Sliding layers keep only their 1,024-row window, not the whole ring.
        #expect(snapshot.byteCount == 25 * config.slidingWindow * 4_096 * 2 + 5 * prefixLength * 2_048 * 2)

        for position in prefixLength..<3_000 {
            writeMarker(kv, position: position, value: 61_000)
            kv.advance()
        }
        kv.reset()
        for position in 0..<prefixLength {
            writeMarker(kv, position: position, value: 62_000)
            kv.advance()
        }

        kv.restorePrefix(snapshot)
        #expect(kv.position == prefixLength)
        #expect(kv.highWaterPosition == prefixLength)
        let windowStart = prefixLength - config.slidingWindow
        for position in [windowStart, 1_500, prefixLength - 1] {
            let expected = UInt16(position % 60_000)
            #expect(readMarkers(kv, layer: 0, position: position) == (expected, expected &+ 1))
        }
        for position in [0, 977, prefixLength - 1] {
            let expected = UInt16(position % 60_000)
            #expect(readMarkers(kv, layer: 5, position: position) == (expected, expected &+ 1))
        }
    }

    @Test func reset_clearsPosition() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        for _ in 0..<100 { kv.advance() }
        #expect(kv.position == 100)
        kv.reset()
        #expect(kv.position == 0)
        #expect(kv.highWaterPosition == 0)
        // Cursor reusable after reset.
        kv.advance()
        #expect(kv.position == 1)
    }


}
