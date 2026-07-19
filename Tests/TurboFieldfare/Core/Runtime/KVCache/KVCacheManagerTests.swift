import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// Tests `KVCacheManager` shape, growth, separate K/V storage, FP16 SWA ring,
/// packed storage, and reset semantics against the Gemma 4 config.
@Suite struct KVCacheManagerTests {

    private let config = ArchConfig.gemma4_26B_A4B

    private func makeManager(maxContext: Int,
                             storageMode: KVCacheStorageMode = .fp16,
                             fp16RingEnabled: Bool = false,
                             fp16RingCapacityOverride: Int? = nil) throws -> (MetalContext, KVCacheManager) {
        let ctx = try MetalContext()
        let kv = try KVCacheManager(device: ctx.device,
                                    config: config,
                                    maxContext: maxContext,
                                    storageMode: storageMode,
                                    fp16RingEnabled: fp16RingEnabled,
                                    slidingWindow: config.slidingWindow,
                                    maxPrefillChunkTokens: 128,
                                    fp16RingCapacityOverride: fp16RingCapacityOverride)
        return (ctx, kv)
    }

    @Test func defaultStorageMode_isFP16() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        #expect(kv.storageMode == .fp16)
        #expect(kv.turboQuantLayout(layer: 0) == nil)
    }

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

    @Test func turboQuantStorage_usesPackedLayerLayouts() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      storageMode: .turboQuant(.k4v4NormCorrected))

        #expect(kv.storageMode == .turboQuant(.k4v4NormCorrected))

        let swa = try #require(kv.turboQuantLayout(layer: 0))
        #expect(swa.key.bytesPerToken == 1_040)
        #expect(swa.value.bytesPerToken == 1_040)
        #expect(kv.quantizedKeyBuffer(layer: 0, validTokenCount: 0).length == 128 * 1_040)
        #expect(kv.quantizedValueBuffer(layer: 0, validTokenCount: 0).length == 128 * 1_040)

        let full = try #require(kv.turboQuantLayout(layer: 5))
        #expect(full.key.bytesPerToken == 516)
        #expect(full.value.bytesPerToken == 516)
        #expect(kv.quantizedKeyBuffer(layer: 5, validTokenCount: 0).length == 128 * 516)
        #expect(kv.quantizedValueBuffer(layer: 5, validTokenCount: 0).length == 128 * 516)
    }

    @Test func turboQuantSlots_areLinearByRoleStride() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      storageMode: .turboQuant(.k4v4NormCorrected))

        let swa = try #require(kv.turboQuantLayout(layer: 0))
        #expect(swa.key.bytesPerToken == 1_040)
        #expect(swa.value.bytesPerToken == 1_040)

        let k0 = kv.quantizedKeySlot(layer: 0, position: 3)
        let v0 = kv.quantizedValueSlot(layer: 0, position: 3)
        #expect(k0.offset == 3 * swa.key.bytesPerToken)
        #expect(v0.offset == 3 * swa.value.bytesPerToken)
        #expect(k0.buffer !== v0.buffer)
    }

    @Test func turboQuantReset_clearsPosition() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      storageMode: .turboQuant(.k4v4NormCorrected))
        for _ in 0..<9 { kv.advance() }
        #expect(kv.position == 9)
        kv.reset()
        #expect(kv.position == 0)
    }
}
